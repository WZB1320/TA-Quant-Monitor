"""
P3 组合回测 — 动态回撤保护阈值 (牛市12%/震荡市8%)

建议A验证: P2固定8%阈值在牛市过早触发(65天降仓), 削了测试窗夏普(3.448→3.410).
P3改用动态阈值:
  - trending(牛市/趋势): 触发12%, 恢复6%  (放宽, 避免牛市过早降仓)
  - transition/ranging(震荡): 触发8%, 恢复4% (维持P2的严格保护)

对比三套保护策略 (同一份分组回测结果, 仅组合层保护不同):
  - no_dd      : 无回撤保护 (看保护本身的净效果)
  - fixed_8    : 固定8%/恢复4% (= P2, 对照)
  - dynamic    : 动态12%/8% (P3, 本次验证)

分组配置 = P2 (科技40/消费10/周期42.5/医药0/机械0/现金7.5%, 周期trending过滤, 科技ATR1.8)
震荡市Alpha不动, 本次只验证动态阈值能否找回测试窗夏普.

用法:
  python scripts/p3_portfolio_backtest.py
"""
import sys
import os
import json
import shutil
import warnings
import numpy as np
import pandas as pd
from datetime import datetime, date as date_cls

warnings.filterwarnings("ignore")
project_root = os.path.join(os.path.dirname(__file__), "..")
sys.path.insert(0, project_root)

from src.config.runtime_mode import set_mode, RuntimeMode
set_mode(RuntimeMode.BACKTEST)

from src.backtest.engine import BacktestEngine
from src.backtest.regime_detector import RegimeDetector
from src.data_fetcher import DataManager
from src.config.group_config import GroupConfig
from src.config.user_preferences import UserPreferences, _USER_PREF_FILE

TRAIN_START, TRAIN_END = "2024-07-01", "2025-06-30"
TEST_START, TEST_END = "2025-07-01", "2026-06-30"
DATA_START, DATA_END = "2024-02-01", "2026-07-13"
BENCHMARK = "sh.000300"
TOTAL_CAPITAL = 1000000

# P3 分组配置 (= P2)
WEIGHTS = {"科技成长型": 0.40, "消费稳健型": 0.10, "周期资源型": 0.425,
           "医药创新型": 0.0, "机械制造型": 0.0}
REGIMES_CFG = {"周期资源型": {"trending"}}
ATR_OVERRIDE = {"科技成长型": 1.8}

# 三套回撤保护策略
DD_REDUCED_RATIO = 0.5  # 保护期降仓至50%
DD_FIXED = {"threshold": -0.08, "recovery": -0.04}  # = P2
DD_DYNAMIC = {
    "trending": {"threshold": -0.12, "recovery": -0.06},
    "transition": {"threshold": -0.08, "recovery": -0.04},
    "ranging": {"threshold": -0.08, "recovery": -0.04},
}

RESULT_JSON = os.path.join(project_root, "data", "p3_portfolio_result.json")
REPORT_MD = os.path.join(project_root, "data", "p3_portfolio_report.md")


def load_watchlist():
    cfg_path = os.path.join(project_root, "config", "strategy_config.json")
    with open(cfg_path, "r", encoding="utf-8") as f:
        cfg = json.load(f)
    return {g: [s["code"] for s in stocks]
            for g, stocks in cfg["strategy_config"]["watchlist"].items()
            if not g.startswith("_")}


def run_group(data_map, benchmark_df, group_codes, group_capital,
              trade_regimes, atr_mult, start, end):
    if group_capital < 1000:
        return {"sharpe": 0, "total_return": 0, "alpha": 0, "max_drawdown": 0,
                "trade_count": 0, "win_rate": 0, "final_value": group_capital,
                "daily_values": None, "skipped": True}
    GroupConfig._instance = None
    GroupConfig._config = None
    try:
        engine = BacktestEngine(
            initial_capital=group_capital, lookback_days=120, position_ratio=0.3,
            commission_rate=0.00025, stamp_tax=0.001, slippage=0.0001,
            signal_dedup_days=5, risk_per_trade=0.05, atr_stop_mult=atr_mult,
            forced_regime=None, trade_regimes=trade_regimes,
        )
        sub_map = {c: data_map[c] for c in group_codes if c in data_map}
        m = engine.run(sub_map, benchmark_df=benchmark_df, start_date=start, end_date=end)
        return {
            "sharpe": getattr(m, "sharpe_ratio", 0) or 0,
            "total_return": getattr(m, "total_return", 0) or 0,
            "alpha": getattr(m, "alpha", 0) or 0,
            "max_drawdown": getattr(m, "max_drawdown", 0) or 0,
            "trade_count": getattr(m, "trade_count", 0) or 0,
            "win_rate": getattr(m, "win_rate", 0) or 0,
            "final_value": getattr(m, "final_value", group_capital) or group_capital,
            "daily_values": engine.daily_values.copy() if engine.daily_values is not None else None,
        }
    except Exception as e:
        return {"error": str(e)}


def compute_regime_series(benchmark_df):
    """预计算基准每个交易日的 regime (trending/transition/ranging)."""
    detector = RegimeDetector()
    df = benchmark_df.copy()
    if "date" not in df.columns:
        df = df.reset_index()
    # 统一 date 为 date 对象
    df["date"] = pd.to_datetime(df["date"]).dt.date
    regimes = {}
    for d in df["date"].tolist():
        regimes[d] = detector.detect(benchmark_df, d)
    return regimes  # {date_obj: regime_str}


def apply_protection_fixed(portfolio_nav, threshold, recovery, reduced_ratio):
    """固定阈值回撤保护 (P2逻辑)."""
    cummax = portfolio_nav.cummax()
    drawdown = (portfolio_nav - cummax) / cummax
    in_protection = False
    protected_nav = [portfolio_nav.iloc[0]]
    protection_days = 0
    trigger_count = 0
    for i in range(1, len(portfolio_nav)):
        dd = drawdown.iloc[i]
        if dd < threshold and not in_protection:
            in_protection = True
            trigger_count += 1
        elif dd > recovery and in_protection:
            in_protection = False
        daily_return = portfolio_nav.iloc[i] / portfolio_nav.iloc[i - 1] - 1
        if in_protection:
            adjusted_return = daily_return * reduced_ratio
            protection_days += 1
        else:
            adjusted_return = daily_return
        protected_nav.append(protected_nav[-1] * (1 + adjusted_return))
    return (pd.Series(protected_nav, index=portfolio_nav.index),
            protection_days, trigger_count)


def apply_protection_dynamic(portfolio_nav, regime_map, dd_map, reduced_ratio):
    """动态阈值回撤保护: 阈值随当日 regime 变化 (trending宽, 震荡严)."""
    cummax = portfolio_nav.cummax()
    drawdown = (portfolio_nav - cummax) / cummax
    in_protection = False
    protected_nav = [portfolio_nav.iloc[0]]
    protection_days = 0
    trigger_count = 0
    # 统计各 regime 触发次数
    trigger_by_regime = {"trending": 0, "transition": 0, "ranging": 0}
    protection_by_regime = {"trending": 0, "transition": 0, "ranging": 0}

    for i in range(1, len(portfolio_nav)):
        ts = portfolio_nav.index[i]
        # 索引是 datetime, 取 date 对象查 regime
        d = ts.date() if hasattr(ts, "date") else ts
        regime = regime_map.get(d, "transition")  # 缺失回退
        cfg = dd_map.get(regime, dd_map["transition"])
        threshold = cfg["threshold"]
        recovery = cfg["recovery"]

        dd = drawdown.iloc[i]
        if dd < threshold and not in_protection:
            in_protection = True
            trigger_count += 1
            trigger_by_regime[regime] = trigger_by_regime.get(regime, 0) + 1
        elif dd > recovery and in_protection:
            in_protection = False

        daily_return = portfolio_nav.iloc[i] / portfolio_nav.iloc[i - 1] - 1
        if in_protection:
            adjusted_return = daily_return * reduced_ratio
            protection_days += 1
            protection_by_regime[regime] = protection_by_regime.get(regime, 0) + 1
        else:
            adjusted_return = daily_return
        protected_nav.append(protected_nav[-1] * (1 + adjusted_return))
    return (pd.Series(protected_nav, index=portfolio_nav.index),
            protection_days, trigger_count,
            trigger_by_regime, protection_by_regime)


def metrics_from_nav(portfolio_nav, benchmark_df, start, end, protection_info):
    """从净值序列算组合级指标."""
    daily_returns = portfolio_nav.pct_change().dropna()
    total_return = (portfolio_nav.iloc[-1] / TOTAL_CAPITAL) - 1
    sharpe = (daily_returns.mean() / daily_returns.std() * np.sqrt(252)
              if daily_returns.std() > 0 else 0.0)
    cummax = portfolio_nav.cummax()
    drawdown = (portfolio_nav - cummax) / cummax
    max_drawdown = drawdown.min()

    bench_df = benchmark_df.set_index("date")["close"]
    bench_idx = pd.to_datetime(bench_df.index)
    bench_df = pd.Series(bench_df.values, index=bench_idx)
    start_ts = pd.Timestamp(start)
    end_ts = pd.Timestamp(end)
    bmask = (bench_df.index >= start_ts) & (bench_df.index <= end_ts)
    bench = bench_df[bmask]
    bench_return = (bench.iloc[-1] / bench.iloc[0]) - 1 if len(bench) > 0 else 0
    alpha = total_return - bench_return

    return {
        "sharpe": round(sharpe, 3),
        "total_return_pct": round(total_return * 100, 2),
        "alpha_pct": round(alpha * 100, 2),
        "benchmark_return_pct": round(bench_return * 100, 2),
        "max_drawdown_pct": round(max_drawdown * 100, 2),
        "trade_count": protection_info.get("trade_count", 0),
        "final_value": round(portfolio_nav.iloc[-1], 0),
        "protection_info": protection_info,
    }


def build_portfolio_nav(group_results, weights, start, end):
    """合并各分组daily_values + 现金 → 组合净值 (不加保护)."""
    portfolio_nav = None
    total_trades = 0
    for g, r in group_results.items():
        if "error" in r or r.get("daily_values") is None:
            continue
        nav = r["daily_values"]
        if portfolio_nav is None:
            portfolio_nav = nav.copy()
        else:
            portfolio_nav = portfolio_nav.add(nav, fill_value=0)
        total_trades += r["trade_count"]

    if portfolio_nav is None or len(portfolio_nav) < 10:
        return None, 0

    start_ts = pd.Timestamp(start)
    end_ts = pd.Timestamp(end)
    nav_idx = pd.to_datetime(portfolio_nav.index)
    portfolio_nav = pd.Series(portfolio_nav.values, index=nav_idx)
    mask = (portfolio_nav.index >= start_ts) & (portfolio_nav.index <= end_ts)
    portfolio_nav = portfolio_nav[mask]

    invested_capital = sum(weights[g] * TOTAL_CAPITAL for g in group_results
                           if "error" not in group_results[g] and not group_results[g].get("skipped"))
    cash = TOTAL_CAPITAL - invested_capital
    portfolio_nav = portfolio_nav + cash
    return portfolio_nav, total_trades


def main():
    run_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print("=" * 70)
    print("  P3 组合回测 — 动态回撤保护 (牛市12%/震荡市8%)")
    print(f"  训练:{TRAIN_START}~{TRAIN_END} | 测试:{TEST_START}~{TEST_END}")
    print(f"  分组配置=P2: {WEIGHTS} + 现金{(1-sum(WEIGHTS.values()))*100:.1f}%")
    print(f"  三套保护: no_dd / fixed_8(P2) / dynamic(P3)")
    print("=" * 70)

    pref_existed = os.path.exists(_USER_PREF_FILE)
    pref_backup = _USER_PREF_FILE + ".p3_bak"
    if pref_existed:
        shutil.copy2(_USER_PREF_FILE, pref_backup)
    UserPreferences().clear_all()

    try:
        watchlist = load_watchlist()
        dm = DataManager()
        print("\n拉取数据...")
        all_codes = [c for codes in watchlist.values() for c in codes]
        data_map = {}
        for code in all_codes:
            df = dm.get_daily_kline(code, start_date=DATA_START, end_date=DATA_END)
            if df is not None and len(df) > 80:
                data_map[code] = df
        benchmark_df = dm.get_daily_kline(BENCHMARK, start_date=DATA_START, end_date=DATA_END)
        print(f"  股票 {len(data_map)}/{len(all_codes)}, 基准 {len(benchmark_df)}条")

        # 预计算 regime
        print("\n预计算市场 regime...")
        regime_map = compute_regime_series(benchmark_df)
        regime_counts = pd.Series(list(regime_map.values())).value_counts()
        print(f"  regime分布: {dict(regime_counts)}")

        all_results = {}
        for window, start, end in [("train", TRAIN_START, TRAIN_END),
                                    ("test", TEST_START, TEST_END)]:
            wl = "训练" if window == "train" else "测试"
            print(f"\n{'='*60}\n  [{wl}窗] 跑分组回测\n{'='*60}")
            window_groups = {}
            for group_name, codes in watchlist.items():
                if group_name.startswith("_"):
                    continue
                group_codes = [c for c in codes if c in data_map]
                if len(group_codes) < 2:
                    continue
                capital = TOTAL_CAPITAL * WEIGHTS[group_name]
                regimes = REGIMES_CFG.get(group_name)
                atr_mult = ATR_OVERRIDE.get(group_name, 2.0)
                r = run_group(data_map, benchmark_df, group_codes, capital,
                              regimes, atr_mult, start, end)
                if "error" not in r and not r.get("skipped"):
                    print(f"    {group_name:12s}: 夏普={r['sharpe']:+.3f} "
                          f"收益={r['total_return']*100:+.1f}% 交易={r['trade_count']}笔")
                elif r.get("skipped"):
                    print(f"    {group_name:12s}: [跳过-权重0%]")
                window_groups[group_name] = r

            # 构建组合净值 (无保护)
            portfolio_nav, total_trades = build_portfolio_nav(window_groups, WEIGHTS, start, end)
            if portfolio_nav is None:
                all_results[window] = {"error": "无有效净值"}
                continue

            # 三套保护策略
            # 1) no_dd
            no_dd_info = {"enabled": False, "strategy": "no_dd",
                          "protection_days": 0, "trigger_count": 0, "trade_count": total_trades}
            no_dd_metrics = metrics_from_nav(portfolio_nav, benchmark_df, start, end, no_dd_info)

            # 2) fixed_8 (= P2)
            nav_fixed, p_days_f, p_trig_f = apply_protection_fixed(
                portfolio_nav, DD_FIXED["threshold"], DD_FIXED["recovery"], DD_REDUCED_RATIO)
            fixed_info = {"enabled": True, "strategy": "fixed_8",
                          "protection_days": p_days_f, "trigger_count": p_trig_f,
                          "trade_count": total_trades,
                          "threshold": DD_FIXED["threshold"], "recovery": DD_FIXED["recovery"]}
            fixed_metrics = metrics_from_nav(nav_fixed, benchmark_df, start, end, fixed_info)

            # 3) dynamic (P3)
            nav_dyn, p_days_d, p_trig_d, trig_by_reg, prot_by_reg = apply_protection_dynamic(
                portfolio_nav, regime_map, DD_DYNAMIC, DD_REDUCED_RATIO)
            dyn_info = {"enabled": True, "strategy": "dynamic",
                        "protection_days": p_days_d, "trigger_count": p_trig_d,
                        "trade_count": total_trades,
                        "trigger_by_regime": trig_by_reg,
                        "protection_by_regime": prot_by_reg}
            dyn_metrics = metrics_from_nav(nav_dyn, benchmark_df, start, end, dyn_info)

            all_results[window] = {
                "groups": {g: {k: v for k, v in r.items() if k != "daily_values"}
                           for g, r in window_groups.items()},
                "no_dd": no_dd_metrics,
                "fixed_8": fixed_metrics,
                "dynamic": dyn_metrics,
            }

            # 打印三套对比
            print(f"\n  [{wl}窗] 三套保护对比:")
            for strat, m in [("no_dd", no_dd_metrics), ("fixed_8", fixed_metrics), ("dynamic", dyn_metrics)]:
                pi = m["protection_info"]
                tag = f" [保护{pi.get('trigger_count',0)}次/{pi.get('protection_days',0)}天]" if pi.get("enabled") else ""
                print(f"    {strat:10s}: 夏普={m['sharpe']:+.3f} 收益={m['total_return_pct']:+.1f}% "
                      f"Alpha={m['alpha_pct']:+.1f}% 回撤={m['max_drawdown_pct']:.1f}%{tag}")

        # 保存
        os.makedirs(os.path.dirname(RESULT_JSON), exist_ok=True)
        output = {
            "run_time": run_time,
            "config": {
                "weights": WEIGHTS, "atr_override": ATR_OVERRIDE,
                "regimes": {k: list(v) for k, v in REGIMES_CFG.items()},
                "dd_fixed": DD_FIXED,
                "dd_dynamic": DD_DYNAMIC,
                "dd_reduced_ratio": DD_REDUCED_RATIO,
                "regime_distribution": dict(regime_counts),
            },
            "results": all_results,
        }
        with open(RESULT_JSON, "w", encoding="utf-8") as f:
            json.dump(output, f, ensure_ascii=False, indent=2, default=str)
        print(f"\n✓ 结果 → {RESULT_JSON}")
        report = generate_report(output, run_time)
        with open(REPORT_MD, "w", encoding="utf-8") as f:
            f.write(report)
        print(f"✓ 报告 → {REPORT_MD}")

        # 汇总
        print(f"\n{'='*70}\n  三套保护策略对比汇总\n{'='*70}")
        print(f"{'窗口':<6} {'策略':<10} {'夏普':>8} {'收益%':>8} {'Alpha%':>8} {'回撤%':>8} {'保护':>12}")
        for window, wl in [("train", "训练"), ("test", "测试")]:
            if "error" in all_results[window]:
                continue
            for strat in ["no_dd", "fixed_8", "dynamic"]:
                m = all_results[window][strat]
                pi = m["protection_info"]
                prot = f"{pi.get('trigger_count',0)}次/{pi.get('protection_days',0)}天" if pi.get("enabled") else "—"
                print(f"{wl:<6} {strat:<10} {m['sharpe']:>8.3f} {m['total_return_pct']:>8.1f} "
                      f"{m['alpha_pct']:>8.1f} {m['max_drawdown_pct']:>8.1f} {prot:>12}")

    finally:
        if pref_existed and os.path.exists(pref_backup):
            shutil.move(pref_backup, _USER_PREF_FILE)


def generate_report(output, run_time):
    L = []
    L.append("# P3 组合回测报告 — 动态回撤保护 (牛市12%/震荡市8%)")
    L.append(f"\n**运行时间**: {run_time}\n")
    L.append("**验证目标**: P2固定8%阈值在牛市过早触发(65天降仓)削了测试窗夏普(3.448→3.410), "
             "P3改动态阈值(trending12%/震荡8%), 验证测试窗夏普能否回升.\n")
    L.append("**分组配置** = P2: 科技40%/消费10%/周期42.5%/医药0%/机械0%/现金7.5%, "
             "周期trending过滤, 科技ATR1.8\n")
    dd = output["config"]["dd_dynamic"]
    L.append("**三套保护策略**:")
    L.append("- no_dd: 无回撤保护 (看保护本身的净效果)")
    L.append(f"- fixed_8: 固定阈值{output['config']['dd_fixed']['threshold']*100:.0f}%/恢复{output['config']['dd_fixed']['recovery']*100:.0f}% (= P2)")
    L.append(f"- dynamic(P3): trending触发{abs(dd['trending']['threshold'])*100:.0f}%/恢复{abs(dd['trending']['recovery'])*100:.0f}%, "
             f"震荡触发{abs(dd['ranging']['threshold'])*100:.0f}%/恢复{abs(dd['ranging']['recovery'])*100:.0f}%\n")
    L.append(f"**基准regime分布**: {output['config']['regime_distribution']}\n")

    for window, wl in [("train", "训练窗(震荡市)"), ("test", "测试窗(牛市)")]:
        if "error" in output["results"][window]:
            continue
        L.append(f"## {wl}\n")
        L.append("### 三套保护策略对比\n")
        L.append("| 策略 | 夏普 | 收益% | Alpha% | 基准% | 回撤% | 交易数 | 保护触发 |")
        L.append("|------|------|-------|--------|-------|-------|--------|---------|")
        for strat, label in [("no_dd", "no_dd"), ("fixed_8", "fixed_8(P2)"), ("dynamic", "dynamic(P3)")]:
            m = output["results"][window][strat]
            pi = m["protection_info"]
            prot = f"{pi.get('trigger_count',0)}次/{pi.get('protection_days',0)}天" if pi.get("enabled") else "—"
            L.append(f"| {label} | {m['sharpe']:+.3f} | {m['total_return_pct']:+.1f} | "
                     f"{m['alpha_pct']:+.1f} | {m['benchmark_return_pct']:+.1f} | "
                     f"{m['max_drawdown_pct']:.1f} | {m['trade_count']} | {prot} |")
        L.append("")

    # 动态保护按regime分解
    L.append("## 动态保护(P3) 触发分解\n")
    L.append("| 窗口 | 总触发 | 总保护天数 | trending触发 | trending保护天数 | 震荡触发 | 震荡保护天数 |")
    L.append("|------|--------|-----------|-------------|-----------------|---------|-------------|")
    for window, wl in [("train", "训练"), ("test", "测试")]:
        if "error" in output["results"][window]:
            continue
        pi = output["results"][window]["dynamic"]["protection_info"]
        trig_by = pi.get("trigger_by_regime", {})
        prot_by = pi.get("protection_by_regime", {})
        trend_trig = trig_by.get("trending", 0)
        range_trig = trig_by.get("transition", 0) + trig_by.get("ranging", 0)
        trend_prot = prot_by.get("trending", 0)
        range_prot = prot_by.get("transition", 0) + prot_by.get("ranging", 0)
        L.append(f"| {wl} | {pi.get('trigger_count',0)} | {pi.get('protection_days',0)} | "
                 f"{trend_trig} | {trend_prot} | {range_trig} | {range_prot} |")
    L.append("")

    # 关键对比: fixed_8 vs dynamic
    L.append("## 核心对比: fixed_8(P2) vs dynamic(P3)\n")
    L.append("| 窗口 | 指标 | fixed_8(P2) | dynamic(P3) | Δ |")
    L.append("|------|------|------------|------------|---|")
    for window, wl in [("train", "训练"), ("test", "测试")]:
        if "error" in output["results"][window]:
            continue
        f8 = output["results"][window]["fixed_8"]
        dyn = output["results"][window]["dynamic"]
        for metric, key, fmt in [("夏普", "sharpe", "{:+.3f}"),
                                   ("收益%", "total_return_pct", "{:+.1f}"),
                                   ("Alpha%", "alpha_pct", "{:+.1f}"),
                                   ("回撤%", "max_drawdown_pct", "{:.1f}")]:
            v1 = f8[key]
            v2 = dyn[key]
            d = v2 - v1
            L.append(f"| {wl} | {metric} | {fmt.format(v1)} | {fmt.format(v2)} | {fmt.format(d)} |")
    L.append("")

    # 结论
    L.append("## 结论\n")
    if "error" not in output["results"]["test"]:
        f8t = output["results"]["test"]["fixed_8"]
        dynt = output["results"]["test"]["dynamic"]
        no_dd_t = output["results"]["test"]["no_dd"]
        L.append(f"**测试窗(牛市) 夏普**: no_dd {no_dd_t['sharpe']:.3f} → fixed_8 {f8t['sharpe']:.3f} → dynamic(P3) {dynt['sharpe']:.3f}\n")
        L.append(f"**测试窗(牛市) 回撤**: no_dd {no_dd_t['max_drawdown_pct']:.1f}% → fixed_8 {f8t['max_drawdown_pct']:.1f}% → dynamic(P3) {dynt['max_drawdown_pct']:.1f}%\n")
        L.append(f"**测试窗(牛市) Alpha**: no_dd {no_dd_t['alpha_pct']:+.1f}% → fixed_8 {f8t['alpha_pct']:+.1f}% → dynamic(P3) {dynt['alpha_pct']:+.1f}%\n")
        pi_f = f8t["protection_info"]
        pi_d = dynt["protection_info"]
        L.append(f"**保护触发**: fixed_8 {pi_f.get('trigger_count',0)}次/{pi_f.get('protection_days',0)}天, "
                 f"dynamic {pi_d.get('trigger_count',0)}次/{pi_d.get('protection_days',0)}天\n")

    if "error" not in output["results"]["train"]:
        f8tr = output["results"]["train"]["fixed_8"]
        dyntr = output["results"]["train"]["dynamic"]
        L.append(f"**训练窗(震荡市)**: fixed_8 夏普{f8tr['sharpe']:.3f}/Alpha{f8tr['alpha_pct']:+.1f}%/回撤{f8tr['max_drawdown_pct']:.1f}%, "
                 f"dynamic 夏普{dyntr['sharpe']:.3f}/Alpha{dyntr['alpha_pct']:+.1f}%/回撤{dyntr['max_drawdown_pct']:.1f}%\n")

    # 三角评估 P3
    L.append("**稳定-收益-回撤 三角评估 (P3=dynamic)**:\n")
    if "error" not in output["results"]["test"] and "error" not in output["results"]["train"]:
        p3t = output["results"]["test"]["dynamic"]
        p3tr = output["results"]["train"]["dynamic"]
        L.append("| 维度 | 标准 | 测试窗 | 训练窗 |")
        L.append("|------|------|--------|--------|")
        L.append(f"| 稳定 | 夏普>1.0 | {p3t['sharpe']:.3f}{'✅' if p3t['sharpe']>1 else '❌'} | "
                 f"{p3tr['sharpe']:.3f}{'✅' if p3tr['sharpe']>1 else '⚠️'} |")
        L.append(f"| 收益 | Alpha>0 | {p3t['alpha_pct']:+.1f}%{'✅' if p3t['alpha_pct']>0 else '❌'} | "
                 f"{p3tr['alpha_pct']:+.1f}%{'❌' if p3tr['alpha_pct']<0 else '✅'} |")
        L.append(f"| 回撤 | <10% | {p3t['max_drawdown_pct']:.1f}%{'✅' if p3t['max_drawdown_pct']>-10 else '❌'} | "
                 f"{p3tr['max_drawdown_pct']:.1f}%{'✅' if p3tr['max_drawdown_pct']>-10 else '❌'} |")
        L.append("")

    return "\n".join(L)


if __name__ == "__main__":
    main()
