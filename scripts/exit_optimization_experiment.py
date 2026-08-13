"""
退出参数优化实验 — 攻坚训练窗震荡市 Alpha 转正

诊断2确认根因: 退出逻辑过紧(非信号方向错)
  - 止损后90.9%反弹(+9.07%) → 止损太紧, 卖在低点
  - 止盈后78.7%继续涨(+7.95%) → 止盈太早, 卖飞高点
  - 信号方向45.2%正确, 仅1.6% "涨了但被打飞" → 入场OK, 退出是病根

本实验扫描不同 stop_loss_params 配置, 在P2基线上寻找:
  - 训练窗: Alpha转正(或接近0)
  - 测试窗: Alpha>0, 回撤<10%, 夏普>1.0
  - 两窗兼顾

参数说明 (position.py DEFAULT_STOP_PARAMS):
  - hard_stop_pct: 硬止损比例 (0.10=10%)
  - trail_tier1_threshold: 盈利进入中档的阈值 (0.10=10%)
  - trail_tier2_threshold: 盈利进入高档的阈值 (0.20=20%)
  - trail_mult_low: 盈利<t1时的移动止盈倍率系数 (×atr_stop_mult)
  - trail_mult_mid: 盈利t1~t2时的移动止盈倍率系数
  - trail_mult_high: 盈利>t2时的移动止盈倍率系数

当前P2基线(科技ATR1.8, 周期ATR2.0):
  科技: trail = 1.8×[1.0/0.8/0.6] = [1.8/1.44/1.08] ATR
  周期: trail = 2.0×[1.0/0.8/0.6] = [2.0/1.6/1.2] ATR
  震荡市正常波动1-2 ATR, 1.08-1.2 ATR的trailing极易被噪声触发.

用法: python scripts/exit_optimization_experiment.py
"""
import sys
import os
import json
import shutil
import warnings
import numpy as np
import pandas as pd
from datetime import datetime

warnings.filterwarnings("ignore")
project_root = os.path.join(os.path.dirname(__file__), "..")
sys.path.insert(0, project_root)

from src.config.runtime_mode import set_mode, RuntimeMode
set_mode(RuntimeMode.BACKTEST)

from src.backtest.engine import BacktestEngine
from src.data_fetcher import DataManager
from src.config.group_config import GroupConfig
from src.config.user_preferences import UserPreferences, _USER_PREF_FILE

TRAIN_START, TRAIN_END = "2024-07-01", "2025-06-30"
TEST_START, TEST_END = "2025-07-01", "2026-06-30"
DATA_START, DATA_END = "2024-02-01", "2026-07-13"
BENCHMARK = "sh.000300"
TOTAL_CAPITAL = 1000000

# P2 配置 (冻结基线)
WEIGHTS = {"科技成长型": 0.40, "消费稳健型": 0.10, "周期资源型": 0.425,
           "医药创新型": 0.0, "机械制造型": 0.0}
REGIMES_CFG = {"周期资源型": {"trending"}}
ATR_OVERRIDE = {"科技成长型": 1.8}

# 回撤保护参数 (P2基线)
DD_THRESHOLD = -0.08
DD_RECOVERY = -0.04
DD_REDUCED_RATIO = 0.5

# ── 退出参数扫描配置 ──
# 每个配置覆盖 DEFAULT_STOP_PARAMS 的部分键
EXIT_CONFIGS = [
    # 基线 (P2默认)
    {"name": "baseline_P2",
     "params": None},  # None=用默认值

    # 放宽移动止盈 30%
    {"name": "wider_trail_30",
     "params": {"trail_mult_low": 1.3, "trail_mult_mid": 1.0, "trail_mult_high": 0.8}},

    # 放宽移动止盈 50%
    {"name": "wider_trail_50",
     "params": {"trail_mult_low": 1.5, "trail_mult_mid": 1.2, "trail_mult_high": 1.0}},

    # 放宽移动止盈 100% (翻倍)
    {"name": "wider_trail_100",
     "params": {"trail_mult_low": 2.0, "trail_mult_mid": 1.5, "trail_mult_high": 1.2}},

    # 放宽硬止损
    {"name": "wider_hardstop_15",
     "params": {"hard_stop_pct": 0.15}},

    # 放宽硬止损 + 移动止盈
    {"name": "wider_both",
     "params": {"trail_mult_low": 1.5, "trail_mult_mid": 1.2, "trail_mult_high": 1.0,
                "hard_stop_pct": 0.15}},

    # 让利润奔跑: 大幅放宽trailing, 稍宽硬止损
    {"name": "let_winners_run",
     "params": {"trail_mult_low": 2.0, "trail_mult_mid": 1.5, "trail_mult_high": 1.0,
                "hard_stop_pct": 0.12}},

    # 放宽档位阈值 (推迟收紧时机)
    {"name": "wider_tiers",
     "params": {"trail_tier1_threshold": 0.15, "trail_tier2_threshold": 0.30}},

    # 组合最优: 放宽trailing + 放宽档位 + 稍宽硬止损
    {"name": "combined_best",
     "params": {"trail_mult_low": 1.5, "trail_mult_mid": 1.2, "trail_mult_high": 1.0,
                "trail_tier1_threshold": 0.15, "trail_tier2_threshold": 0.30,
                "hard_stop_pct": 0.12}},

    # 极宽trailing (测试上限)
    {"name": "ultra_wide_trail",
     "params": {"trail_mult_low": 2.5, "trail_mult_mid": 2.0, "trail_mult_high": 1.5,
                "hard_stop_pct": 0.12}},
]

REPORT_MD = os.path.join(project_root, "data", "exit_optimization_report.md")
RESULT_JSON = os.path.join(project_root, "data", "exit_optimization_result.json")


def load_watchlist():
    cfg_path = os.path.join(project_root, "config", "strategy_config.json")
    with open(cfg_path, "r", encoding="utf-8") as f:
        cfg = json.load(f)
    return {g: [s["code"] for s in stocks]
            for g, stocks in cfg["strategy_config"]["watchlist"].items()
            if not g.startswith("_")}


def run_group(data_map, benchmark_df, group_codes, group_capital,
              trade_regimes, atr_mult, start, end, stop_loss_params):
    """跑单组回测, 返回 daily_values + 指标."""
    if group_capital < 1000:
        return None
    GroupConfig._instance = None
    GroupConfig._config = None
    engine = BacktestEngine(
        initial_capital=group_capital, lookback_days=120, position_ratio=0.3,
        commission_rate=0.00025, stamp_tax=0.001, slippage=0.0001,
        signal_dedup_days=5, risk_per_trade=0.05, atr_stop_mult=atr_mult,
        forced_regime=None, trade_regimes=trade_regimes,
        stop_loss_params=stop_loss_params,
    )
    sub_map = {c: data_map[c] for c in group_codes if c in data_map}
    engine.run(sub_map, benchmark_df=benchmark_df, start_date=start, end_date=end)
    if engine.daily_values is None:
        return None
    dv = engine.daily_values.copy()
    dv_idx = pd.to_datetime(dv.index)
    return pd.Series(dv.values, index=dv_idx)


def apply_drawdown_protection(portfolio_nav):
    """组合级回撤保护 (P2基线: >8%降仓50%, 恢复4%)."""
    cummax = portfolio_nav.cummax()
    drawdown = (portfolio_nav - cummax) / cummax
    in_protection = False
    protected_nav = [portfolio_nav.iloc[0]]
    protection_days = 0
    trigger_count = 0
    for i in range(1, len(portfolio_nav)):
        dd = drawdown.iloc[i]
        if dd < DD_THRESHOLD and not in_protection:
            in_protection = True
            trigger_count += 1
        elif dd > DD_RECOVERY and in_protection:
            in_protection = False
        daily_return = portfolio_nav.iloc[i] / portfolio_nav.iloc[i - 1] - 1
        adjusted = daily_return * DD_REDUCED_RATIO if in_protection else daily_return
        protected_nav.append(protected_nav[-1] * (1 + adjusted))
        if in_protection:
            protection_days += 1
    return (pd.Series(protected_nav, index=portfolio_nav.index),
            protection_days, trigger_count)


def compute_metrics(portfolio_nav, benchmark_df, start, end, dd_protection=True):
    """算组合级指标."""
    start_ts = pd.Timestamp(start)
    end_ts = pd.Timestamp(end)
    mask = (portfolio_nav.index >= start_ts) & (portfolio_nav.index <= end_ts)
    portfolio_nav = portfolio_nav[mask]
    if len(portfolio_nav) < 10:
        return None

    # 现金部分
    invested = sum(w * TOTAL_CAPITAL for g, w in WEIGHTS.items() if w > 0)
    cash = TOTAL_CAPITAL - invested
    portfolio_nav = portfolio_nav + cash

    prot_info = {"protection_days": 0, "trigger_count": 0}
    if dd_protection:
        portfolio_nav, p_days, p_trigs = apply_drawdown_protection(portfolio_nav)
        prot_info = {"protection_days": p_days, "trigger_count": p_trigs}

    daily_ret = portfolio_nav.pct_change().dropna()
    total_return = (portfolio_nav.iloc[-1] / TOTAL_CAPITAL) - 1
    sharpe = (daily_ret.mean() / daily_ret.std() * np.sqrt(252)
              if daily_ret.std() > 0 else 0.0)
    cummax = portfolio_nav.cummax()
    drawdown = (portfolio_nav - cummax) / cummax
    max_dd = drawdown.min()

    bench = benchmark_df.copy()
    if "date" not in bench.columns:
        bench = bench.reset_index()
    bench["date"] = pd.to_datetime(bench["date"])
    bench_s = bench.set_index("date")["close"].astype(float)
    bench_s = bench_s[(bench_s.index >= start_ts) & (bench_s.index <= end_ts)]
    bench_ret = (bench_s.iloc[-1] / bench_s.iloc[0]) - 1 if len(bench_s) > 0 else 0
    alpha = total_return - bench_ret

    return {
        "total_return_pct": round(total_return * 100, 2),
        "sharpe": round(sharpe, 3),
        "alpha_pct": round(alpha * 100, 2),
        "benchmark_return_pct": round(bench_ret * 100, 2),
        "max_drawdown_pct": round(max_dd * 100, 2),
        **prot_info,
    }


def main():
    run_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print("=" * 80)
    print("  退出参数优化实验 — 攻坚训练窗震荡市 Alpha 转正")
    print(f"  P2基线: 训练Alpha-4.9%/夏普1.000/回撤-7.5%, 测试Alpha+33.1%/夏普3.410/回撤-8.7%")
    print(f"  诊断根因: 止损踏空90.9%, 止盈过早78.7%, 信号方向OK(45.2%正确)")
    print("=" * 80)

    pref_existed = os.path.exists(_USER_PREF_FILE)
    pref_backup = _USER_PREF_FILE + ".exit_bak"
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

        all_results = []
        for cfg in EXIT_CONFIGS:
            name = cfg["name"]
            params = cfg["params"]
            print(f"\n{'='*80}")
            print(f"  配置: {name}")
            print(f"  参数: {params or '默认(P2基线)'}")
            print(f"{'='*80}")

            # 跑训练窗
            print(f"  [训练窗 {TRAIN_START}~{TRAIN_END}]...")
            train_nav = None
            for g, codes in watchlist.items():
                if g not in WEIGHTS or WEIGHTS[g] == 0:
                    continue
                g_codes = [c for c in codes if c in data_map]
                if len(g_codes) < 2:
                    continue
                capital = TOTAL_CAPITAL * WEIGHTS[g]
                regimes = REGIMES_CFG.get(g)
                atr_mult = ATR_OVERRIDE.get(g, 2.0)
                nav = run_group(data_map, benchmark_df, g_codes, capital,
                                regimes, atr_mult, TRAIN_START, TRAIN_END, params)
                if nav is not None:
                    if train_nav is None:
                        train_nav = nav.copy()
                    else:
                        train_nav = train_nav.add(nav, fill_value=0)

            train_m = compute_metrics(train_nav, benchmark_df, TRAIN_START, TRAIN_END) if train_nav is not None else None

            # 跑测试窗
            print(f"  [测试窗 {TEST_START}~{TEST_END}]...")
            test_nav = None
            for g, codes in watchlist.items():
                if g not in WEIGHTS or WEIGHTS[g] == 0:
                    continue
                g_codes = [c for c in codes if c in data_map]
                if len(g_codes) < 2:
                    continue
                capital = TOTAL_CAPITAL * WEIGHTS[g]
                regimes = REGIMES_CFG.get(g)
                atr_mult = ATR_OVERRIDE.get(g, 2.0)
                nav = run_group(data_map, benchmark_df, g_codes, capital,
                                regimes, atr_mult, TEST_START, TEST_END, params)
                if nav is not None:
                    if test_nav is None:
                        test_nav = nav.copy()
                    else:
                        test_nav = test_nav.add(nav, fill_value=0)

            test_m = compute_metrics(test_nav, benchmark_df, TEST_START, TEST_END) if test_nav is not None else None

            if train_m and test_m:
                # 达标判断
                train_ok = train_m["alpha_pct"] > 0
                test_ok = test_m["alpha_pct"] > 0 and test_m["max_drawdown_pct"] > -10 and test_m["sharpe"] > 1.0
                both_ok = train_ok and test_ok
                tag = "✅✅" if both_ok else ("✅" if (train_ok or test_ok) else "❌")

                print(f"\n  训练窗: Alpha{train_m['alpha_pct']:+.2f}% 夏普{train_m['sharpe']:.3f} 回撤{train_m['max_drawdown_pct']:.1f}%")
                print(f"  测试窗: Alpha{test_m['alpha_pct']:+.2f}% 夏普{test_m['sharpe']:.3f} 回撤{test_m['max_drawdown_pct']:.1f}%")
                print(f"  达标: 训练{'✅' if train_ok else '❌'} 测试{'✅' if test_ok else '❌'} {tag}")

                all_results.append({
                    "name": name, "params": params,
                    "train": train_m, "test": test_m,
                    "train_ok": train_ok, "test_ok": test_ok, "both_ok": both_ok,
                })

        # ── 汇总 ──
        print("\n" + "=" * 80)
        print("  扫描汇总")
        print("=" * 80)
        print(f"{'配置':<22} {'训练Alpha%':>10} {'训练夏普':>8} {'训练回撤%':>9} {'测试Alpha%':>10} {'测试夏普':>8} {'测试回撤%':>9} {'达标':>6}")
        print("-" * 95)
        for r in all_results:
            t, s = r["train"], r["test"]
            tag = "✅✅" if r["both_ok"] else ("✅" if (r["train_ok"] or r["test_ok"]) else "❌")
            print(f"{r['name']:<22} {t['alpha_pct']:>+10.2f} {t['sharpe']:>8.3f} {t['max_drawdown_pct']:>9.1f} "
                  f"{s['alpha_pct']:>+10.2f} {s['sharpe']:>8.3f} {s['max_drawdown_pct']:>9.1f} {tag:>6}")

        # 找最优
        best = None
        for r in all_results:
            if r["both_ok"]:
                if best is None or r["train"]["alpha_pct"] > best["train"]["alpha_pct"]:
                    best = r
        if best is None:
            # 没有双达标, 找训练Alpha最高且测试不崩的
            candidates = [r for r in all_results if r["test_ok"] and r["name"] != "baseline_P2"]
            if candidates:
                best = max(candidates, key=lambda r: r["train"]["alpha_pct"])

        print(f"\n{'='*80}")
        if best:
            print(f"  最优配置: {best['name']}")
            print(f"  参数: {best['params']}")
            print(f"  训练窗: Alpha{best['train']['alpha_pct']:+.2f}% 夏普{best['train']['sharpe']:.3f} 回撤{best['train']['max_drawdown_pct']:.1f}%")
            print(f"  测试窗: Alpha{best['test']['alpha_pct']:+.2f}% 夏普{best['test']['sharpe']:.3f} 回撤{best['test']['max_drawdown_pct']:.1f}%")
        else:
            print("  未找到达标配置, 需进一步调整或引入regime-adaptive退出逻辑.")

        # 报告
        report = generate_report(run_time, all_results, best)
        with open(REPORT_MD, "w", encoding="utf-8") as f:
            f.write(report)
        with open(RESULT_JSON, "w", encoding="utf-8") as f:
            json.dump(all_results, f, ensure_ascii=False, indent=2, default=str)
        print(f"\n✓ 报告 → {REPORT_MD}")
        print(f"✓ 数据 → {RESULT_JSON}")

    finally:
        if pref_existed and os.path.exists(pref_backup):
            shutil.move(pref_backup, _USER_PREF_FILE)


def generate_report(run_time, results, best):
    L = []
    L.append("# 退出参数优化实验报告\n")
    L.append(f"**运行时间**: {run_time}")
    L.append(f"**基线**: P2 (科技40/消费10/周期42.5, 周期trending过滤, 科技ATR1.8, 回撤保护>8%降仓)")
    L.append(f"**P2基线指标**: 训练Alpha-4.9%/夏普1.000/回撤-7.5%, 测试Alpha+33.1%/夏普3.410/回撤-8.7%\n")

    L.append("## 诊断根因\n")
    L.append("诊断2确认退出逻辑过紧(非信号方向错):")
    L.append("- 止损后90.9%反弹(+9.07%) → 止损太紧, 卖在低点")
    L.append("- 止盈后78.7%继续涨(+7.95%) → 止盈太早, 卖飞高点")
    L.append("- 信号方向45.2%正确, 仅1.6%被打飞 → 入场OK, 退出是病根\n")

    L.append("## 扫描配置\n")
    L.append("| 配置 | trail_mult[low/mid/high] | tier[t1/t2] | hard_stop |")
    L.append("|------|-------------------------|-------------|-----------|")
    for r in results:
        p = r["params"]
        if p is None:
            L.append(f"| {r['name']} | [1.0/0.8/0.6](默认) | [0.10/0.20](默认) | 0.10(默认) |")
        else:
            tl = p.get("trail_mult_low", 1.0)
            tm = p.get("trail_mult_mid", 0.8)
            th = p.get("trail_mult_high", 0.6)
            t1 = p.get("trail_tier1_threshold", 0.10)
            t2 = p.get("trail_tier2_threshold", 0.20)
            hs = p.get("hard_stop_pct", 0.10)
            L.append(f"| {r['name']} | [{tl}/{tm}/{th}] | [{t1}/{t2}] | {hs} |")
    L.append("")

    L.append("## 扫描结果\n")
    L.append("### 训练窗(震荡市)\n")
    L.append("| 配置 | Alpha% | 夏普 | 回撤% | 达标(Alpha>0) |")
    L.append("|------|--------|------|-------|--------------|")
    for r in results:
        t = r["train"]
        tag = "✅" if r["train_ok"] else "❌"
        L.append(f"| {r['name']} | {t['alpha_pct']:+.2f} | {t['sharpe']:.3f} | {t['max_drawdown_pct']:.1f} | {tag} |")
    L.append("")

    L.append("### 测试窗(牛市)\n")
    L.append("| 配置 | Alpha% | 夏普 | 回撤% | 达标(Alpha>0&回撤<10%) |")
    L.append("|------|--------|------|-------|-----------------------|")
    for r in results:
        s = r["test"]
        tag = "✅" if r["test_ok"] else "❌"
        L.append(f"| {r['name']} | {s['alpha_pct']:+.2f} | {s['sharpe']:.3f} | {s['max_drawdown_pct']:.1f} | {tag} |")
    L.append("")

    L.append("## 最优配置\n")
    if best:
        L.append(f"**配置名**: {best['name']}")
        L.append(f"**参数**: {best['params']}\n")
        L.append("| 窗口 | Alpha% | 夏普 | 回撤% | 收益% |")
        L.append("|------|--------|------|-------|-------|")
        L.append(f"| 训练窗 | {best['train']['alpha_pct']:+.2f} | {best['train']['sharpe']:.3f} | {best['train']['max_drawdown_pct']:.1f} | {best['train']['total_return_pct']:+.2f} |")
        L.append(f"| 测试窗 | {best['test']['alpha_pct']:+.2f} | {best['test']['sharpe']:.3f} | {best['test']['max_drawdown_pct']:.1f} | {best['test']['total_return_pct']:+.2f} |\n")

        # 对比基线
        base = next((r for r in results if r["name"] == "baseline_P2"), None)
        if base and base != best:
            L.append("### vs P2基线改进\n")
            L.append("| 指标 | P2基线 | 最优 | 改进 |")
            L.append("|------|--------|------|------|")
            tr_d = best["train"]["alpha_pct"] - base["train"]["alpha_pct"]
            ts_d = best["test"]["alpha_pct"] - base["test"]["alpha_pct"]
            tr_s = best["train"]["sharpe"] - base["train"]["sharpe"]
            ts_s = best["test"]["sharpe"] - base["test"]["sharpe"]
            L.append(f"| 训练Alpha | {base['train']['alpha_pct']:+.2f}% | {best['train']['alpha_pct']:+.2f}% | {tr_d:+.2f}% |")
            L.append(f"| 测试Alpha | {base['test']['alpha_pct']:+.2f}% | {best['test']['alpha_pct']:+.2f}% | {ts_d:+.2f}% |")
            L.append(f"| 训练夏普 | {base['train']['sharpe']:.3f} | {best['train']['sharpe']:.3f} | {tr_s:+.3f} |")
            L.append(f"| 测试夏普 | {base['test']['sharpe']:.3f} | {best['test']['sharpe']:.3f} | {ts_s:+.3f} |")
    else:
        L.append("未找到双达标配置, 需进一步调整或引入regime-adaptive退出逻辑.")
    L.append("")

    L.append("## 结论与下一步\n")
    if best and best["both_ok"]:
        L.append("**✅ 成功找到双达标配置**, 可作为P3候选基线.")
        L.append("下一步: 将最优参数固化到引擎, 跑全量验证.")
    elif best:
        L.append("**⚠️ 部分达标**, 最优配置改善了训练窗但测试窗有妥协.")
        L.append("下一步建议: 引入regime-adaptive退出逻辑(震荡市放宽/趋势市保持紧), 避免全局放宽影响牛市表现.")
    else:
        L.append("**❌ 全局参数扫描未达标**, 需引入regime-adaptive退出逻辑:")
        L.append("- 震荡市(transition/ranging): 放宽trailing stop + 硬止损")
        L.append("- 趋势市(trending): 保持P2基线参数")
        L.append("- 需修改PositionManager.check_stop_loss 支持regime-adaptive参数")
    return "\n".join(L)


if __name__ == "__main__":
    main()
