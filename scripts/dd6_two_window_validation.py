"""回撤保护阈值 6% 两窗口验证 — 训练窗(2024-07~2025-06) + 测试窗(2025-07~2026-06)

基于 P3 固化退出参数, 在两个窗口分别对比三种回撤保护配置:
  1. 无保护 (对照)
  2. 8% 阈值 (P3基线, recovery4%, 降仓50%)
  3. 6% 阈值 (候选, recovery3%, 降仓50%)

每个窗口只跑一次各组回测拿到组合净值, 然后对同一净值应用三种保护配置,
确认 6% 阈值在两窗的三角指标(夏普>1 / Alpha>0 / 回撤<10%)都达标后才固化.

用法: python scripts/dd6_two_window_validation.py
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
from src.backtest.position import PositionManager

TRAIN_START, TRAIN_END = "2024-07-01", "2025-06-30"
TEST_START, TEST_END = "2025-07-01", "2026-06-30"
DATA_START, DATA_END = "2024-02-01", "2026-07-13"
BENCHMARK = "sh.000300"
TOTAL_CAPITAL = 1000000

# P3 组合配置 (继承P2, 不变)
WEIGHTS = {"科技成长型": 0.40, "消费稳健型": 0.10, "周期资源型": 0.425,
           "医药创新型": 0.0, "机械制造型": 0.0}
REGIMES_CFG = {"周期资源型": {"trending"}}
ATR_OVERRIDE = {"科技成长型": 1.8}
DD_REDUCED_RATIO = 0.5

# 三种回撤保护配置
DD_CONFIGS = [
    ("无保护", None, None),
    ("8%(P3基线)", -0.08, -0.04),
    ("6%(候选)", -0.06, -0.03),
]

# P3基线(8%阈值)已知指标 (来自 p3_full_validation, 用于校验重跑一致性)
P3_BASELINE_8 = {
    "train": {"alpha_pct": 1.59, "sharpe": 1.437, "max_drawdown_pct": -7.6,
              "total_return_pct": 14.76},
    "test": {"alpha_pct": 47.13, "sharpe": 3.362, "max_drawdown_pct": -7.5,
             "total_return_pct": 73.43},
}

REPORT_MD = os.path.join(project_root, "data", "dd6_two_window_validation_report.md")


def load_watchlist():
    cfg_path = os.path.join(project_root, "config", "strategy_config.json")
    with open(cfg_path, "r", encoding="utf-8") as f:
        cfg = json.load(f)
    return {g: [s["code"] for s in stocks]
            for g, stocks in cfg["strategy_config"]["watchlist"].items()
            if not g.startswith("_")}


def run_group(data_map, benchmark_df, group_codes, group_capital,
              trade_regimes, atr_mult, start, end):
    """跑单组回测 — 使用P3默认退出参数(已固化到PositionManager)."""
    if group_capital < 1000:
        return {"skipped": True, "daily_values": None, "trade_count": 0}
    GroupConfig._instance = None
    GroupConfig._config = None
    engine = BacktestEngine(
        initial_capital=group_capital, lookback_days=120, position_ratio=0.3,
        commission_rate=0.00025, stamp_tax=0.001, slippage=0.0001,
        signal_dedup_days=5, risk_per_trade=0.05, atr_stop_mult=atr_mult,
        forced_regime=None, trade_regimes=trade_regimes,
    )
    sub_map = {c: data_map[c] for c in group_codes if c in data_map}
    m = engine.run(sub_map, benchmark_df=benchmark_df, start_date=start, end_date=end)
    return {
        "trade_count": getattr(m, "trade_count", 0) or 0,
        "daily_values": engine.daily_values.copy() if engine.daily_values is not None else None,
        "skipped": False,
    }


def apply_protection(portfolio_nav, threshold, recovery):
    """组合级回撤保护 (参数化版本, 降仓比例固定50%)."""
    if threshold is None:
        return portfolio_nav, 0, 0
    cummax = portfolio_nav.cummax()
    drawdown = (portfolio_nav - cummax) / cummax
    in_prot = False
    prot_nav = [portfolio_nav.iloc[0]]
    p_days = 0; p_trigs = 0
    for i in range(1, len(portfolio_nav)):
        dd = drawdown.iloc[i]
        if dd < threshold and not in_prot:
            in_prot = True; p_trigs += 1
        elif dd > recovery and in_prot:
            in_prot = False
        r = portfolio_nav.iloc[i] / portfolio_nav.iloc[i-1] - 1
        a = r * DD_REDUCED_RATIO if in_prot else r
        prot_nav.append(prot_nav[-1] * (1 + a))
        if in_prot: p_days += 1
    return pd.Series(prot_nav, index=portfolio_nav.index), p_days, p_trigs


def build_portfolio_nav(group_results, start, end):
    """由各组 daily_values 拼出组合净值(含现金, 未应用保护)."""
    portfolio_nav = None
    total_trades = 0
    for g, r in group_results.items():
        if r.get("skipped") or r.get("daily_values") is None:
            continue
        nav = r["daily_values"]
        portfolio_nav = nav if portfolio_nav is None else portfolio_nav.add(nav, fill_value=0)
        total_trades += r["trade_count"]

    if portfolio_nav is None or len(portfolio_nav) < 10:
        return None, 0

    start_ts = pd.Timestamp(start)
    end_ts = pd.Timestamp(end)
    nav_idx = pd.to_datetime(portfolio_nav.index)
    portfolio_nav = pd.Series(portfolio_nav.values, index=nav_idx)
    mask = (portfolio_nav.index >= start_ts) & (portfolio_nav.index <= end_ts)
    portfolio_nav = portfolio_nav[mask]

    invested = sum(w * TOTAL_CAPITAL for g, w in WEIGHTS.items() if w > 0)
    cash = TOTAL_CAPITAL - invested
    portfolio_nav = portfolio_nav + cash
    return portfolio_nav, total_trades


def metrics_from_nav(portfolio_nav, benchmark_df, start, end, total_trades,
                     p_days=0, p_trigs=0):
    daily_ret = portfolio_nav.pct_change().dropna()
    total_return = (portfolio_nav.iloc[-1] / TOTAL_CAPITAL) - 1
    sharpe = (daily_ret.mean() / daily_ret.std() * np.sqrt(252)
              if daily_ret.std() > 0 else 0.0)
    cummax = portfolio_nav.cummax()
    max_dd = ((portfolio_nav - cummax) / cummax).min()

    bench = benchmark_df.copy()
    if "date" not in bench.columns:
        bench = bench.reset_index()
    bench["date"] = pd.to_datetime(bench["date"])
    bench_s = bench.set_index("date")["close"].astype(float)
    bench_s = bench_s[(bench_s.index >= pd.Timestamp(start)) & (bench_s.index <= pd.Timestamp(end))]
    bench_ret = (bench_s.iloc[-1] / bench_s.iloc[0]) - 1 if len(bench_s) > 0 else 0
    alpha = total_return - bench_ret

    return {
        "sharpe": round(sharpe, 3),
        "total_return_pct": round(total_return * 100, 2),
        "alpha_pct": round(alpha * 100, 2),
        "benchmark_return_pct": round(bench_ret * 100, 2),
        "max_drawdown_pct": round(max_dd * 100, 2),
        "trade_count": total_trades,
        "protection_days": p_days,
        "trigger_count": p_trigs,
    }


def run_window(data_map, benchmark_df, watchlist, start, end, window_name):
    """跑一个窗口: 各组回测 → 组合净值 → 三种保护配置对比."""
    print(f"\n{'='*85}")
    print(f"  [{window_name}] {start}~{end}")
    print(f"{'='*85}")

    group_results = {}
    for g, codes in watchlist.items():
        if g not in WEIGHTS or WEIGHTS[g] == 0:
            print(f"  {g:12s}: [跳过-权重0%]")
            group_results[g] = {"skipped": True}
            continue
        g_codes = [c for c in codes if c in data_map]
        if len(g_codes) < 2:
            continue
        capital = TOTAL_CAPITAL * WEIGHTS[g]
        regimes = REGIMES_CFG.get(g)
        atr_mult = ATR_OVERRIDE.get(g, 2.0)
        r = run_group(data_map, benchmark_df, g_codes, capital,
                      regimes, atr_mult, start, end)
        group_results[g] = r
        if not r.get("skipped"):
            print(f"  {g:12s}: 交易{r['trade_count']}笔")

    portfolio_nav, total_trades = build_portfolio_nav(group_results, start, end)
    if portfolio_nav is None:
        return None

    # 三种保护配置
    config_metrics = {}
    for label, th, rec in DD_CONFIGS:
        nav_p, p_days, p_trigs = apply_protection(portfolio_nav, th, rec)
        m = metrics_from_nav(nav_p, benchmark_df, start, end, total_trades, p_days, p_trigs)
        config_metrics[label] = m
        print(f"  {label:14s}: 收益{m['total_return_pct']:+6.2f}% Alpha{m['alpha_pct']:+6.2f}% "
              f"夏普{m['sharpe']:.3f} 回撤{m['max_drawdown_pct']:.1f}% 触发{p_trigs}次/{p_days}天")
    return config_metrics


def triangle_ok(m):
    """三角指标是否达标."""
    return (m["sharpe"] > 1.0 and m["alpha_pct"] > 0 and m["max_drawdown_pct"] > -10)


def main():
    run_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print("=" * 85)
    print("  回撤保护阈值 6% 两窗口验证")
    print("  P3退出参数(已固化) + 三种回撤保护配置对比")
    print("  候选: 6%阈值(recovery3%, 降仓50%) — 需两窗三角全达标才固化")
    print("=" * 85)

    # 确认P3默认参数已固化
    print(f"\n引擎默认退出参数 (P3固化):")
    print(f"  trail_mult: [low={PositionManager.DEFAULT_STOP_PARAMS['trail_mult_low']}, "
          f"mid={PositionManager.DEFAULT_STOP_PARAMS['trail_mult_mid']}, "
          f"high={PositionManager.DEFAULT_STOP_PARAMS['trail_mult_high']}]")
    print(f"  hard_stop_pct: {PositionManager.DEFAULT_STOP_PARAMS['hard_stop_pct']}")
    print(f"  regime_exit_config: {PositionManager.DEFAULT_REGIME_EXIT_CONFIG}")

    pref_existed = os.path.exists(_USER_PREF_FILE)
    pref_backup = _USER_PREF_FILE + ".dd6val_bak"
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

        train_m = run_window(data_map, benchmark_df, watchlist, TRAIN_START, TRAIN_END, "训练窗(震荡市)")
        test_m = run_window(data_map, benchmark_df, watchlist, TEST_START, TEST_END, "测试窗(牛市)")

        if train_m is None or test_m is None:
            print("\n❌ 净数据不足, 退出")
            return

        # ── 汇总对比 ──
        print(f"\n{'='*85}")
        print(f"  两窗口汇总对比")
        print(f"{'='*85}")

        print(f"\n{'配置':<16} {'窗口':<8} {'收益%':>8} {'Alpha%':>8} {'夏普':>7} {'回撤%':>7} {'触发':>5} {'降仓天':>6} {'达标':>5}")
        print("-" * 80)
        for label, _, _ in DD_CONFIGS:
            for wn, m in [("训练", train_m[label]), ("测试", test_m[label])]:
                ok = "✅" if triangle_ok(m) else "❌"
                print(f"{label:<16} {wn:<8} {m['total_return_pct']:>+8.2f} {m['alpha_pct']:>+8.2f} "
                      f"{m['sharpe']:>7.3f} {m['max_drawdown_pct']:>7.1f} {m['trigger_count']:>5} "
                      f"{m['protection_days']:>6} {ok:>5}")

        # 6%候选两窗达标判断
        cand_label = "6%(候选)"
        train_ok = triangle_ok(train_m[cand_label])
        test_ok = triangle_ok(test_m[cand_label])
        both_ok = train_ok and test_ok

        print(f"\n{'='*85}")
        print(f"  6%候选阈值 — 两窗达标判断")
        print(f"{'='*85}")
        print(f"  训练窗: 夏普{train_m[cand_label]['sharpe']:.3f}{'✅' if train_ok else '❌'} "
              f"Alpha{train_m[cand_label]['alpha_pct']:+.2f}%{'✅' if train_m[cand_label]['alpha_pct']>0 else '❌'} "
              f"回撤{train_m[cand_label]['max_drawdown_pct']:.1f}%{'✅' if train_m[cand_label]['max_drawdown_pct']>-10 else '❌'}")
        print(f"  测试窗: 夏普{test_m[cand_label]['sharpe']:.3f}{'✅' if test_ok else '❌'} "
              f"Alpha{test_m[cand_label]['alpha_pct']:+.2f}%{'✅' if test_m[cand_label]['alpha_pct']>0 else '❌'} "
              f"回撤{test_m[cand_label]['max_drawdown_pct']:.1f}%{'✅' if test_m[cand_label]['max_drawdown_pct']>-10 else '❌'}")
        print(f"\n  两窗全达标: {'✅ 是 → 可固化6%阈值' if both_ok else '❌ 否 → 维持8%基线'}")

        # 6% vs 8% 改进对比
        print(f"\n{'='*85}")
        print(f"  6% vs 8%(P3基线) 改进对比")
        print(f"{'='*85}")
        print(f"\n{'指标':<10} {'窗口':<6} {'8%基线':>10} {'6%候选':>10} {'改进':>10}")
        print("-" * 50)
        for wn, m8, m6 in [("训练", train_m["8%(P3基线)"], train_m[cand_label]),
                           ("测试", test_m["8%(P3基线)"], test_m[cand_label])]:
            for key, label in [("total_return_pct", "收益%"), ("alpha_pct", "Alpha%"),
                               ("sharpe", "夏普"), ("max_drawdown_pct", "回撤%")]:
                v8 = m8[key]; v6 = m6[key]; d = v6 - v8
                print(f"{label:<10} {wn:<6} {v8:>+10.2f} {v6:>+10.2f} {d:>+10.2f}")

        # 报告
        report = generate_report(run_time, train_m, test_m, both_ok)
        with open(REPORT_MD, "w", encoding="utf-8") as f:
            f.write(report)
        print(f"\n✓ 报告 → {REPORT_MD}")

    finally:
        if pref_existed and os.path.exists(pref_backup):
            shutil.move(pref_backup, _USER_PREF_FILE)


def generate_report(run_time, train_m, test_m, both_ok):
    L = []
    L.append("# 回撤保护阈值 6% 两窗口验证报告\n")
    L.append(f"**运行时间**: {run_time}")
    L.append(f"**目的**: 验证6%阈值(recovery3%,降仓50%)在训练/测试两窗是否都达标, 通过后固化\n")
    L.append(f"**P3退出参数**(已固化): trail_mult[2.0/1.5/1.0] + hard_stop 0.12 + 震荡市禁用trailing\n")

    L.append("## 两窗口三配置对比\n")
    for wn, m in [("训练窗(2024-07~2025-06,震荡市)", train_m),
                  ("测试窗(2025-07~2026-06,牛市)", test_m)]:
        L.append(f"### {wn}\n")
        L.append("| 配置 | 收益% | Alpha% | 夏普 | 回撤% | 触发次数 | 降仓天数 | 三角达标 |")
        L.append("|------|-------|--------|------|-------|---------|---------|---------|")
        for label, _, _ in DD_CONFIGS:
            r = m[label]
            ok = "✅" if triangle_ok(r) else "❌"
            L.append(f"| {label} | {r['total_return_pct']:+.2f} | {r['alpha_pct']:+.2f} | "
                     f"{r['sharpe']:.3f} | {r['max_drawdown_pct']:.1f} | {r['trigger_count']} | "
                     f"{r['protection_days']} | {ok} |")
        L.append("")

    L.append("## 6%候选 vs 8%基线 改进对比\n")
    L.append("| 指标 | 窗口 | 8%基线 | 6%候选 | 改进 |")
    L.append("|------|------|--------|--------|------|")
    for wn, m8, m6 in [("训练", train_m["8%(P3基线)"], train_m["6%(候选)"]),
                       ("测试", test_m["8%(P3基线)"], test_m["6%(候选)"])]:
        for key, label in [("total_return_pct", "收益%"), ("alpha_pct", "Alpha%"),
                           ("sharpe", "夏普"), ("max_drawdown_pct", "回撤%")]:
            v8 = m8[key]; v6 = m6[key]; d = v6 - v8
            L.append(f"| {label} | {wn} | {v8:+.2f} | {v6:+.2f} | {d:+.2f} |")
    L.append("")

    L.append("## 两窗达标判断\n")
    cand = "6%(候选)"
    train_ok = triangle_ok(train_m[cand])
    test_ok = triangle_ok(test_m[cand])
    L.append("| 窗口 | 夏普(>1) | Alpha(>0) | 回撤(<10%) | 达标 |")
    L.append("|------|---------|-----------|-----------|------|")
    t = train_m[cand]
    s = test_m[cand]
    L.append(f"| 训练窗 | {t['sharpe']:.3f}{'✅' if t['sharpe']>1 else '❌'} | "
             f"{t['alpha_pct']:+.2f}%{'✅' if t['alpha_pct']>0 else '❌'} | "
             f"{t['max_drawdown_pct']:.1f}%{'✅' if t['max_drawdown_pct']>-10 else '❌'} | "
             f"{'✅' if train_ok else '❌'} |")
    L.append(f"| 测试窗 | {s['sharpe']:.3f}{'✅' if s['sharpe']>1 else '❌'} | "
             f"{s['alpha_pct']:+.2f}%{'✅' if s['alpha_pct']>0 else '❌'} | "
             f"{s['max_drawdown_pct']:.1f}%{'✅' if s['max_drawdown_pct']>-10 else '❌'} | "
             f"{'✅' if test_ok else '❌'} |")
    L.append("")

    L.append("## 结论\n")
    if both_ok:
        L.append("**✅ 6%阈值两窗三角全达标, 可固化!**\n")
        L.append("建议将回撤保护阈值从8%降到6%(recovery3%, 降仓50%保持不变).\n")
    else:
        L.append("**❌ 6%阈值未能在两窗全部达标, 维持8%基线.**\n")
        if not train_ok:
            L.append("- 训练窗未达标\n")
        if not test_ok:
            L.append("- 测试窗未达标\n")

    L.append("### 配置参数\n")
    L.append("| 配置 | 阈值 | recovery | 降仓比例 |")
    L.append("|------|------|----------|---------|")
    L.append("| 8%(P3基线) | -0.08 | -0.04 | 0.5 |")
    L.append("| 6%(候选) | -0.06 | -0.03 | 0.5 |")
    return "\n".join(L)


if __name__ == "__main__":
    main()
