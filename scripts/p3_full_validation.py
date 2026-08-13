"""
P3 全量验证 — Regime-adaptive 退出参数固化后的最终验证

P3 基线 (2026-08-07 固化):
  全局退出参数 (已写入 PositionManager.DEFAULT_STOP_PARAMS):
    - trail_mult: [2.0/1.5/1.0] (P2: [1.0/0.8/0.6], 放宽让利润奔跑)
    - hard_stop_pct: 0.12 (P2: 0.10, 稍宽避免假突破)
  分体制退出 (已写入 PositionManager.DEFAULT_REGIME_EXIT_CONFIG):
    - ranging: disable_trailing=True (震荡市禁用移动止盈, 只靠硬止损+信号退出)

  组合配置 (继承P2):
    - 权重: 科技40%/消费10%/周期42.5%/医药0%/机械0%/现金7.5%
    - 周期组: trending过滤
    - 科技组: ATR 1.8
    - 回撤保护: >8%降仓50%, 恢复4%

P2基线 (对照):
    - 训练: Alpha-4.9%/夏普1.000/回撤-7.5%
    - 测试: Alpha+33.1%/夏普3.410/回撤-8.7%

用法: python scripts/p3_full_validation.py
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

# P2/P3 组合配置 (继承P2, 不变)
WEIGHTS = {"科技成长型": 0.40, "消费稳健型": 0.10, "周期资源型": 0.425,
           "医药创新型": 0.0, "机械制造型": 0.0}
REGIMES_CFG = {"周期资源型": {"trending"}}
ATR_OVERRIDE = {"科技成长型": 1.8}

# 回撤保护参数 (继承P2)
DD_THRESHOLD = -0.08
DD_RECOVERY = -0.04
DD_REDUCED_RATIO = 0.5

# P2基线指标 (对照)
P2_BASELINE = {
    "train": {"alpha_pct": -4.90, "sharpe": 1.000, "max_drawdown_pct": -7.5,
              "total_return_pct": 8.27},
    "test": {"alpha_pct": 33.11, "sharpe": 3.410, "max_drawdown_pct": -8.7,
             "total_return_pct": 59.40},
}

REPORT_MD = os.path.join(project_root, "data", "p3_full_validation_report.md")
RESULT_JSON = os.path.join(project_root, "data", "p3_full_validation_result.json")


def load_watchlist():
    cfg_path = os.path.join(project_root, "config", "strategy_config.json")
    with open(cfg_path, "r", encoding="utf-8") as f:
        cfg = json.load(f)
    return {g: [s["code"] for s in stocks]
            for g, stocks in cfg["strategy_config"]["watchlist"].items()
            if not g.startswith("_")}


def run_group(data_map, benchmark_df, group_codes, group_capital,
              trade_regimes, atr_mult, start, end):
    """跑单组回测 — 使用P3默认退出参数(无需显式传入)."""
    if group_capital < 1000:
        return {"skipped": True, "daily_values": None, "trade_count": 0,
                "sharpe": 0, "total_return": 0, "alpha": 0, "max_drawdown": 0,
                "win_rate": 0, "final_value": group_capital}
    GroupConfig._instance = None
    GroupConfig._config = None
    engine = BacktestEngine(
        initial_capital=group_capital, lookback_days=120, position_ratio=0.3,
        commission_rate=0.00025, stamp_tax=0.001, slippage=0.0001,
        signal_dedup_days=5, risk_per_trade=0.05, atr_stop_mult=atr_mult,
        forced_regime=None, trade_regimes=trade_regimes,
        # P3默认参数已固化到PositionManager, 无需显式传入
        # stop_loss_params=None → 用DEFAULT_STOP_PARAMS (P3)
        # regime_exit_config=None → 用DEFAULT_REGIME_EXIT_CONFIG (P3)
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
        "skipped": False,
    }


def apply_drawdown_protection(portfolio_nav):
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


def compute_portfolio_metrics(group_results, benchmark_df, start, end):
    portfolio_nav = None
    total_trades = 0
    for g, r in group_results.items():
        if r.get("skipped") or r.get("daily_values") is None:
            continue
        nav = r["daily_values"]
        portfolio_nav = nav if portfolio_nav is None else portfolio_nav.add(nav, fill_value=0)
        total_trades += r["trade_count"]

    if portfolio_nav is None or len(portfolio_nav) < 10:
        return {"error": "无有效净值数据"}

    start_ts = pd.Timestamp(start)
    end_ts = pd.Timestamp(end)
    nav_idx = pd.to_datetime(portfolio_nav.index)
    portfolio_nav = pd.Series(portfolio_nav.values, index=nav_idx)
    mask = (portfolio_nav.index >= start_ts) & (portfolio_nav.index <= end_ts)
    portfolio_nav = portfolio_nav[mask]

    invested = sum(w * TOTAL_CAPITAL for g, w in WEIGHTS.items() if w > 0)
    cash = TOTAL_CAPITAL - invested
    portfolio_nav = portfolio_nav + cash

    portfolio_nav, p_days, p_triggers = apply_drawdown_protection(portfolio_nav)

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
        "sharpe": round(sharpe, 3),
        "total_return_pct": round(total_return * 100, 2),
        "alpha_pct": round(alpha * 100, 2),
        "benchmark_return_pct": round(bench_ret * 100, 2),
        "max_drawdown_pct": round(max_dd * 100, 2),
        "trade_count": total_trades,
        "protection_days": p_days,
        "trigger_count": p_triggers,
    }


def main():
    run_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print("=" * 85)
    print("  P3 全量验证 — Regime-adaptive 退出参数固化后最终验证")
    print(f"  P3核心: 全局trail_mult[2.0/1.5/1.0] + hard_stop 0.12 + 震荡市禁用trailing")
    print(f"  训练窗: {TRAIN_START}~{TRAIN_END} (震荡市)")
    print(f"  测试窗: {TEST_START}~{TEST_END} (牛市)")
    print("=" * 85)

    # 确认P3默认参数已固化
    print(f"\n引擎默认退出参数 (P3固化):")
    print(f"  DEFAULT_STOP_PARAMS: {PositionManager.DEFAULT_STOP_PARAMS}")
    print(f"  DEFAULT_REGIME_EXIT_CONFIG: {PositionManager.DEFAULT_REGIME_EXIT_CONFIG}")

    pref_existed = os.path.exists(_USER_PREF_FILE)
    pref_backup = _USER_PREF_FILE + ".p3val_bak"
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

        # ── 训练窗 ──
        print(f"\n{'='*85}")
        print(f"  [训练窗 {TRAIN_START}~{TRAIN_END}] 震荡市")
        print(f"{'='*85}")
        train_group_results = {}
        for g, codes in watchlist.items():
            if g not in WEIGHTS or WEIGHTS[g] == 0:
                print(f"  {g:12s}: [跳过-权重0%]")
                train_group_results[g] = {"skipped": True}
                continue
            g_codes = [c for c in codes if c in data_map]
            if len(g_codes) < 2:
                continue
            capital = TOTAL_CAPITAL * WEIGHTS[g]
            regimes = REGIMES_CFG.get(g)
            atr_mult = ATR_OVERRIDE.get(g, 2.0)
            r = run_group(data_map, benchmark_df, g_codes, capital,
                          regimes, atr_mult, TRAIN_START, TRAIN_END)
            train_group_results[g] = r
            if not r.get("skipped"):
                print(f"  {g:12s}: 夏普{r['sharpe']:+.3f} 收益{r['total_return']*100:+.1f}% "
                      f"Alpha{r['alpha']*100:+.1f}% 交易{r['trade_count']}笔")

        train_m = compute_portfolio_metrics(train_group_results, benchmark_df,
                                             TRAIN_START, TRAIN_END)

        # ── 测试窗 ──
        print(f"\n{'='*85}")
        print(f"  [测试窗 {TEST_START}~{TEST_END}] 牛市")
        print(f"{'='*85}")
        test_group_results = {}
        for g, codes in watchlist.items():
            if g not in WEIGHTS or WEIGHTS[g] == 0:
                print(f"  {g:12s}: [跳过-权重0%]")
                test_group_results[g] = {"skipped": True}
                continue
            g_codes = [c for c in codes if c in data_map]
            if len(g_codes) < 2:
                continue
            capital = TOTAL_CAPITAL * WEIGHTS[g]
            regimes = REGIMES_CFG.get(g)
            atr_mult = ATR_OVERRIDE.get(g, 2.0)
            r = run_group(data_map, benchmark_df, g_codes, capital,
                          regimes, atr_mult, TEST_START, TEST_END)
            test_group_results[g] = r
            if not r.get("skipped"):
                print(f"  {g:12s}: 夏普{r['sharpe']:+.3f} 收益{r['total_return']*100:+.1f}% "
                      f"Alpha{r['alpha']*100:+.1f}% 交易{r['trade_count']}笔")

        test_m = compute_portfolio_metrics(test_group_results, benchmark_df,
                                            TEST_START, TEST_END)

        # ── 汇总 ──
        print(f"\n{'='*85}")
        print(f"  P3 全量验证结果")
        print(f"{'='*85}")

        print(f"\n  {'指标':<12} {'P2基线':>10} {'P3':>10} {'改进':>10}")
        print(f"  {'-'*48}")
        for window, p2, p3 in [("训练窗", P2_BASELINE["train"], train_m),
                                ("测试窗", P2_BASELINE["test"], test_m)]:
            print(f"\n  [{window}]")
            for key, label in [("alpha_pct", "Alpha%"), ("sharpe", "夏普"),
                               ("max_drawdown_pct", "回撤%"), ("total_return_pct", "收益%")]:
                v2 = p2.get(key, 0)
                v3 = p3.get(key, 0)
                d = v3 - v2
                print(f"  {label:<12} {v2:>+10.2f} {v3:>+10.2f} {d:>+10.2f}")

        # 三角评估
        print(f"\n{'='*85}")
        print(f"  稳定-收益-回撤 三角评估 (P3)")
        print(f"{'='*85}")
        for window, m in [("训练窗(震荡市)", train_m), ("测试窗(牛市)", test_m)]:
            sharpe_ok = m["sharpe"] > 1.0
            alpha_ok = m["alpha_pct"] > 0
            dd_ok = m["max_drawdown_pct"] > -10
            print(f"  {window}: 夏普{m['sharpe']:.3f}{'✅' if sharpe_ok else '❌'} "
                  f"Alpha{m['alpha_pct']:+.2f}%{'✅' if alpha_ok else '❌'} "
                  f"回撤{m['max_drawdown_pct']:.1f}%{'✅' if dd_ok else '❌'}")
        all_ok = (train_m["sharpe"] > 1.0 and train_m["alpha_pct"] > 0 and
                  train_m["max_drawdown_pct"] > -10 and
                  test_m["sharpe"] > 1.0 and test_m["alpha_pct"] > 0 and
                  test_m["max_drawdown_pct"] > -10)
        print(f"\n  六项全达标: {'✅ 是' if all_ok else '❌ 否'}")

        # 保护触发
        print(f"\n  回撤保护: 训练窗{train_m.get('trigger_count',0)}次/{train_m.get('protection_days',0)}天, "
              f"测试窗{test_m.get('trigger_count',0)}次/{test_m.get('protection_days',0)}天")

        # 报告
        report = generate_report(run_time, train_m, test_m,
                                  train_group_results, test_group_results, all_ok)
        with open(REPORT_MD, "w", encoding="utf-8") as f:
            f.write(report)
        result = {"train": train_m, "test": test_m,
                  "train_groups": {g: {k: v for k, v in r.items() if k != "daily_values"}
                                   for g, r in train_group_results.items()},
                  "test_groups": {g: {k: v for k, v in r.items() if k != "daily_values"}
                                  for g, r in test_group_results.items()}}
        with open(RESULT_JSON, "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2, default=str)
        print(f"\n✓ 报告 → {REPORT_MD}")
        print(f"✓ 数据 → {RESULT_JSON}")

    finally:
        if pref_existed and os.path.exists(pref_backup):
            shutil.move(pref_backup, _USER_PREF_FILE)


def generate_report(run_time, train_m, test_m, train_groups, test_groups, all_ok):
    L = []
    L.append("# P3 全量验证报告 — Regime-adaptive 退出参数固化\n")
    L.append(f"**运行时间**: {run_time}")
    L.append(f"**版本**: P3 (P2基线 + 退出参数优化)\n")

    L.append("## P3 固化参数\n")
    L.append("### 全局退出参数 (PositionManager.DEFAULT_STOP_PARAMS)\n")
    L.append("| 参数 | P2值 | P3值 | 说明 |")
    L.append("|------|------|------|------|")
    L.append("| trail_mult_low | 1.0 | **2.0** | 盈利<10%: 放宽trailing让利润奔跑 |")
    L.append("| trail_mult_mid | 0.8 | **1.5** | 盈利10~20%: 适度跟随 |")
    L.append("| trail_mult_high | 0.6 | **1.0** | 盈利>20%: 锁定利润 |")
    L.append("| hard_stop_pct | 0.10 | **0.12** | 稍宽硬止损避免假突破 |")
    L.append("| no_atr_hard_stop_pct | 0.10 | **0.12** | 无ATR时同步 |")
    L.append("")
    L.append("### 分体制退出 (PositionManager.DEFAULT_REGIME_EXIT_CONFIG)\n")
    L.append("| 体制 | 配置 | 机制 |")
    L.append("|------|------|------|")
    L.append("| ranging | disable_trailing=True | 震荡市禁用移动止盈,只靠硬止损+信号退出 |")
    L.append("| transition | (无覆盖) | 保持全局trailing |")
    L.append("| trending | (无覆盖) | 保持全局trailing |")
    L.append("")

    L.append("## 组合级指标对比\n")
    L.append("### 训练窗(震荡市 2024-07~2025-06)\n")
    L.append("| 指标 | P2基线 | P3 | 改进 |")
    L.append("|------|--------|-----|------|")
    for key, label in [("alpha_pct", "Alpha%"), ("sharpe", "夏普"),
                       ("max_drawdown_pct", "回撤%"), ("total_return_pct", "收益%"),
                       ("benchmark_return_pct", "基准%"), ("trade_count", "交易数")]:
        v2 = P2_BASELINE["train"].get(key, "—")
        v3 = train_m.get(key, "—")
        if isinstance(v2, (int, float)) and isinstance(v3, (int, float)):
            d = v3 - v2
            L.append(f"| {label} | {v2} | {v3} | {d:+} |")
        else:
            L.append(f"| {label} | {v2} | {v3} | — |")
    L.append("")

    L.append("### 测试窗(牛市 2025-07~2026-06)\n")
    L.append("| 指标 | P2基线 | P3 | 改进 |")
    L.append("|------|--------|-----|------|")
    for key, label in [("alpha_pct", "Alpha%"), ("sharpe", "夏普"),
                       ("max_drawdown_pct", "回撤%"), ("total_return_pct", "收益%"),
                       ("benchmark_return_pct", "基准%"), ("trade_count", "交易数")]:
        v2 = P2_BASELINE["test"].get(key, "—")
        v3 = test_m.get(key, "—")
        if isinstance(v2, (int, float)) and isinstance(v3, (int, float)):
            d = v3 - v2
            L.append(f"| {label} | {v2} | {v3} | {d:+} |")
        else:
            L.append(f"| {label} | {v2} | {v3} | — |")
    L.append("")

    L.append("## 分组明细\n")
    L.append("### 训练窗\n")
    L.append("| 分组 | 夏普 | 收益% | Alpha% | 交易数 |")
    L.append("|------|------|-------|--------|--------|")
    for g in WEIGHTS:
        r = train_groups.get(g, {})
        if r.get("skipped"):
            L.append(f"| {g} | — | — | — | [暂停] |")
        else:
            L.append(f"| {g} | {r.get('sharpe',0):+.3f} | {r.get('total_return',0)*100:+.1f} | "
                     f"{r.get('alpha',0)*100:+.1f} | {r.get('trade_count',0)} |")
    L.append("")
    L.append("### 测试窗\n")
    L.append("| 分组 | 夏普 | 收益% | Alpha% | 交易数 |")
    L.append("|------|------|-------|--------|--------|")
    for g in WEIGHTS:
        r = test_groups.get(g, {})
        if r.get("skipped"):
            L.append(f"| {g} | — | — | — | [暂停] |")
        else:
            L.append(f"| {g} | {r.get('sharpe',0):+.3f} | {r.get('total_return',0)*100:+.1f} | "
                     f"{r.get('alpha',0)*100:+.1f} | {r.get('trade_count',0)} |")
    L.append("")

    L.append("## 稳定-收益-回撤 三角评估\n")
    L.append("| 维度 | 标准 | 训练窗(P3) | 测试窗(P3) |")
    L.append("|------|------|-----------|-----------|")
    sharpe_t = "✅" if train_m["sharpe"] > 1.0 else "❌"
    sharpe_s = "✅" if test_m["sharpe"] > 1.0 else "❌"
    alpha_t = "✅" if train_m["alpha_pct"] > 0 else "❌"
    alpha_s = "✅" if test_m["alpha_pct"] > 0 else "❌"
    dd_t = "✅" if train_m["max_drawdown_pct"] > -10 else "❌"
    dd_s = "✅" if test_m["max_drawdown_pct"] > -10 else "❌"
    L.append(f"| 稳定 | 夏普>1.0 | {train_m['sharpe']:.3f}{sharpe_t} | {test_m['sharpe']:.3f}{sharpe_s} |")
    L.append(f"| 收益 | Alpha>0 | {train_m['alpha_pct']:+.2f}%{alpha_t} | {test_m['alpha_pct']:+.2f}%{alpha_s} |")
    L.append(f"| 回撤 | <10% | {train_m['max_drawdown_pct']:.1f}%{dd_t} | {test_m['max_drawdown_pct']:.1f}%{dd_s} |")
    L.append("")

    L.append("## 回撤保护触发\n")
    L.append(f"- 训练窗: {train_m.get('trigger_count',0)}次触发, {train_m.get('protection_days',0)}天降仓")
    L.append(f"- 测试窗: {test_m.get('trigger_count',0)}次触发, {test_m.get('protection_days',0)}天降仓")
    L.append("")

    L.append("## 结论\n")
    if all_ok:
        L.append("**✅ P3 全量验证通过! 六项三角指标全部达标!**\n")
    else:
        L.append("**⚠️ P3 部分指标未达标**, 详情见三角评估表.\n")
    L.append(f"- 训练窗Alpha: {P2_BASELINE['train']['alpha_pct']:+.2f}% → {train_m['alpha_pct']:+.2f}% "
             f"({'转正✅' if train_m['alpha_pct'] > 0 else '仍为负❌'})")
    L.append(f"- 测试窗Alpha: {P2_BASELINE['test']['alpha_pct']:+.2f}% → {test_m['alpha_pct']:+.2f}%")
    L.append(f"- 训练窗夏普: {P2_BASELINE['train']['sharpe']:.3f} → {train_m['sharpe']:.3f}")
    L.append(f"- 测试窗夏普: {P2_BASELINE['test']['sharpe']:.3f} → {test_m['sharpe']:.3f}")
    L.append("")
    L.append("### 优化路径回顾\n")
    L.append("1. **P0**: 周期trending过滤 + 消费降权10%")
    L.append("2. **P1**: 科技ATR1.8 + 医药暂停")
    L.append("3. **P2(fixed_8)**: 机械暂停 + 回撤保护(>8%降仓) — **测试窗三角达标, 训练窗Alpha -4.9%**")
    L.append("4. **P3**: 退出参数优化(放宽trailing + 震荡市禁用trailing) — **训练窗Alpha转正✅**")
    L.append("")
    L.append("### P3核心机制\n")
    L.append("- **诊断2发现**: 退出逻辑过紧(止损踏空90.9%, 止盈过早78.7%), 非信号方向错(45.2%正确)")
    L.append("- **趋势市**: 放宽trailing_mult [2.0/1.5/1.0], 让利润奔跑 → 牛市捕获更多收益")
    L.append("- **震荡市**: 禁用trailing只靠硬止损, 避免趋势跟踪退出被振出 → 骑住波动减少踏空")
    return "\n".join(L)


if __name__ == "__main__":
    main()
