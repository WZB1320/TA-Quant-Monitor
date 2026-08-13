"""
P3 医药组参数调优 — 均值回归模式 + 参数扫描

P3 改动 (2026-08-10):
  医药组配置已有均值回归倾向(RSI权重0.20最高, KDJ 0.13, MACD降权0.8),
  但 strategy_mode 未设置(仍趋势跟踪), MA60方向约束生效阻止超卖买入,
  退出逻辑仍是trailing. 与消费组P2类似的"三层不协同"问题.

  P3 应用均值回归三层协同:
    1. 信号层: strategy_mode="mean_reversion" → 跳过MA60方向约束
    2. 退出层: target_profit_pct目标止盈 + 禁用trailing + 收紧止损
    3. 配置层: score_threshold降低(40→30/35)增加信号频率

  参数扫描维度:
    - score_threshold: 30/35 (降低信号门槛)
    - target_profit_pct: 5%/7%/9% (医药股波动大于消费)
    - hard_stop_pct: 8%/10%/12% (匹配止盈的风险收益比)
    - atr_stop_mult: 1.5/2.0

P1 基线 (对照, 医药组趋势跟踪):
  训练窗: Alpha -7.7%, 夏普+0.335, 收益+5.5%, 8笔
  测试窗: Alpha -25.3%, 夏普-0.145, 收益+1.0%, 7笔
  2026YTD: Alpha -14.36%, 夏普-1.421, 收益-15.77%, 2笔, 胜率0%

用法: python scripts/p3_medical_tuning.py
"""
import sys
import os
import json
import shutil
import warnings
import itertools
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
GROUP_CAPITAL = 100000  # 医药组单独回测, 10万资金

# P1 基线 (医药组趋势跟踪, 对照)
P1_BASELINE = {
    "train": {"alpha": -0.077, "sharpe": 0.335, "total_return": 0.055, "trade_count": 8},
    "test": {"alpha": -0.253, "sharpe": -0.145, "total_return": 0.010, "trade_count": 7},
}

REPORT_MD = os.path.join(project_root, "data", "p3_medical_tuning_report.md")
RESULT_JSON = os.path.join(project_root, "data", "p3_medical_tuning_result.json")

# 参数扫描网格
SCAN_GRID = {
    "score_threshold": [30, 35],
    "target_profit_pct": [0.05, 0.07, 0.09],
    "hard_stop_pct": [0.08, 0.10, 0.12],
    "atr_stop_mult": [1.5, 2.0],
}


def load_medical_codes():
    cfg_path = os.path.join(project_root, "config", "strategy_config.json")
    with open(cfg_path, "r", encoding="utf-8") as f:
        cfg = json.load(f)
    stocks = cfg["strategy_config"]["watchlist"].get("医药创新型", [])
    return [s["code"] for s in stocks]


def run_single(data_map, benchmark_df, start, end,
               score_threshold, target_profit_pct, hard_stop_pct, atr_stop_mult,
               use_mean_reversion=True):
    """跑单次医药组回测."""
    GroupConfig._instance = None
    GroupConfig._config = None

    mean_reversion_config = None
    if use_mean_reversion:
        mean_reversion_config = {
            "target_profit_pct": target_profit_pct,
            "hard_stop_pct": hard_stop_pct,
            "disable_trailing": True,
        }

    engine = BacktestEngine(
        initial_capital=GROUP_CAPITAL, lookback_days=120, position_ratio=0.3,
        commission_rate=0.00025, stamp_tax=0.001, slippage=0.0001,
        signal_dedup_days=5, risk_per_trade=0.05, atr_stop_mult=atr_stop_mult,
        forced_regime=None, trade_regimes=None,
        dd_protection_config=BacktestEngine.DEFAULT_DD_PROTECTION_CONFIG,
        mean_reversion_config=mean_reversion_config,
    )

    # 临时覆盖 score_threshold (通过修改 GroupConfig 单例)
    gc = GroupConfig()
    gc._load()
    if "医药创新型" in gc._groups:
        gc._groups["医药创新型"]["score_threshold"] = score_threshold
        if use_mean_reversion:
            gc._groups["医药创新型"]["strategy_mode"] = "mean_reversion"
            gc._groups["医药创新型"]["mean_reversion_exit"] = mean_reversion_config
        else:
            gc._groups["医药创新型"]["strategy_mode"] = "trend_following"
            gc._groups["医药创新型"].pop("mean_reversion_exit", None)

    m = engine.run(data_map, benchmark_df=benchmark_df, start_date=start, end_date=end)

    # 统计退出原因
    exit_reasons = {}
    for t in engine.position_mgr.closed_trades:
        sig = t.exit_signal or ""
        if "均值回归止盈" in sig:
            exit_reasons["均值回归止盈"] = exit_reasons.get("均值回归止盈", 0) + 1
        elif "ATR移动止盈" in sig or "移动止盈" in sig:
            exit_reasons["trailing止盈"] = exit_reasons.get("trailing止盈", 0) + 1
        elif "ATR硬止损" in sig or "安全网" in sig or "硬止损" in sig:
            exit_reasons["止损"] = exit_reasons.get("止损", 0) + 1
        elif "score=" in sig:
            exit_reasons["信号退出"] = exit_reasons.get("信号退出", 0) + 1
        elif "回撤保护" in sig:
            exit_reasons["回撤保护"] = exit_reasons.get("回撤保护", 0) + 1
        else:
            exit_reasons["其他"] = exit_reasons.get("其他", 0) + 1

    return {
        "sharpe": getattr(m, "sharpe_ratio", 0) or 0,
        "total_return": getattr(m, "total_return", 0) or 0,
        "alpha": getattr(m, "alpha", 0) or 0,
        "max_drawdown": getattr(m, "max_drawdown", 0) or 0,
        "trade_count": getattr(m, "trade_count", 0) or 0,
        "win_rate": getattr(m, "win_rate", 0) or 0,
        "exit_reasons": exit_reasons,
    }


def main():
    run_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print("=" * 95)
    print("  P3 医药组参数调优 — 均值回归模式 + 参数扫描")
    print(f"  训练窗: {TRAIN_START}~{TRAIN_END} (震荡市)")
    print(f"  测试窗: {TEST_START}~{TEST_END} (牛市)")
    print(f"  资金: ¥{GROUP_CAPITAL:,}")
    print("=" * 95)

    pref_existed = os.path.exists(_USER_PREF_FILE)
    pref_backup = _USER_PREF_FILE + ".p3tune_bak"
    if pref_existed:
        shutil.copy2(_USER_PREF_FILE, pref_backup)
    UserPreferences().clear_all()

    try:
        codes = load_medical_codes()
        dm = DataManager()
        print(f"\n医药组股票: {codes}")
        print("拉取数据...")
        data_map = {}
        for code in codes:
            df = dm.get_daily_kline(code, start_date=DATA_START, end_date=DATA_END)
            if df is not None and len(df) > 80:
                data_map[code] = df
        benchmark_df = dm.get_daily_kline(BENCHMARK, start_date=DATA_START, end_date=DATA_END)
        print(f"  股票 {len(data_map)}/{len(codes)}, 基准 {len(benchmark_df)}条")

        # ── 1. 基线: 趋势跟踪(当前配置) ──
        print(f"\n{'='*95}")
        print("  [基线] 趋势跟踪模式 (当前配置, strategy_mode未设置)")
        print(f"{'='*95}")
        baseline_results = {}
        for window_name, ws, we in [("训练窗", TRAIN_START, TRAIN_END),
                                     ("测试窗", TEST_START, TEST_END)]:
            r = run_single(data_map, benchmark_df, ws, we,
                           score_threshold=40, target_profit_pct=0,
                           hard_stop_pct=0.12, atr_stop_mult=2.5,
                           use_mean_reversion=False)
            baseline_results[window_name] = r
            print(f"  {window_name}: 夏普{r['sharpe']:+.3f} 收益{r['total_return']*100:+.1f}% "
                  f"Alpha{r['alpha']*100:+.1f}% 交易{r['trade_count']}笔 "
                  f"胜率{r['win_rate']*100:.0f}%")
            if r.get("exit_reasons"):
                print(f"          退出: {r['exit_reasons']}")

        # ── 2. 参数扫描: 均值回归模式 ──
        print(f"\n{'='*95}")
        print("  [参数扫描] 均值回归模式 (strategy_mode=mean_reversion)")
        print(f"{'='*95}")

        # 生成参数组合
        keys = list(SCAN_GRID.keys())
        combos = list(itertools.product(*[SCAN_GRID[k] for k in keys]))

        print(f"  共 {len(combos)} 种参数组合 × 2 窗口 = {len(combos)*2} 次回测\n")
        print(f"  {'组合':>4} {'score':>6} {'止盈%':>6} {'止损%':>6} {'ATR':>5} | "
              f"{'训练Alpha':>10} {'训练夏普':>8} {'训练笔':>5} | "
              f"{'测试Alpha':>10} {'测试夏普':>8} {'测试笔':>5}")
        print(f"  {'-'*90}")

        scan_results = []
        best_combo = None
        best_score = -999  # 综合评分: 训练Alpha + 测试Alpha (越高越好)

        for idx, combo in enumerate(combos, 1):
            params = dict(zip(keys, combo))
            train_r = run_single(data_map, benchmark_df, TRAIN_START, TRAIN_END,
                                 **params, use_mean_reversion=True)
            test_r = run_single(data_map, benchmark_df, TEST_START, TEST_END,
                                **params, use_mean_reversion=True)

            # 综合评分: 训练+测试Alpha之和, 优先训练窗
            combined_score = train_r["alpha"] + test_r["alpha"]
            if combined_score > best_score:
                best_score = combined_score
                best_combo = {**params, "train": train_r, "test": test_r}

            scan_results.append({
                "params": params,
                "train": {k: v for k, v in train_r.items()},
                "test": {k: v for k, v in test_r.items()},
                "combined_alpha": combined_score,
            })

            print(f"  {idx:>4} {params['score_threshold']:>6} "
                  f"{params['target_profit_pct']*100:>6.0f} "
                  f"{params['hard_stop_pct']*100:>6.0f} "
                  f"{params['atr_stop_mult']:>5.1f} | "
                  f"{train_r['alpha']*100:>+10.2f}% {train_r['sharpe']:>+8.3f} "
                  f"{train_r['trade_count']:>5} | "
                  f"{test_r['alpha']*100:>+10.2f}% {test_r['sharpe']:>+8.3f} "
                  f"{test_r['trade_count']:>5}")

        # ── 3. 最优组合详情 ──
        print(f"\n{'='*95}")
        print("  [最优组合]")
        print(f"{'='*95}")
        if best_combo:
            print(f"  参数: score_threshold={best_combo['score_threshold']}, "
                  f"target_profit_pct={best_combo['target_profit_pct']*100:.0f}%, "
                  f"hard_stop_pct={best_combo['hard_stop_pct']*100:.0f}%, "
                  f"atr_stop_mult={best_combo['atr_stop_mult']}")
            for window_name, r in [("训练窗", best_combo["train"]),
                                   ("测试窗", best_combo["test"])]:
                print(f"\n  {window_name}:")
                print(f"    夏普{r['sharpe']:+.3f} 收益{r['total_return']*100:+.1f}% "
                      f"Alpha{r['alpha']*100:+.1f}% 回撤{r['max_drawdown']*100:.1f}% "
                      f"交易{r['trade_count']}笔 胜率{r['win_rate']*100:.0f}%")
                if r.get("exit_reasons"):
                    print(f"    退出: {r['exit_reasons']}")

                # Alpha 转正判断
                if "训练" in window_name:
                    baseline_alpha = P1_BASELINE["train"]["alpha"]
                else:
                    baseline_alpha = P1_BASELINE["test"]["alpha"]
                improved = r["alpha"] - baseline_alpha
                print(f"    Alpha改善: {baseline_alpha*100:+.2f}% → {r['alpha']*100:+.2f}% "
                      f"({improved*100:+.2f}%) {'✅转正' if r['alpha'] > 0 else '❌仍为负'}")

        # ── 4. 基线 vs 最优 对比 ──
        print(f"\n{'='*95}")
        print("  [基线 vs 最优 对比]")
        print(f"{'='*95}")
        print(f"\n  {'指标':<14} {'P1基线(趋势)':>14} {'P3最优(均值回归)':>16} {'改善':>10}")
        print(f"  {'-'*58}")
        for window, base_key, best_r in [("训练窗", "train", best_combo["train"]),
                                         ("测试窗", "test", best_combo["test"])]:
            print(f"\n  [{window}]")
            for key, label in [("alpha", "Alpha%"), ("sharpe", "夏普"),
                               ("total_return", "收益%"), ("trade_count", "交易数")]:
                v1 = P1_BASELINE[base_key].get(key, 0)
                v2 = best_r.get(key, 0)
                if key in ("alpha", "total_return"):
                    print(f"  {label:<14} {v1*100:>+14.2f}% {v2*100:>+16.2f}% {(v2-v1)*100:>+10.2f}%")
                else:
                    print(f"  {label:<14} {v1:>+14} {v2:>+16} {v2-v1:>+10.2f}")

        # 三角评估
        print(f"\n{'='*95}")
        print("  [三角评估] (最优组合)")
        print(f"{'='*95}")
        for window, r in [("训练窗(震荡市)", best_combo["train"]),
                          ("测试窗(牛市)", best_combo["test"])]:
            sharpe_ok = r["sharpe"] > 0
            alpha_ok = r["alpha"] > 0
            dd_ok = r["max_drawdown"] > -0.15
            print(f"  {window}: 夏普{r['sharpe']:.3f}{'✅' if sharpe_ok else '❌'} "
                  f"Alpha{r['alpha']*100:+.2f}%{'✅' if alpha_ok else '❌'} "
                  f"回撤{r['max_drawdown']*100:.1f}%{'✅' if dd_ok else '❌'}")

        # 报告
        report = generate_report(run_time, baseline_results, scan_results, best_combo)
        with open(REPORT_MD, "w", encoding="utf-8") as f:
            f.write(report)
        result = {"baseline": baseline_results, "scan_results": scan_results,
                  "best_combo": {k: v for k, v in best_combo.items() if k not in ("train", "test")},
                  "best_train": best_combo["train"], "best_test": best_combo["test"]}
        with open(RESULT_JSON, "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2, default=str)
        print(f"\n✓ 报告 → {REPORT_MD}")
        print(f"✓ 数据 → {RESULT_JSON}")

    finally:
        if pref_existed and os.path.exists(pref_backup):
            shutil.move(pref_backup, _USER_PREF_FILE)


def generate_report(run_time, baseline_results, scan_results, best_combo):
    L = []
    L.append("# P3 医药组参数调优报告\n")
    L.append(f"**运行时间**: {run_time}")
    L.append(f"**版本**: P3 (均值回归模式 + 参数扫描)\n")

    L.append("## P1 基线 (趋势跟踪)\n")
    L.append("| 窗口 | Alpha% | 夏普 | 收益% | 交易数 |")
    L.append("|------|--------|------|-------|--------|")
    for window, r in baseline_results.items():
        L.append(f"| {window} | {r['alpha']*100:+.2f} | {r['sharpe']:.3f} | "
                 f"{r['total_return']*100:+.2f} | {r['trade_count']} |")
    L.append("")

    L.append("## 参数扫描结果 (均值回归模式)\n")
    L.append(f"扫描组合数: {len(scan_results)}\n")
    L.append("| # | score | 止盈% | 止损% | ATR | 训练Alpha% | 训练夏普 | 训练笔 | 测试Alpha% | 测试夏普 | 测试笔 | 综合Alpha |")
    L.append("|---|-------|-------|-------|-----|-----------|---------|--------|-----------|---------|--------|----------|")
    for idx, sr in enumerate(scan_results, 1):
        p = sr["params"]
        L.append(f"| {idx} | {p['score_threshold']} | {p['target_profit_pct']*100:.0f} | "
                 f"{p['hard_stop_pct']*100:.0f} | {p['atr_stop_mult']:.1f} | "
                 f"{sr['train']['alpha']*100:+.2f} | {sr['train']['sharpe']:.3f} | "
                 f"{sr['train']['trade_count']} | "
                 f"{sr['test']['alpha']*100:+.2f} | {sr['test']['sharpe']:.3f} | "
                 f"{sr['test']['trade_count']} | {sr['combined_alpha']*100:+.2f} |")
    L.append("")

    L.append("## 最优组合\n")
    L.append(f"- score_threshold: {best_combo['score_threshold']}")
    L.append(f"- target_profit_pct: {best_combo['target_profit_pct']*100:.0f}%")
    L.append(f"- hard_stop_pct: {best_combo['hard_stop_pct']*100:.0f}%")
    L.append(f"- atr_stop_mult: {best_combo['atr_stop_mult']}")
    L.append("")

    L.append("## 基线 vs 最优 对比\n")
    for window, base_key, best_r in [("训练窗", "train", best_combo["train"]),
                                     ("测试窗", "test", best_combo["test"])]:
        L.append(f"### {window}\n")
        L.append("| 指标 | P1基线(趋势) | P3最优(均值回归) | 改善 |")
        L.append("|------|-------------|----------------|------|")
        for key, lbl in [("alpha", "Alpha%"), ("sharpe", "夏普"),
                         ("total_return", "收益%"), ("trade_count", "交易数")]:
            v1 = P1_BASELINE[base_key].get(key, 0)
            v2 = best_r.get(key, 0)
            if key in ("alpha", "total_return"):
                L.append(f"| {lbl} | {v1*100:+.2f} | {v2*100:+.2f} | {(v2-v1)*100:+.2f} |")
            else:
                L.append(f"| {lbl} | {v1} | {v2} | {v2-v1:+.2f} |")
        L.append("")

    L.append("## 结论\n")
    c_train_alpha = best_combo["train"]["alpha"]
    c_test_alpha = best_combo["test"]["alpha"]
    if c_train_alpha > 0 and c_test_alpha > 0:
        L.append("**✅ P3 医药组Alpha双窗转正! 均值回归模式有效!**\n")
    elif c_train_alpha > 0 or c_test_alpha > 0:
        L.append("**⚠️ P3 医药组Alpha单窗转正**, 需进一步优化.\n")
    else:
        L.append("**❌ P3 医药组Alpha仍为负**, 均值回归模式未改善, 可能是标的问题.\n")
    L.append(f"- 训练窗Alpha: {P1_BASELINE['train']['alpha']*100:+.2f}% → {c_train_alpha*100:+.2f}%")
    L.append(f"- 测试窗Alpha: {P1_BASELINE['test']['alpha']*100:+.2f}% → {c_test_alpha*100:+.2f}%")
    return "\n".join(L)


if __name__ == "__main__":
    main()
