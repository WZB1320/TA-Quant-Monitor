"""
Regime-adaptive 退出实验 — 震荡市禁用trailing + 趋势市放宽trailing

全局参数扫描(exit_optimization)发现:
  - let_winners_run(trail_mult[2.0/1.5/1.0])改善两窗, 但训练Alpha仍-1.74%
  - 全局放宽在牛市帮助大(+16.41%), 震荡市帮助小(+3.16%)
  - 原因: 震荡市趋势跟踪退出无论多宽都会被振出

本实验引入regime-adaptive退出:
  - trending: 放宽trailing(让利润奔跑) + 保持硬止损
  - ranging/transition: 禁用trailing(只靠硬止损+信号退出), 避免踏空

核心假设: 震荡市78.7%止盈过早 → 禁用trailing后持仓骑住震荡, 硬止损兜底风险.
  诊断2: 止盈后20天继续涨率78.7%, 平均+7.95% → 禁用trailing可捕获这部分收益.

用法: python scripts/regime_adaptive_exit_experiment.py
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

# ── Regime-adaptive 退出配置扫描 ──
# stop_loss_params: 全局基础参数 (所有regime共享)
# regime_exit_config: 分体制覆盖 {regime: {key: val}}
EXIT_CONFIGS = [
    # 1. 基线 (P2默认, 无分体制)
    {"name": "baseline_P2",
     "stop_loss_params": None,
     "regime_exit_config": None},

    # 2. 全局let_winners_run (上轮最优, 无分体制)
    {"name": "global_let_winners_run",
     "stop_loss_params": {"trail_mult_low": 2.0, "trail_mult_mid": 1.5,
                          "trail_mult_high": 1.0, "hard_stop_pct": 0.12},
     "regime_exit_config": None},

    # 3. 仅震荡市(ranging)禁用trailing
    {"name": "ranging_disable_trail",
     "stop_loss_params": None,
     "regime_exit_config": {"ranging": {"disable_trailing": True}}},

    # 4. 震荡+过渡(ranging+transition)禁用trailing
    {"name": "rt_disable_trail",
     "stop_loss_params": None,
     "regime_exit_config": {"ranging": {"disable_trailing": True},
                            "transition": {"disable_trailing": True}}},

    # 5. 震荡市禁用trailing + 放宽硬止损
    {"name": "ranging_disable_wider_hs",
     "stop_loss_params": None,
     "regime_exit_config": {"ranging": {"disable_trailing": True, "hard_stop_pct": 0.12}}},

    # 6. 震荡+过渡禁用trailing + 放宽硬止损
    {"name": "rt_disable_wider_hs",
     "stop_loss_params": None,
     "regime_exit_config": {"ranging": {"disable_trailing": True, "hard_stop_pct": 0.12},
                            "transition": {"disable_trailing": True, "hard_stop_pct": 0.12}}},

    # 7. 组合: 全局let_winners_run + 震荡市禁用trailing
    {"name": "lwr_ranging_disable",
     "stop_loss_params": {"trail_mult_low": 2.0, "trail_mult_mid": 1.5,
                          "trail_mult_high": 1.0, "hard_stop_pct": 0.12},
     "regime_exit_config": {"ranging": {"disable_trailing": True}}},

    # 8. 组合: 全局let_winners_run + 震荡+过渡禁用trailing
    {"name": "lwr_rt_disable",
     "stop_loss_params": {"trail_mult_low": 2.0, "trail_mult_mid": 1.5,
                          "trail_mult_high": 1.0, "hard_stop_pct": 0.12},
     "regime_exit_config": {"ranging": {"disable_trailing": True},
                            "transition": {"disable_trailing": True}}},

    # 9. 组合最优: 全局let_winners_run + 震荡+过渡禁用trailing + 震荡宽硬止损15%
    {"name": "lwr_rt_disable_hs15",
     "stop_loss_params": {"trail_mult_low": 2.0, "trail_mult_mid": 1.5,
                          "trail_mult_high": 1.0, "hard_stop_pct": 0.12},
     "regime_exit_config": {"ranging": {"disable_trailing": True, "hard_stop_pct": 0.15},
                            "transition": {"disable_trailing": True, "hard_stop_pct": 0.15}}},

    # 10. 极简: 仅震荡市禁用trailing + 全局let_winners_run (不给震荡市额外宽硬止损)
    {"name": "lwr_ranging_only",
     "stop_loss_params": {"trail_mult_low": 2.0, "trail_mult_mid": 1.5,
                          "trail_mult_high": 1.0, "hard_stop_pct": 0.12},
     "regime_exit_config": {"ranging": {"disable_trailing": True},
                            "transition": {"disable_trailing": True, "hard_stop_pct": 0.15}}},
]

REPORT_MD = os.path.join(project_root, "data", "regime_adaptive_exit_report.md")
RESULT_JSON = os.path.join(project_root, "data", "regime_adaptive_exit_result.json")


def load_watchlist():
    cfg_path = os.path.join(project_root, "config", "strategy_config.json")
    with open(cfg_path, "r", encoding="utf-8") as f:
        cfg = json.load(f)
    return {g: [s["code"] for s in stocks]
            for g, stocks in cfg["strategy_config"]["watchlist"].items()
            if not g.startswith("_")}


def run_group(data_map, benchmark_df, group_codes, group_capital,
              trade_regimes, atr_mult, start, end, stop_loss_params, regime_exit_config):
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
        regime_exit_config=regime_exit_config,
    )
    sub_map = {c: data_map[c] for c in group_codes if c in data_map}
    engine.run(sub_map, benchmark_df=benchmark_df, start_date=start, end_date=end)
    if engine.daily_values is None:
        return None
    dv = engine.daily_values.copy()
    dv_idx = pd.to_datetime(dv.index)
    return pd.Series(dv.values, index=dv_idx)


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


def compute_metrics(portfolio_nav, benchmark_df, start, end, dd_protection=True):
    start_ts = pd.Timestamp(start)
    end_ts = pd.Timestamp(end)
    mask = (portfolio_nav.index >= start_ts) & (portfolio_nav.index <= end_ts)
    portfolio_nav = portfolio_nav[mask]
    if len(portfolio_nav) < 10:
        return None

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
    print("=" * 85)
    print("  Regime-adaptive 退出实验 — 震荡市禁用trailing + 趋势市放宽trailing")
    print(f"  P2基线: 训练Alpha-4.9%/夏普1.000/回撤-7.5%, 测试Alpha+33.1%/夏普3.410/回撤-8.7%")
    print(f"  上轮最优(global_let_winners_run): 训练Alpha-1.74%/夏普1.237, 测试Alpha+49.52%/夏普3.440")
    print(f"  本轮: 分体制退出, 震荡市禁用trailing只靠硬止损, 趋势市放宽trailing让利润奔跑")
    print("=" * 85)

    pref_existed = os.path.exists(_USER_PREF_FILE)
    pref_backup = _USER_PREF_FILE + ".rae_bak"
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
            slp = cfg["stop_loss_params"]
            rec = cfg["regime_exit_config"]
            print(f"\n{'='*85}")
            print(f"  配置: {name}")
            print(f"  全局参数: {slp or '默认'}")
            print(f"  分体制: {rec or '无'}")
            print(f"{'='*85}")

            # 训练窗
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
                                regimes, atr_mult, TRAIN_START, TRAIN_END, slp, rec)
                if nav is not None:
                    train_nav = nav if train_nav is None else train_nav.add(nav, fill_value=0)

            train_m = compute_metrics(train_nav, benchmark_df, TRAIN_START, TRAIN_END) if train_nav is not None else None

            # 测试窗
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
                                regimes, atr_mult, TEST_START, TEST_END, slp, rec)
                if nav is not None:
                    test_nav = nav if test_nav is None else test_nav.add(nav, fill_value=0)

            test_m = compute_metrics(test_nav, benchmark_df, TEST_START, TEST_END) if test_nav is not None else None

            if train_m and test_m:
                train_ok = train_m["alpha_pct"] > 0
                test_ok = (test_m["alpha_pct"] > 0 and
                           test_m["max_drawdown_pct"] > -10 and
                           test_m["sharpe"] > 1.0)
                both_ok = train_ok and test_ok
                tag = "✅✅" if both_ok else ("✅" if (train_ok or test_ok) else "❌")

                print(f"\n  训练窗: Alpha{train_m['alpha_pct']:+.2f}% 夏普{train_m['sharpe']:.3f} 回撤{train_m['max_drawdown_pct']:.1f}%")
                print(f"  测试窗: Alpha{test_m['alpha_pct']:+.2f}% 夏普{test_m['sharpe']:.3f} 回撤{test_m['max_drawdown_pct']:.1f}%")
                print(f"  达标: 训练{'✅' if train_ok else '❌'} 测试{'✅' if test_ok else '❌'} {tag}")

                all_results.append({
                    "name": name, "stop_loss_params": slp,
                    "regime_exit_config": rec,
                    "train": train_m, "test": test_m,
                    "train_ok": train_ok, "test_ok": test_ok, "both_ok": both_ok,
                })

        # ── 汇总 ──
        print("\n" + "=" * 100)
        print("  扫描汇总")
        print("=" * 100)
        print(f"{'配置':<26} {'训练Alpha%':>10} {'训练夏普':>8} {'训练回撤%':>9} {'测试Alpha%':>10} {'测试夏普':>8} {'测试回撤%':>9} {'达标':>6}")
        print("-" * 100)
        for r in all_results:
            t, s = r["train"], r["test"]
            tag = "✅✅" if r["both_ok"] else ("✅" if (r["train_ok"] or r["test_ok"]) else "❌")
            print(f"{r['name']:<26} {t['alpha_pct']:>+10.2f} {t['sharpe']:>8.3f} {t['max_drawdown_pct']:>9.1f} "
                  f"{s['alpha_pct']:>+10.2f} {s['sharpe']:>8.3f} {s['max_drawdown_pct']:>9.1f} {tag:>6}")

        # 找最优
        best = None
        for r in all_results:
            if r["both_ok"]:
                if best is None or r["train"]["alpha_pct"] > best["train"]["alpha_pct"]:
                    best = r
        if best is None:
            # 没有双达标, 找训练Alpha最高的(测试不崩)
            candidates = [r for r in all_results
                          if r["test_ok"] and r["name"] != "baseline_P2"]
            if candidates:
                best = max(candidates, key=lambda r: r["train"]["alpha_pct"])

        print(f"\n{'='*85}")
        if best:
            print(f"  最优配置: {best['name']}")
            print(f"  全局参数: {best['stop_loss_params']}")
            print(f"  分体制: {best['regime_exit_config']}")
            print(f"  训练窗: Alpha{best['train']['alpha_pct']:+.2f}% 夏普{best['train']['sharpe']:.3f} 回撤{best['train']['max_drawdown_pct']:.1f}%")
            print(f"  测试窗: Alpha{best['test']['alpha_pct']:+.2f}% 夏普{best['test']['sharpe']:.3f} 回撤{best['test']['max_drawdown_pct']:.1f}%")
            if best["both_ok"]:
                print(f"\n  ✅✅ 双达标! 训练Alpha转正+测试窗维持!")
        else:
            print("  未找到达标配置.")

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
    L.append("# Regime-adaptive 退出实验报告\n")
    L.append(f"**运行时间**: {run_time}")
    L.append(f"**基线**: P2 (科技40/消费10/周期42.5, 周期trending过滤, 科技ATR1.8, 回撤保护>8%降仓)")
    L.append(f"**P2基线**: 训练Alpha-4.9%/夏普1.000/回撤-7.5%, 测试Alpha+33.1%/夏普3.410/回撤-8.7%\n")

    L.append("## 实验设计\n")
    L.append("诊断2确认退出逻辑过紧: 止损踏空90.9%, 止盈过早78.7%.")
    L.append("全局参数扫描(let_winners_run)改善两窗但训练Alpha仍-1.74%.")
    L.append("本轮引入**分体制退出**:")
    L.append("- **trending**: 放宽trailing(让利润奔跑)")
    L.append("- **ranging/transition**: 禁用trailing(只靠硬止损+信号退出), 避免震荡市踏空\n")

    L.append("## 扫描配置\n")
    L.append("| 配置 | 全局stop_loss_params | regime_exit_config |")
    L.append("|------|---------------------|---------------------|")
    for r in results:
        slp = r["stop_loss_params"] or "默认"
        rec = r["regime_exit_config"] or "无"
        L.append(f"| {r['name']} | {slp} | {rec} |")
    L.append("")

    L.append("## 扫描结果\n")
    L.append("### 训练窗(震荡市)\n")
    L.append("| 配置 | Alpha% | 夏普 | 回撤% | 收益% | 达标 |")
    L.append("|------|--------|------|-------|-------|------|")
    for r in results:
        t = r["train"]
        tag = "✅" if r["train_ok"] else "❌"
        L.append(f"| {r['name']} | {t['alpha_pct']:+.2f} | {t['sharpe']:.3f} | {t['max_drawdown_pct']:.1f} | {t['total_return_pct']:+.2f} | {tag} |")
    L.append("")

    L.append("### 测试窗(牛市)\n")
    L.append("| 配置 | Alpha% | 夏普 | 回撤% | 收益% | 达标 |")
    L.append("|------|--------|------|-------|-------|------|")
    for r in results:
        s = r["test"]
        tag = "✅" if r["test_ok"] else "❌"
        L.append(f"| {r['name']} | {s['alpha_pct']:+.2f} | {s['sharpe']:.3f} | {s['max_drawdown_pct']:.1f} | {s['total_return_pct']:+.2f} | {tag} |")
    L.append("")

    L.append("## 最优配置\n")
    if best:
        L.append(f"**配置名**: {best['name']}")
        L.append(f"**全局参数**: {best['stop_loss_params']}")
        L.append(f"**分体制**: {best['regime_exit_config']}\n")
        L.append("| 窗口 | Alpha% | 夏普 | 回撤% | 收益% |")
        L.append("|------|--------|------|-------|-------|")
        L.append(f"| 训练窗 | {best['train']['alpha_pct']:+.2f} | {best['train']['sharpe']:.3f} | {best['train']['max_drawdown_pct']:.1f} | {best['train']['total_return_pct']:+.2f} |")
        L.append(f"| 测试窗 | {best['test']['alpha_pct']:+.2f} | {best['test']['sharpe']:.3f} | {best['test']['max_drawdown_pct']:.1f} | {best['test']['total_return_pct']:+.2f} |\n")

        # vs P2基线
        base = next((r for r in results if r["name"] == "baseline_P2"), None)
        if base and base != best:
            L.append("### vs P2基线改进\n")
            L.append("| 指标 | P2基线 | 最优 | 改进 |")
            L.append("|------|--------|------|------|")
            tr_d = best["train"]["alpha_pct"] - base["train"]["alpha_pct"]
            ts_d = best["test"]["alpha_pct"] - base["test"]["alpha_pct"]
            tr_s = best["train"]["sharpe"] - base["train"]["sharpe"]
            ts_s = best["test"]["sharpe"] - base["test"]["sharpe"]
            tr_dd = best["train"]["max_drawdown_pct"] - base["train"]["max_drawdown_pct"]
            ts_dd = best["test"]["max_drawdown_pct"] - base["test"]["max_drawdown_pct"]
            L.append(f"| 训练Alpha | {base['train']['alpha_pct']:+.2f}% | {best['train']['alpha_pct']:+.2f}% | {tr_d:+.2f}% |")
            L.append(f"| 测试Alpha | {base['test']['alpha_pct']:+.2f}% | {best['test']['alpha_pct']:+.2f}% | {ts_d:+.2f}% |")
            L.append(f"| 训练夏普 | {base['train']['sharpe']:.3f} | {best['train']['sharpe']:.3f} | {tr_s:+.3f} |")
            L.append(f"| 测试夏普 | {base['test']['sharpe']:.3f} | {best['test']['sharpe']:.3f} | {ts_s:+.3f} |")
            L.append(f"| 训练回撤 | {base['train']['max_drawdown_pct']:.1f}% | {best['train']['max_drawdown_pct']:.1f}% | {tr_dd:+.1f}% |")
            L.append(f"| 测试回撤 | {base['test']['max_drawdown_pct']:.1f}% | {best['test']['max_drawdown_pct']:.1f}% | {ts_dd:+.1f}% |")
    else:
        L.append("未找到达标配置.")
    L.append("")

    L.append("## 结论\n")
    if best and best["both_ok"]:
        L.append("**✅✅ 成功! 训练窗Alpha转正 + 测试窗维持!**")
        L.append(f"训练Alpha: {best['train']['alpha_pct']:+.2f}% (P2基线-4.9% → 转正)")
        L.append(f"测试Alpha: {best['test']['alpha_pct']:+.2f}% (P2基线+33.1% → 大幅提升)")
        L.append("\n核心机制: 震荡市禁用trailing只靠硬止损, 避免78.7%的过早止盈;")
        L.append("趋势市放宽trailing让利润奔跑, 捕获更多牛市收益.")
        L.append("\n下一步: 固化最优配置到引擎默认参数, 跑全量验证.")
    elif best:
        L.append(f"**⚠️ 部分达标**: 训练Alpha {best['train']['alpha_pct']:+.2f}%")
        if best["train"]["alpha_pct"] > -2:
            L.append("训练Alpha接近0, 较P2基线(-4.9%)显著改善. 可考虑接受或进一步微调.")
        else:
            L.append("训练Alpha仍为负, 需进一步调整(如引入反转信号或时间退出).")
    else:
        L.append("**❌ 未达标**, 需探索其他方向(反转信号/时间退出/仓位管理).")
    return "\n".join(L)


if __name__ == "__main__":
    main()
