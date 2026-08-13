"""
P5 消费组暂停验证 — 消费组权重 10%→0%, 资金转现金

P4 均值回归验证失败 (消费组Alpha -6.5%→-12.8% 恶化), 已回滚.
P5 决策: 暂停消费组 (2024-2026 消费股基本面承压期), 等基本面拐点再启用.

权重变化:
  P3+ : 科技40% / 消费10% / 周期42.5% / 医药0% / 机械0% / 现金7.5%
  P5  : 科技40% / 消费0%  / 周期42.5% / 医药0% / 机械0% / 现金17.5%
  (消费组10%转现金, 科技/周期不变, 降低集中度风险)

P3+ 基线 (对照):
  训练窗: Alpha+1.59%/夏普1.437/回撤-7.6%/收益+14.76%
  测试窗: Alpha+46.11%/夏普3.341/回撤-7.5%/收益+72.40%

用法: python scripts/p5_consumer_paused_validation.py
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

# P5: 消费组暂停 (10%→0%), 转现金
WEIGHTS = {"科技成长型": 0.40, "消费稳健型": 0.0, "周期资源型": 0.425,
           "医药创新型": 0.0, "机械制造型": 0.0}
REGIMES_CFG = {"周期资源型": {"trending"}}
ATR_OVERRIDE = {"科技成长型": 1.8}
DD_CONFIG = BacktestEngine.DEFAULT_DD_PROTECTION_CONFIG

# P3+ 基线 (对照)
P3PLUS_BASELINE = {
    "train": {"alpha_pct": 1.59, "sharpe": 1.437, "max_drawdown_pct": -7.6,
              "total_return_pct": 14.76},
    "test": {"alpha_pct": 46.11, "sharpe": 3.341, "max_drawdown_pct": -7.5,
             "total_return_pct": 72.40},
}

REPORT_MD = os.path.join(project_root, "data", "p5_consumer_paused_validation_report.md")
RESULT_JSON = os.path.join(project_root, "data", "p5_consumer_paused_validation_result.json")


def load_watchlist():
    cfg_path = os.path.join(project_root, "config", "strategy_config.json")
    with open(cfg_path, "r", encoding="utf-8") as f:
        cfg = json.load(f)
    return {g: [s["code"] for s in stocks]
            for g, stocks in cfg["strategy_config"]["watchlist"].items()
            if not g.startswith("_")}


def run_group(data_map, benchmark_df, group_codes, group_capital,
              trade_regimes, atr_mult, start, end, dd_config):
    if group_capital < 1000:
        return {"skipped": True, "daily_values": None, "trade_count": 0,
                "sharpe": 0, "total_return": 0, "alpha": 0, "max_drawdown": 0,
                "win_rate": 0, "final_value": group_capital,
                "dd_stats": {"enabled": False, "triggers": 0, "reduce_days": 0},
                "dd_reduce_trades": 0}
    GroupConfig._instance = None
    GroupConfig._config = None
    engine = BacktestEngine(
        initial_capital=group_capital, lookback_days=120, position_ratio=0.3,
        commission_rate=0.00025, stamp_tax=0.001, slippage=0.0001,
        signal_dedup_days=5, risk_per_trade=0.05, atr_stop_mult=atr_mult,
        forced_regime=None, trade_regimes=trade_regimes,
        dd_protection_config=dd_config,
    )
    sub_map = {c: data_map[c] for c in group_codes if c in data_map}
    m = engine.run(sub_map, benchmark_df=benchmark_df, start_date=start, end_date=end)
    dd_reduce_trades = sum(1 for t in engine.position_mgr.closed_trades
                           if "回撤保护" in (t.exit_signal or ""))
    return {
        "sharpe": getattr(m, "sharpe_ratio", 0) or 0,
        "total_return": getattr(m, "total_return", 0) or 0,
        "alpha": getattr(m, "alpha", 0) or 0,
        "max_drawdown": getattr(m, "max_drawdown", 0) or 0,
        "trade_count": getattr(m, "trade_count", 0) or 0,
        "win_rate": getattr(m, "win_rate", 0) or 0,
        "final_value": getattr(m, "final_value", group_capital) or group_capital,
        "daily_values": engine.daily_values.copy() if engine.daily_values is not None else None,
        "dd_stats": engine.dd_protection_stats,
        "dd_reduce_trades": dd_reduce_trades,
        "skipped": False,
    }


def compute_portfolio_metrics(group_results, benchmark_df, start, end):
    portfolio_nav = None
    total_trades = 0
    total_dd_triggers = 0
    total_dd_reduce_trades = 0
    for g, r in group_results.items():
        if r.get("skipped") or r.get("daily_values") is None:
            continue
        nav = r["daily_values"]
        portfolio_nav = nav if portfolio_nav is None else portfolio_nav.add(nav, fill_value=0)
        total_trades += r["trade_count"]
        ds = r.get("dd_stats", {})
        total_dd_triggers += ds.get("triggers", 0)
        total_dd_reduce_trades += r.get("dd_reduce_trades", 0)

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
        "dd_triggers": total_dd_triggers,
        "dd_reduce_trades": total_dd_reduce_trades,
    }


def main():
    run_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print("=" * 95)
    print("  P5 消费组暂停验证 — 消费组权重 10%→0%, 资金转现金")
    print(f"  权重: 科技40% / 周期42.5% / 现金17.5% (消费/医药/机械暂停)")
    print(f"  训练窗: {TRAIN_START}~{TRAIN_END} (震荡市)")
    print(f"  测试窗: {TEST_START}~{TEST_END} (牛市)")
    print("=" * 95)

    pref_existed = os.path.exists(_USER_PREF_FILE)
    pref_backup = _USER_PREF_FILE + ".p5val_bak"
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

        for window_name, ws, we in [("训练窗", TRAIN_START, TRAIN_END),
                                     ("测试窗", TEST_START, TEST_END)]:
            print(f"\n{'='*95}")
            print(f"  [{window_name} {ws}~{we}]")
            print(f"{'='*95}")
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
                              regimes, atr_mult, ws, we, DD_CONFIG)
                group_results[g] = r
                if not r.get("skipped"):
                    print(f"  {g:12s}: 夏普{r['sharpe']:+.3f} 收益{r['total_return']*100:+.1f}% "
                          f"Alpha{r['alpha']*100:+.1f}% 交易{r['trade_count']}笔 "
                          f"胜率{r['win_rate']*100:.0f}%")

            m = compute_portfolio_metrics(group_results, benchmark_df, ws, we)
            if window_name == "训练窗":
                train_m = m
            else:
                test_m = m

            print(f"\n  组合: 夏普{m['sharpe']:.3f} 收益{m['total_return_pct']:+.2f}% "
                  f"Alpha{m['alpha_pct']:+.2f}% 回撤{m['max_drawdown_pct']:.1f}% "
                  f"交易{m['trade_count']}笔")

        # ── 汇总对比 ──
        print(f"\n{'='*95}")
        print(f"  P5 消费组暂停 vs P3+ 基线 对比")
        print(f"{'='*95}")

        print(f"\n  {'指标':<14} {'P3+基线':>10} {'P5暂停消费':>12} {'改进':>10}")
        print(f"  {'-'*52}")
        for window, p3p, p5 in [("训练窗", P3PLUS_BASELINE["train"], train_m),
                                ("测试窗", P3PLUS_BASELINE["test"], test_m)]:
            print(f"\n  [{window} 组合]")
            for key, label in [("alpha_pct", "Alpha%"), ("sharpe", "夏普"),
                               ("max_drawdown_pct", "回撤%"), ("total_return_pct", "收益%")]:
                v3 = p3p.get(key, 0)
                v5 = p5.get(key, 0)
                d = v5 - v3
                print(f"  {label:<14} {v3:>+10.2f} {v5:>+12.2f} {d:>+10.2f}")

        # 三角评估
        print(f"\n{'='*95}")
        print(f"  稳定-收益-回撤 三角评估 (P5)")
        print(f"{'='*95}")
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

        # 报告
        report = generate_report(run_time, train_m, test_m, all_ok)
        with open(REPORT_MD, "w", encoding="utf-8") as f:
            f.write(report)
        result = {"train": train_m, "test": test_m}
        with open(RESULT_JSON, "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2, default=str)
        print(f"\n✓ 报告 → {REPORT_MD}")
        print(f"✓ 数据 → {RESULT_JSON}")

    finally:
        if pref_existed and os.path.exists(pref_backup):
            shutil.move(pref_backup, _USER_PREF_FILE)


def generate_report(run_time, train_m, test_m, all_ok):
    L = []
    L.append("# P5 消费组暂停验证报告\n")
    L.append(f"**运行时间**: {run_time}")
    L.append(f"**版本**: P5 (P3+ + 消费组暂停)\n")
    L.append("## 改动\n")
    L.append("| 项 | P3+ | P5 |")
    L.append("|----|-----|-----|")
    L.append("| 消费组权重 | 10% | 0% (暂停) |")
    L.append("| 现金比例 | 7.5% | 17.5% |")
    L.append("| 科技/周期权重 | 40%/42.5% | 40%/42.5% (不变) |")
    L.append("")
    L.append("> P4 均值回归验证失败已回滚, 消费组配置恢复P3+趋势跟踪, 仅暂停权重.\n")

    L.append("## 组合级指标对比\n")
    for window, p3p, p5 in [("训练窗(震荡市)", P3PLUS_BASELINE["train"], train_m),
                            ("测试窗(牛市)", P3PLUS_BASELINE["test"], test_m)]:
        L.append(f"### {window}\n")
        L.append("| 指标 | P3+基线 | P5暂停消费 | 改进 |")
        L.append("|------|---------|-----------|------|")
        for key, lbl in [("alpha_pct", "Alpha%"), ("sharpe", "夏普"),
                         ("max_drawdown_pct", "回撤%"), ("total_return_pct", "收益%")]:
            v3 = p3p.get(key, 0)
            v5 = p5.get(key, 0)
            d = v5 - v3
            L.append(f"| {lbl} | {v3} | {v5} | {d:+} |")
        L.append("")

    L.append("## 三角评估\n")
    L.append("| 维度 | 标准 | 训练窗 | 测试窗 |")
    L.append("|------|------|--------|--------|")
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
    L.append("## 结论\n")
    if all_ok:
        L.append("**✅ P5 全量验证通过! 六项三角指标全部达标!**\n")
    else:
        L.append("**⚠️ P5 部分指标未达标**, 详情见三角评估表.\n")
    return "\n".join(L)


if __name__ == "__main__":
    main()
