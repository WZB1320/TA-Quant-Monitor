"""
2026年以来各组收益对比沪深300 — P5配置(消费组暂停)

回测区间: 2026-01-01 ~ 2026-07-13 (数据可用范围)
权重: 科技40% / 消费0%(暂停) / 周期42.5% / 现金17.5%
对比: 各组单独收益 + 组合收益 vs 沪深300

用法: python scripts/ytd_2026_group_comparison.py
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

# 2026年以来: 回测区间 + 预热数据范围
BT_START, BT_END = "2026-01-01", "2026-08-06"
DATA_START, DATA_END = "2025-06-01", "2026-08-06"
BENCHMARK = "sh.000300"
TOTAL_CAPITAL = 1000000

# P5 权重 (消费组暂停)
WEIGHTS = {"科技成长型": 0.40, "消费稳健型": 0.0, "周期资源型": 0.425,
           "医药创新型": 0.0, "机械制造型": 0.0}
REGIMES_CFG = {"周期资源型": {"trending"}}
ATR_OVERRIDE = {"科技成长型": 1.8}
DD_CONFIG = BacktestEngine.DEFAULT_DD_PROTECTION_CONFIG

# 即使权重为0, 也单独跑回测作为参考 (标注"已暂停")
RUN_ALL_GROUPS_AS_REFERENCE = True

REPORT_MD = os.path.join(project_root, "data", "ytd_2026_group_comparison_report.md")


def load_watchlist():
    cfg_path = os.path.join(project_root, "config", "strategy_config.json")
    with open(cfg_path, "r", encoding="utf-8") as f:
        cfg = json.load(f)
    names = {}
    wl = {}
    for g, stocks in cfg["strategy_config"]["watchlist"].items():
        if g.startswith("_"):
            continue
        if not isinstance(stocks, list):
            continue
        wl[g] = [s["code"] for s in stocks]
        for s in stocks:
            names[s["code"]] = s["name"]
    return wl, names


def run_group(data_map, benchmark_df, group_codes, group_capital,
              trade_regimes, atr_mult, start, end, dd_config):
    if group_capital < 1000:
        return {"skipped": True, "sharpe": 0, "total_return": 0, "alpha": 0,
                "max_drawdown": 0, "trade_count": 0, "win_rate": 0,
                "daily_values": None, "dd_triggers": 0}
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
    return {
        "sharpe": getattr(m, "sharpe_ratio", 0) or 0,
        "total_return": getattr(m, "total_return", 0) or 0,
        "alpha": getattr(m, "alpha", 0) or 0,
        "max_drawdown": getattr(m, "max_drawdown", 0) or 0,
        "trade_count": getattr(m, "trade_count", 0) or 0,
        "win_rate": getattr(m, "win_rate", 0) or 0,
        "daily_values": engine.daily_values.copy() if engine.daily_values is not None else None,
        "dd_triggers": engine.dd_protection_stats.get("triggers", 0),
        "skipped": False,
    }


def main():
    run_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print("=" * 95)
    print(f"  2026年以来各组收益对比沪深300")
    print(f"  回测区间: {BT_START} ~ {BT_END}")
    print(f"  配置: P5 (消费组暂停, 科技40%/周期42.5%/现金17.5%)")
    print("=" * 95)

    pref_existed = os.path.exists(_USER_PREF_FILE)
    pref_backup = _USER_PREF_FILE + ".ytd26_bak"
    if pref_existed:
        shutil.copy2(_USER_PREF_FILE, pref_backup)
    UserPreferences().clear_all()

    try:
        watchlist, names = load_watchlist()
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

        # 沪深300同期收益
        bench = benchmark_df.copy()
        if "date" not in bench.columns:
            bench = bench.reset_index()
        bench["date"] = pd.to_datetime(bench["date"])
        bench_s = bench.set_index("date")["close"].astype(float)
        bt_start_ts = pd.Timestamp(BT_START)
        bt_end_ts = pd.Timestamp(BT_END)
        bench_s = bench_s[(bench_s.index >= bt_start_ts) & (bench_s.index <= bt_end_ts)]
        bench_return = (bench_s.iloc[-1] / bench_s.iloc[0]) - 1 if len(bench_s) > 0 else 0
        print(f"\n  沪深300 ({BT_START}~{BT_END}): {bench_return*100:+.2f}%")

        # 各组回测
        print(f"\n{'='*95}")
        print(f"  各组单独回测 (资金={TOTAL_CAPITAL}模拟满仓该组)")
        print(f"{'='*95}")
        group_results = {}
        for g, codes in watchlist.items():
            g_codes = [c for c in codes if c in data_map]
            if len(g_codes) < 2:
                continue
            # 权重为0的组也跑(作为参考), 用模拟资金
            sim_capital = TOTAL_CAPITAL if RUN_ALL_GROUPS_AS_REFERENCE else TOTAL_CAPITAL * WEIGHTS.get(g, 0)
            if sim_capital < 1000:
                group_results[g] = {"skipped": True}
                print(f"  {g:12s}: [跳过-权重0%]")
                continue
            regimes = REGIMES_CFG.get(g)
            atr_mult = ATR_OVERRIDE.get(g, 2.0)
            r = run_group(data_map, benchmark_df, g_codes, sim_capital,
                          regimes, atr_mult, BT_START, BT_END, DD_CONFIG)
            group_results[g] = r
            weight_str = f"权重{WEIGHTS.get(g,0)*100:.0f}%" if WEIGHTS.get(g,0) > 0 else "已暂停(参考)"
            if not r.get("skipped"):
                alpha_vs_bench = r["total_return"] - bench_return
                print(f"  {g:12s}: 收益{r['total_return']*100:+.2f}% Alpha{alpha_vs_bench*100:+.2f}% "
                      f"夏普{r['sharpe']:.3f} 回撤{r['max_drawdown']*100:.1f}% "
                      f"交易{r['trade_count']}笔 胜率{r['win_rate']*100:.0f}% [{weight_str}]")

        # 组合回测 (P5权重)
        print(f"\n{'='*95}")
        print(f"  组合回测 (P5权重: 科技40%/周期42.5%/现金17.5%)")
        print(f"{'='*95}")
        portfolio_nav = None
        total_trades = 0
        total_dd_triggers = 0
        for g, r in group_results.items():
            if r.get("skipped") or r.get("daily_values") is None:
                continue
            w = WEIGHTS.get(g, 0)
            if w == 0:
                continue
            # 用实际权重资金跑
            capital = TOTAL_CAPITAL * w
            regimes = REGIMES_CFG.get(g)
            atr_mult = ATR_OVERRIDE.get(g, 2.0)
            g_codes = [c for c in watchlist[g] if c in data_map]
            r_actual = run_group(data_map, benchmark_df, g_codes, capital,
                                 regimes, atr_mult, BT_START, BT_END, DD_CONFIG)
            nav = r_actual.get("daily_values")
            if nav is not None:
                portfolio_nav = nav if portfolio_nav is None else portfolio_nav.add(nav, fill_value=0)
            total_trades += r_actual.get("trade_count", 0)
            total_dd_triggers += r_actual.get("dd_triggers", 0)

        if portfolio_nav is not None:
            invested = sum(w * TOTAL_CAPITAL for w in WEIGHTS.values() if w > 0)
            cash = TOTAL_CAPITAL - invested
            portfolio_nav = portfolio_nav + cash
            nav_idx = pd.to_datetime(portfolio_nav.index)
            portfolio_nav = pd.Series(portfolio_nav.values, index=nav_idx)
            mask = (portfolio_nav.index >= bt_start_ts) & (portfolio_nav.index <= bt_end_ts)
            portfolio_nav = portfolio_nav[mask]

            daily_ret = portfolio_nav.pct_change().dropna()
            port_return = (portfolio_nav.iloc[-1] / TOTAL_CAPITAL) - 1
            port_sharpe = (daily_ret.mean() / daily_ret.std() * np.sqrt(252)
                           if daily_ret.std() > 0 else 0.0)
            cummax = portfolio_nav.cummax()
            port_dd = ((portfolio_nav - cummax) / cummax).min()
            port_alpha = port_return - bench_return

            print(f"\n  组合: 收益{port_return*100:+.2f}% Alpha{port_alpha*100:+.2f}% "
                  f"夏普{port_sharpe:.3f} 回撤{port_dd*100:.1f}% 交易{total_trades}笔")
            print(f"  沪深300: 收益{bench_return*100:+.2f}%")
            print(f"  组合 vs 沪深300: {'跑赢' if port_return > bench_return else '跑输'} "
                  f"{abs(port_return - bench_return)*100:.2f}个百分点")

        # 汇总表
        print(f"\n{'='*95}")
        print(f"  2026年以来收益汇总 ({BT_START} ~ {BT_END})")
        print(f"{'='*95}")
        print(f"\n  {'组别':<14} {'收益%':>8} {'Alpha%':>8} {'夏普':>7} {'回撤%':>7} {'交易':>5} {'胜率':>5} {'状态':<10}")
        print(f"  {'-'*70}")
        for g, r in group_results.items():
            if r.get("skipped"):
                continue
            alpha_vs = r["total_return"] - bench_return
            w = WEIGHTS.get(g, 0)
            status = f"权重{w*100:.0f}%" if w > 0 else "已暂停"
            print(f"  {g:<14} {r['total_return']*100:>+8.2f} {alpha_vs*100:>+8.2f} "
                  f"{r['sharpe']:>7.3f} {r['max_drawdown']*100:>7.1f} "
                  f"{r['trade_count']:>5} {r['win_rate']*100:>4.0f}% {status:<10}")
        if portfolio_nav is not None:
            print(f"  {'-'*70}")
            print(f"  {'组合(P5)':<14} {port_return*100:>+8.2f} {port_alpha*100:>+8.2f} "
                  f"{port_sharpe:>7.3f} {port_dd*100:>7.1f} {total_trades:>5} {'':>5} {'实盘':<10}")
        print(f"  {'沪深300':<14} {bench_return*100:>+8.2f} {'—':>8} {'—':>7} {'—':>7} {'—':>5} {'—':>5} {'基准':<10}")

        # 报告
        report = generate_report(run_time, group_results, bench_return,
                                 port_return if portfolio_nav is not None else 0,
                                 port_sharpe if portfolio_nav is not None else 0,
                                 port_dd if portfolio_nav is not None else 0,
                                 port_alpha if portfolio_nav is not None else 0,
                                 total_trades, total_dd_triggers)
        with open(REPORT_MD, "w", encoding="utf-8") as f:
            f.write(report)
        print(f"\n✓ 报告 → {REPORT_MD}")

    finally:
        if pref_existed and os.path.exists(pref_backup):
            shutil.move(pref_backup, _USER_PREF_FILE)


def generate_report(run_time, group_results, bench_return, port_return,
                    port_sharpe, port_dd, port_alpha, total_trades, total_dd_triggers):
    L = []
    L.append(f"# 2026年以来各组收益对比沪深300\n")
    L.append(f"**运行时间**: {run_time}")
    L.append(f"**回测区间**: {BT_START} ~ {BT_END}")
    L.append(f"**配置**: P5 (消费组暂停, 科技40%/周期42.5%/现金17.5%)\n")

    L.append("## 收益汇总\n")
    L.append("| 组别 | 收益% | Alpha%(vs沪深300) | 夏普 | 回撤% | 交易数 | 胜率 | 状态 |")
    L.append("|------|-------|------------------|------|-------|--------|------|------|")
    for g, r in group_results.items():
        if r.get("skipped"):
            continue
        alpha_vs = r["total_return"] - bench_return
        w = WEIGHTS.get(g, 0)
        status = f"权重{w*100:.0f}%" if w > 0 else "已暂停(参考)"
        L.append(f"| {g} | {r['total_return']*100:+.2f} | {alpha_vs*100:+.2f} | "
                 f"{r['sharpe']:.3f} | {r['max_drawdown']*100:.1f} | "
                 f"{r['trade_count']} | {r['win_rate']*100:.0f}% | {status} |")
    L.append(f"| **组合(P5)** | **{port_return*100:+.2f}** | **{port_alpha*100:+.2f}** | "
             f"**{port_sharpe:.3f}** | **{port_dd*100:.1f}** | **{total_trades}** | — | 实盘 |")
    L.append(f"| 沪深300 | {bench_return*100:+.2f} | — | — | — | — | — | 基准 |")
    L.append("")

    L.append("## 结论\n")
    L.append(f"- 沪深300同期收益: **{bench_return*100:+.2f}%**")
    L.append(f"- P5组合收益: **{port_return*100:+.2f}%**, Alpha: **{port_alpha*100:+.2f}%**")
    outperform = port_return > bench_return
    L.append(f"- 组合{'跑赢' if outperform else '跑输'}沪深300 {abs(port_return-bench_return)*100:.2f}个百分点")
    L.append(f"- 最大回撤: {port_dd*100:.1f}%, 夏普: {port_sharpe:.3f}")
    L.append(f"- 回撤保护触发: {total_dd_triggers}次")
    return "\n".join(L)


if __name__ == "__main__":
    main()
