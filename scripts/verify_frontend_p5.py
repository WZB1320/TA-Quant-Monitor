"""
验证前端 backtest.py P5 配置同步后的回测结果
模拟前端路由逻辑: 分组独立回测 + 权重合并
对比: 前端配置 vs 脚本配置, 确认收益一致(~34%)

用法: python scripts/verify_frontend_p5.py
"""
import sys, os, json, shutil, warnings
import pandas as pd
import numpy as np

warnings.filterwarnings("ignore")
project_root = os.path.join(os.path.dirname(__file__), "..")
sys.path.insert(0, project_root)

from src.config.runtime_mode import set_mode, RuntimeMode
set_mode(RuntimeMode.BACKTEST)

from src.backtest.engine import BacktestEngine
from src.backtest.metrics import compute_metrics
from src.data_fetcher import DataManager, Watchlist
from src.config.group_config import GroupConfig
from src.config.user_preferences import UserPreferences, _USER_PREF_FILE

# ── 完全复制 backtest.py 的 P5 配置 ──
PORTFOLIO_WEIGHTS = {
    "科技成长型": 0.40, "消费稳健型": 0.0, "周期资源型": 0.425,
    "医药创新型": 0.0, "机械制造型": 0.0,
}
REGIMES_CFG = {"周期资源型": {"trending"}}
ATR_OVERRIDE = {"科技成长型": 1.8}
DD_CONFIG = BacktestEngine.DEFAULT_DD_PROTECTION_CONFIG

BT_START, BT_END = "2026-01-01", "2026-08-06"
DATA_START = "2025-06-01"
INITIAL_CAPITAL = 100000  # 前端默认10万


def main():
    print("=" * 90)
    print("  验证前端 backtest.py P5 配置同步后的回测结果")
    print(f"  区间: {BT_START} ~ {BT_END}, 初始资金: {INITIAL_CAPITAL}")
    print("=" * 90)

    pref_existed = os.path.exists(_USER_PREF_FILE)
    pref_backup = _USER_PREF_FILE + ".vf_bak"
    if pref_existed:
        shutil.copy2(_USER_PREF_FILE, pref_backup)
    UserPreferences().clear_all()

    try:
        wl = Watchlist()
        all_stocks = wl.get_all()
        gc = GroupConfig()

        dm = DataManager()
        data_map = {}
        stock_info = {}
        for stock in all_stocks:
            code = stock["code"]
            df = dm.get_daily_kline(code, start_date=DATA_START, end_date=BT_END)
            if df is not None and len(df) >= 120:
                data_map[code] = df
                stock_info[code] = {
                    "name": stock.get("name", code),
                    "group": gc.get_group(code),
                }

        bench_df = dm.get_daily_kline("sh.000300", start_date=DATA_START, end_date=BT_END)
        print(f"  股票 {len(data_map)}/{len(all_stocks)}, 基准 {len(bench_df)}条")

        # ── 模拟前端 backtest.py 的分组独立回测逻辑 (修复后) ──
        active_groups = [g for g, w in PORTFOLIO_WEIGHTS.items() if w > 0]
        # 全部组参与时不归一化, 保留现金缓冲
        weights_normalized = {g: PORTFOLIO_WEIGHTS[g] for g in active_groups}

        portfolio_nav = None
        all_closed_trades = []
        group_results = {}

        for group_name in active_groups:
            weight = weights_normalized[group_name]
            group_capital = INITIAL_CAPITAL * weight

            group_codes = [c for c, info in stock_info.items() if info["group"] == group_name and c in data_map]
            if not group_codes:
                continue

            GroupConfig._instance = None
            GroupConfig._config = None

            engine = BacktestEngine(
                initial_capital=group_capital, lookback_days=120, position_ratio=0.3,
                commission_rate=0.00025, stamp_tax=0.001, slippage=0.0001,
                signal_dedup_days=5, risk_per_trade=0.05,
                atr_stop_mult=ATR_OVERRIDE.get(group_name, 2.0),
                group_config=gc, forced_regime="auto",
                benchmark_df_for_memory=bench_df,
                trade_regimes=REGIMES_CFG.get(group_name),
                dd_protection_config=DD_CONFIG,
            )
            sub_map = {c: data_map[c] for c in group_codes}
            m = engine.run(sub_map, benchmark_df=bench_df, start_date=BT_START, end_date=BT_END)

            if engine.daily_values is not None:
                portfolio_nav = engine.daily_values.copy() if portfolio_nav is None else portfolio_nav.add(engine.daily_values, fill_value=0)
            all_closed_trades.extend(engine.position_mgr.closed_trades)

            group_results[group_name] = {
                "return": m.total_return, "sharpe": m.sharpe_ratio,
                "max_dd": m.max_drawdown, "trades": m.trade_count,
                "capital": group_capital, "weight": weight,
            }
            print(f"  [{group_name}] 资金{group_capital:.0f} 权重{weight:.1%}: "
                  f"收益{m.total_return*100:+.2f}% 夏普{m.sharpe_ratio:.3f} 回撤{m.max_drawdown*100:.1f}% 交易{m.trade_count}笔")

        # 加现金
        cash = INITIAL_CAPITAL - sum(group_results[g]["capital"] for g in group_results)
        portfolio_nav = portfolio_nav + cash

        # 组合级 metrics
        bench_series = BacktestEngine._align_benchmark(bench_df, portfolio_nav.index)
        metrics = compute_metrics(
            daily_values=portfolio_nav,
            trades=all_closed_trades,
            initial_capital=INITIAL_CAPITAL,
            benchmark_values=bench_series if bench_series is not None else None,
        )

        print(f"\n{'='*90}")
        print(f"  前端P5配置回测结果 (模拟 backtest.py 逻辑)")
        print(f"{'='*90}")
        print(f"  组合收益:   {metrics.total_return*100:+.2f}%")
        print(f"  Alpha:     {metrics.alpha*100:+.2f}%")
        print(f"  夏普:      {metrics.sharpe_ratio:.3f}")
        print(f"  回撤:      {metrics.max_drawdown*100:.1f}%")
        print(f"  交易笔数:   {metrics.trade_count}")
        print(f"  胜率:      {metrics.win_rate*100:.1f}%")
        print(f"  基准收益:   {metrics.benchmark_return*100:+.2f}%")
        print(f"  现金比例:   {cash/INITIAL_CAPITAL*100:.1f}%")

        print(f"\n  对比脚本回测(P5): +34.22% (2026-01-01 ~ 2026-08-06)")
        print(f"  差异: {abs(metrics.total_return*100 - 34.22):.2f}% (应<1%, 因资金规模不同略有差异)")

    finally:
        if pref_existed and os.path.exists(pref_backup):
            shutil.move(pref_backup, _USER_PREF_FILE)


if __name__ == "__main__":
    main()
