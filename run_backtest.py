"""量化策略回测脚本 — 一键运行"""
import os
import sys
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.data_fetcher import DataManager, Watchlist
from src.backtest import BacktestEngine
from src.config.group_config import GroupConfig


def main():
    # 清除信号历史，避免跨次回测信号去重干扰
    hist_file = os.path.join('data', 'signal_history.json')
    if os.path.exists(hist_file):
        os.remove(hist_file)

    # 加载股票数据
    dm = DataManager()
    wl = Watchlist()

    data_map = {}
    for s in wl.get_all():
        df = dm.get_daily_kline(s['code'], start_date='2024-07-01')
        if df is not None and len(df) >= 120:
            data_map[s['code']] = df

    print(f"加载 {len(data_map)} 只股票")

    # 回测引擎
    engine = BacktestEngine(
        initial_capital=100_000,
        lookback_days=120,
        position_ratio=0.30,
        signal_dedup_days=5,
        risk_per_trade=0.05,
        atr_stop_mult=2.5,
    )
    metrics = engine.run(data_map)
    trades = engine.position_mgr.closed_trades

    # ── 分组统计 ──
    gc = GroupConfig()
    group_stats = defaultdict(lambda: {'wins': 0, 'losses': 0, 'total_pnl': 0, 'trades': []})
    for t in trades:
        g = gc.get_group(t.symbol)
        group_stats[g]['trades'].append(t)
        if t.pnl > 0:
            group_stats[g]['wins'] += 1
        else:
            group_stats[g]['losses'] += 1
        group_stats[g]['total_pnl'] += t.pnl

    print("\n" + "=" * 70)
    print("  回测统计 (2025-01 ~ 2026-06)")
    print("=" * 70)

    for g in sorted(group_stats.keys()):
        s = group_stats[g]
        total = s['wins'] + s['losses']
        if total == 0:
            print(f"  {g}: 无交易")
            continue
        wr = s['wins'] / total * 100
        aw = sum(t.pnl_pct for t in s['trades'] if t.pnl > 0) / s['wins'] * 100 if s['wins'] > 0 else 0
        al = sum(t.pnl_pct for t in s['trades'] if t.pnl <= 0) / s['losses'] * 100 if s['losses'] > 0 else 0
        ah = sum(t.holding_days for t in s['trades']) / total
        print(f"  {g}: {total}笔 | 胜率{wr:.0f}% | PnL:{s['total_pnl']:+,.0f} | 均盈+{aw:.1f}% | 均亏{al:.1f}% | 均持{ah:.0f}d")

    print("\n" + "=" * 70)
    print("  整体绩效")
    print("=" * 70)
    print(f"  总收益率: {metrics.total_return:.2%}")
    print(f"  年化收益: {metrics.annual_return:.2%}")
    print(f"  最大回撤: {metrics.max_drawdown:.2%}")
    print(f"  夏普比率: {metrics.sharpe_ratio:.2f}")
    print(f"  总交易数: {len(trades)}")
    print(f"  总盈亏:   {sum(t.pnl for t in trades):+,.0f}")

    # ── 各分组详细交易记录 ──
    for g in sorted(group_stats.keys()):
        s = group_stats[g]
        if not s['trades']:
            continue
        print("\n" + "=" * 70)
        print(f"  {g} — 详细交易记录 ({len(s['trades'])}笔)")
        print("=" * 70)
        sorted_trades = sorted(s['trades'], key=lambda t: t.entry_date)
        for i, t in enumerate(sorted_trades, 1):
            sig = t.entry_signal[:30] if t.entry_signal else ""
            result = "盈" if t.pnl > 0 else "亏"
            print(f"  {i:>2}. {t.symbol} | {t.entry_date} -> {t.exit_date} | {sig} | "
                  f"PnL:{t.pnl:+,.0f}({t.pnl_pct:+.2f}%) | 持{t.holding_days}d [{result}]")


if __name__ == '__main__':
    main()