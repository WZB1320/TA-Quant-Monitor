"""
交易复盘 — 逐笔分析买卖信号的技术指标明细
找出亏损共性原因，提出优化建议
"""
import os, sys

# 修复: 此前启动即删除 data/signal_history.json, 会摧毁 LIVE 模式实盘去重数据.
# 改为回测模式运行(内存去重, 不读写磁盘), 既不污染实盘, 又保证可复现.
from src.config.runtime_mode import set_mode, RuntimeMode
set_mode(RuntimeMode.BACKTEST)

from src.data_fetcher import DataManager, Watchlist
from src.backtest import BacktestEngine
from src.signal_engine import SignalEngine
from src.indicators import IndicatorPipeline
from datetime import date
import pandas as pd
import numpy as np

# ── 运行回测 ──
dm = DataManager()
wl = Watchlist()
data_map = {}
for s in wl.get_all():
    code = s["code"]
    df = dm.get_daily_kline(code, start_date="2024-01-01")
    if df is not None and len(df) >= 120:
        data_map[code] = df

engine = BacktestEngine(initial_capital=100000, lookback_days=120,
                        position_ratio=0.3, signal_dedup_days=5)
metrics = engine.run(data_map)
trades = engine.position_mgr.closed_trades

print("=" * 80)
print("  16笔交易全面复盘 — 技术指标明细")
print("=" * 80)

# ── 获取每笔交易在买入日和卖出日的指标 ──
sig = SignalEngine(dedup_days=5)
pipeline = IndicatorPipeline()

# 统计
wins = []
losses = []
buy_trigger_stats = {"trend": 0, "strength": 0, "momentum": 0, "volume": 0}
sell_trigger_stats = {"trend": 0, "strength": 0, "momentum": 0, "volume": 0}

for i, t in enumerate(trades):
    df = data_map.get(t.symbol)
    if df is None:
        continue

    # ── 买入日指标 ──
    entry_mask = df["date"] == t.entry_date.strftime("%Y-%m-%d")
    if entry_mask.any():
        entry_idx = int(entry_mask.idxmax())
        df_entry = df.iloc[:entry_idx + 1].copy()
        ind_entry = pipeline.run(df_entry)

        # 注意: 本脚本已在顶部 set_mode(BACKTEST), SignalEngine 使用内存去重,
        # 不读写磁盘, 因此无需 (也绝不应) 删除 data/signal_history.json.
        # 下方仅做指标复盘, 不再重建信号, 避免误删实盘去重数据.
        sig_entry = SignalEngine(dedup_days=5)

    print(f"\n{'─' * 80}")
    print(f"  交易 #{i+1}: {t.symbol}  |  {t.entry_signal}")
    print(f"  买入: {t.entry_date} @ {t.entry_price:.2f}  |  "
          f"卖出: {t.exit_date} @ {t.exit_price:.2f}  |  "
          f"持仓: {t.holding_days}天")
    print(f"  盈亏: {t.pnl:+,.0f} ({t.pnl_pct*100:+.2f}%)  |  "
          f"{'WIN' if t.pnl > 0 else 'LOSS'}")

    if t.pnl > 0:
        wins.append(t)
    else:
        losses.append(t)

    # ── 买入日指标分析 ──
    entry_mask = df["date"] == t.entry_date.strftime("%Y-%m-%d")
    if entry_mask.any():
        entry_idx = int(entry_mask.idxmax())
        if entry_idx >= 120:
            df_entry = df.iloc[:entry_idx + 1].copy()
            ind = pipeline.run(df_entry)

            print(f"\n  >>> 买入日 ({t.entry_date}) 指标:")
            for name, r in ind.items():
                d = {1: "▲", -1: "▼", 0: "—"}[r.direction]
                cat_tag = f"[{r.category}]"
                print(f"    {d} {cat_tag:12s} {r.name:<12s} "
                      f"strength={r.strength:.1f}  |  {r.description}")

    # ── 卖出日指标分析 ──
    exit_mask = df["date"] == t.exit_date.strftime("%Y-%m-%d")
    if exit_mask.any():
        exit_idx = int(exit_mask.idxmax())
        if exit_idx >= 120:
            df_exit = df.iloc[:exit_idx + 1].copy()
            ind = pipeline.run(df_exit)

            print(f"\n  >>> 卖出日 ({t.exit_date}) 指标:")
            for name, r in ind.items():
                d = {1: "▲", -1: "▼", 0: "—"}[r.direction]
                cat_tag = f"[{r.category}]"
                print(f"    {d} {cat_tag:12s} {r.name:<12s} "
                      f"strength={r.strength:.1f}  |  {r.description}")

    # ── 价格走势上下文 ──
    entry_mask2 = df["date"] == t.entry_date.strftime("%Y-%m-%d")
    if entry_mask2.any():
        entry_idx2 = int(entry_mask2.idxmax())
        # 买入前后各5日价格
        start = max(0, entry_idx2 - 5)
        end = min(len(df) - 1, entry_idx2 + 5)
        prices = []
        for j in range(start, end + 1):
            row = df.iloc[j]
            marker = " << BUY" if j == entry_idx2 else ""
            prices.append(f"    {row['date']}: close={float(row['close']):.2f}{marker}")
        print(f"\n  >>> 买入前后价格:")
        for p in prices:
            print(p)

print("\n" + "=" * 80)
print("  盈亏分类汇总")
print("=" * 80)

if wins:
    print(f"\n  ✓ 盈利交易 ({len(wins)}笔):")
    avg_hold = sum(t.holding_days for t in wins) / len(wins)
    avg_pnl = sum(t.pnl for t in wins) / len(wins)
    avg_pct = sum(t.pnl_pct for t in wins) / len(wins) * 100
    print(f"    平均持仓: {avg_hold:.0f}天")
    print(f"    平均盈利: {avg_pnl:+,.0f} ({avg_pct:+.1f}%)")
    print(f"    股票分布: {set(t.symbol for t in wins)}")

if losses:
    print(f"\n  ✗ 亏损交易 ({len(losses)}笔):")
    avg_hold2 = sum(t.holding_days for t in losses) / len(losses)
    avg_pnl2 = sum(t.pnl for t in losses) / len(losses)
    avg_pct2 = sum(t.pnl_pct for t in losses) / len(losses) * 100
    print(f"    平均持仓: {avg_hold2:.0f}天")
    print(f"    平均亏损: {avg_pnl2:+,.0f} ({avg_pct2:+.1f}%)")
    print(f"    股票分布: {set(t.symbol for t in losses)}")

# ── 按信号类型分组 ──
print(f"\n{'=' * 80}")
print("  按买入信号类型分组")
print("=" * 80)

signal_groups = {}
for t in trades:
    sig_label = t.entry_signal.split("score=")[0].strip()
    signal_groups.setdefault(sig_label, []).append(t)

for sig, group in sorted(signal_groups.items()):
    win_count = sum(1 for t in group if t.pnl > 0)
    avg_pnl = sum(t.pnl for t in group) / len(group)
    avg_pct = sum(t.pnl_pct for t in group) / len(group) * 100
    print(f"  [{sig}]  {len(group)}笔  "
          f"win: {win_count}/{len(group)}  "
          f"avg: {avg_pnl:+,.0f} ({avg_pct:+.1f}%)")

# ── 按股票分组 ──
print(f"\n{'=' * 80}")
print("  按股票分组")
print("=" * 80)
stock_groups = {}
for t in trades:
    stock_groups.setdefault(t.symbol, []).append(t)
for sym, group in sorted(stock_groups.items()):
    win_count = sum(1 for t in group if t.pnl > 0)
    total_pnl = sum(t.pnl for t in group)
    print(f"  {sym}: {len(group)}笔 | win:{win_count}/{len(group)} | total: {total_pnl:+,.0f}")