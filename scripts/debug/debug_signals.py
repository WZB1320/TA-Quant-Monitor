"""对比诊断: 手动追踪 vs 回测引擎 信号差异"""
import os
json_path = "data/signal_history.json"
if os.path.exists(json_path):
    os.remove(json_path)

from datetime import date
from src.data_fetcher import DataManager, Watchlist
from src.signal_engine import SignalEngine
from src.backtest import BacktestEngine
import pandas as pd

dm = DataManager()
wl = Watchlist()

data_map = {}
for s in wl.get_all():
    code = s["code"]
    df = dm.get_daily_kline(code, start_date="2024-01-01")
    if df is not None and len(df) >= 120:
        data_map[code] = df

# ── 方法1: 手动追踪, 记录所有信号 ──
print("=" * 80)
print("方法1: 手动追踪 — 逐日生成信号并记录")
print("=" * 80)

sig1 = SignalEngine(dedup_days=5)
all_dates = set()
for df in data_map.values():
    for d in df["date"]:
        all_dates.add(pd.Timestamp(d).date())
all_dates = sorted(all_dates)

manual_signals = []  # [(date, symbol, level, score)]
for today in all_dates:
    for symbol, df in data_map.items():
        idx = None
        for j in range(len(df)):
            if pd.Timestamp(df["date"].iloc[j]).date() == today:
                idx = j
                break
        if idx is None or idx < 120:
            continue
        df_slice = df.iloc[:idx + 1].copy()
        try:
            result = sig1.analyze(symbol, df_slice)
        except:
            continue
        if result.level.is_actionable:
            manual_signals.append((today, symbol, result.level.label, result.score))

print(f"手动追踪共生成 {len(manual_signals)} 个信号:")
for s in manual_signals:
    print(f"  {s[0]}  {s[1]:<8}  {s[2]:<8}  score={s[3]:+.1f}")

# ── 方法2: 回测引擎内部, 模拟 run() 的信号生成部分 ──
print(f"\n{'='*80}")
print("方法2: 回测引擎 — 模拟 run() 中的信号生成")
print("=" * 80)

sig2 = SignalEngine(dedup_days=5)
engine_signals = []
for today in all_dates:
    for symbol, df in data_map.items():
        # 使用 _locate_date 方法
        target_str = today.strftime("%Y-%m-%d")
        mask = df["date"] == target_str
        if not mask.any():
            continue
        idx = int(mask.idxmax())
        if idx < 120:
            continue
        df_slice = df.iloc[:idx + 1].copy()
        try:
            result = sig2.analyze(symbol, df_slice)
        except:
            continue
        if result.level.is_actionable:
            engine_signals.append((today, symbol, result.level.label, result.score))

print(f"回测引擎方式共生成 {len(engine_signals)} 个信号:")
for s in engine_signals:
    print(f"  {s[0]}  {s[1]:<8}  {s[2]:<8}  score={s[3]:+.1f}")

# ── 对比 ──
print(f"\n{'='*80}")
print("信号对比")
print("=" * 80)

manual_set = set((s[0], s[1], s[2]) for s in manual_signals)
engine_set = set((s[0], s[1], s[2]) for s in engine_signals)

only_manual = manual_set - engine_set
only_engine = engine_set - manual_set

if only_manual:
    print(f"\n仅在手动追踪中出现 ({len(only_manual)} 个):")
    for s in sorted(only_manual):
        print(f"  {s[0]}  {s[1]}  {s[2]}")
else:
    print("\n手动追踪无多余信号")

if only_engine:
    print(f"\n仅在回测引擎中出现 ({len(only_engine)} 个):")
    for s in sorted(only_engine):
        print(f"  {s[0]}  {s[1]}  {s[2]}")
else:
    print("回测引擎无多余信号")

if not only_manual and not only_engine:
    print("\n信号完全一致! 问题出在 _execute_signals 的执行机制。")
else:
    print(f"\n信号不一致! 差异={len(only_manual)}个")