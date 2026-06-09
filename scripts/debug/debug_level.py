"""诊断: 打印所有非NEUTRAL的analyze结果"""
from src.data_fetcher import DataManager, Watchlist
from src.signal_engine import SignalEngine
import pandas as pd

dm = DataManager()
wl = Watchlist()

data_map = {}
for s in wl.get_all():
    code = s["code"]
    df = dm.get_daily_kline(code, start_date="2024-01-01")
    if df is not None and len(df) >= 120:
        data_map[code] = df

all_dates = set()
for df in data_map.values():
    for d in df["date"]:
        all_dates.add(pd.Timestamp(d).date())
all_dates = sorted(all_dates)

sig = SignalEngine(dedup_days=5)

non_neutral = 0
actionable = 0
count = 0

for today in all_dates:
    for symbol, df in data_map.items():
        target_str = today.strftime("%Y-%m-%d")
        mask = df["date"] == target_str
        if not mask.any():
            continue
        idx = int(mask.idxmax())
        if idx < 120:
            continue
        df_slice = df.iloc[:idx + 1].copy()
        result = sig.analyze(symbol, df_slice)
        count += 1
        if result.level.value != 0:
            non_neutral += 1
            if non_neutral <= 30:
                print(f"  {today} {symbol} level={result.level.name} score={result.score:+.1f} actionable={result.level.is_actionable}")
        if result.level.is_actionable:
            actionable += 1

print(f"\n共调用 {count} 次")
print(f"非NEUTRAL: {non_neutral} 次")
print(f"可操作: {actionable} 次")

# 对比: 用手动方式
print(f"\n--- 手动方式对比 ---")
sig2 = SignalEngine(dedup_days=5)
manual_actionable = 0
manual_count = 0
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
        result = sig2.analyze(symbol, df_slice)
        manual_count += 1
        if result.level.is_actionable:
            manual_actionable += 1

print(f"手动方式: 调用 {manual_count} 次, 可操作 {manual_actionable} 次")

# 对比 idx 是否一致
print(f"\n--- idx 对比 ---")
for today in all_dates[:5]:
    for symbol, df in data_map.items():
        target_str = today.strftime("%Y-%m-%d")
        mask = df["date"] == target_str
        if not mask.any():
            continue
        idx1 = int(mask.idxmax())
        
        idx2 = None
        for j in range(len(df)):
            if pd.Timestamp(df["date"].iloc[j]).date() == today:
                idx2 = j
                break
        
        print(f"  {today} {symbol}: mask.idxmax()={idx1}, 手动={idx2}, 一致={idx1==idx2}")