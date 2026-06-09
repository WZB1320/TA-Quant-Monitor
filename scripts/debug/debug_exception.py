"""诊断: 去掉 try/except, 看 sig2.analyze() 是否抛异常"""
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

sig2 = SignalEngine(dedup_days=5)
count = 0
error_count = 0

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
        
        # 不用 try/except, 直接调用
        result = sig2.analyze(symbol, df_slice)
        count += 1
        if result.level.is_actionable:
            print(f"  {today} {symbol} {result.level.label} score={result.score:+.1f}")

print(f"\n共调用 {count} 次 analyze, 抛异常 {error_count} 次")