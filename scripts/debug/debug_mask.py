"""最小化测试: 直接检查 mask.any()"""
from src.data_fetcher import DataManager, Watchlist
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

print(f"日历共 {len(all_dates)} 天, 从 {all_dates[0]} 到 {all_dates[-1]}")
print(f"股票: {list(data_map.keys())}")

# 取第一个股票测试
symbol = list(data_map.keys())[0]
df = data_map[symbol]
print(f"\n{symbol} date列: dtype={df['date'].dtype}, 前3={df['date'].head(3).tolist()}")

# 测试前10个日期
found = 0
for today in all_dates[:10]:
    target_str = today.strftime("%Y-%m-%d")
    mask = df["date"] == target_str
    result = mask.any()
    if result:
        found += 1
        print(f"  {today} -> '{target_str}' -> mask.any()={result}, idx={mask.idxmax()}")
    else:
        print(f"  {today} -> '{target_str}' -> mask.any()={result}")

print(f"\n前10天中匹配到 {found} 天")

# 测试一个已知应该存在的日期
test = pd.Timestamp("2025-07-18").date()
target_str = test.strftime("%Y-%m-%d")
mask = df["date"] == target_str
print(f"\n测试 {test}: mask.any()={mask.any()}, sum={mask.sum()}, dtype={mask.dtype}")

# 测试所有日期
total_found = 0
for today in all_dates:
    target_str = today.strftime("%Y-%m-%d")
    for sym, df2 in data_map.items():
        mask = df2["date"] == target_str
        if mask.any():
            total_found += 1
print(f"\n总共匹配到 {total_found} 个 (股票,日期) 对")