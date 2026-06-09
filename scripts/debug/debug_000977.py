"""检查000977信号为何被拦截"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from src.data_fetcher.manager import DataManager
import pandas as pd

dm = DataManager()
df = dm.get_daily_kline("sz000977")
print(f"000977 data: {type(df)}")
if df is not None:
    print(f"  len={len(df)}")
    print(f"  columns={list(df.columns)}")
    print(f"  date range: {df['date'].iloc[0]} ~ {df['date'].iloc[-1]}")
    print(f"  first 3 rows:")
    print(df.head(3))
else:
    print("  No data!")
