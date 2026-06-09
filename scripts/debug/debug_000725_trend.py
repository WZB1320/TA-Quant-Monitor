"""查找000725真正的行情启动时间"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import pandas as pd
import numpy as np
from src.data_fetcher import DataManager

dm = DataManager()
code = "000725"
df = dm.get_daily_kline(code, start_date='2024-07-01')
if df is None:
    print(f"无法加载 {code}")
    sys.exit(1)

# 找到收盘价>4.5的所有日期
hot = df[df['close'] > 4.5].copy()
print("=" * 70)
print(f"  000725 收盘价>4.5的日期")
print("=" * 70)
for _, row in hot.iterrows():
    print(f"  {row['date']}  收盘:{row['close']:.2f}  成交量:{row.get('volume', 0):.0f}")

# 找到涨幅>5%的日期
df['pct_change'] = df['close'].pct_change() * 100
big_up = df[df['pct_change'] > 5].copy()
print(f"\n  000725 涨幅>5%的日期 (2025年以来)")
print("=" * 70)
big_up_2025 = big_up[big_up['date'] >= '2025-01-01']
for _, row in big_up_2025.iterrows():
    print(f"  {row['date']}  收盘:{row['close']:.2f}  涨幅:{row['pct_change']:.1f}%  成交量:{row.get('volume', 0):.0f}")

# 找到收盘价创60日新高的日期
df['high_60d'] = df['close'].rolling(60).max().shift(1)
breakout = df[df['close'] > df['high_60d']].copy()
print(f"\n  000725 创60日新高的日期 (2025年以来)")
print("=" * 70)
breakout_2025 = breakout[breakout['date'] >= '2025-01-01']
for _, row in breakout_2025.iterrows():
    print(f"  {row['date']}  收盘:{row['close']:.2f}  前60日最高:{row['high_60d']:.2f}")
