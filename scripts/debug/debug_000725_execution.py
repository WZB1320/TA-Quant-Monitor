"""检查000725在05-22信号T+1执行日的涨跌停和仓位"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import pandas as pd
from src.data_fetcher import DataManager, Watchlist
from src.backtest.broker import Broker

dm = DataManager()
df = dm.get_daily_kline('000725', start_date='2024-07-01')

broker = Broker()

# 05-22是信号日, T+1执行是05-25
# 找到05-22在df中的位置
idx_22 = df[df['date'] == '2026-05-22'].index[0]
print(f"05-22 在df中的位置: idx={idx_22}")

next_info = broker.get_next_open(df, idx_22)
if next_info:
    print(f"次日信息: date={next_info['date']} open={next_info['open']:.2f} prev_close={next_info['prev_close']:.2f}")
    can = broker.can_trade(next_info['open'], next_info['prev_close'])
    print(f"can_trade={can}")

    # 手动计算涨跌停
    open_price = next_info['open']
    prev_close = next_info['prev_close']
    limit_up = prev_close * 1.1
    print(f"  涨停价={limit_up:.2f}  开盘价={open_price:.2f}")
    print(f"  开盘/昨收={open_price/prev_close:.4f}  涨幅={((open_price/prev_close)-1)*100:.2f}%")

# 也检查05-21 (一字涨停日)
idx_21 = df[df['date'] == '2026-05-21'].index[0]
next_info_21 = broker.get_next_open(df, idx_21)
if next_info_21:
    print(f"\n05-21次日: date={next_info_21['date']} open={next_info_21['open']:.2f} prev_close={next_info_21['prev_close']:.2f}")
    can_21 = broker.can_trade(next_info_21['open'], next_info_21['prev_close'])
    print(f"can_trade={can_21}")

# 检查Broker的can_trade逻辑
print(f"\n--- Broker can_trade 逻辑 ---")
import inspect
print(inspect.getsource(broker.can_trade))
