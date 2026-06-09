"""模拟回测引擎逻辑, 排查000725在05-22信号为何没执行"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import pandas as pd
import numpy as np
from datetime import date
from src.data_fetcher import DataManager, Watchlist
from src.signal_engine import SignalEngine
from src.config.group_config import GroupConfig

dm = DataManager()
wl = Watchlist()

# 加载所有股票数据 (和回测一样)
data_map = {}
for s in wl.get_all():
    df = dm.get_daily_kline(s['code'], start_date='2024-07-01')
    if df is not None and len(df) >= 120:
        data_map[s['code']] = df

print(f"加载 {len(data_map)} 只股票")
print(f"包含000725: {'000725' in data_map}")

# 检查000725数据
df_725 = data_map.get('000725')
if df_725 is not None:
    # 找到05-22附近的数据
    mask = (df_725['date'] >= '2026-05-18') & (df_725['date'] <= '2026-05-28')
    print(f"\n000725 05-18~05-28 数据:")
    print(df_725[mask][['date', 'open', 'close', 'high', 'low']].to_string())

# 模拟回测引擎的信号计算
signal_engine = SignalEngine(dedup_days=5, group_config=GroupConfig())

# 检查05-21和05-22的信号
for target_date_str in ['2026-05-21', '2026-05-22', '2026-05-25']:
    target_date = pd.Timestamp(target_date_str).date()

    # 截取到该日的数据
    df_slice = df_725[df_725['date'] <= target_date_str].copy()
    if len(df_slice) < 120:
        print(f"\n{target_date_str}: 数据不足 ({len(df_slice)}行)")
        continue

    try:
        result = signal_engine.analyze('000725', df_slice, analysis_date=target_date)
        print(f"\n{target_date_str}: 信号={result.level.name} 得分={result.score:.1f}")
        print(f"  is_actionable={result.level.is_actionable} is_bullish={result.level.is_bullish}")
    except Exception as e:
        print(f"\n{target_date_str}: 异常 {e}")

# 检查回测引擎在05-22时的持仓状态
# 需要运行完整回测才能看到, 但可以先检查信号去重
print(f"\n--- 信号去重检查 ---")
# 重新创建signal_engine模拟回测
signal_engine2 = SignalEngine(dedup_days=5, group_config=GroupConfig())

# 模拟回测逐日计算, 只看000725
lookback = 120
dates_725 = df_725['date'].values

signals_found = []
for i in range(lookback, len(df_725)):
    d = df_725['date'].iloc[i]
    if d < pd.Timestamp('2026-05-01'):
        continue  # 只看5月以后

    df_slice = df_725.iloc[:i+1].copy()
    try:
        result = signal_engine2.analyze('000725', df_slice,
                                         analysis_date=pd.Timestamp(d).date())
        if result.level.is_actionable:
            signals_found.append((str(d)[:10], result.level.name, result.score))
            print(f"  {str(d)[:10]}: {result.level.name} score={result.score:.1f}")
    except Exception as e:
        pass

print(f"\n5月后000725共发现 {len(signals_found)} 个可操作信号")

# 检查涨跌停
print(f"\n--- 涨跌停检查 ---")
from src.backtest.broker import Broker
broker = Broker()
for target_date_str in ['2026-05-22', '2026-05-25', '2026-05-26']:
    df_slice = df_725[df_725['date'] <= target_date_str]
    idx = len(df_slice) - 1
    next_info = broker.get_next_open(df_725, idx)
    if next_info:
        print(f"  {target_date_str} 次日: date={next_info['date']} open={next_info['open']:.2f} prev_close={next_info['prev_close']:.2f}")
        can = broker.can_trade(next_info['open'], next_info['prev_close'])
        print(f"    can_trade={can}")
