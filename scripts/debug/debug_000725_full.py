"""完整回测, 但在05-22附近打印000725的详细状态"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import pandas as pd
import numpy as np
from datetime import date
from collections import defaultdict
from src.data_fetcher import DataManager, Watchlist
from src.backtest import BacktestEngine
from src.backtest.position import PositionManager
from src.signal_engine import SignalEngine
from src.config.group_config import GroupConfig

dm = DataManager()
wl = Watchlist()

data_map = {}
for s in wl.get_all():
    df = dm.get_daily_kline(s['code'], start_date='2024-07-01')
    if df is not None and len(df) >= 120:
        data_map[s['code']] = df

# 手动运行回测, 在关键日期打印状态
gc = GroupConfig()
signal_engine = SignalEngine(dedup_days=5, group_config=gc)
position_mgr = PositionManager(
    initial_capital=100_000,
    position_ratio=0.30,
    risk_per_trade=0.05,
    atr_stop_mult=2.5,
)

from src.backtest.broker import Broker
broker = Broker()

# 构建交易日历
all_dates = sorted(set(d for df in data_map.values() for d in df['date'].values))
all_dates = [pd.Timestamp(d).date() if not isinstance(d, date) else d for d in all_dates]

# 只看05-19到05-28
focus_start = date(2026, 5, 19)
focus_end = date(2026, 5, 28)

# 先运行到05-18, 建立状态
pending_signals = {}
lookback = 120

for i, today in enumerate(all_dates):
    if today > focus_end:
        break

    # 每日更新
    prices_today = {}
    for symbol, df in data_map.items():
        mask = df['date'] <= pd.Timestamp(today)
        df_today = df[mask]
        if len(df_today) > 0:
            prices_today[symbol] = df_today['close'].values[-1]

    # 检查持仓止损
    for symbol in list(position_mgr.open_positions.keys()):
        if symbol in prices_today:
            closed = position_mgr.check_stop_loss(symbol, prices_today[symbol], today)
            if closed is not None:
                signal_engine.filter.record_exit(symbol, today)
                if closed.pnl > 0:
                    signal_engine.filter.record_win(symbol)
                else:
                    signal_engine.filter.record_loss(symbol, today)

    # T+1执行昨日信号
    if i > 0:
        prev_date = all_dates[i-1]
        prev_signals = pending_signals.pop(prev_date, {})
        if prev_signals and today >= focus_start:
            print(f"\n=== {today} 执行 {prev_date} 的信号 ===")
            for sym, res in prev_signals.items():
                print(f"  {sym}: {res.level.name} score={res.score:.1f}")

    # 计算今日信号
    signals_today = {}
    for symbol, df in data_map.items():
        idx_arr = df.index[df['date'] <= pd.Timestamp(today)].tolist()
        if not idx_arr:
            continue
        idx = idx_arr[-1]
        if idx < lookback:
            continue
        df_slice = df.iloc[:idx+1].copy()
        try:
            result = signal_engine.analyze(symbol, df_slice, analysis_date=today)
            if result.level.is_actionable:
                signals_today[symbol] = result
        except Exception:
            continue

    pending_signals[today] = signals_today

    # 在关注日期打印状态
    if focus_start <= today <= focus_end:
        open_pos = list(position_mgr.open_positions.keys())
        print(f"\n--- {today} 状态 ---")
        print(f"  持仓: {open_pos if open_pos else '空仓'}")
        print(f"  现金: {position_mgr.cash:,.0f}")
        print(f"  总资产: {position_mgr.total_value(prices_today):,.0f}")
        print(f"  今日信号: {[(s, r.level.name, f'{r.score:.1f}') for s, r in signals_today.items()]}")
        if '000725' in signals_today:
            print(f"  >>> 000725信号: {signals_today['000725'].level.name} score={signals_today['000725'].score:.1f}")
            # 检查冷却期
            in_cd = signal_engine.filter.is_in_cooldown('000725', today, True)
            print(f"  >>> 冷却期: {in_cd}")
            # 检查连亏暂停
            gp = gc.get_all_group_params('000725')
            suspended = signal_engine.filter.is_suspended('000725', today,
                                                          gp.get('max_consecutive_losses', 0),
                                                          gp.get('consecutive_loss_suspend', 0))
            print(f"  >>> 连亏暂停: {suspended}")
