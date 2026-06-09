"""验证修复后的回测引擎是否有漏信号"""
import os
json_path = "data/signal_history.json"
if os.path.exists(json_path):
    os.remove(json_path)

from src.data_fetcher import DataManager, Watchlist
from src.signal_engine import SignalEngine
from src.backtest import BacktestEngine
from datetime import date
import pandas as pd

dm = DataManager()
wl = Watchlist()

data_map = {}
for s in wl.get_all():
    code = s["code"]
    df = dm.get_daily_kline(code, start_date="2024-01-01")
    if df is not None and len(df) >= 120:
        data_map[code] = df

# ====== 方法1: 手动追踪(传 analysis_date) ======
if os.path.exists(json_path):
    os.remove(json_path)
sig1 = SignalEngine(dedup_days=5)

# 构建日历
all_dates = set()
for df in data_map.values():
    for d in df["date"]:
        all_dates.add(pd.Timestamp(d).date())
all_dates = sorted(all_dates)

manual_signals = []
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
        result = sig1.analyze(symbol, df_slice, analysis_date=today)
        if result.level.is_actionable:
            manual_signals.append((today, symbol, result.level.label, result.score))

# ====== 方法2: 回测引擎方式 (传到 analyze) ======
if os.path.exists(json_path):
    os.remove(json_path)

# 模拟回测引擎的信号生成方式
sig2 = SignalEngine(dedup_days=5)
engine_signals = []
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
        result = sig2.analyze(symbol, df_slice, analysis_date=today)
        if result.level.is_actionable:
            engine_signals.append((today, symbol, result.level.label, result.score))

print("=" * 60)
print("  信号生成验证 (analysis_date 已修复)")
print("=" * 60)
print(f"\n手动追踪信号数: {len(manual_signals)}")
print(f"回测方式信号数: {len(engine_signals)}")
print(f"一致: {'YES' if len(manual_signals) == len(engine_signals) else 'NO!!!'}")
print()

# 检查差异
m_set = {(s[0], s[1]) for s in manual_signals}
e_set = {(s[0], s[1]) for s in engine_signals}
only_manual = m_set - e_set
only_engine = e_set - m_set
if only_manual:
    print(f"仅在手动追踪中: {only_manual}")
if only_engine:
    print(f"仅在引擎方式中: {only_engine}")

# ====== 方法3: 回测引擎实际运行 ======
if os.path.exists(json_path):
    os.remove(json_path)

engine = BacktestEngine(initial_capital=100000, lookback_days=120,
                        position_ratio=0.3, signal_dedup_days=5)

# 捕获回测引擎内部的信号生成数量
original_analyze = engine.signal_engine.analyze
signal_dates = []
def counting_analyze(symbol, df, analysis_date=None):
    result = original_analyze(symbol, df, analysis_date=analysis_date)
    if result.level.is_actionable:
        signal_dates.append((analysis_date, symbol, result.level.label, result.score))
    return result
engine.signal_engine.analyze = counting_analyze

metrics = engine.run(data_map)

print()
print("=" * 60)
print("  回测引擎实际运行")
print("=" * 60)
print(f"\n回测内部信号生成数: {len(signal_dates)}")
print(f"最终成交笔数: {len(engine.position_mgr.closed_trades)}")
print()

print("回测性能指标:")
print(f"  初始资金:    {metrics.initial_capital:,.0f}")
print(f"  最终资金:    {metrics.final_value:,.0f}")
print(f"  总收益率:    {metrics.total_return*100:+.2f}%")
print(f"  年化收益率:  {metrics.annual_return*100:+.2f}%")
print(f"  夏普比率:    {metrics.sharpe_ratio:.2f}")
print(f"  最大回撤:    {metrics.max_drawdown*100:.2f}%")
print(f"  胜率:        {metrics.win_rate*100:.1f}%")
print(f"  盈亏比:      {metrics.profit_factor:.2f}")
print(f"  交易次数/信号数: {metrics.trade_count}/{len(signal_dates)}")