"""调试: 检查信号生成"""
import os, json

json_path = "data/signal_history.json"
if os.path.exists(json_path):
    os.remove(json_path)

from src.data_fetcher import DataManager, Watchlist
from src.signal_engine import SignalEngine
from src.indicators import IndicatorPipeline
import pandas as pd

dm = DataManager()
wl = Watchlist()
data_map = {}
for s in wl.get_all():
    code = s["code"]
    df = dm.get_daily_kline(code, start_date="2024-01-01")
    if df is not None and len(df) >= 120:
        data_map[code] = df

sig = SignalEngine(dedup_days=5)

# 取一个样本日期, 检测一只股票
symbol = list(data_map.keys())[0]
df = data_map[symbol]

# 取最近有数据的120天
df_slice = df.iloc[:150].copy()

print(f"调试: {symbol}, 数据范围: {df_slice.iloc[0]['date']} ~ {df_slice.iloc[-1]['date']}")
print(f"数据长度: {len(df_slice)}")

# 手动跑指标
pipeline = IndicatorPipeline()
ind = pipeline.run(df_slice)

print("\n--- 指标结果 ---")
for name, r in ind.items():
    if hasattr(r, 'direction'):
        print(f"  {name}: direction={r.direction}, strength={r.strength:.1f}, desc={r.description}")
    else:
        print(f"  {name}: type={type(r).__name__}, value={r}")

# 跑信号
result = sig.analyze(symbol, df_slice)
print(f"\n--- 信号结果 ---")
print(f"  level: {result.level.label}")
print(f"  score: {result.score:+.1f}")
print(f"  blocked: {result.hard_filter_blocked}")
print(f"  block_reason: {result.block_reason}")
print(f"  reason: {result.reason}")

# 遍历所有股票所有日期, 看看有多少信号
print("\n\n--- 全量信号统计 ---")
if os.path.exists(json_path):
    os.remove(json_path)

sig2 = SignalEngine(dedup_days=5)
all_dates = set()
for df in data_map.values():
    for d in df["date"]:
        all_dates.add(pd.Timestamp(d).date())
all_dates = sorted(all_dates)

signal_count = 0
blocked_count = 0
neutral_count = 0
score_blocked = 0
vol_blocked = 0
ma60_blocked = 0
adx_blocked = 0

for today in all_dates[-60:]:  # 只检查最近60个交易日
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
            signal_count += 1
            print(f"  SIGNAL: {today} {symbol} {result.level.label} score={result.score:+.1f}")
        elif result.hard_filter_blocked:
            blocked_count += 1
            if "ADX" in result.block_reason:
                adx_blocked += 1
        elif result.level == __import__('src.signal_engine.signals', fromlist=['SignalLevel']).SignalLevel.NEUTRAL:
            neutral_count += 1

print(f"\n信号统计 (最近60天):")
print(f"  可操作信号: {signal_count}")
print(f"  硬过滤拦截: {blocked_count}")
print(f"  中性/观望:  {neutral_count}")