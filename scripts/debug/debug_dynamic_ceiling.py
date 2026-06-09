"""调试动态Ceiling: 检查000725在5月21-22日的突破确认和均线排列"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import pandas as pd
import numpy as np
from src.data_fetcher import DataManager, Watchlist
from src.indicators.pipeline import IndicatorPipeline
from src.signal_engine.filter import SignalFilter
from src.config.group_config import GroupConfig

# 加载数据
dm = DataManager()
wl = Watchlist()

code = "000725"
df = dm.get_daily_kline(code, start_date='2024-07-01')
if df is None:
    print(f"无法加载 {code}")
    sys.exit(1)

# 找到5月19-30日的数据
may_dates = df[(df['date'] >= '2025-05-19') & (df['date'] <= '2025-05-30')]
print("=" * 70)
print(f"  000725 5月19-30日行情")
print("=" * 70)
for _, row in may_dates.iterrows():
    print(f"  {row['date']}  收盘:{row['close']:.2f}  成交量:{row.get('volume', 0):.0f}")

# 加载分组配置
gc = GroupConfig()
group_params = gc.get_all_group_params(code)
print(f"\n分组参数: score_ceiling={group_params.get('score_ceiling')}")
print(f"  breakout_ceiling_bonus={group_params.get('breakout_ceiling_bonus', 0)}")
print(f"  ma_alignment_ceiling_bonus={group_params.get('ma_alignment_ceiling_bonus', 0)}")

# 对5月19-26日逐日分析
pipeline = IndicatorPipeline()
filter_engine = SignalFilter()

print("\n" + "=" * 70)
print(f"  逐日动态Ceiling分析")
print("=" * 70)

for target_date in ['2025-05-19', '2025-05-20', '2025-05-21', '2025-05-22', '2025-05-23', '2025-05-26']:
    mask = df['date'] <= target_date
    if mask.sum() < 60:
        print(f"  {target_date}: 数据不足60天, 跳过")
        continue

    df_slice = df[mask].copy()
    close = df_slice['close'].values.astype(float)
    latest_close = close[-1]

    # 计算指标
    indicators = pipeline.run(df_slice)
    score = indicators.get("SCORE")

    # 突破确认: 创20日新高
    high_20d = np.max(close[-21:-1]) if len(close) >= 21 else 0
    is_breakout = latest_close > high_20d

    # 均线排列
    ma5 = np.mean(close[-5:])
    ma10 = np.mean(close[-10:])
    ma20 = np.mean(close[-20:])
    ma60 = np.mean(close[-60:])
    full_alignment = ma5 > ma10 > ma20 > ma60
    short_alignment = ma5 > ma10 > ma20

    # 动态Ceiling
    base_ceiling = group_params.get('score_ceiling', 0)
    dynamic_ceiling = filter_engine.calc_dynamic_ceiling(
        base_ceiling, df_slice, indicators, group_params)

    # 得分
    score_val = score.values.get('score', 0) if score and hasattr(score, 'values') else 0
    if not isinstance(score_val, (int, float)):
        score_val = 0

    # 信号方向
    ma60_dir = indicators.get("MA60")
    ma60_direction = ma60_dir.direction if ma60_dir else 0

    print(f"\n  {target_date}:")
    print(f"    收盘={latest_close:.2f}  20日最高={high_20d:.2f}  创新高={'Y' if is_breakout else 'N'}")
    print(f"    MA5={ma5:.2f} MA10={ma10:.2f} MA20={ma20:.2f} MA60={ma60:.2f}")
    print(f"    完全多头排列={'Y' if full_alignment else 'N'}  短中期多头={'Y' if short_alignment else 'N'}")
    print(f"    MA60方向={'多头' if ma60_direction==1 else '空头' if ma60_direction==-1 else '中性'}")
    print(f"    得分={score_val:.1f}  基础ceiling={base_ceiling}  动态ceiling={dynamic_ceiling}")
    if score_val > base_ceiling:
        print(f"    >>> 超过基础ceiling! 动态放行={'Y' if score_val <= dynamic_ceiling else 'N'}")

# 也检查之前被拦截的高分信号
print("\n" + "=" * 70)
print(f"  历史高分信号动态Ceiling检查")
print("=" * 70)

high_score_dates = {
    '300433': ['2025-09-26', '2025-08-12'],
    '000725': ['2025-09-01', '2025-12-08'],
    '002138': ['2026-03-03'],
    '600522': ['2025-08-12'],
}

for stock_code, dates in high_score_dates.items():
    df_stock = dm.get_daily_kline(stock_code, start_date='2024-07-01')
    if df_stock is None:
        continue
    gp = gc.get_all_group_params(stock_code)

    for target_date in dates:
        mask = df_stock['date'] <= target_date
        if mask.sum() < 60:
            continue
        df_slice = df_stock[mask].copy()
        close = df_slice['close'].values.astype(float)
        latest_close = close[-1]

        indicators = pipeline.run(df_slice)
        score = indicators.get("SCORE")
        score_val = score.values.get('score', 0) if score and hasattr(score, 'values') else 0
        if not isinstance(score_val, (int, float)):
            score_val = 0

        high_20d = np.max(close[-21:-1]) if len(close) >= 21 else 0
        is_breakout = latest_close > high_20d

        ma5 = np.mean(close[-5:])
        ma10 = np.mean(close[-10:])
        ma20 = np.mean(close[-20:])
        ma60 = np.mean(close[-60:])
        full_alignment = ma5 > ma10 > ma20 > ma60
        short_alignment = ma5 > ma10 > ma20

        base_ceiling = gp.get('score_ceiling', 0)
        dynamic_ceiling = filter_engine.calc_dynamic_ceiling(
            base_ceiling, df_slice, indicators, gp)

        print(f"\n  {stock_code} {target_date}:")
        print(f"    收盘={latest_close:.2f}  20日最高={high_20d:.2f}  创新高={'Y' if is_breakout else 'N'}")
        print(f"    完全多头={'Y' if full_alignment else 'N'}  短中期多头={'Y' if short_alignment else 'N'}")
        print(f"    得分={score_val:.1f}  基础ceiling={base_ceiling}  动态ceiling={dynamic_ceiling}")
        if score_val > base_ceiling:
            print(f"    >>> 超过基础ceiling! 动态放行={'Y' if score_val <= dynamic_ceiling else 'N'}")
