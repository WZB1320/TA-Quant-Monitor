"""检查000725在2026-05-21涨停日的信号状态"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import pandas as pd
import numpy as np
from src.data_fetcher import DataManager
from src.indicators.pipeline import IndicatorPipeline
from src.signal_engine.filter import SignalFilter
from src.signal_engine.scorer import Scorer
from src.signal_engine.validator import Validator
from src.config.group_config import GroupConfig

dm = DataManager()
code = "000725"
df = dm.get_daily_kline(code, start_date='2024-07-01')

# 检查5月19-6月5日
for target_date in ['2026-05-19', '2026-05-20', '2026-05-21', '2026-05-22', '2026-05-25', '2026-05-26', '2026-05-27', '2026-06-01', '2026-06-02']:
    mask = df['date'] <= target_date
    if mask.sum() < 60:
        print(f"  {target_date}: 数据不足, 跳过")
        continue

    df_slice = df[mask].copy()
    close = df_slice['close'].values.astype(float)
    latest_close = close[-1]

    # 完整信号流程
    pipeline = IndicatorPipeline()
    filter_engine = SignalFilter()
    scorer = Scorer()
    validator = Validator()
    gc = GroupConfig()

    indicators = pipeline.run(df_slice)

    # Step 2: 硬过滤
    blocked, block_reason = filter_engine.hard_filter(indicators)

    # Step 3: 评分
    group_weights = gc.get_regime_weights(code)
    score = scorer.score(indicators, regime_weights=group_weights)
    indicators["SCORE"] = score

    # Step 4: 交叉验证
    level = validator.validate(indicators, hard_blocked=blocked)

    # Step 5: 硬过滤方向约束
    group_params = gc.get_all_group_params(code)
    level_after = filter_engine.apply_hard_constraint(level, indicators,
                                                      score_threshold=25,
                                                      group_params=group_params,
                                                      df=df_slice)

    # 突破确认
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

    # MA60方向
    ma60_ind = indicators.get("MA60")
    ma60_dir = ma60_ind.direction if ma60_ind else 0

    print(f"\n  {target_date}: 收盘={latest_close:.2f}")
    print(f"    MA60方向={'多头' if ma60_dir==1 else '空头' if ma60_dir==-1 else '中性'}  MA60值={ma60:.2f}")
    print(f"    硬过滤blocked={blocked}  原因={block_reason}")
    print(f"    得分={score:.1f}  信号级别={level.name}")
    print(f"    约束后级别={level_after.name}")
    print(f"    创20日新高={'Y' if is_breakout else 'N'}(20日高={high_20d:.2f})")
    print(f"    完全多头={'Y' if full_alignment else 'N'}  短中期多头={'Y' if short_alignment else 'N'}")
    print(f"    基础ceiling={base_ceiling}  动态ceiling={dynamic_ceiling}")
    print(f"    bonus: breakout={group_params.get('breakout_ceiling_bonus',0)}  ma_align={group_params.get('ma_alignment_ceiling_bonus',0)}")

    # 如果被拦截, 详细分析原因
    if level.is_bullish and not level_after.is_bullish:
        print(f"    >>> 信号被拦截! 原因分析:")
        if ma60_dir == -1:
            print(f"        - MA60空头区域, 不能出看多信号")
        if base_ceiling > 0 and score > base_ceiling:
            if score > dynamic_ceiling:
                print(f"        - 得分{score:.1f} > 动态ceiling{dynamic_ceiling}, 过热拦截")
            else:
                print(f"        - 得分{score:.1f} > 基础ceiling{base_ceiling}, 但动态ceiling={dynamic_ceiling}放行")
