"""检查000725在05-26的动态Ceiling计算"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import numpy as np
from src.data_fetcher import DataManager
from src.indicators.pipeline import IndicatorPipeline
from src.signal_engine.filter import SignalFilter
from src.config.group_config import GroupConfig

dm = DataManager()
df = dm.get_daily_kline('000725', start_date='2024-07-01')

for target_date in ['2026-05-22', '2026-05-26', '2026-06-04', '2026-06-05']:
    mask = df['date'] <= target_date
    df_slice = df[mask].copy()

    pipeline = IndicatorPipeline()
    filter_engine = SignalFilter()
    gc = GroupConfig()

    indicators = pipeline.run(df_slice)
    group_params = gc.get_all_group_params('000725')

    close = df_slice['close'].values.astype(float)
    high_20d = np.max(close[-21:-1]) if len(close) >= 21 else 0
    is_new_high = close[-1] > high_20d

    # 检查均线排列
    ma5 = indicators.get("MA5")
    ma10 = indicators.get("MA10")
    ma20 = indicators.get("MA20")

    ma5_val = ma5.values.get("ma5") if ma5 else None
    ma10_val = ma10.values.get("ma10") if ma10 else None
    ma20_val = ma20.values.get("ma20") if ma20 else None

    full_alignment = ma5_val and ma10_val and ma20_val and ma5_val > ma10_val > ma20_val
    short_mid = ma5_val and ma10_val and ma5_val > ma10_val

    # 动态Ceiling计算
    base_ceiling = group_params.get("score_ceiling", 0)
    dynamic_ceiling = filter_engine.calc_dynamic_ceiling(base_ceiling, df_slice, indicators, group_params)

    from src.signal_engine.scorer import Scorer
    scorer = Scorer()
    group_weights = gc.get_regime_weights('000725')
    score_val = scorer.score(indicators, regime_weights=group_weights)

    print(f"\n{target_date}: 收盘={close[-1]:.2f} 得分={score_val:.1f}")
    print(f"  创20日新高={'Y' if is_new_high else 'N'} (20日高={high_20d:.2f})")
    print(f"  MA5={ma5_val} MA10={ma10_val} MA20={ma20_val}")
    print(f"  完全多头排列={'Y' if full_alignment else 'N'} 短中期多头={'Y' if short_mid else 'N'}")
    print(f"  基础ceiling={base_ceiling} 动态ceiling={dynamic_ceiling}")
    print(f"  得分vs动态ceiling: {'通过' if score_val <= dynamic_ceiling else '拦截'}")
