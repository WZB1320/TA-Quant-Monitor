"""检查000725在2026-05-22的信号"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.data_fetcher import DataManager
from src.indicators.pipeline import IndicatorPipeline
from src.signal_engine.filter import SignalFilter
from src.signal_engine.scorer import Scorer
from src.signal_engine.validator import Validator
from src.config.group_config import GroupConfig

dm = DataManager()
code = "000725"
df = dm.get_daily_kline(code, start_date='2024-07-01')

for target_date in ['2026-05-21', '2026-05-22', '2026-05-25', '2026-05-26']:
    mask = df['date'] <= target_date
    df_slice = df[mask].copy()

    pipeline = IndicatorPipeline()
    filter_engine = SignalFilter()
    scorer = Scorer()
    validator = Validator()
    gc = GroupConfig()

    indicators = pipeline.run(df_slice)
    blocked, block_reason = filter_engine.hard_filter(indicators)

    group_weights = gc.get_regime_weights(code)
    score = scorer.score(indicators, regime_weights=group_weights)
    indicators["SCORE"] = score

    level = validator.validate(indicators, hard_blocked=blocked)

    group_params = gc.get_all_group_params(code)

    # 逐层检查
    import numpy as np
    close = df_slice['close'].values.astype(float)

    # ADX体制覆盖
    effective_params = filter_engine._apply_regime_filter_overrides(group_params, indicators)

    # score threshold
    st = group_params.get("score_threshold", 25)
    is_breakout = close[-1] > np.max(close[-21:-1]) if len(close) >= 21 else False
    effective_st = st * 0.7 if is_breakout else st
    threshold_pass = score >= effective_st

    # ceiling
    base_ceiling = group_params.get("score_ceiling", 0)
    dynamic_ceiling = filter_engine.calc_dynamic_ceiling(base_ceiling, df_slice, indicators, group_params)
    ceiling_pass = score <= dynamic_ceiling if base_ceiling > 0 else True

    # 完整约束
    level_after = filter_engine.apply_hard_constraint(level, indicators,
                                                      score_threshold=25,
                                                      group_params=group_params,
                                                      df=df_slice)

    print(f"\n{target_date}: 收盘={close[-1]:.2f} 得分={score:.1f}")
    print(f"  创20日新高={'Y' if is_breakout else 'N'}")
    print(f"  得分阈值: 基础={st} 有效={effective_st:.1f} {'通过' if threshold_pass else '拦截'}")
    print(f"  Ceiling: 基础={base_ceiling} 动态={dynamic_ceiling} {'通过' if ceiling_pass else '拦截'}")
    print(f"  约束后级别: {level_after.name}")
    print(f"  冷却期: {filter_engine.is_in_cooldown(code, target_date, True)}")
    print(f"  连亏暂停: {filter_engine.is_suspended(code, target_date, 2, 15)}")
    print(f"  去重: {filter_engine.is_duplicate(code, level_after, analysis_date=target_date)}")
