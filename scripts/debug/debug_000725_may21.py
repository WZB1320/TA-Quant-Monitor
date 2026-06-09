"""检查000725在2026-05-21为什么没有交易"""
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

# 检查数据最后几天
print("000725 最后10天数据:")
print(df.tail(10)[['date', 'close', 'volume']].to_string())

# 2026-05-21信号分析
target_date = '2026-05-21'
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
level_after = filter_engine.apply_hard_constraint(level, indicators,
                                                  score_threshold=25,
                                                  group_params=group_params,
                                                  df=df_slice)

close = df_slice['close'].values.astype(float)
high_20d = np.max(close[-21:-1])
is_breakout = close[-1] > high_20d

print(f"\n2026-05-21 信号分析:")
print(f"  收盘={close[-1]:.2f}  20日最高={high_20d:.2f}  创新高={'Y' if is_breakout else 'N'}")
print(f"  硬过滤: blocked={blocked}  原因={block_reason}")
print(f"  得分={score:.1f}  验证级别={level.name}  约束后={level_after.name}")

# 检查冷却期
in_cooldown = filter_engine.is_in_cooldown(code, target_date, True)
print(f"  冷却期: {in_cooldown}")

# 检查连亏暂停
max_cl = group_params.get('max_consecutive_losses', 0)
suspend_d = group_params.get('consecutive_loss_suspend', 0)
suspended = filter_engine.is_suspended(code, target_date, max_cl, suspend_d)
print(f"  连亏暂停: {suspended} (max_cl={max_cl}, suspend_d={suspend_d})")

# 检查信号去重
is_dup = filter_engine.is_duplicate(code, level_after, analysis_date=target_date)
print(f"  信号去重: {is_dup}")

# 检查各过滤层
print(f"\n  各过滤层检查:")
ma60_ind = indicators.get("MA60")
print(f"    MA60方向: {ma60_ind.direction} ({'多头' if ma60_ind.direction==1 else '空头'})")

vol_ind = indicators.get("VOL_RATIO")
if vol_ind:
    vr = vol_ind.values.get('vol_ratio', 1.0)
    vol_threshold = group_params.get('vol_ratio_threshold', 0.6)
    print(f"    量比: {vr:.2f}  阈值: {vol_threshold}  {'通过' if vr >= vol_threshold else '拦截'}")

atr_ind = indicators.get("ATR")
if atr_ind:
    atr_pct = atr_ind.values.get('atr_pct', 0)
    atr_ratio = atr_pct / 100.0
    atr_max = group_params.get('atr_price_ratio_max', 0)
    effective_atr_max = atr_max * 1.5 if is_breakout else atr_max
    print(f"    ATR/Price: {atr_ratio:.4f}  阈值: {effective_atr_max}  {'通过' if atr_ratio <= effective_atr_max else '拦截'}")

ma20_ind = indicators.get("MA20")
if ma20_ind:
    ma20_val = ma20_ind.values.get('ma20')
    price = ma20_ind.values.get('price')
    if ma20_val and price:
        deviation = (price - ma20_val) / ma20_val
        max_dev = group_params.get('price_ma20_max_deviation', 0)
        effective_max_dev = max_dev * 2.0 if is_breakout else max_dev
        print(f"    MA20偏离: {deviation:.4f}  阈值: {effective_max_dev}  {'通过' if deviation <= effective_max_dev else '拦截'}")

rsi_ind = indicators.get("RSI")
if rsi_ind:
    rsi_val = rsi_ind.values.get('rsi', 0)
    rsi_limit = group_params.get('rsi_overbought', 0)
    print(f"    RSI: {rsi_val:.1f}  阈值: {rsi_limit}  {'通过' if rsi_limit == 0 or rsi_val <= rsi_limit else '拦截'}")
