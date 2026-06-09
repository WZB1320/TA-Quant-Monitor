"""
指标管道 (Pipeline)

核心职责:
  1. 按类别分组执行所有指标
  2. 按优先级顺序执行: MA60(硬过滤) → ADX(硬过滤) → 趋势指标 → 动量指标 → 量价指标
  3. 收集所有指标结果，输出汇总

管道是"分层管道架构"的核心组件，连接数据层和信号引擎。
"""
from typing import List, Dict
import pandas as pd

from .base import BaseIndicator, IndicatorResult
from .trend import MA60Indicator, EMADualIndicator, MACDIndicator, MA20Indicator
from .strength import ADXIndicator, ATRIndicator
from .momentum import RSIIndicator, KDJIndicator
from .volume import OBVIndicator, VolumeRatioIndicator


class IndicatorPipeline:
    """指标计算管道"""

    def __init__(self):
        # 按执行顺序排列（硬过滤在前）
        self._indicators: List[BaseIndicator] = [
            MA60Indicator(),          # 1. 牛熊分界（硬过滤）
            ADXIndicator(),           # 2. 趋势强度（硬过滤）
            ATRIndicator(),           # 3. 波动率（仓位&止损）
            MA20Indicator(),          # 4. 短期均线（偏离度过滤）
            EMADualIndicator(),       # 5. 趋势方向
            MACDIndicator(),          # 6. 趋势确认
            RSIIndicator(),           # 7. 动量位置
            KDJIndicator(),           # 8. 动量和拐点
            OBVIndicator(),           # 9. 量价关系
            VolumeRatioIndicator(),   # 10. 放量确认
        ]

    def run(self, df: pd.DataFrame, indicator_params: dict = None) -> Dict[str, IndicatorResult]:
        """
        对一支股票执行全部指标计算

        Args:
            df: 标准化日线数据 (date, open, high, low, close, volume)
            indicator_params: 分组专属指标参数 {指标名: {参数键: 参数值}}

        Returns:
            {指标名: IndicatorResult} 字典
        """
        params = indicator_params or {}
        results = {}
        for indicator in self._indicators:
            try:
                ind_params = params.get(indicator.name)
                result = indicator.calculate(df, params=ind_params)
                results[indicator.name] = result
            except Exception as e:
                # 单个指标异常不中断管道
                results[indicator.name] = IndicatorResult(
                    name=indicator.name,
                    category=indicator.category,
                    direction=0,
                    signal="neutral",
                    strength=0.0,
                    description=f"计算异常: {e}",
                )
        return results

    def run_batch(self, stock_data: Dict[str, pd.DataFrame]) -> Dict[str, Dict[str, IndicatorResult]]:
        """
        批量执行多只股票的指标计算

        Args:
            stock_data: {股票代码: DataFrame}

        Returns:
            {股票代码: {指标名: IndicatorResult}}
        """
        batch_results = {}
        for code, df in stock_data.items():
            batch_results[code] = self.run(df)
        return batch_results

    @staticmethod
    def get_hard_filters() -> List[str]:
        """返回硬过滤指标名称（这些指标不通过则直接抑制信号）"""
        return ["MA60", "ADX"]