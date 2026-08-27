"""大盘 MA60 过滤器

提取自 BacktestEngine._compute_market_ma60, 用 pandas rolling 替代手动循环。
原实现 O(N²) 嵌套循环, 本实现 O(N) 一次性计算所有日期的 MA60 趋势。
"""
from datetime import date
from typing import Dict, List, Optional

import pandas as pd
import numpy as np


class MarketFilter:
    """大盘 MA60 趋势过滤器"""

    def __init__(self, benchmark_df: Optional[pd.DataFrame] = None):
        """
        Args:
            benchmark_df: 基准指数日线 (需含 date, close 列)
        """
        import logging
        self._logger = logging.getLogger(__name__)
        self._trend_map: Dict[date, int] = self._compute(benchmark_df) if benchmark_df is not None else {}
        # 基准数据不足(行数<60或缺列)时无法计算 MA60, 大盘保护整体失效.
        # 此前 is_bearish 默认按"多头"处理, 会静默放行所有开仓. 这里显式告警,
        # 避免熊市保护在无数据时悄悄失效而用户无感知.
        if benchmark_df is not None and not self._trend_map:
            self._logger.warning(
                "MarketFilter: 基准数据不足(需>=60个交易日)或无 close 列, "
                "大盘 MA60 保护失效, 回测将按未知体制处理(不开空仓保护)."
            )

    @staticmethod
    def _compute(benchmark_df: pd.DataFrame) -> Dict[date, int]:
        """计算基准指数每日 MA60 趋势

        Returns:
            {date: 1(多头) / -1(空头)}, 仅包含 MA60 可计算的日期
        """
        if benchmark_df is None or len(benchmark_df) < 60:
            return {}

        df = benchmark_df.copy()
        # 统一 date 列为 date 对象
        if "date" in df.columns:
            df["date"] = pd.to_datetime(df["date"]).dt.date
        else:
            # 用索引
            df["date"] = pd.to_datetime(df.index).date

        close = df["close"].astype(np.float64)
        # pandas rolling 一次性计算, O(N)
        ma60 = close.rolling(window=60, min_periods=60).mean()
        trend = np.where(close > ma60, 1, -1)

        # 构建映射 (仅 MA60 可计算的日期)
        result: Dict[date, int] = {}
        for i, d in enumerate(df["date"]):
            if not np.isnan(ma60.iloc[i]):
                result[d] = int(trend[i])
        return result

    def is_bearish(self, target_date) -> bool:
        """指定日期大盘是否空头 (close < MA60)

        未知日期(基准数据缺失/未覆盖)返回 False(不开空仓保护), 但语义上
        是"未知"而非"确认多头", 与 get_direction 的 0 保持一致; 真正的
        失效已在 __init__ 处告警. 此前的默认 1(假设多头)会静默放行所有开仓.
        """
        if isinstance(target_date, pd.Timestamp):
            target_date = target_date.date()
        elif isinstance(target_date, str):
            target_date = pd.Timestamp(target_date).date()
        return self._trend_map.get(target_date, 0) == -1

    def get_direction(self, target_date) -> int:
        """获取指定日期的大盘趋势方向: 1 多头 / -1 空头 / 0 未知"""
        if isinstance(target_date, pd.Timestamp):
            target_date = target_date.date()
        elif isinstance(target_date, str):
            target_date = pd.Timestamp(target_date).date()
        return self._trend_map.get(target_date, 0)

    @property
    def trend_map(self) -> Dict[date, int]:
        """完整趋势映射 (只读)"""
        return dict(self._trend_map)
