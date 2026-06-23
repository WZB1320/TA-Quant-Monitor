"""市场体制检测器

提取自 BacktestEngine._get_market_regime, 统一体制判断逻辑。
原 BacktestEngine 用波动率+均线判断, 而 Scorer._get_regime 用 ADX,
两套逻辑并存且不一致。本模块统一入口, 消除双套逻辑。

注意: 本模块保留原 BacktestEngine 的波动率+均线判断算法 (向后兼容),
后续可统一为 ADX 方案。当前优先拆分, 不改变既有回测结果。
"""
from datetime import date
from typing import Optional

import pandas as pd
import numpy as np


class RegimeDetector:
    """市场体制检测器"""

    def __init__(self,
                 trend_threshold: float = 0.01,
                 weak_trend_threshold: float = 0.005,
                 vol_ratio_high: float = 1.3,
                 vol_ratio_extreme: float = 1.5):
        """
        Args:
            trend_threshold: 趋势强度阈值 (ma20-ma60)/ma60, 超过则判为趋势市
            weak_trend_threshold: 弱趋势阈值
            vol_ratio_high: 近期波动率 / 长期波动率 > 此值判为高波动
            vol_ratio_extreme: 波动率极端倍数
        """
        self._trend_th = trend_threshold
        self._weak_th = weak_trend_threshold
        self._vol_high = vol_ratio_high
        self._vol_extreme = vol_ratio_extreme

    def detect(self, benchmark_df: Optional[pd.DataFrame],
               today,
               calendar=None) -> str:
        """检测指定日期的市场体制

        Args:
            benchmark_df: 基准指数日线 (需含 close 列, 可选 high/low)
            today: 目标日期
            calendar: 可选, 提供则用 O(1) 定位; 否则回退到线性扫描

        Returns:
            "trending" / "transition" / "ranging"
        """
        if benchmark_df is None or len(benchmark_df) < 30:
            return "transition"

        # 定位 today 在 benchmark_df 中的位置
        if calendar is not None:
            idx = calendar.locate("__benchmark__", today)
        else:
            idx = self._locate(benchmark_df, today)

        if idx is None or idx < 28:
            return "transition"

        close = benchmark_df["close"].values[:idx + 1].astype(np.float64)
        if len(close) < 40:
            return "transition"

        # 最近20日收益率波动率
        returns = np.diff(close[-21:]) / close[-21:-1]
        recent_vol = np.std(returns) * np.sqrt(252)

        # 最近60日波动率
        if len(close) >= 61:
            long_returns = np.diff(close[-61:]) / close[-61:-1]
            long_vol = np.std(long_returns) * np.sqrt(252)
        else:
            long_vol = recent_vol

        # 趋势强度: 20日均线 vs 60日均线
        ma20 = np.mean(close[-20:])
        ma60 = np.mean(close[-60:]) if len(close) >= 60 else ma20
        trend_strength = (ma20 - ma60) / ma60

        # 综合判断 (保留原 BacktestEngine 逻辑)
        if abs(trend_strength) > self._trend_th and recent_vol < long_vol * self._vol_extreme:
            return "trending"
        elif recent_vol > long_vol * self._vol_high:
            return "ranging"
        elif abs(trend_strength) > self._weak_th:
            return "transition"
        else:
            return "ranging"

    @staticmethod
    def _locate(df: pd.DataFrame, target) -> Optional[int]:
        """线性扫描定位 (回退方案)"""
        if df.index.name == "date" or isinstance(df.index, pd.DatetimeIndex):
            if isinstance(target, date) and not isinstance(target, pd.Timestamp):
                target = pd.Timestamp(target)
            try:
                loc = df.index.get_loc(target)
                if isinstance(loc, slice):
                    return loc.start
                if isinstance(loc, np.ndarray):
                    return int(loc[0]) if len(loc) > 0 else None
                return int(loc)
            except KeyError:
                return None
        elif "date" in df.columns:
            target_str = target.strftime("%Y-%m-%d") if isinstance(target, date) else str(target)
            mask = df["date"] == target_str
            if mask.any():
                return int(mask.idxmax())
            return None
        return None
