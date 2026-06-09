"""
动量类指标: RSI(21) + KDJ(9,3,3)

RSI(21): 21周期 + EMA5平滑 + 中性缓冲区40-60
  - 对RSI原始值做EMA5平滑, 用平滑后曲线判断区间
  - 40-60 为绝对中性区, 不输出方向信号 (只输出中性)
  - 平滑RSI < 30 后拐头 → 超卖反弹信号
  - 平滑RSI > 70 后拐头 → 超买回调信号
  - 平滑RSI 60~70 为多头区间; 30~40 为空头区间

KDJ(9,3,3): 标准参数
  K上穿D + J < 20 → 超卖金叉
  K下穿D + J > 80 → 超买死叉
"""
import numpy as np
import pandas as pd
from .base import BaseIndicator, IndicatorResult


class RSIIndicator(BaseIndicator):
    """RSI 相对强弱指标 — EMA平滑 + 中性缓冲区"""
    name = "RSI"
    category = "momentum"

    # 默认参数
    DEFAULT_PARAMS = {
        "period": 21,
        "smooth_ema": 5,
        "oversold": 30,
        "overbought": 70,
        "neutral_zone": [40, 60],
    }

    def calculate(self, df: pd.DataFrame, params: dict = None) -> IndicatorResult:
        p = {**self.DEFAULT_PARAMS, **(params or {})}
        period = p["period"]
        smooth = p["smooth_ema"]
        oversold = p["oversold"]
        overbought = p["overbought"]
        nz_low, nz_high = p["neutral_zone"]

        close = df["close"].values.astype(np.float64)

        min_len = period + smooth + 2
        if len(close) < min_len:
            return self._make_result(0, "neutral", 0.0, f"数据不足{min_len}天")

        rsi_raw = self._rsi(close, period=period)
        rsi = self._ema(rsi_raw, period=smooth)
        latest = rsi[-1]
        prev = rsi[-2]

        # 中性缓冲区: 绝对观望, 不输出方向
        if nz_low <= latest <= nz_high:
            return self._make_result(
                0, "neutral", 0.0,
                f"RSI中性缓冲区 (平滑RSI={latest:.1f}), 观望",
                rsi=round(latest, 1), rsi_raw=round(rsi_raw[-1], 1)
            )

        # 超卖区域拐头向上
        if latest < oversold and latest > prev:
            return self._make_result(
                1, "buy", 0.7,
                f"RSI超卖反弹! 平滑RSI={latest:.1f}, 从低位拐头向上",
                rsi=round(latest, 1), rsi_raw=round(rsi_raw[-1], 1)
            )
        # 超买区域拐头向下
        elif latest > overbought and latest < prev:
            return self._make_result(
                -1, "sell", 0.7,
                f"RSI超买回落! 平滑RSI={latest:.1f}, 从高位拐头向下",
                rsi=round(latest, 1), rsi_raw=round(rsi_raw[-1], 1)
            )
        # 多头区间
        elif latest > nz_high:
            return self._make_result(
                1, "buy", 0.3,
                f"RSI强多头区间 (平滑RSI={latest:.1f})",
                rsi=round(latest, 1), rsi_raw=round(rsi_raw[-1], 1)
            )
        # 空头区间
        elif latest < nz_low:
            return self._make_result(
                -1, "sell", 0.3,
                f"RSI强空头区间 (平滑RSI={latest:.1f})",
                rsi=round(latest, 1), rsi_raw=round(rsi_raw[-1], 1)
            )
        else:
            return self._make_result(
                0, "neutral", 0.0,
                f"RSI中性 (平滑RSI={latest:.1f})",
                rsi=round(latest, 1), rsi_raw=round(rsi_raw[-1], 1)
            )

    @staticmethod
    def _rsi(data: np.ndarray, period: int = 21) -> np.ndarray:
        delta = np.diff(data, prepend=data[0])
        gain = np.where(delta > 0, delta, 0)
        loss = np.where(delta < 0, -delta, 0)
        avg_gain = np.zeros_like(data)
        avg_loss = np.zeros_like(data)
        avg_gain[period] = np.mean(gain[1:period + 1])
        avg_loss[period] = np.mean(loss[1:period + 1])
        for i in range(period + 1, len(data)):
            avg_gain[i] = (avg_gain[i - 1] * (period - 1) + gain[i]) / period
            avg_loss[i] = (avg_loss[i - 1] * (period - 1) + loss[i]) / period
        rs = np.divide(avg_gain, avg_loss + 1e-10)
        return 100 - 100 / (1 + rs)

    @staticmethod
    def _ema(data: np.ndarray, period: int = 5) -> np.ndarray:
        alpha = 2.0 / (period + 1)
        result = np.zeros_like(data)
        result[0] = data[0]
        for i in range(1, len(data)):
            result[i] = alpha * data[i] + (1 - alpha) * result[i - 1]
        return result


class KDJIndicator(BaseIndicator):
    """KDJ 随机指标"""
    name = "KDJ"
    category = "momentum"

    DEFAULT_PARAMS = {
        "k_period": 9,
        "k_smooth": 3,
        "d_smooth": 3,
        "oversold_j": 20,
        "overbought_j": 80,
    }

    def calculate(self, df: pd.DataFrame, params: dict = None) -> IndicatorResult:
        p = {**self.DEFAULT_PARAMS, **(params or {})}
        n = p["k_period"]
        m1 = p["k_smooth"]
        m2 = p["d_smooth"]
        oversold_j = p["oversold_j"]
        overbought_j = p["overbought_j"]

        high = df["high"].values.astype(np.float64)
        low = df["low"].values.astype(np.float64)
        close = df["close"].values.astype(np.float64)

        if len(close) < n + 3:
            return self._make_result(0, "neutral", 0.0, f"数据不足{n + 3}天")

        k, d, j = self._kdj(high, low, close, n=n, m1=m1, m2=m2)
        latest_k, latest_d, latest_j = k[-1], d[-1], j[-1]
        prev_k, prev_d, prev_j = k[-2], d[-2], j[-2]

        # 超卖区金叉 (K上穿D 且 J<oversold_j)
        if prev_k <= prev_d and latest_k > latest_d and latest_j < 50:
            return self._make_result(
                1, "buy", 0.8 if latest_j < oversold_j else 0.5,
                f"KDJ{'超卖' if latest_j < oversold_j else ''}金叉! K={latest_k:.1f}, D={latest_d:.1f}, J={latest_j:.1f}",
                K=round(latest_k, 1), D=round(latest_d, 1), J=round(latest_j, 1)
            )
        # 超买区死叉 (K下穿D 且 J>overbought_j)
        elif prev_k >= prev_d and latest_k < latest_d and latest_j > 50:
            return self._make_result(
                -1, "sell", 0.8 if latest_j > overbought_j else 0.5,
                f"KDJ{'超买' if latest_j > overbought_j else ''}死叉! K={latest_k:.1f}, D={latest_d:.1f}, J={latest_j:.1f}",
                K=round(latest_k, 1), D=round(latest_d, 1), J=round(latest_j, 1)
            )
        # 金叉后多头
        elif latest_k > latest_d:
            return self._make_result(
                1, "buy", 0.3,
                f"KDJ多头排列 (K={latest_k:.1f} > D={latest_d:.1f})",
                K=round(latest_k, 1), D=round(latest_d, 1), J=round(latest_j, 1)
            )
        else:
            return self._make_result(
                -1, "sell", 0.3,
                f"KDJ空头排列 (K={latest_k:.1f} < D={latest_d:.1f})",
                K=round(latest_k, 1), D=round(latest_d, 1), J=round(latest_j, 1)
            )

    @staticmethod
    def _kdj(high, low, close, n=9, m1=3, m2=3):
        """计算 KDJ"""
        lowest_low = np.array([low[max(0, i - n + 1):i + 1].min() for i in range(len(low))])
        highest_high = np.array([high[max(0, i - n + 1):i + 1].max() for i in range(len(high))])

        rsv = np.where(
            highest_high - lowest_low > 0,
            (close - lowest_low) / (highest_high - lowest_low + 1e-10) * 100,
            50  # 平盘默认50
        )

        k = np.zeros_like(close)
        d = np.zeros_like(close)
        k[0], d[0] = 50, 50
        for i in range(1, len(close)):
            k[i] = k[i - 1] * (m1 - 1) / m1 + rsv[i] / m1
            d[i] = d[i - 1] * (m2 - 1) / m2 + k[i] / m2
        j = 3 * k - 2 * d
        return k, d, j