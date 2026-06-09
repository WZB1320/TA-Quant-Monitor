"""
趋势强度 & 波动率指标: ADX(14), ATR(14)

ADX: 判断当前是否有趋势，以及趋势强度
ATR: 平均真实波幅, 用于动态仓位管理和止损
"""
import numpy as np
import pandas as pd
from .base import BaseIndicator, IndicatorResult


class ADXIndicator(BaseIndicator):
    """ADX 趋势强度"""
    name = "ADX"
    category = "strength"

    DEFAULT_PARAMS = {
        "period": 14,
        "weak": 20,
        "trending": 25,
        "strong": 40,
    }

    def calculate(self, df: pd.DataFrame, params: dict = None) -> IndicatorResult:
        p = {**self.DEFAULT_PARAMS, **(params or {})}
        period = p["period"]
        weak_th = p["weak"]
        trending_th = p["trending"]
        strong_th = p["strong"]

        high = df["high"].values.astype(np.float64)
        low = df["low"].values.astype(np.float64)
        close = df["close"].values.astype(np.float64)

        if len(close) < period * 2 + 2:
            return self._make_result(0, "neutral", 0.0, f"数据不足{period * 2 + 2}天")

        adx, plus_di, minus_di = self._adx(high, low, close, period=period)

        latest_adx = adx[-1]
        latest_plus = plus_di[-1]
        latest_minus = minus_di[-1]
        prev_adx = adx[-2] if len(adx) > 2 and not np.isnan(adx[-2]) else None

        # 判断方向
        if latest_plus > latest_minus:
            direction = 1
            dir_text = "+DI主导"
        else:
            direction = -1
            dir_text = "-DI主导"

        # 判断强度 (使用分组阈值)
        if latest_adx > strong_th:
            desc = f"ADX={latest_adx:.1f}, 强趋势 ({dir_text})"
            strength = 0.9
        elif latest_adx > trending_th:
            desc = f"ADX={latest_adx:.1f}, 趋势运行中 ({dir_text})"
            strength = 0.6
        elif latest_adx > weak_th:
            desc = f"ADX={latest_adx:.1f}, 弱势趋势, 谨慎参与 ({dir_text})"
            strength = 0.3
        else:
            desc = f"ADX={latest_adx:.1f}, 无趋势/震荡市, 不建议交易"
            strength = 0.0
            direction = 0

        return self._make_result(
            direction, "buy" if direction > 0 else "sell" if direction < 0 else "neutral",
            strength, desc,
            adx=round(latest_adx, 1), plus_di=round(latest_plus, 1),
            minus_di=round(latest_minus, 1),
            adx_prev=round(prev_adx, 1) if prev_adx is not None else None
        )

    @staticmethod
    def _adx(high, low, close, period=14):
        """计算 ADX / +DI / -DI"""
        n = len(close)
        if n < period + 1:
            return np.full(n, np.nan), np.full(n, np.nan), np.full(n, np.nan)

        tr = np.zeros(n)
        plus_dm = np.zeros(n)
        minus_dm = np.zeros(n)

        for i in range(1, n):
            hl = high[i] - low[i]
            hc = abs(high[i] - close[i - 1])
            lc = abs(low[i] - close[i - 1])
            tr[i] = max(hl, hc, lc)

            up = high[i] - high[i - 1]
            down = low[i - 1] - low[i]
            plus_dm[i] = up if (up > down and up > 0) else 0
            minus_dm[i] = down if (down > up and down > 0) else 0

        # Wilder's smoothing: 初始值用第一个 period 的 sum
        atr = np.zeros(n)
        atr_sm = np.zeros(n)
        dm_plus_sm = np.zeros(n)
        dm_minus_sm = np.zeros(n)

        atr[period] = np.sum(tr[1:period + 1])
        atr_sm[period] = np.sum(tr[1:period + 1])
        dm_plus_sm[period] = np.sum(plus_dm[1:period + 1])
        dm_minus_sm[period] = np.sum(minus_dm[1:period + 1])

        for i in range(period + 1, n):
            atr_sm[i] = atr_sm[i - 1] - atr_sm[i - 1] / period + tr[i]
            dm_plus_sm[i] = dm_plus_sm[i - 1] - dm_plus_sm[i - 1] / period + plus_dm[i]
            dm_minus_sm[i] = dm_minus_sm[i - 1] - dm_minus_sm[i - 1] / period + minus_dm[i]
            atr[i] = atr_sm[i] / period if period > 0 else atr_sm[i]

        atr[period] = atr_sm[period] / period

        eps = 1e-10
        plus_di = np.where(atr > eps, dm_plus_sm / (period * atr + eps) * 100, 50.0)
        minus_di = np.where(atr > eps, dm_minus_sm / (period * atr + eps) * 100, 50.0)

        dx = np.abs(plus_di - minus_di) / (plus_di + minus_di + eps) * 100

        adx = np.zeros(n)
        adx[2 * period] = np.mean(dx[period + 1:2 * period + 1])
        for i in range(2 * period + 1, n):
            adx[i] = (adx[i - 1] * (period - 1) + dx[i]) / period

        return adx, plus_di, minus_di


class ATRIndicator(BaseIndicator):
    """ATR(14) 平均真实波幅 — 动态仓位 & 止损的核心参数

    ATR 衡量的是股票每日平均波动幅度, 是专业量化交易最基础的波动率指标。

    用途:
      - ATR/Price 比值越大 → 波动越大 → 仓位应越小, 止损应越宽
      - ATR/Price 比值越小 → 波动越小 → 仓位可适当放大, 止损可收紧

    direction 跟随整体趋势 (由ADX提供), 本指标主要提供 atr_raw 和 atr_pct
    """
    name = "ATR"
    category = "strength"

    def calculate(self, df: pd.DataFrame, params: dict = None) -> IndicatorResult:
        high = df["high"].values.astype(np.float64)
        low = df["low"].values.astype(np.float64)
        close = df["close"].values.astype(np.float64)

        if len(close) < 20:
            return self._make_result(0, "neutral", 0.0, "数据不足20天")

        atr = self._atr(high, low, close, period=14)
        latest_atr = atr[-1]
        latest_price = close[-1]
        atr_pct = (latest_atr / latest_price) * 100  # ATR 占价格的百分比

        # 波动率分级 (用于仓位调整)
        if atr_pct > 5:
            vol_level = "极高波动"
            vol_str = 0.2  # 降仓80%
        elif atr_pct > 3:
            vol_level = "高波动"
            vol_str = 0.5  # 降仓50%
        elif atr_pct > 1.5:
            vol_level = "正常波动"
            vol_str = 0.8  # 标准仓位
        else:
            vol_level = "低波动"
            vol_str = 1.0  # 全额仓位

        # position_mult 用于仓位计算: pos = base_ratio * vol_str
        return self._make_result(
            0, "neutral", vol_str,
            f"ATR={latest_atr:.2f} ({atr_pct:.1f}%), {vol_level}, 仓位系数={vol_str:.1f}",
            atr=round(latest_atr, 2), atr_pct=round(atr_pct, 2),
            vol_level=vol_level, position_mult=round(vol_str, 2)
        )

    @staticmethod
    def _atr(high: np.ndarray, low: np.ndarray, close: np.ndarray,
             period: int = 14) -> np.ndarray:
        """计算 ATR (Wilder's smoothing)"""
        n = len(close)
        tr = np.zeros(n)
        for i in range(1, n):
            hl = high[i] - low[i]
            hc = abs(high[i] - close[i - 1])
            lc = abs(low[i] - close[i - 1])
            tr[i] = max(hl, hc, lc)

        atr = np.zeros(n)
        atr[period] = np.mean(tr[1:period + 1])
        for i in range(period + 1, n):
            atr[i] = (atr[i - 1] * (period - 1) + tr[i]) / period
        return atr