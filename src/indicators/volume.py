"""
量价类指标: OBV + 量比

OBV (On-Balance Volume): 能量潮
  逻辑: 价格涨 → OBV +当日成交量，价格跌 → OBV -当日成交量
  信号: OBV创新高 → 资金流入，真突破确认
       价格新高但OBV未新高 → 量价背离，假突破预警

量比: Vol / MA(Vol, 20)
  > 1.5: 放量
  < 0.5: 缩量
  用于确认突破/跌破的真实性
"""
import numpy as np
import pandas as pd
from .base import BaseIndicator, IndicatorResult


class OBVIndicator(BaseIndicator):
    """OBV 能量潮"""
    name = "OBV"
    category = "volume"

    def calculate(self, df: pd.DataFrame, params: dict = None) -> IndicatorResult:
        close = df["close"].values.astype(np.float64)
        volume = df["volume"].values.astype(np.float64)

        if len(close) < 20:
            return self._make_result(0, "neutral", 0.0, "数据不足20天")

        obv = self._obv(close, volume)

        # 比较最近10天内的OBV高低点
        recent_obv = obv[-10:]
        recent_close = close[-10:]

        obv_new_high = obv[-1] >= np.max(obv[-20:])
        price_new_high = close[-1] >= np.max(close[-20:])

        obv_trend = obv[-1] - np.mean(obv[-20:])
        obv_direction = 1 if obv_trend > 0 else -1

        # 量价共振：价格新高 + OBV新高
        if price_new_high and obv_new_high:
            return self._make_result(
                1, "buy", 0.8,
                "OBV与价格同步创新高, 量价共振, 上涨有资金支撑",
                obv=round(obv[-1], 0)
            )
        # 量价背离：价格新高但 OBV 未新高
        elif price_new_high and not obv_new_high:
            return self._make_result(
                -1, "sell", 0.7,
                "OBV未随价格创新高, 量价背离! 警惕假突破/见顶",
                obv=round(obv[-1], 0)
            )
        # OBV下降趋势
        elif obv_direction < 0:
            return self._make_result(
                -1, "sell", 0.4,
                "OBV下降趋势, 资金在流出",
                obv=round(obv[-1], 0)
            )
        # OBV上升趋势
        else:
            return self._make_result(
                1, "buy", 0.4,
                "OBV上升趋势, 资金在流入",
                obv=round(obv[-1], 0)
            )

    @staticmethod
    def _obv(close: np.ndarray, volume: np.ndarray) -> np.ndarray:
        obv = np.zeros_like(close)
        obv[0] = volume[0]
        for i in range(1, len(close)):
            if close[i] > close[i - 1]:
                obv[i] = obv[i - 1] + volume[i]
            elif close[i] < close[i - 1]:
                obv[i] = obv[i - 1] - volume[i]
            else:
                obv[i] = obv[i - 1]
        return obv


class VolumeRatioIndicator(BaseIndicator):
    """量比 = 当日成交量 / 20日均量"""
    name = "VOL_RATIO"
    category = "volume"

    def calculate(self, df: pd.DataFrame, params: dict = None) -> IndicatorResult:
        volume = df["volume"].values.astype(np.float64)
        close = df["close"].values.astype(np.float64)

        if len(volume) < 21:
            return self._make_result(0, "neutral", 0.0, "数据不足21天")

        ma_vol20 = np.mean(volume[-21:-1])
        latest_vol = volume[-1]
        vol_ratio = latest_vol / ma_vol20 if ma_vol20 > 0 else 1

        price_up = close[-1] > close[-2] if len(close) >= 2 else False

        if vol_ratio > 2.0 and price_up:
            return self._make_result(
                1, "buy", 0.9,
                f"大幅放量上涨! 量比={vol_ratio:.1f}倍, 突破信号强烈",
                vol_ratio=round(vol_ratio, 1)
            )
        elif vol_ratio > 1.5 and price_up:
            return self._make_result(
                1, "buy", 0.5,
                f"放量上涨 (量比={vol_ratio:.1f}), 量价配合良好",
                vol_ratio=round(vol_ratio, 1)
            )
        elif vol_ratio > 2.0 and not price_up:
            return self._make_result(
                -1, "sell", 0.7,
                f"放量下跌! 量比={vol_ratio:.1f}倍, 有资金出逃迹象",
                vol_ratio=round(vol_ratio, 1)
            )
        elif vol_ratio > 1.5 and not price_up:
            return self._make_result(
                -1, "sell", 0.4,
                f"放量下跌 (量比={vol_ratio:.1f}), 注意风险",
                vol_ratio=round(vol_ratio, 1)
            )
        elif vol_ratio < 0.5:
            return self._make_result(
                0, "neutral", 0.0,
                f"严重缩量 (量比={vol_ratio:.1f}), 交投清淡",
                vol_ratio=round(vol_ratio, 1)
            )
        else:
            return self._make_result(
                0, "neutral", 0.0,
                f"正常量能 (量比={vol_ratio:.1f})",
                vol_ratio=round(vol_ratio, 1)
            )