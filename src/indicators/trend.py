"""
趋势类指标: EMA12/26 双均线 + MA60 + MACD

趋势骨架三部曲:
  MA60  — 牛熊分界线（硬过滤）
  EMA12/26 — 短期/中期趋势方向
  MACD — 趋势动量确认

信号逻辑（看多）:
  价格 > MA60 且 EMA12 > EMA26 且 MACD 零轴上方 → 看多
  价格 < MA60 → 看空（硬约束）
  均线缠绕且 MACD 接近零轴 → 中性
"""
import numpy as np
import pandas as pd
from .base import BaseIndicator, IndicatorResult


class MA60Indicator(BaseIndicator):
    """MA60 牛熊分界线"""
    name = "MA60"
    category = "trend"

    def calculate(self, df: pd.DataFrame, params: dict = None) -> IndicatorResult:
        close = df["close"].values.astype(np.float64)

        if len(close) < 60:
            return self._make_result(0, "neutral", 0.0, "数据不足60天，无法计算MA60")

        ma60 = np.mean(close[-60:])
        latest = close[-1]
        pct = (latest - ma60) / ma60 * 100

        if latest > ma60:
            return self._make_result(
                1, "buy", min(abs(pct) / 5, 1.0),
                f"价格在MA60上方 ({latest:.2f} > {ma60:.2f}), 多头区域",
                ma60=round(ma60, 2), price=latest, deviation_pct=round(pct, 2)
            )
        else:
            return self._make_result(
                -1, "sell", min(abs(pct) / 5, 1.0),
                f"价格在MA60下方 ({latest:.2f} < {ma60:.2f}), 空头区域",
                ma60=round(ma60, 2), price=latest, deviation_pct=round(pct, 2)
            )


class EMADualIndicator(BaseIndicator):
    """EMA双均线 — 差值变化率预判"""
    name = "EMA_DUAL"
    category = "trend"

    DEFAULT_PARAMS = {
        "fast": 12,
        "slow": 26,
    }

    def calculate(self, df: pd.DataFrame, params: dict = None) -> IndicatorResult:
        p = {**self.DEFAULT_PARAMS, **(params or {})}
        fast_period = p["fast"]
        slow_period = p["slow"]

        close = df["close"].values.astype(np.float64)

        if len(close) < slow_period:
            return self._make_result(0, "neutral", 0.0, f"数据不足{slow_period}天")

        ema_fast = self._ema(close, fast_period)
        ema_slow = self._ema(close, slow_period)
        latest_f, latest_s = ema_fast[-1], ema_slow[-1]
        prev_f, prev_s = ema_fast[-2], ema_slow[-2]

        diff_pct = (latest_f - latest_s) / latest_s * 100

        # 计算差值变化率 (过去5天差值的一阶差分, 判断开口方向)
        diffs = ema_fast[-5:] - ema_slow[-5:]
        diff_trend = diffs[-1] - diffs[0]  # 正=开口扩大, 负=收缩

        # --- 左侧潜伏买点: 差值虽为负但持续回升 (开口收缩) ---
        if latest_f < latest_s:
            diff_3d = ema_fast[-3:] - ema_slow[-3:]
            narrowing = all(diff_3d[i] > diff_3d[i-1] for i in range(1, len(diff_3d)))
            price_above_short = close[-1] > ema_fast[-1]
            if narrowing and price_above_short:
                return self._make_result(
                    1, "buy", 0.5,
                    f"EMA左侧潜伏买点! 差值收缩中 (diff={diff_pct:+.2f}%), 价格站上EMA{fast_period}",
                    ema12=round(latest_f, 2), ema26=round(latest_s, 2),
                    diff_pct=round(diff_pct, 2)
                )

        # 金叉
        if prev_f <= prev_s and latest_f > latest_s:
            return self._make_result(
                1, "buy", 0.8, f"EMA金叉! EMA{fast_period}({latest_f:.2f}) 上穿 EMA{slow_period}({latest_s:.2f})",
                ema12=round(latest_f, 2), ema26=round(latest_s, 2), diff_pct=round(diff_pct, 2)
            )
        # 死叉
        elif prev_f >= prev_s and latest_f < latest_s:
            return self._make_result(
                -1, "sell", 0.8, f"EMA死叉! EMA{fast_period}({latest_f:.2f}) 下穿 EMA{slow_period}({latest_s:.2f})",
                ema12=round(latest_f, 2), ema26=round(latest_s, 2), diff_pct=round(diff_pct, 2)
            )
        # 多头排列
        elif latest_f > latest_s:
            strength = 0.8 if diff_trend > 0 else 0.4
            return self._make_result(
                1, "buy", strength,
                f"EMA多头排列 (EMA{fast_period}={latest_f:.2f} > EMA{slow_period}={latest_s:.2f}), "
                f"开口{'扩大' if diff_trend > 0 else '持平/收缩'}",
                ema12=round(latest_f, 2), ema26=round(latest_s, 2), diff_pct=round(diff_pct, 2)
            )
        else:
            return self._make_result(
                -1, "sell", 0.4, f"EMA空头排列 (EMA{fast_period}={latest_f:.2f} < EMA{slow_period}={latest_s:.2f})",
                ema12=round(latest_f, 2), ema26=round(latest_s, 2), diff_pct=round(diff_pct, 2)
            )

    @staticmethod
    def _ema(data: np.ndarray, period: int) -> np.ndarray:
        alpha = 2.0 / (period + 1)
        result = np.zeros_like(data)
        result[0] = data[0]
        for i in range(1, len(data)):
            result[i] = alpha * data[i] + (1 - alpha) * result[i - 1]
        return result


class MACDIndicator(BaseIndicator):
    """MACD — DIF差值变化率预判

    信号逻辑:
      - 左侧预判: DIF虽然 < DEA, 但DIF连续回升且柱状图缩窄→止跌企稳
      - DIF 上穿 DEA 且 MACD 柱 > 0 → 看多
      - DIF 下穿 DEA → 看空
      - DIF 在零轴上方/下方作为趋势方向确认
    """
    name = "MACD"
    category = "trend"

    DEFAULT_PARAMS = {
        "fast": 12,
        "slow": 26,
        "signal": 9,
        # ── 新增: 柱状线二次放大参数 ──
        "bar_expansion_lookback": 10,    # 二次放大检测窗口
        "bar_expansion_max_bonus": 0.20, # 二次放大最大强度加成
    }

    def calculate(self, df: pd.DataFrame, params: dict = None) -> IndicatorResult:
        p = {**self.DEFAULT_PARAMS, **(params or {})}
        fast = p["fast"]
        slow = p["slow"]
        signal = p["signal"]
        bar_lookback = p["bar_expansion_lookback"]
        bar_max_bonus = p["bar_expansion_max_bonus"]

        close = df["close"].values.astype(np.float64)

        min_len = slow + signal
        if len(close) < min_len:
            return self._make_result(0, "neutral", 0.0, f"数据不足{min_len}天")

        dif, dea, macd_bar = self._macd(close, fast=fast, slow=slow, signal=signal)

        latest_dif, latest_dea = dif[-1], dea[-1]
        prev_dif, prev_dea = dif[-2], dea[-2]
        latest_bar = macd_bar[-1]
        prev_bar = macd_bar[-2]

        # 顶背离检测: 价格创新高但DIF未创新高
        bearish_div = self._detect_bearish_divergence(close, dif, lookback=30)

        # 底背离检测: 价格创新低但DIF未创新低
        bullish_div = self._detect_bullish_divergence(close, dif, lookback=30)

        # --- 左侧潜伏买点: DIF < DEA 但 DIF 连续3天回升 + 柱状图缩窄 ---
        if latest_dif < latest_dea and latest_dif > 0:
            dif_rising = (dif[-1] > dif[-2] > dif[-3])  # DIF 连续3天回升
            bar_narrowing = [abs(b) for b in macd_bar[-3:]]
            bar_shrinking = bar_narrowing[-1] < bar_narrowing[-2] < bar_narrowing[-3]
            if dif_rising and bar_shrinking:
                # 底背离加成: 底部企稳信号更可靠
                strength = 0.60 if bullish_div else 0.45
                div_note = ", 底背离加成" if bullish_div else ""
                return self._make_result(
                    1, "buy", strength,
                    f"MACD左侧潜伏: DIF回升+柱收缩 (DIF={latest_dif:.3f}), 止跌企稳信号{div_note}",
                    dif=round(latest_dif, 3), dea=round(latest_dea, 3),
                    bar=round(latest_bar, 3),
                    bearish_divergence=bearish_div, bullish_divergence=bullish_div
                )

        # 顶背离时降低看多强度
        div_suffix = f", ⚠顶背离!" if bearish_div else ""

        # 金叉
        if prev_dif <= prev_dea and latest_dif > latest_dea:
            if latest_dif > 0:
                strength = 0.6 if bearish_div else 0.9  # 顶背离时金叉降级
                return self._make_result(
                    1, "buy", strength,
                    f"MACD零轴上方金叉! DIF({latest_dif:.3f}) 上穿 DEA({latest_dea:.3f}){div_suffix}",
                    dif=round(latest_dif, 3), dea=round(latest_dea, 3),
                    bar=round(latest_bar, 3),
                    bearish_divergence=bearish_div, bullish_divergence=bullish_div
                )
            else:
                # 底背离加成: 零轴下金叉+底背离 = 反转信号更可靠
                strength = 0.70 if bullish_div else 0.5
                div_note = ", 底背离反转" if bullish_div else ""
                return self._make_result(
                    1, "buy", strength,
                    f"MACD零轴下方金叉 (弱势反弹信号){div_note}",
                    dif=round(latest_dif, 3), dea=round(latest_dea, 3),
                    bar=round(latest_bar, 3),
                    bearish_divergence=bearish_div, bullish_divergence=bullish_div
                )
        # 死叉
        elif prev_dif >= prev_dea and latest_dif < latest_dea:
            return self._make_result(
                -1, "sell", 0.7,
                f"MACD死叉! DIF({latest_dif:.3f}) 下穿 DEA({latest_dea:.3f})",
                dif=round(latest_dif, 3), dea=round(latest_dea, 3),
                bar=round(latest_bar, 3),
                bearish_divergence=bearish_div, bullish_divergence=bullish_div
            )
        # 多头区域
        elif latest_dif > 0 and latest_dif > latest_dea:
            bar_strengthening = latest_bar > prev_bar
            base_strength = 0.4 + (0.2 if bar_strengthening else 0)

            # 柱状线二次放大加成: 趋势休整后再次加速
            second_exp, exp_strength = self._detect_bar_second_expansion(
                macd_bar, lookback=bar_lookback
            )
            if second_exp:
                base_strength = min(base_strength * (1 + exp_strength * bar_max_bonus), 1.0)

            strength = base_strength * 0.6 if bearish_div else base_strength  # 顶背离降级
            exp_note = f", 柱二次放大({exp_strength:.0%})" if second_exp else ""
            return self._make_result(
                1, "buy", strength,
                f"MACD多头区域, DIF={latest_dif:.3f}, 柱{'在增强' if bar_strengthening else '在减弱'}{exp_note}{div_suffix}",
                dif=round(latest_dif, 3), dea=round(latest_dea, 3),
                bar=round(latest_bar, 3),
                bearish_divergence=bearish_div, bullish_divergence=bullish_div,
                bar_second_expansion=second_exp, bar_expansion_strength=round(exp_strength, 2)
            )
        elif latest_dif > 0:
            return self._make_result(
                1, "buy", 0.2,
                f"MACD零轴上方但DIF < DEA, 多头减弱",
                dif=round(latest_dif, 3), dea=round(latest_dea, 3),
                bar=round(latest_bar, 3),
                bearish_divergence=bearish_div, bullish_divergence=bullish_div
            )
        else:
            # 底背离降级: 空头区域+底背离 = 下跌动能减弱, 看空强度降低
            strength = 0.15 if bullish_div else 0.3
            div_note = ", 底背离(动能减弱)" if bullish_div else ""
            return self._make_result(
                -1, "sell", strength,
                f"MACD空头区域, DIF={latest_dif:.3f}{div_note}",
                dif=round(latest_dif, 3), dea=round(latest_dea, 3),
                bar=round(latest_bar, 3),
                bearish_divergence=bearish_div, bullish_divergence=bullish_div
            )

    @staticmethod
    def _detect_bearish_divergence(close: np.ndarray, dif: np.ndarray,
                                    lookback: int = 30) -> bool:
        """顶背离检测: 近期价格创新高但DIF未创新高"""
        n = len(close)
        if n < lookback:
            return False
        recent_close = close[-lookback:]
        recent_dif = dif[-lookback:]
        # 找近期最高价位置
        peak_idx = np.argmax(recent_close)
        # 如果最高价在最近5天内(当前高位区域)
        if peak_idx < lookback - 5:
            return False
        # 在最高价之前找前一个高点
        before_peak_close = recent_close[:peak_idx + 1]
        before_peak_dif = recent_dif[:peak_idx + 1]
        if len(before_peak_close) < 5:
            return False
        prev_peak_idx = np.argmax(before_peak_close[:-3])  # 排除当前峰值附近
        # 价格创新高但DIF未创新高 → 顶背离
        if (recent_close[peak_idx] > before_peak_close[prev_peak_idx] and
                recent_dif[peak_idx] < before_peak_dif[prev_peak_idx]):
            return True
        return False

    @staticmethod
    def _detect_bullish_divergence(close: np.ndarray, dif: np.ndarray,
                                    lookback: int = 30) -> bool:
        """底背离检测: 近期价格创新低但DIF未创新低"""
        n = len(close)
        if n < lookback:
            return False
        recent_close = close[-lookback:]
        recent_dif = dif[-lookback:]
        # 找近期最低价位置
        trough_idx = np.argmin(recent_close)
        if trough_idx < lookback - 5:
            return False
        before_trough_close = recent_close[:trough_idx + 1]
        before_trough_dif = recent_dif[:trough_idx + 1]
        if len(before_trough_close) < 5:
            return False
        prev_trough_idx = np.argmin(before_trough_close[:-3])
        # 价格创新低但DIF未创新低 → 底背离
        if (recent_close[trough_idx] < before_trough_close[prev_trough_idx] and
                recent_dif[trough_idx] > before_trough_dif[prev_trough_idx]):
            return True
        return False

    @staticmethod
    def _detect_bar_second_expansion(macd_bar: np.ndarray,
                                      lookback: int = 10) -> tuple:
        """检测柱状线二次放大

        逻辑: 柱状线经历 放大→收缩→再放大 的过程, 表示趋势休整后再次加速

        Returns:
            (is_second_expansion: bool, expansion_strength: float 0~1)
        """
        if len(macd_bar) < lookback:
            return False, 0.0

        bars = macd_bar[-lookback:]

        # 当前柱状线必须为正且在放大
        if bars[-1] <= 0 or bars[-1] <= bars[-2]:
            return False, 0.0

        # 在前 lookback-3 天找柱状线的局部极小值(收缩点)
        # 极小值: 比前一天小, 且后一天开始放大
        contraction_idx = None
        search_range = min(lookback - 3, 7)  # 最多往前找7天
        for i in range(1, search_range):
            if bars[i] < bars[i - 1] and bars[i + 1] > bars[i]:
                contraction_idx = i
                break

        if contraction_idx is None:
            return False, 0.0

        # 从收缩点开始, 柱状线应整体放大趋势(允许单日小幅回调)
        expansion_bars = bars[contraction_idx:]
        if len(expansion_bars) < 3:
            return False, 0.0

        expanding = True
        for i in range(1, len(expansion_bars)):
            if expansion_bars[i] < expansion_bars[i - 1] * 0.7:  # 允许30%回调
                expanding = False
                break

        if not expanding:
            return False, 0.0

        # 二次放大幅度: 当前柱状线 / 收缩点柱状线绝对值, 3倍封顶
        contraction_val = abs(bars[contraction_idx])
        current_val = abs(bars[-1])
        if contraction_val > 1e-10:
            expansion_ratio = current_val / contraction_val
        else:
            expansion_ratio = 1.0

        expansion_strength = min(expansion_ratio / 3.0, 1.0)
        return True, expansion_strength

    @staticmethod
    def _macd(data: np.ndarray, fast=12, slow=26, signal=9):
        ema_fast = _ema(data, fast)
        ema_slow = _ema(data, slow)
        dif = ema_fast - ema_slow
        dea = _ema(dif, signal)
        macd_bar = 2 * (dif - dea)
        return dif, dea, macd_bar


def _ema(data: np.ndarray, period: int) -> np.ndarray:
    alpha = 2.0 / (period + 1)
    result = np.zeros_like(data)
    result[0] = data[0]
    for i in range(1, len(data)):
        result[i] = alpha * data[i] + (1 - alpha) * result[i - 1]
    return result


class MA20Indicator(BaseIndicator):
    """MA20 短期均线 — 用于价格偏离度过滤

    不参与评分，仅用于过滤：
      - 价格偏离MA20过大时拦截买入信号（避免追高）
    """
    name = "MA20"
    category = "trend"

    def calculate(self, df: pd.DataFrame, params: dict = None) -> IndicatorResult:
        close = df["close"].values.astype(np.float64)

        if len(close) < 20:
            return self._make_result(0, "neutral", 0.0, "数据不足20天")

        ma20 = np.mean(close[-20:])
        latest = close[-1]
        deviation = (latest - ma20) / ma20

        return self._make_result(
            0, "neutral", 0.0,
            f"MA20={ma20:.2f}, 价格={latest:.2f}, 偏离={deviation:.2%}",
            ma20=round(ma20, 2), price=round(float(latest), 2),
            deviation=round(float(deviation), 4)
        )