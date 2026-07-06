"""
信号过滤器 (Filter)

过滤规则:
  1. 硬过滤 (Hard Filter): MA60 定牛熊 + ADX 判断有没有趋势
     - 看多信号: 必须价格 > MA60, 否则降为 NEUTRAL
     - 看空信号: 必须价格 < MA60, 否则降为 NEUTRAL
     - ADX < 20: 全部降为观望 (震荡市不交易)
  2. 信号去重: 同一只股票同一方向的信号, N天内不重复发出
  3. 大盘环境 (可选 V2): 指数 < MA60 时抑制买入信号

注意: is_duplicate / record 使用实际分析日期而非 datetime.now(),
     避免回测时所有信号被误判为同日重复。

运行时模式:
  - LIVE 模式: 信号历史读写 signal_history.json, 跨会话保留
  - BACKTEST 模式: 信号历史仅存内存, 不写盘, 避免污染实时数据
  由 src.config.runtime_mode 控制, 默认 LIVE (向后兼容)
"""
from datetime import datetime, date, timedelta
from typing import Dict, Optional
import json
import os

from src.indicators.base import IndicatorResult
from src.signal_engine.signals import SignalLevel, SignalResult
from src.config.runtime_mode import get_mode, RuntimeMode

# 去重记录文件 (仅用于实时交易, 回测禁用)
_DEDUP_FILE = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))), "data", "signal_history.json")


class SignalFilter:
    """信号过滤器"""

    def __init__(self, dedup_days: int = 5, market_ma60_filter: bool = False,
                 cooldown_days: int = 4):
        """
        Args:
            dedup_days: 同一股票同方向信号去重天数
            market_ma60_filter: 是否启用大盘环境过滤 (需要外部传入大盘指标)
            cooldown_days: 卖出/止损后的冷却期天数, 期间禁止反向开仓 (默认4天)
        """
        self.dedup_days = dedup_days
        self.market_ma60_filter = market_ma60_filter
        self._market_ma60_direction: int = 0  # 大盘MA60方向: 1多头, -1空头, 0未知
        self.cooldown_days = cooldown_days
        self._last_exit: Dict[str, date] = {}  # {symbol: 最后一次卖出日期}
        self._history: Dict[str, str] = {}
        self._history_loaded: bool = False  # 懒加载标记
        self._consecutive_losses: Dict[str, int] = {}  # {symbol: 连续亏损次数}
        self._symbol_suspended_until: Dict[str, date] = {}  # {symbol: 暂停至日期}

    def _ensure_history_loaded(self) -> None:
        """懒加载信号历史 — 仅 LIVE 模式从磁盘读取, BACKTEST 模式用空内存"""
        if self._history_loaded:
            return
        self._history_loaded = True
        if get_mode() == RuntimeMode.LIVE:
            self._history = self._load_history()

    def set_market_ma60(self, direction: int):
        """设置大盘MA60方向 (由外部在每次分析前调用)"""
        self._market_ma60_direction = direction

    def record_exit(self, symbol: str, exit_date: date):
        """记录卖出/止损事件, 用于冷却期计算"""
        self._last_exit[symbol] = exit_date

    def record_loss(self, symbol: str, loss_date: date):
        """记录亏损交易, 用于连亏保护"""
        prev = self._consecutive_losses.get(symbol, 0)
        self._consecutive_losses[symbol] = prev + 1

    def record_win(self, symbol: str):
        """记录盈利交易, 重置连亏计数"""
        self._consecutive_losses[symbol] = 0

    def is_suspended(self, symbol: str, current_date: date,
                     max_consecutive_losses: int = 0,
                     suspend_days: int = 0) -> bool:
        """
        检查是否因连续亏损被暂停

        Args:
            symbol: 股票代码
            current_date: 当前日期
            max_consecutive_losses: 连亏触发阈值 (0=不启用)
            suspend_days: 暂停天数

        Returns:
            True 如果当前被暂停
        """
        if max_consecutive_losses <= 0 or suspend_days <= 0:
            return False

        # 检查是否在暂停期内
        suspended_until = self._symbol_suspended_until.get(symbol)
        if suspended_until is not None and current_date <= suspended_until:
            return True

        # 暂停期已过, 重置连亏计数 (给股票重新机会)
        if suspended_until is not None and current_date > suspended_until:
            self._consecutive_losses[symbol] = 0
            del self._symbol_suspended_until[symbol]

        # 检查连亏是否触发暂停
        losses = self._consecutive_losses.get(symbol, 0)
        if losses >= max_consecutive_losses:
            self._symbol_suspended_until[symbol] = current_date + timedelta(days=suspend_days)
            return True

        return False

    def is_in_cooldown(self, symbol: str, current_date: date,
                       is_bullish: bool,
                       group_cooldown: int = None) -> bool:
        """
        检查是否在冷却期

        冷却期规则: 卖出/止损后 cooldown_days 天内禁止同方向开仓
        如果是买入信号(is_bullish=True)且最近刚卖出, 则拦截

        Args:
            group_cooldown: 分组专属冷却天数 (可选, 覆盖全局默认)
        """
        if not is_bullish:
            return False  # 卖出信号不受冷却期限制

        last_exit = self._last_exit.get(symbol)
        if last_exit is None:
            return False

        cd = group_cooldown if group_cooldown is not None else self.cooldown_days
        days_since = (current_date - last_exit).days
        return days_since <= cd

    # ── 硬过滤 (仅 ADX 数据可用性检查, 不再硬拦截震荡市) ──
    def hard_filter(self, indicator_results: Dict[str, IndicatorResult]
                    ) -> tuple[bool, str]:
        """
        硬过滤: 仅检查数据可用性
        体制自适应评分器会根据 ADX 动态调整权重:
          - 趋势市(ADX>25): 趋势权重↑, 趋势策略
          - 震荡市(ADX<20): 动量权重↑, 均值回归策略

        Returns:
            (是否被拦截, 拦截原因)
        """
        adx = indicator_results.get("ADX")

        # ADX 未计算出来 → 数据不足, 无法评分
        if adx is None:
            return True, "ADX未计算, 数据不足"

        return False, ""

    def _apply_regime_filter_overrides(self, group_params: dict,
                                        indicator_results: Dict[str, IndicatorResult]) -> dict:
        """
        ADX+MA60 体制自适应: 根据当前ADX值和MA60方向, 用 regime_filter_overrides 中的参数覆盖默认过滤参数

        逻辑 (MA60方向作为辅助判断, 防止ADX滞后导致趋势末端误判):
          - ADX > 25 AND MA60多头: trending (真趋势, 放宽过滤)
          - ADX > 25 AND MA60空头: transition (趋势可能衰减, 不放宽)
          - ADX 20-25 AND MA60多头: transition (有支撑, 适中)
          - ADX 20-25 AND MA60空头: ranging (无支撑, 收紧)
          - ADX < 20: ranging (震荡市, 收紧)
          - 无 regime_filter_overrides: 保持原参数不变
        """
        overrides = group_params.get("regime_filter_overrides")
        if not overrides:
            return group_params

        # 获取当前ADX值
        adx_ind = indicator_results.get("ADX")
        if adx_ind is None:
            return group_params

        adx_val = adx_ind.values.get("adx", 0)
        if adx_val is None:
            return group_params

        # 获取MA60方向
        ma60_ind = indicator_results.get("MA60")
        ma60_bullish = False
        if ma60_ind is not None:
            ma60_bullish = ma60_ind.direction == 1  # 1=多头, -1=空头

        # 判断当前体制 (ADX + MA60 双重判断)
        if adx_val > 25:
            if ma60_bullish:
                regime = "trending"      # 真趋势: ADX强 + MA60多头
            else:
                regime = "transition"    # 趋势衰减: ADX强但MA60空头, 不放宽
        elif adx_val >= 20:
            if ma60_bullish:
                regime = "transition"    # 有支撑: ADX中等 + MA60多头
            else:
                regime = "ranging"       # 无支撑: ADX中等 + MA60空头, 收紧
        else:
            regime = "ranging"           # 震荡市: ADX弱, 无论MA60方向都收紧

        regime_params = overrides.get(regime)
        if not regime_params:
            return group_params

        # 用体制覆盖参数替换 group_params 中的对应字段
        merged = dict(group_params)
        for key in ("score_ceiling", "cooldown_days", "max_consecutive_losses",
                     "consecutive_loss_suspend", "rsi_overbought",
                     "vol_ratio_threshold", "atr_price_ratio_max",
                     "price_ma20_max_deviation"):
            if key in regime_params:
                merged[key] = regime_params[key]

        return merged

    def calc_dynamic_ceiling(self, base_ceiling: float, df: "pd.DataFrame",
                             indicator_results: Dict[str, IndicatorResult],
                             group_params: dict) -> float:
        """
        动态Ceiling计算: 突破确认 + 均线排列加成

        规则:
          1. 如果价格创20日新高 → 突破确认, ceiling提升
          2. 如果均线多头排列 → 趋势健康, ceiling额外提升
          3. 不满足条件 → 保持原ceiling不变

        Args:
            base_ceiling: 基础ceiling值 (如52)
            df: 日线数据
            indicator_results: 指标结果
            group_params: 分组参数 (含 breakout_ceiling_bonus, ma_alignment_bonus 等)

        Returns:
            动态调整后的ceiling值
        """
        import numpy as np

        # 如果未启用动态ceiling (bonus参数为0或不存在), 直接返回原值
        breakout_bonus = group_params.get("breakout_ceiling_bonus", 0)
        ma_bonus = group_params.get("ma_alignment_ceiling_bonus", 0)

        if breakout_bonus <= 0 and ma_bonus <= 0:
            return base_ceiling

        if df is None or len(df) < 20:
            return base_ceiling

        dynamic_ceiling = base_ceiling

        # ── 突破确认: 当前收盘价是否创20日新高 ──
        if breakout_bonus > 0:
            close = df["close"].values.astype(float)
            latest_close = close[-1]
            high_20d = np.max(close[-21:-1])  # 前20日最高价 (不含当日)
            if latest_close > high_20d:
                dynamic_ceiling += breakout_bonus

        # ── 均线排列加成: MA5 > MA10 > MA20 > MA60 ──
        if ma_bonus > 0 and len(df) >= 60:
            close = df["close"].values.astype(float)
            ma5 = np.mean(close[-5:])
            ma10 = np.mean(close[-10:])
            ma20 = np.mean(close[-20:])
            ma60 = np.mean(close[-60:])

            if ma5 > ma10 > ma20 > ma60:
                # 完全多头排列: 最大加成
                dynamic_ceiling += ma_bonus
            elif ma5 > ma10 > ma20:
                # 短中期多头: 半额加成
                dynamic_ceiling += ma_bonus // 2

        return dynamic_ceiling
    def apply_hard_constraint(self, level: SignalLevel, indicator_results:
                              Dict[str, IndicatorResult],
                              score_threshold: float = 25,
                              group_params: dict = None,
                              df: "pd.DataFrame" = None) -> tuple:
        """硬过滤后，限制信号方向 + 最低得分阈值 + 成交量过滤 + MACD顶背离 + 分组专属过滤

        Returns:
            (SignalLevel, str): 信号级别 + 拦截原因(空字符串表示未拦截)
        """
        if group_params is None:
            group_params = {}

        # ── ADX体制自适应: 根据当前ADX值动态覆盖过滤参数 ──
        # 保留原始score_ceiling供动态Ceiling计算使用
        original_ceiling = group_params.get("score_ceiling", 0) if group_params else 0
        group_params = self._apply_regime_filter_overrides(group_params, indicator_results)

        ma60 = indicator_results.get("MA60")
        if ma60 is None:
            return level, ""

        # 空头区域不能出看多信号 → 直接降为观望
        if ma60.direction == -1 and level.is_bullish:
            return SignalLevel.NEUTRAL, "价格在MA60下方(空头区域)，不发看多信号"
        # 多头区域不能出看空信号 → 直接降为观望
        if ma60.direction == 1 and level.is_bearish:
            return SignalLevel.NEUTRAL, "价格在MA60上方(多头区域)，不发看空信号"

        # 最低得分阈值 (分组专属)
        score = indicator_results.get("SCORE")
        st = group_params.get("score_threshold", score_threshold)
        if score is not None and abs(score) < st:
            return SignalLevel.NEUTRAL, f"得分{score:.1f}低于阈值{st}"

        # ── 第1层: 过热信号拦截 (动态Ceiling: 突破确认+均线排列加成) ──
        is_breakout_signal = False
        if level.is_bullish:
            base_ceiling = original_ceiling if original_ceiling > 0 else group_params.get("score_ceiling", 0)
            if base_ceiling > 0 and score is not None and score > base_ceiling:
                dynamic_ceiling = self.calc_dynamic_ceiling(
                    base_ceiling, df, indicator_results, group_params)
                if score > dynamic_ceiling:
                    return SignalLevel.NEUTRAL, f"得分{score:.1f}超过上限{dynamic_ceiling:.0f}(过热信号)"
                else:
                    is_breakout_signal = True
            elif df is not None and len(df) >= 20:
                import numpy as np
                close_arr = df["close"].values.astype(float)
                if close_arr[-1] > np.max(close_arr[-21:-1]):
                    is_breakout_signal = True

        # ── 第2层: 成交量过滤 (分组专属量比阈值) ──
        if level.is_bullish:
            vol = indicator_results.get("VOL_RATIO")
            if vol is not None:
                vr = vol.values.get("vol_ratio", 1.0)
                vol_threshold = group_params.get("vol_ratio_threshold", 0.6)
                if vr < vol_threshold:
                    return SignalLevel.NEUTRAL, f"量比{vr:.2f}低于阈值{vol_threshold}"

        # ── 第3层: ATR/Price波动率门槛 (突破信号放宽) ──
        if level.is_bullish:
            atr_max = group_params.get("atr_price_ratio_max", 0)
            if atr_max > 0:
                atr_ind = indicator_results.get("ATR")
                if atr_ind is not None:
                    atr_pct = atr_ind.values.get("atr_pct")
                    if atr_pct is not None:
                        atr_ratio = atr_pct / 100.0
                        effective_atr_max = atr_max * 1.5 if is_breakout_signal else atr_max
                        if atr_ratio > effective_atr_max:
                            return SignalLevel.NEUTRAL, f"ATR波动率{atr_ratio*100:.1f}%超过上限{effective_atr_max*100:.1f}%"

        # ── 第4层: MACD零轴位置过滤 ──
        if level.is_bullish:
            require_dif = group_params.get("require_macd_dif_above_zero", False)
            if require_dif:
                macd = indicator_results.get("MACD")
                if macd is not None:
                    dif = macd.values.get("dif")
                    if dif is not None and dif <= 0:
                        return SignalLevel.NEUTRAL, f"MACD DIF({dif:.3f})在零轴下方"

        # ── 第5层: 价格偏离MA20过滤 (突破信号放宽) ──
        if level.is_bullish:
            max_dev = group_params.get("price_ma20_max_deviation", 0)
            if max_dev > 0:
                ma20_ind = indicator_results.get("MA20")
                if ma20_ind is not None:
                    ma20_val = ma20_ind.values.get("ma20")
                    price = ma20_ind.values.get("price")
                    if ma20_val and price and ma20_val > 0:
                        deviation = (price - ma20_val) / ma20_val
                        effective_max_dev = max_dev * 3.0 if is_breakout_signal else max_dev
                        if deviation > effective_max_dev:
                            return SignalLevel.NEUTRAL, f"价格偏离MA20达{deviation*100:.1f}%超过上限{effective_max_dev*100:.1f}%"

        # ── 第6层: RSI超买过滤 ──
        if level.is_bullish:
            rsi_limit = group_params.get("rsi_overbought", 0)
            if rsi_limit > 0:
                rsi_ind = indicator_results.get("RSI")
                if rsi_ind is not None:
                    rsi_val = rsi_ind.values.get("rsi")
                    if rsi_val is not None and rsi_val > rsi_limit:
                        return SignalLevel.NEUTRAL, f"RSI({rsi_val:.0f})超买(>{rsi_limit})"

        # MACD顶背离降级: 强买入→买入, 买入→观望
        if level.is_bullish:
            macd = indicator_results.get("MACD")
            if macd is not None and macd.values.get("bearish_divergence", False):
                if level == SignalLevel.STRONG_BUY:
                    return SignalLevel.BUY, "MACD顶背离，强买入降级为买入"
                elif level == SignalLevel.BUY:
                    return SignalLevel.NEUTRAL, "MACD顶背离，买入降级为观望"

        # 大盘环境过滤: 指数在MA60下方时抑制买入信号
        if self.market_ma60_filter and level.is_bullish and self._market_ma60_direction == -1:
            return SignalLevel.NEUTRAL, "大盘指数在MA60下方，抑制买入信号"

        return level, ""

    # ── 信号去重 ──
    def is_duplicate(self, symbol: str, level: SignalLevel,
                     analysis_date: date = None) -> bool:
        """
        检查是否是重复信号

        Args:
            symbol: 股票代码
            level: 信号级别
            analysis_date: 实际分析日期 (date 对象), 回测时必须传入
        """
        if not level.is_actionable:
            return False

        self._ensure_history_loaded()

        direction = "bull" if level.is_bullish else "bear"
        key = f"{symbol}_{direction}"

        if key in self._history:
            last_date = self._history[key]
            if analysis_date is not None:
                ref_date = analysis_date
            else:
                ref_date = date.today()
            last_dt = datetime.strptime(last_date, "%Y-%m-%d").date()
            if (ref_date - last_dt).days < self.dedup_days:
                return True

        return False

    def record(self, symbol: str, level: SignalLevel,
               analysis_date: date = None):
        """
        记录本次信号

        Args:
            symbol: 股票代码
            level: 信号级别
            analysis_date: 实际分析日期 (date 对象), 回测时必须传入
        """
        if not level.is_actionable:
            return
        self._ensure_history_loaded()
        direction = "bull" if level.is_bullish else "bear"
        record_date = analysis_date if analysis_date is not None else date.today()
        self._history[f"{symbol}_{direction}"] = record_date.strftime("%Y-%m-%d")
        # 仅 LIVE 模式写盘, BACKTEST 模式纯内存
        if get_mode() == RuntimeMode.LIVE:
            self._save_history()

    def _load_history(self) -> dict:
        try:
            if os.path.exists(_DEDUP_FILE):
                with open(_DEDUP_FILE, "r") as f:
                    return json.load(f)
        except Exception:
            pass
        return {}

    def _save_history(self):
        try:
            os.makedirs(os.path.dirname(_DEDUP_FILE), exist_ok=True)
            with open(_DEDUP_FILE, "w") as f:
                json.dump(self._history, f, indent=2)
        except Exception:
            pass

    def clear_history(self):
        """清除信号去重历史

        - LIVE 模式: 清内存 + 清磁盘 (新分析前调用)
        - BACKTEST 模式: 仅清内存 (不触碰磁盘, 保护实时数据)
        """
        self._history = {}
        self._history_loaded = True  # 标记已加载, 避免回测模式从磁盘读取
        if get_mode() == RuntimeMode.LIVE:
            self._save_history()