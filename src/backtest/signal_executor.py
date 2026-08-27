"""T+1 信号执行器

提取自 BacktestEngine._apply_pending / _flush_pending,
负责在信号生成次日以开盘价执行买卖, 包含涨跌停检查、MA60 重验证、
冷却期检查、分组专属参数应用。
"""
from datetime import date
from typing import Dict, Optional

import pandas as pd

from src.signal_engine import SignalEngine, SignalResult
from src.config.group_config import GroupConfig
from .broker import Broker
from .position import PositionManager
from .calendar import TradingCalendar
from .market_filter import MarketFilter


class SignalExecutor:
    """T+1 信号执行器"""

    def __init__(self,
                 broker: Broker,
                 position_mgr: PositionManager,
                 signal_engine: SignalEngine,
                 group_config: GroupConfig,
                 market_filter: MarketFilter,
                 calendar: TradingCalendar):
        self.broker = broker
        self.position_mgr = position_mgr
        self.signal_engine = signal_engine
        self.group_config = group_config
        self.market_filter = market_filter
        self.calendar = calendar

    def execute(self, signals: Dict[str, SignalResult],
                data_map: Dict[str, pd.DataFrame], signal_date) -> None:
        """执行 pending 信号 (T+1 成交)

        Args:
            signals: {symbol: SignalResult} 信号日生成的信号
            data_map: 全量数据
            signal_date: 信号生成日期 (T 日)
        """
        if not signals:
            return

        # 按得分绝对值排序, 优先执行强信号
        sorted_signals = sorted(
            signals.items(),
            key=lambda x: abs(x[1].score), reverse=True
        )

        for symbol, result in sorted_signals:
            df = data_map.get(symbol)
            if df is None:
                continue

            idx = self.calendar.locate(symbol, signal_date)
            if idx is None:
                continue

            next_info = self.broker.get_next_open(df, idx)
            if next_info is None:
                continue

            try:
                next_date = next_info["date"]
                if isinstance(next_date, pd.Timestamp):
                    next_date = next_date.date()
                elif isinstance(next_date, str):
                    next_date = pd.Timestamp(next_date).date()
            except Exception:
                continue

            open_price = next_info["open"]
            prev_close = next_info["prev_close"]

            # 涨跌停检查 (按板块自动推断涨跌停幅度, 如创业板/科创板20%)
            if not self.broker.can_trade(open_price, prev_close, symbol=symbol):
                continue

            if result.level.is_bullish:
                self._execute_buy(symbol, result, df, next_date, next_info)
            elif result.level.is_bearish:
                self._execute_sell(symbol, result, next_date, open_price)

    def _execute_buy(self, symbol: str, result: SignalResult,
                     df: pd.DataFrame, next_date, next_info: dict) -> None:
        """执行买入信号"""
        # 大盘MA60过滤: 指数在MA60下方时, 仓位减半
        market_bearish = self.market_filter.is_bearish(next_date)

        # 冷却期检查: 最近卖出/止损后 N 天内不反向买入
        if self.signal_engine.filter.is_in_cooldown(symbol, next_date, True):
            return

        # MA60 执行日重验证: 信号日MA60多头, 但执行日可能已转空
        # 前视偏差修复: 成交在执行日(next_date)开盘价, 此时只能用到
        # 执行日之前(含前一日)的数据, 不能包含执行日当天的收盘价.
        # 故切片取 iloc[:exec_idx] (执行日之前的所有K线), 而非 iloc[:exec_idx+1].
        exec_idx = self.calendar.locate(symbol, next_date)
        exec_atr = None
        if exec_idx is not None and exec_idx >= 60:
            exec_df = df.iloc[:exec_idx]
            try:
                exec_ind = self.signal_engine.pipeline.run(exec_df)
                exec_ma60 = exec_ind.get("MA60")
                if exec_ma60 and exec_ma60.direction != 1:
                    return  # 执行日MA60已转空, 放弃买入
                exec_atr_ind = exec_ind.get("ATR")
                if exec_atr_ind:
                    exec_atr = exec_atr_ind.values.get("atr")
            except Exception:
                pass

        if self.position_mgr.has_position(symbol):
            return  # 已有持仓, 不重复买

        buy_price = self.broker.buy_price(next_info["open"])
        # 信号强度加成: 强买入 ×1.3, 普通买入 ×1.0
        signal_strength = 1.3 if "强买入" in result.level.label else 1.0
        # 分组专属仓位加成
        group_boost = self.group_config.get_max_per_stock_boost(symbol)
        signal_strength *= group_boost
        # 分组专属止损倍率
        group_stop_mult = self.group_config.get_atr_stop_mult(symbol)

        self.position_mgr.open_long(
            symbol=symbol,
            entry_date=next_date,
            entry_price=buy_price,
            signal=f"{result.level.label} score={result.score:+.1f}",
            atr_value=exec_atr,
            bearish_market=market_bearish,
            signal_strength=signal_strength,
            atr_stop_mult=group_stop_mult,
        )

    def _execute_sell(self, symbol: str, result: SignalResult,
                      next_date, open_price: float) -> None:
        """执行卖出信号"""
        if not self.position_mgr.has_position(symbol):
            return

        sell_price = self.broker.sell_price(open_price)
        closed = self.position_mgr.close_position(
            symbol=symbol,
            exit_date=next_date,
            exit_price=sell_price,
            signal=f"{result.level.label} score={result.score:+.1f}",
        )
        # 卖出后记录冷却期 + 连亏保护
        if closed is not None:
            self.signal_engine.filter.record_exit(symbol, next_date)
            if closed.pnl > 0:
                self.signal_engine.filter.record_win(symbol)
            else:
                self.signal_engine.filter.record_loss(symbol, next_date)

    def flush(self, pending_signals: Dict, data_map: Dict[str, pd.DataFrame]) -> None:
        """回测结束时清理待处理信号

        边界修复: 主循环已对每一天执行其前一日(T)产生的信号(T+1开盘价成交)。
        仅在回测区间内最后一天生成的信号, 其 T+1 执行日已超出 bt_end,
        没有合法的成交窗口, 不应在区间外开仓。此前直接以"执行日之后"的
        开盘价开仓并把仓位计入期末净值, 会扭曲回测结果。这里统一丢弃,
        不执行任何越界信号。
        """
        pending_signals.clear()
