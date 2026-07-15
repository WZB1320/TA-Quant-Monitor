"""个股回测持仓状态追踪器

在个股信号回测中模拟持仓状态机, 让止损/止盈/冷却期真正发挥作用。

状态流转:
  IDLE (空仓) --买入信号--> HOLDING (持仓) --卖出/止损/止盈--> COOLDOWN (冷却) --期满--> IDLE

核心价值:
  - 买入后每日检查止损/止盈 (复用 PositionManager 逻辑, 与组合回测一致)
  - 止损触发后进入冷却期, 期间拒绝买入信号
  - 信号生成感知持仓状态, 不再孤立

设计说明:
  - 执行价用信号日收盘价 (个股回测为展示工具, 简化T+1)
  - 止损规则与组合回测完全一致 (复用 PositionManager.check_stop_loss)
  - 同时调用 SignalFilter.record_exit/record_loss, 让 SignalEngine 内部冷却也生效
  - SignalEngine.analyze 内部已将不可执行的 bullish 信号降级为 NEUTRAL,
    因此 level.is_bullish 即表示"可执行的买入信号"
"""
from datetime import date, timedelta
from typing import Dict, Optional, Any

from src.backtest.position import PositionManager, Trade
from src.config.group_config import GroupConfig
from src.signal_engine.filter import SignalFilter
from src.signal_engine.signals import SignalResult


class PositionStateTracker:
    """个股回测持仓状态追踪器 — 复用 PositionManager 的止损逻辑

    三状态机:
      IDLE     — 空仓, 可接受买入信号
      HOLDING  — 持仓, 每日检查止损/止盈, 可接受卖出信号
      COOLDOWN — 冷却期, 拒绝买入信号
    """

    IDLE = 'IDLE'
    HOLDING = 'HOLDING'
    COOLDOWN = 'COOLDOWN'

    def __init__(self, symbol: str, group_config: GroupConfig,
                 signal_filter: SignalFilter):
        """
        Args:
            symbol: 股票代码
            group_config: 分组配置 (获取冷却天数、止损倍率)
            signal_filter: SignalEngine 的 filter 实例 (同步冷却/连亏状态)
        """
        self.symbol = symbol
        self.group_config = group_config
        self.signal_filter = signal_filter

        # 分组专属参数
        self.cooldown_days = group_config.get_cooldown_days(symbol)
        self.atr_stop_mult = group_config.get_atr_stop_mult(symbol)

        # 复用 PositionManager 的止损逻辑
        # 资金设大, 避免因资金不足导致开仓失败 (个股回测不关心资金, 只用止损逻辑)
        self.position_mgr = PositionManager(
            initial_capital=1_000_000,
            position_ratio=0.3,
            risk_per_trade=0.05,
            atr_stop_mult=self.atr_stop_mult,
        )

        # 状态机
        self.state = self.IDLE
        self.cooldown_until: Optional[date] = None
        self.entry_date: Optional[date] = None

    def process_day(self, signal_result: SignalResult,
                    close_price: float,
                    atr_value: Optional[float],
                    today: date) -> Dict[str, Any]:
        """处理单日信号, 返回操作和持仓状态信息

        Args:
            signal_result: SignalEngine.analyze 的返回值
            close_price: 当日收盘价 (作为执行价)
            atr_value: 当日ATR值 (None 则无法计算ATR止损)
            today: 分析日期

        Returns:
            持仓状态字典, 含 state/action/entry_price/stop_loss_price 等字段
        """
        result = self._empty_result()

        if self.state == self.IDLE:
            self._handle_idle(signal_result, close_price, atr_value, today, result)

        elif self.state == self.HOLDING:
            self._handle_holding(signal_result, close_price, today, result)

        elif self.state == self.COOLDOWN:
            self._handle_cooldown(today, result)

        result['state'] = self.state
        return result

    # ── 状态处理 ──

    def _handle_idle(self, signal_result: SignalResult, close_price: float,
                     atr_value: Optional[float], today: date,
                     result: Dict) -> None:
        """空仓: 检查买入信号"""
        # level.is_bullish 即表示可执行的买入信号 (analyze 内部已降级不可执行的)
        if signal_result.level.is_bullish:
            self._open_position(signal_result, close_price, atr_value, today)
            if self.state == self.HOLDING:
                result['action'] = 'BUY'
                result.update(self._holding_info(close_price, today))

    def _handle_holding(self, signal_result: SignalResult, close_price: float,
                        today: date, result: Dict) -> None:
        """持仓: 先检查止损/止盈, 再检查卖出信号"""
        # 1. 止损/止盈检查 (复用 PositionManager, 与组合回测完全一致)
        closed = self.position_mgr.check_stop_loss(
            self.symbol, close_price, today,
            signal_score=signal_result.score,
        )
        if closed is not None:
            self._on_position_closed(closed, today)
            result['action'] = 'TAKE_PROFIT' if closed.pnl > 0 else 'STOP_LOSS'
            result['exit_reason'] = closed.exit_signal
            result['exit_pnl_pct'] = round(closed.pnl_pct * 100, 2)
            result['cooldown_remaining'] = self._cooldown_remaining(today)
            return

        # 2. 卖出信号检查 (bearish 信号不受冷却/去重约束)
        if signal_result.level.is_bearish:
            closed = self.position_mgr.close_position(
                self.symbol, today, close_price,
                signal=f"{signal_result.level.label} score={signal_result.score:+.1f}",
            )
            if closed is not None:
                self._on_position_closed(closed, today)
                result['action'] = 'SELL'
                result['exit_reason'] = closed.exit_signal
                result['exit_pnl_pct'] = round(closed.pnl_pct * 100, 2)
                result['cooldown_remaining'] = self._cooldown_remaining(today)
                return

        # 3. 继续持有
        result['action'] = 'HOLD'
        result.update(self._holding_info(close_price, today))

    def _handle_cooldown(self, today: date, result: Dict) -> None:
        """冷却期: 检查是否期满"""
        if self.cooldown_until and today >= self.cooldown_until:
            self.state = self.IDLE
            self.cooldown_until = None
            result['action'] = 'NONE'
        else:
            result['action'] = 'COOLDOWN_BLOCKED'
            result['cooldown_remaining'] = self._cooldown_remaining(today)

    # ── 开仓/平仓 ──

    def _open_position(self, signal_result: SignalResult, close_price: float,
                       atr_value: Optional[float], today: date) -> None:
        """开仓 (复用 PositionManager.open_long)"""
        trade = self.position_mgr.open_long(
            symbol=self.symbol,
            entry_date=today,
            entry_price=close_price,
            signal=f"{signal_result.level.label} score={signal_result.score:+.1f}",
            atr_value=atr_value,
            bearish_market=False,
            signal_strength=1.0,
            atr_stop_mult=self.atr_stop_mult,
        )
        if trade is not None:
            self.state = self.HOLDING
            self.entry_date = today

    def _on_position_closed(self, trade: Trade, today: date) -> None:
        """平仓后处理: 进入冷却期, 同步 filter 状态"""
        self.state = self.COOLDOWN
        self.cooldown_until = today + timedelta(days=self.cooldown_days)
        self.entry_date = None

        # 同步 SignalFilter 状态 (让 SignalEngine 内部冷却/连亏也生效, 双重保险)
        self.signal_filter.record_exit(self.symbol, today)
        if trade.pnl > 0:
            self.signal_filter.record_win(self.symbol)
        else:
            self.signal_filter.record_loss(self.symbol, today)

    # ── 状态信息计算 ──

    def _holding_info(self, close_price: float, today: date) -> Dict[str, Any]:
        """计算持仓状态信息 (止损价/移动止盈价/浮盈亏等)"""
        trade = self.position_mgr.open_positions.get(self.symbol)
        if trade is None:
            return {}

        # 持仓天数 (自然日)
        holding_days = (today - trade.entry_date).days if trade.entry_date else 0

        # 浮盈亏%
        pnl_pct = ((close_price - trade.entry_price) / trade.entry_price * 100
                    if trade.entry_price else 0.0)

        # 止损价计算 (与 PositionManager.check_stop_loss 逻辑一致)
        atr_val = getattr(trade, '_atr_value', None)
        stop_mult = getattr(trade, '_atr_stop_mult', self.atr_stop_mult)

        # 10%硬止损价
        hard_stop = trade.entry_price * 0.90
        # ATR硬止损价
        atr_stop = trade.entry_price - atr_val * stop_mult if atr_val else None
        # 有效止损价 = 较高者 (先触发)
        stop_loss_price = max(hard_stop, atr_stop) if atr_stop else hard_stop

        # 移动止盈价 (仅盈利时有效, 盈利自适应倍率)
        trailing_stop_price = None
        if atr_val and atr_val > 0 and trade.highest_price > trade.entry_price:
            profit_pct = (close_price - trade.entry_price) / trade.entry_price
            if profit_pct > 0.20:
                trailing_mult = stop_mult * 0.6   # 盈利>20%, 锁定利润
            elif profit_pct > 0.10:
                trailing_mult = stop_mult * 0.8   # 盈利10~20%, 适度收紧
            else:
                trailing_mult = stop_mult          # 盈利<10%, 给足空间
            trailing_stop_price = trade.highest_price - atr_val * trailing_mult

        return {
            'entry_price': round(trade.entry_price, 2),
            'stop_loss_price': round(stop_loss_price, 2),
            'trailing_stop_price': round(trailing_stop_price, 2) if trailing_stop_price else None,
            'highest_price': round(trade.highest_price, 2),
            'holding_pnl_pct': round(pnl_pct, 2),
            'holding_days': holding_days,
        }

    def _cooldown_remaining(self, today: date) -> int:
        """剩余冷却天数"""
        if self.cooldown_until is None:
            return 0
        return max((self.cooldown_until - today).days, 0)

    def _empty_result(self) -> Dict[str, Any]:
        return {
            'state': self.state,
            'action': 'NONE',
            'entry_price': None,
            'stop_loss_price': None,
            'trailing_stop_price': None,
            'highest_price': None,
            'holding_pnl_pct': None,
            'holding_days': None,
            'cooldown_remaining': None,
            'exit_reason': None,
            'exit_pnl_pct': None,
        }

    # ── 交易摘要 ──

    def get_trade_summary(self) -> Dict[str, Any]:
        """获取完整交易摘要 (回测结束后调用)"""
        trades = self.position_mgr.closed_trades
        total = len(trades)
        wins = sum(1 for t in trades if t.pnl > 0)
        losses = sum(1 for t in trades if t.pnl <= 0)
        stop_loss_count = sum(1 for t in trades if '止损' in t.exit_signal)
        take_profit_count = sum(1 for t in trades if '止盈' in t.exit_signal)
        signal_exit_count = sum(1 for t in trades if 'score=' in t.exit_signal)

        max_pnl = max((t.pnl_pct for t in trades), default=0.0)
        min_pnl = min((t.pnl_pct for t in trades), default=0.0)
        avg_holding = (
            sum((t.exit_date - t.entry_date).days for t in trades if t.is_closed) / total
            if total > 0 else 0.0
        )

        return {
            'total_trades': total,
            'win_count': wins,
            'loss_count': losses,
            'stop_loss_count': stop_loss_count,
            'take_profit_count': take_profit_count,
            'signal_exit_count': signal_exit_count,
            'max_pnl_pct': round(max_pnl * 100, 2),
            'min_pnl_pct': round(min_pnl * 100, 2),
            'avg_holding_days': round(avg_holding, 1),
        }
