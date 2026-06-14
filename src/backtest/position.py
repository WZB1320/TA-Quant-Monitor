"""
持仓 & 交易记录管理

- Trade: 单笔完整交易 (开仓→平仓)
- PositionManager: 管理所有持仓状态, 处理开仓/平仓
"""
from dataclasses import dataclass, field
from datetime import date
from typing import Dict, List, Optional
from enum import Enum


class Side(Enum):
    LONG = "long"
    SHORT = "short"


@dataclass
class Trade:
    """单笔交易记录"""
    symbol: str
    side: Side
    entry_date: date
    entry_price: float
    exit_date: Optional[date] = None
    exit_price: Optional[float] = None
    shares: int = 0
    entry_signal: str = ""            # 开仓信号
    exit_signal: str = ""             # 平仓信号
    commission: float = 0.0           # 总手续费
    pnl: float = 0.0                  # 盈亏金额
    pnl_pct: float = 0.0              # 盈亏百分比
    holding_days: int = 0             # 持仓天数
    highest_price: float = 0.0        # 持仓期间最高价 (用于移动止盈)

    @property
    def is_closed(self) -> bool:
        return self.exit_date is not None

    def close(self, exit_date: date, exit_price: float, signal: str,
              commission: float = 0.0):
        self.exit_date = exit_date
        self.exit_price = exit_price
        self.exit_signal = signal
        self.commission += commission
        # 计算盈亏
        if self.side == Side.LONG:
            self.pnl = (exit_price - self.entry_price) * self.shares - self.commission
        else:
            self.pnl = (self.entry_price - exit_price) * self.shares - self.commission
        self.pnl_pct = self.pnl / (self.entry_price * self.shares) if self.shares > 0 else 0.0
        self.holding_days = (exit_date - self.entry_date).days


class PositionManager:
    """持仓管理器 — 体制自适应仓位 & ATR 动态止损

    专业级风险控制:
      - 整体仓位范围: 30% ~ 80%, 根据市场体制动态调整
        - 趋势市 (ADX>25): 目标仓位 80%, 单票上限 30%
        - 过渡期 (20≤ADX≤25): 目标仓位 55%, 单票上限 20%
        - 震荡市 (ADX<20): 目标仓位 30%, 单票上限 15%
      - 信号强度加成: 强买入信号仓位 ×1.3, 普通买入 ×1.0
      - ATR 止损距离 = ATR × stop_multiplier
      - 移动止盈 = 从最高点回撤超过 2×ATR 即平仓
    """

    # 体制等级 (数值越小越保守, 用于取 min)
    _REGIME_LEVEL = {"ranging": 0, "transition": 1, "trending": 2}

    # 体制 → 目标整体仓位 & 单票上限
    REGIME_CONFIG = {
        "trending":   {"target_ratio": 0.80, "max_per_stock": 0.30},
        "transition": {"target_ratio": 0.55, "max_per_stock": 0.25},
        "ranging":    {"target_ratio": 0.30, "max_per_stock": 0.15},
    }

    def _effective_regime(self, stock_regime: str = None) -> str:
        """取市场体制与个股体制的较低值 (更保守)"""
        if stock_regime is None:
            return self._regime
        market_level = self._REGIME_LEVEL.get(self._regime, 1)
        stock_level = self._REGIME_LEVEL.get(stock_regime, 1)
        effective_level = min(market_level, stock_level)
        for name, level in self._REGIME_LEVEL.items():
            if level == effective_level:
                return name
        return "transition"

    def __init__(self, initial_capital: float, position_ratio: float = 0.3,
                 commission_rate: float = 0.0003, slippage: float = 0.0001,
                 risk_per_trade: float = 0.01, atr_stop_mult: float = 2.0):
        """
        Args:
            initial_capital: 初始资金
            position_ratio: 单只股票最大仓位占比 (0~1), 作为绝对上限
            commission_rate: 手续费率 (默认万三)
            slippage: 滑点率 (默认万一)
            risk_per_trade: 每笔交易最大风险敞口 (默认 1%, 即每笔最多亏总资金的1%)
            atr_stop_mult: ATR 止损倍率 (默认 2.0x ATR)
        """
        self.initial_capital = initial_capital
        self.cash = initial_capital
        self.position_ratio = position_ratio
        self.commission_rate = commission_rate
        self.slippage = slippage
        self.risk_per_trade = risk_per_trade
        self.atr_stop_mult = atr_stop_mult
        self._open: Dict[str, Trade] = {}
        self.closed_trades: List[Trade] = []
        self._regime: str = "transition"  # 当前市场体制

    def set_regime(self, regime: str):
        """设置当前市场体制 (由回测引擎每日更新)"""
        self._regime = regime if regime in self.REGIME_CONFIG else "transition"

    @property
    def _current_config(self) -> dict:
        return self.REGIME_CONFIG.get(self._regime, self.REGIME_CONFIG["transition"])

    @property
    def current_position_ratio(self) -> float:
        """当前体制下的单票最大仓位"""
        return min(self._current_config["max_per_stock"], self.position_ratio)

    # ── 持仓查询 ──

    @property
    def open_positions(self) -> Dict[str, Trade]:
        return self._open

    def has_position(self, symbol: str) -> bool:
        return symbol in self._open

    # ── 开仓 ──

    def open_long(self, symbol: str, entry_date: date, entry_price: float,
                  signal: str, atr_value: float = None,
                  bearish_market: bool = False,
                  signal_strength: float = 1.0,
                  stock_regime: str = None,
                  atr_stop_mult: float = None) -> Optional[Trade]:
        """
        开多仓 (体制自适应 + ATR 动态仓位)

        核心逻辑:
          1. 取市场体制与个股体制的较低值 → 确定有效仓位上限
          2. 根据信号强度调整单票仓位 (强买入 ×1.3, 普通 ×1.0)
          3. ATR 控制单票止损距离, 波动大的自动降仓
          4. 熊市(大盘<MA60)额外减半

        Args:
            symbol: 股票代码
            entry_date: 入场日期
            entry_price: 入场价 (已含滑点)
            signal: 触发信号描述
            atr_value: ATR 值 (用于动态仓位), None 则用固定仓位
            bearish_market: 大盘是否在MA60下方, 是则仓位减半
            signal_strength: 信号强度加成 (1.0~1.3), 强买入用1.3
            stock_regime: 个股体制 (trending/transition/ranging), 与市场体制取较低值

        Returns:
            Trade 或 None
        """
        if self.has_position(symbol):
            return None

        total_capital = self.total_value({symbol: entry_price})

        # ── 体制自适应: 取市场与个股的较低值 ──
        effective_regime = self._effective_regime(stock_regime)
        config = self.REGIME_CONFIG[effective_regime]
        target_ratio = config["target_ratio"]
        max_per_stock = min(config["max_per_stock"], self.position_ratio)

        # 已有持仓市值 (用持仓的 entry_price 近似, 因为 open_long 时没有全部行情)
        existing_mv = sum(t.shares * t.entry_price for t in self._open.values())
        current_exposure = existing_mv / total_capital if total_capital > 0 else 0
        remaining_capacity = max(target_ratio - current_exposure, 0.05)  # 至少留 5% 空间

        if atr_value is not None and atr_value > 0:
            # ── ATR 动态仓位 ──
            risk_amount = total_capital * self.risk_per_trade
            stop_distance = atr_value * self.atr_stop_mult
            # ATR 计算的基础仓位: risk_amount / stop_distance = 可买股数
            shares_by_risk = risk_amount / stop_distance
            atr_position = shares_by_risk * entry_price
            # 限制在目标仓位范围内
            position_value = min(atr_position, total_capital * remaining_capacity)
            # 信号强度加成: 强买入 ×1.3
            position_value *= signal_strength
        else:
            # 无 ATR 时用目标仓位比例
            position_value = total_capital * remaining_capacity
            position_value *= signal_strength

        # 大盘在MA60下方 → 仓位减半
        if bearish_market:
            position_value *= 0.5

        # 不超过单票上限
        max_position = total_capital * max_per_stock
        position_value = min(position_value, max_position)

        # 不超过可用现金
        position_value = min(position_value, self.cash * 0.95)  # 留5%现金缓冲

        raw_shares = position_value / entry_price
        shares = int(raw_shares / 100) * 100

        if shares < 100:
            return None

        # 计算实际成本和手续费
        cost = shares * entry_price
        comm = max(cost * self.commission_rate, 5)

        if cost + comm > self.cash:
            shares -= 100
            if shares < 100:
                return None
            cost = shares * entry_price
            comm = max(cost * self.commission_rate, 5)

        self.cash -= (cost + comm)

        trade = Trade(
            symbol=symbol,
            side=Side.LONG,
            entry_date=entry_date,
            entry_price=entry_price,
            shares=shares,
            entry_signal=signal,
            commission=comm,
        )
        # 保存 ATR 止损参数
        stop_mult = atr_stop_mult if atr_stop_mult is not None else self.atr_stop_mult
        if atr_value is not None:
            trade._atr_stop_price = entry_price - atr_value * stop_mult
            trade._atr_value = atr_value
            trade._atr_stop_mult = stop_mult
        self._open[symbol] = trade
        return trade

    # ── 平仓 ──

    def close_position(self, symbol: str, exit_date: date, exit_price: float,
                       signal: str) -> Optional[Trade]:
        """
        平仓

        Returns:
            已平仓的 Trade 或 None
        """
        trade = self._open.pop(symbol, None)
        if trade is None:
            return None

        # 计算卖出手续费 (佣金 + 印花税)
        revenue = trade.shares * exit_price
        comm = max(revenue * self.commission_rate, 5)
        stamp = revenue * 0.001  # 印花税千1, 仅卖出收取
        total_comm = comm + stamp

        trade.close(exit_date, exit_price, signal, commission=total_comm)

        # 回笼资金
        self.cash += (revenue - total_comm)

        self.closed_trades.append(trade)
        return trade

    # ── 市值计算 ──

    def market_value(self, current_prices: Dict[str, float]) -> float:
        """计算持仓市值"""
        mv = 0.0
        for symbol, trade in self._open.items():
            if symbol in current_prices:
                mv += trade.shares * current_prices[symbol]
        return mv

    def total_value(self, current_prices: Dict[str, float]) -> float:
        """总资产 = 现金 + 持仓市值"""
        return self.cash + self.market_value(current_prices)

    def total_return(self, current_prices: Dict[str, float]) -> float:
        """当前总收益率"""
        return (self.total_value(current_prices) / self.initial_capital) - 1.0

    # ── 统计 ──

    @property
    def trade_count(self) -> int:
        return len(self.closed_trades)

    @property
    def win_count(self) -> int:
        return sum(1 for t in self.closed_trades if t.pnl > 0)

    @property
    def win_rate(self) -> float:
        if not self.closed_trades:
            return 0.0
        return self.win_count / len(self.closed_trades)

    @property
    def total_pnl(self) -> float:
        return sum(t.pnl for t in self.closed_trades)

    @property
    def avg_pnl_pct(self) -> float:
        if not self.closed_trades:
            return 0.0
        return sum(t.pnl_pct for t in self.closed_trades) / len(self.closed_trades)

    @property
    def avg_holding_days(self) -> float:
        closed = [t for t in self.closed_trades if t.is_closed]
        if not closed:
            return 0.0
        return sum(t.holding_days for t in closed) / len(closed)

    # ── 止损 / 止盈 (ATR 动态) ──

    def check_stop_loss(self, symbol: str, current_price: float,
                        current_date: date,
                        signal_score: float = None) -> Optional[Trade]:
        """
        检查是否触发止损/移动止盈 (基于 ATR 动态调整)

        规则:
          1. 硬止损: 价格触及 entry_price - 2.5×ATR → 平仓
          2. 移动止盈: 从最高点回撤超过 stop_dist → 平仓 (仅在盈利时)
             - 盈利 < 10%: stop_dist = 2.5×ATR (给足空间, 让利润奔跑)
             - 盈利 10~20%: stop_dist = 2.0×ATR (适度收紧)
             - 盈利 > 20%: stop_dist = 1.5×ATR (锁定大部分利润)

        Args:
            symbol: 股票代码
            current_price: 当前收盘价
            current_date: 当前日期
            signal_score: 未使用 (保留接口兼容)

        Returns:
            触发时返回已平仓的 Trade, 否则 None
        """
        trade = self._open.get(symbol)
        if trade is None:
            return None

        # 更新最高价
        if current_price > trade.highest_price:
            trade.highest_price = current_price

        # ── 安全网: 10%硬止损 (所有模式永远生效) ──
        if current_price <= trade.entry_price * 0.90:
            pnl_pct = (current_price - trade.entry_price) / trade.entry_price
            return self.close_position(symbol, current_date, current_price,
                                       f"安全网硬止损 ({pnl_pct*100:.1f}%, 保本底线)")

        # 获取 ATR 止损距离
        atr_val = getattr(trade, '_atr_value', None)
        # 使用交易级别的 atr_stop_mult (分组专属), 否则用全局默认值
        trade_stop_mult = getattr(trade, '_atr_stop_mult', self.atr_stop_mult)
        if atr_val is not None and atr_val > 0:
            # ── 盈利自适应移动止盈倍率 ──
            profit_pct = (current_price - trade.entry_price) / trade.entry_price
            if profit_pct > 0.20:
                trailing_mult = trade_stop_mult * 0.6   # 1.5 (盈利>20%, 锁定利润)
            elif profit_pct > 0.10:
                trailing_mult = trade_stop_mult * 0.8   # 2.0 (盈利10~20%, 适度收紧)
            else:
                trailing_mult = trade_stop_mult          # 2.5 (盈利<10%, 给足空间)

            stop_dist = atr_val * trade_stop_mult  # 硬止损始终用原始倍率
            trailing_dist = atr_val * trailing_mult    # 移动止盈用自适应倍率

            # ATR 硬止损
            if current_price <= trade.entry_price - stop_dist:
                pnl_pct = (current_price - trade.entry_price) / trade.entry_price
                return self.close_position(symbol, current_date, current_price,
                                           f"ATR硬止损 ({pnl_pct*100:.1f}%, ATR={atr_val:.2f})")

            # ATR 移动止盈 (盈利自适应倍率)
            if trade.highest_price > trade.entry_price:
                if current_price <= trade.highest_price - trailing_dist:
                    drawdown = (current_price - trade.highest_price) / trade.highest_price
                    return self.close_position(symbol, current_date, current_price,
                                               f"ATR移动止盈 (最高{trade.highest_price:.2f}, "
                                               f"回撤{drawdown*100:.1f}%, 盈利{profit_pct*100:.1f}%)")
        else:
            # 无 ATR 时回退到固定百分比
            pnl_pct = (current_price - trade.entry_price) / trade.entry_price
            if pnl_pct <= -0.10:
                return self.close_position(symbol, current_date, current_price,
                                           f"硬止损 ({pnl_pct*100:.1f}%)")
            if trade.highest_price > trade.entry_price:
                drawdown = (current_price - trade.highest_price) / trade.highest_price
                if drawdown <= -0.05:
                    return self.close_position(symbol, current_date, current_price,
                                               f"移动止盈 (回撤{drawdown*100:.1f}%)")

        return None