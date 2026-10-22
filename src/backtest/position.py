"""
持仓 & 交易记录管理

- Trade: 单笔完整交易 (开仓→平仓)
- PositionManager: 管理所有持仓状态, 处理开仓/平仓
"""
from dataclasses import dataclass, field
from datetime import date
from typing import Dict, List, Optional
from enum import Enum

from src.data_fetcher.event_book import dividend_tax_rate


class Side(Enum):
    LONG = "long"
    SHORT = "short"


@dataclass
class Trade:
    """单笔交易记录

    复权改造新增字段 (M2, 详见 outputs/adjust_reform_project_plan_20261022.md):
        ref_price       除权调整后的成本基准. 与 entry_price 分离:
                        entry_price 是真实成交价 (算本金), ref_price 是止损/止盈
                        的百分比基准. 无除权事件时两者相等, 故旧行为不变.
        entry_amount    开仓本金 = entry_price × 开仓股数. 送股后 shares 会变,
                        但本金不变, 故 pnl_pct 的分母必须单存.
        dividend_gross  持仓期内累计**实际到账**的现金分红 (s>0 时是税后值)
        dividend_taxable 需在卖出时补缴个税的分红额 (s>0 时已在源头代扣, 记 0)
        dividend_tax    已缴红利税 = 源头代扣 + 平仓时补缴

    记账口径 (易错, 勿改):
        ref_price 与 P&L **必须解耦**, 二者是不同语义的两个量:
        - ref_price 按市场乘数**比例**下移 (ref /= mult), 使"百分比回撤"在除权日
          前后连续 —— 只服务止损/止盈, **不进 P&L**.
        - P&L 走现金流: 卖出所得 − 开仓本金 + 实收股息 − 红利税 − 费用.
        二者不能混用: 比例下移量 = ref × D/前收 ≠ 实收股息 D×股数, 混用会
        漏算股息; 而"先用比例下移的基准、再加回股息"又会重复计算
        (两种错误都实测过, 单笔偏差达 58 元).
    """
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
    ref_price: float = 0.0            # 除权调整后的成本基准 (默认=entry_price)
    entry_amount: float = 0.0         # 开仓本金 (送股后不变)
    dividend_gross: float = 0.0        # 累计实际到账的现金分红
    dividend_taxable: float = 0.0     # 待卖出时补缴个税的分红基数
    dividend_tax: float = 0.0         # 已缴红利税 (源头代扣 + 平仓补缴)

    @property
    def is_closed(self) -> bool:
        return self.exit_date is not None

    @property
    def cost_basis(self) -> float:
        """百分比判定的成本基准 (无除权事件时 == entry_price)"""
        return self.ref_price if self.ref_price > 0 else self.entry_price

    @property
    def dividend_net(self) -> float:
        """税后分红"""
        return self.dividend_gross - self.dividend_tax

    def close(self, exit_date: date, exit_price: float, signal: str,
              commission: float = 0.0):
        self.exit_date = exit_date
        self.exit_price = exit_price
        self.exit_signal = signal
        self.commission += commission
        self.holding_days = (exit_date - self.entry_date).days
        # 红利税: 持股 ≤1 年时派息环节不扣缴, 卖出时按持股期限补收 ——
        # 故税率在平仓这一刻才确定 (财税〔2015〕101号 三档 20%/10%/0%).
        # 用 += : s>0 时送股按面值代扣的税在除权日已计入 dividend_tax.
        self.dividend_tax += self.dividend_taxable * dividend_tax_rate(self.holding_days)
        # 计算盈亏 —— **价格口径**: 卖出所得 − 开仓本金 + 股息 − 税 − 费
        #
        # 这里刻意**不用 ref_price** 做成本基准. ref_price 是按市场乘数**比例**下移的
        # (ref /= mult), 比例下移量 = ref × D/前收, 与实收股息 D×股数 并不相等
        # (持仓越久、涨幅越大, 二者差得越多). 例: 10 元买入涨到 20 元后每股派 1 元,
        # 比例下移只有 0.5 元, 而实收 1 元 —— 用基准算会把股息算漏.
        # 反过来若"基准下移 + 再加股息"又会重复计算 (实测一笔差 58 元).
        # 故 P&L 直接走现金流, ref_price 只服务止损/止盈的百分比语义, 两者互不干扰.
        proceeds = exit_price * self.shares
        if self.side == Side.LONG:
            self.pnl = (proceeds - self.entry_amount - self.commission
                        + self.dividend_gross - self.dividend_tax)
        else:
            # 融券方需支付股息 (本项目无做空路径, 保持对称形式)
            self.pnl = (self.entry_amount - proceeds - self.commission
                        - self.dividend_gross)
        denom = self.entry_amount if self.entry_amount > 0 else (
            self.entry_price * self.shares)
        self.pnl_pct = self.pnl / denom if denom > 0 else 0.0


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

    # 执行层参数默认值 (P3基线, 可被 stop_loss_params 覆盖, 用于敏感性扫描)
    # P3优化 (2026-08-07): 诊断2发现退出逻辑过紧(止损踏空90.9%, 止盈过早78.7%),
    # 放宽trailing倍率让利润奔跑 + 稍宽硬止损, 训练窗Alpha从-4.9%转正至+1.59%.
    DEFAULT_STOP_PARAMS = {
        'hard_stop_pct': 0.12,        # 硬止损比例 (P3: 0.10→0.12, 稍宽避免假突破)
        'trail_tier1_threshold': 0.10,  # 盈利10%进入第二档
        'trail_tier2_threshold': 0.20,  # 盈利20%进入第三档
        'trail_mult_low': 2.0,         # 盈利<10%: trailing_mult = stop_mult × 2.0 (P3: 1.0→2.0, 让利润奔跑)
        'trail_mult_mid': 1.5,         # 盈利10~20%: trailing_mult = stop_mult × 1.5 (P3: 0.8→1.5)
        'trail_mult_high': 1.0,        # 盈利>20%: trailing_mult = stop_mult × 1.0 (P3: 0.6→1.0)
        'no_atr_hard_stop_pct': 0.12,  # 无ATR时硬止损比例 (同步P3)
        'no_atr_trail_drawdown': 0.05, # 无ATR时移动止盈回撤比例
    }

    # 分体制退出默认配置 (P3基线)
    # 震荡市(ranging)禁用移动止盈: 诊断2发现78.7%止盈过早, 震荡市趋势跟踪退出
    # 无论多宽都会被振出, 禁用后只靠硬止损+信号退出, 骑住波动.
    # 趋势市(trending)/过渡市(transition)保持trailing, 放宽后让利润奔跑.
    DEFAULT_REGIME_EXIT_CONFIG = {
        "ranging": {"disable_trailing": True},
    }

    def __init__(self, initial_capital: float, position_ratio: float = 0.3,
                 commission_rate: float = 0.0003, slippage: float = 0.0001,
                 risk_per_trade: float = 0.01, atr_stop_mult: float = 2.0,
                 stop_loss_params: dict = None,
                 regime_exit_config: dict = None,
                 event_map: Dict[str, Dict[str, object]] = None):
        """
        Args:
            initial_capital: 初始资金
            position_ratio: 单只股票最大仓位占比 (0~1), 作为绝对上限
            commission_rate: 手续费率 (默认万三)
            slippage: 滑点率 (默认万一)
            risk_per_trade: 每笔交易最大风险敞口 (默认 1%, 即每笔最多亏总资金的1%)
            atr_stop_mult: ATR 止损倍率 (默认 2.0x ATR)
            stop_loss_params: 止损止盈参数覆盖 (用于敏感性扫描), None=用默认值.
                可含键: hard_stop_pct/trail_tier1_threshold/trail_tier2_threshold/
                trail_mult_low/trail_mult_mid/trail_mult_high/
                no_atr_hard_stop_pct/no_atr_trail_drawdown
            regime_exit_config: 分体制退出参数覆盖, 格式 {regime: {key: val}}.
                可含键: disable_trailing(True=禁用移动止盈,仅硬止损), 以及stop_loss_params所有键.
                例: {"ranging": {"disable_trailing": True, "hard_stop_pct": 0.12}}
                None=不分体制(向后兼容). 诊断2发现震荡市trailing过紧致踏空78.7%.
            event_map: 除权事件表 {symbol: {ex_date: CorporateAction}}.
                None=不处理公司行动 (向后兼容, 等价改造前行为).
                由 BacktestEngine 从 EventBook 一次性预加载后传入 —— 成交层按日
                循环调用, 不能每次查库.
        """
        self.initial_capital = initial_capital
        self.cash = initial_capital
        self.position_ratio = position_ratio
        self.commission_rate = commission_rate
        self.slippage = slippage
        self.risk_per_trade = risk_per_trade
        self.atr_stop_mult = atr_stop_mult
        # 合并执行层参数 (默认值 + 用户覆盖)
        self._stop_params = {**self.DEFAULT_STOP_PARAMS,
                             **(stop_loss_params or {})}
        # 分体制退出参数覆盖 (regime-adaptive exit)
        # None=用默认配置(P3: 震荡市禁用trailing), {}=禁用分体制退出
        if regime_exit_config is None:
            self._regime_exit_config = dict(self.DEFAULT_REGIME_EXIT_CONFIG)
        else:
            self._regime_exit_config = regime_exit_config
        self._open: Dict[str, Trade] = {}
        self.closed_trades: List[Trade] = []
        self._regime: str = "transition"  # 当前市场体制
        self._event_map = event_map or {}   # {symbol: {ex_date: CorporateAction}}

    def set_regime(self, regime: str):
        """设置当前市场体制 (由回测引擎每日更新)"""
        self._regime = regime if regime in self.REGIME_CONFIG else "transition"

    # ── 公司行动 (除权除息) ──

    def apply_corporate_actions(self, today) -> Dict[str, float]:
        """对当日为除权除息日的持仓施加调整 (必须在止损检查之前调用)

        A股规则: 股权登记日收盘仍持有即有权领息, 除权日当天卖出也不影响.
        引擎日循环把本调用放在最前 —— 此时持仓是"昨日收盘持有"的集合,
        而当日 T+1 买入的仓位尚未建立, 天然被排除 (买入日=除权日无领息权).

        处理:
          - 现金分红: cash += 每股派息 × 除权日持有股数; 税在平仓时才扣
          - 送转:     shares ×= (1 + s)
          - ref_price / highest_price 同步按市场乘数下移 —— 否则除权日的
            价格跳空会被当成暴跌, 直接误触发硬止损 (plan 风险 R1/R2)

        Returns:
            {symbol: 当日入账的含税分红现金}
        """
        ex_date = today.strftime("%Y-%m-%d") if hasattr(today, "strftime") else str(today)
        credited = {}
        for symbol, trade in list(self._open.items()):
            ev = self._event_map.get(symbol, {}).get(ex_date)
            if ev is None:
                continue
            s = ev.s
            dps = ev.dps_pre if s == 0 else ev.dps_after
            cash_div = dps * trade.shares
            trade.dividend_gross += cash_div
            self.cash += cash_div
            if s == 0:
                # 纯派息: 派息时不扣缴, 卖出时按持股期限补收 → 记入待税基数
                trade.dividend_taxable += cash_div
            else:
                # 含送转: 送股按面值 1 元/股征的个税已从现金派息里代扣
                # (故 dps_after < dps_pre), 不再叠加自算税率.
                # 但成本基准下移用的是**含税**派息 (交易所除权参考价口径),
                # 与到账现金的差额即为源头已代扣的税, 须显式记为成本.
                trade.dividend_tax += max(ev.dps_pre - ev.dps_after, 0.0) * trade.shares

            # 成本基准与最高价下移 (价格侧一律用市场乘数, 不用公告 D/s)
            mult = getattr(ev, "multiplier", None)
            if mult and mult > 0:
                trade.ref_price /= mult
                if trade.highest_price:
                    trade.highest_price /= mult
            elif s:
                trade.ref_price = (trade.ref_price - dps) / (1 + s)
                if trade.highest_price:
                    trade.highest_price = (trade.highest_price - dps) / (1 + s)
            else:
                trade.ref_price = max(trade.ref_price - dps, 1e-9)
                if trade.highest_price:
                    trade.highest_price = max(trade.highest_price - dps, 1e-9)

            trade.shares = int(round(trade.shares * (1 + s)))
            credited[symbol] = cash_div
        return credited

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
                  atr_stop_mult: float = None,
                  atr_ratio: float = None) -> Optional[Trade]:
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
            atr_ratio: ATR / close 的**无量纲比率** (复权改造后的推荐入口).
                信号层在 qfq 尺度算指标, 成交层在 raw 尺度成交, 直接传 ATR 绝对
                值会差一个复权常数 (实测 5~9%). 传比率可让成交层用 entry_price
                自行还原成元, 从结构上杜绝"忘记换算"的 bug.
                与 atr_value 二选一; 同时给定时 atr_ratio 优先.

        Returns:
            Trade 或 None
        """
        if self.has_position(symbol):
            return None

        # ── 体制自适应: 取市场与个股的较低值 ──
        effective_regime = self._effective_regime(stock_regime)
        config = self.REGIME_CONFIG[effective_regime]
        target_ratio = config["target_ratio"]
        max_per_stock = min(config["max_per_stock"], self.position_ratio)

        # 已有持仓市值 (用持仓的成本基准近似, 因为 open_long 时没有全部行情)
        # 送股后 shares 变而 entry_price 不变, 故必须用已除权调整的 cost_basis.
        existing_mv = sum(t.shares * t.cost_basis for t in self._open.values())
        # 总资产 = 现金 + 已有持仓市值
        # 注意: 不能用 total_value({symbol: entry_price}) — 新股票尚未持仓,
        # market_value 会忽略它, 而已有持仓不在 prices 里也不计市值,
        # 结果只剩现金 → current_exposure 持仓后极易 >100%, 新仓被系统性压缩
        total_capital = self.cash + existing_mv
        current_exposure = existing_mv / total_capital if total_capital > 0 else 0
        remaining_capacity = max(target_ratio - current_exposure, 0.05)  # 至少留 5% 空间

        # ATR 归一化: 比率 → 与 entry_price 同尺度的元
        if atr_ratio is not None and atr_ratio > 0:
            atr_value = atr_ratio * entry_price

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
            ref_price=entry_price,                 # 无除权事件时 == entry_price
            entry_amount=shares * entry_price,      # 送股后 shares 变, 本金不变
        )
        # 保存 ATR 止损参数
        stop_mult = atr_stop_mult if atr_stop_mult is not None else self.atr_stop_mult
        if atr_value is not None:
            trade._atr_stop_price = entry_price - atr_value * stop_mult
            trade._atr_value = atr_value
            trade._atr_stop_mult = stop_mult
        if atr_ratio is not None:
            # 记录比率: 除权后 ref_price 变化, 止损距离须由 ref_price 现算,
            # 不能沿用开仓时的绝对 ATR (那是旧尺度的元).
            trade._atr_ratio = atr_ratio
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

        # 回笼资金 (红利税在此补缴: A股持股≤1年时派息环节不扣, 卖出时才收)
        self.cash += (revenue - total_comm - trade.dividend_tax)

        self.closed_trades.append(trade)
        return trade

    def reduce_position(self, symbol: str, exit_date: date, exit_price: float,
                        reduce_ratio: float, signal: str = "回撤保护降仓") -> Optional[Trade]:
        """
        部分平仓 — 按比例卖出部分股数, 剩余继续持有 (组合级回撤保护真实降仓用).

        机制:
          - 计算 reduce_ratio 比例的卖出股数 (向下取整到100股)
          - 创建一个"部分平仓" Trade (复制 entry 信息, shares=卖出股数), close 它, 加入 closed_trades
          - 减少 open Trade 的 shares (entry_price/highest_price/ATR止损参数保持不变)
          - 回笼现金 (扣佣金+印花税)

        单向降仓: 只卖不买, 恢复时由调用方切换状态, 不在此处买回.

        Returns:
            已部分平仓的 Trade 或 None (无持仓/股数不足时)
        """
        trade = self._open.get(symbol)
        if trade is None:
            return None
        sell_shares = int(trade.shares * reduce_ratio / 100) * 100
        if sell_shares < 100:
            return None
        # 部分平仓 Trade (复制 entry 信息, shares=卖出股数)
        partial = Trade(
            symbol=symbol, side=trade.side,
            entry_date=trade.entry_date, entry_price=trade.entry_price,
            shares=sell_shares, entry_signal=trade.entry_signal,
            ref_price=trade.ref_price or trade.entry_price,
            entry_amount=trade.entry_price * sell_shares,
            highest_price=trade.highest_price,
        )
        # 累计分红按卖出股数比例拆分: 卖出部分对应的股息由本次平仓计税,
        # 剩余部分留给仍在持有的仓位 (未来平仓时再按那时的持股期限计税).
        # 已知简化: 同一笔持仓若分多次卖出, 各批持股天数不同, 严格应按 FIFO
        # 分档计税. 20%/10% 在小额股息上是二阶小量, 本期按整笔 holding_days 处理.
        if trade.shares > 0:
            share_of_div = trade.dividend_gross * (sell_shares / trade.shares)
        else:
            share_of_div = 0.0
        partial.dividend_gross = share_of_div
        trade.dividend_gross -= share_of_div
        # 待税基数与已代扣税额按同比例拆分 (partial.close 时才计补充征税)
        partial.dividend_taxable = trade.dividend_taxable * (sell_shares / trade.shares)
        trade.dividend_taxable -= partial.dividend_taxable
        partial.dividend_tax = trade.dividend_tax * (sell_shares / trade.shares)
        trade.dividend_tax -= partial.dividend_tax

        revenue = sell_shares * exit_price
        comm = max(revenue * self.commission_rate, 5)
        stamp = revenue * 0.001  # 印花税千1, 仅卖出收取
        total_comm = comm + stamp
        partial.close(exit_date, exit_price, signal, commission=total_comm)
        self.cash += (revenue - total_comm - partial.dividend_tax)
        self.closed_trades.append(partial)
        # 减少open Trade股数 (entry_price/highest_price/_atr_*不变, 止损逻辑对剩余持仓延续)
        trade.shares -= sell_shares
        return partial

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
        检查是否触发止损/移动止盈 (基于 ATR 动态调整, 支持分体制退出)

        规则:
          1. 硬止损: 价格触及 entry_price - 2.5×ATR → 平仓
          2. 移动止盈: 从最高点回撤超过 stop_dist → 平仓 (仅在盈利时)
             - 盈利 < 10%: stop_dist = 2.5×ATR (给足空间, 让利润奔跑)
             - 盈利 10~20%: stop_dist = 2.0×ATR (适度收紧)
             - 盈利 > 20%: stop_dist = 1.5×ATR (锁定大部分利润)
          3. 分体制退出 (regime_exit_config): 震荡市可禁用trailing只靠硬止损,
             避免趋势跟踪退出在震荡市频繁踏空 (诊断2: 78.7%过早止盈)

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

        # 成本基准: 除权后 ref_price 已下移, 所有百分比判定都基于它.
        # 无除权事件时 ref_price == entry_price, 与改造前逐位一致.
        base = trade.cost_basis

        # 更新最高价
        if current_price > trade.highest_price:
            trade.highest_price = current_price

        # ── 分体制退出参数: 合并基础参数 + 当前regime覆盖 ──
        regime_overrides = self._regime_exit_config.get(self._regime, {})
        effective_params = {**self._stop_params, **regime_overrides}
        disable_trailing = regime_overrides.get("disable_trailing", False)

        # ── 安全网: 硬止损 (所有模式永远生效) ──
        hard_stop_pct = effective_params['hard_stop_pct']
        if current_price <= base * (1 - hard_stop_pct):
            pnl_pct = (current_price - base) / base
            return self.close_position(symbol, current_date, current_price,
                                       f"安全网硬止损 ({pnl_pct*100:.1f}%, 保本底线)")

        # ── 均值回归: 目标价止盈 (盈利达到目标即平仓, 不等trailing) ──
        # 均值回归策略利润不奔跑, 到达目标即止盈.
        # target_profit_pct > 0 时启用, 在 trailing 检查之前触发.
        target_profit_pct = effective_params.get('target_profit_pct', 0)
        if target_profit_pct > 0:
            profit_pct = (current_price - base) / base
            if profit_pct >= target_profit_pct:
                return self.close_position(symbol, current_date, current_price,
                                           f"均值回归止盈 ({profit_pct*100:.1f}%, 目标{target_profit_pct*100:.0f}%)")

        # 获取 ATR 止损距离
        # 复权改造: 优先用比率 (ATR/close) × 当前成本基准现算成元. 这样除权后
        # 止损距离自动随 ref_price 下移, 不需要在除权处理里逐个修正止损价.
        atr_ratio = getattr(trade, '_atr_ratio', None)
        # 使用交易级别的 atr_stop_mult (分组专属), 否则用全局默认值
        trade_stop_mult = getattr(trade, '_atr_stop_mult', self.atr_stop_mult)
        if atr_ratio is not None and atr_ratio > 0:
            atr_val = atr_ratio * base
        else:
            atr_val = getattr(trade, '_atr_value', None)
        if atr_val is not None and atr_val > 0:
            # ── 盈利自适应移动止盈倍率 (参数可配置, 用于敏感性扫描) ──
            profit_pct = (current_price - base) / base
            t1 = effective_params['trail_tier1_threshold']
            t2 = effective_params['trail_tier2_threshold']
            if profit_pct > t2:
                trailing_mult = trade_stop_mult * effective_params['trail_mult_high']
            elif profit_pct > t1:
                trailing_mult = trade_stop_mult * effective_params['trail_mult_mid']
            else:
                trailing_mult = trade_stop_mult * effective_params['trail_mult_low']

            stop_dist = atr_val * trade_stop_mult  # 硬止损始终用原始倍率
            trailing_dist = atr_val * trailing_mult    # 移动止盈用自适应倍率

            # ATR 硬止损
            if current_price <= base - stop_dist:
                pnl_pct = (current_price - base) / base
                return self.close_position(symbol, current_date, current_price,
                                           f"ATR硬止损 ({pnl_pct*100:.1f}%, ATR={atr_val:.2f})")

            # ATR 移动止盈 (盈利自适应倍率) — 震荡市可禁用
            if not disable_trailing and trade.highest_price > base:
                if current_price <= trade.highest_price - trailing_dist:
                    drawdown = (current_price - trade.highest_price) / trade.highest_price
                    return self.close_position(symbol, current_date, current_price,
                                               f"ATR移动止盈 (最高{trade.highest_price:.2f}, "
                                               f"回撤{drawdown*100:.1f}%, 盈利{profit_pct*100:.1f}%)")
        else:
            # 无 ATR 时回退到固定百分比 (参数可配置)
            pnl_pct = (current_price - base) / base
            if pnl_pct <= -effective_params['no_atr_hard_stop_pct']:
                return self.close_position(symbol, current_date, current_price,
                                           f"硬止损 ({pnl_pct*100:.1f}%)")
            # 移动止盈 — 震荡市可禁用
            if not disable_trailing and trade.highest_price > base:
                drawdown = (current_price - trade.highest_price) / trade.highest_price
                if drawdown <= -effective_params['no_atr_trail_drawdown']:
                    return self.close_position(symbol, current_date, current_price,
                                               f"移动止盈 (回撤{drawdown*100:.1f}%)")

        return None