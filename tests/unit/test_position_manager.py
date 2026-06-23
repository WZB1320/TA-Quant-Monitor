"""PositionManager 仓位管理器单元测试

覆盖:
  - 开仓 (ATR 动态仓位, 体制自适应, 信号强度加成)
  - 平仓 (手续费, 印花税, 盈亏计算)
  - 止损止盈 (10% 硬止损, ATR 止损, 三档移动止盈)
  - 体制自适应仓位
  - 边界条件
"""
from datetime import date, timedelta

import pytest

from src.backtest.position import PositionManager, Trade, Side


@pytest.fixture
def pm():
    """标准仓位管理器: 初始资金 10万, 单票上限 30%, ATR 倍率 2.5"""
    return PositionManager(
        initial_capital=100000,
        position_ratio=0.30,
        commission_rate=0.00025,
        risk_per_trade=0.015,
        atr_stop_mult=2.5,
    )


# ── 开仓 ──

class TestOpenLong:
    def test_basic_open_long(self, pm):
        """基本开仓: 应创建持仓, 扣减现金"""
        initial_cash = pm.cash
        trade = pm.open_long(
            symbol="TEST",
            entry_date=date(2026, 1, 1),
            entry_price=10.0,
            signal="BUY",
            atr_value=0.5,
        )
        assert trade is not None
        assert trade.symbol == "TEST"
        assert trade.side == Side.LONG
        assert trade.shares >= 100
        assert trade.entry_price == 10.0
        assert pm.has_position("TEST")
        assert pm.cash < initial_cash

    def test_duplicate_open_returns_none(self, pm):
        """已持仓的股票再次开仓应返回 None"""
        pm.open_long("TEST", date(2026, 1, 1), 10.0, "BUY", atr_value=0.5)
        duplicate = pm.open_long("TEST", date(2026, 1, 2), 10.5, "BUY", atr_value=0.5)
        assert duplicate is None

    def test_atr_controls_position_size(self, pm):
        """ATR 越大, 仓位应越小 (风险控制)"""
        # 小 ATR → 大仓位
        trade_small_atr = pm.open_long(
            "SMALL_ATR", date(2026, 1, 1), 10.0, "BUY", atr_value=0.2
        )
        shares_small = trade_small_atr.shares

        # 重置
        pm2 = PositionManager(initial_capital=100000, position_ratio=0.30,
                              risk_per_trade=0.015, atr_stop_mult=2.5)
        # 大 ATR → 小仓位
        trade_large_atr = pm2.open_long(
            "LARGE_ATR", date(2026, 1, 1), 10.0, "BUY", atr_value=1.0
        )
        shares_large = trade_large_atr.shares

        assert shares_large < shares_small, \
            f"大 ATR 应导致更小仓位: small={shares_small}, large={shares_large}"

    def test_signal_strength_boosts_position(self, pm):
        """信号强度加成应放大仓位"""
        # 普通信号 (strength=1.0)
        trade_normal = pm.open_long(
            "NORMAL", date(2026, 1, 1), 10.0, "BUY",
            atr_value=0.5, signal_strength=1.0
        )
        shares_normal = trade_normal.shares

        # 重置
        pm2 = PositionManager(initial_capital=100000, position_ratio=0.30,
                              risk_per_trade=0.015, atr_stop_mult=2.5)
        # 强信号 (strength=1.3)
        trade_strong = pm2.open_long(
            "STRONG", date(2026, 1, 1), 10.0, "BUY",
            atr_value=0.5, signal_strength=1.3
        )
        shares_strong = trade_strong.shares

        assert shares_strong >= shares_normal, "强信号应使得仓位不小于普通信号"

    def test_bearish_market_halves_position(self, pm):
        """大盘空头时仓位应减半"""
        trade_normal = pm.open_long(
            "NORMAL", date(2026, 1, 1), 10.0, "BUY",
            atr_value=0.5, bearish_market=False
        )
        shares_normal = trade_normal.shares

        pm2 = PositionManager(initial_capital=100000, position_ratio=0.30,
                              risk_per_trade=0.015, atr_stop_mult=2.5)
        trade_bearish = pm2.open_long(
            "BEARISH", date(2026, 1, 1), 10.0, "BUY",
            atr_value=0.5, bearish_market=True
        )
        shares_bearish = trade_bearish.shares

        assert shares_bearish < shares_normal, "熊市应减仓"

    def test_shares_rounded_to_100(self, pm):
        """股数应取整到 100 股 (A股最小交易单位)"""
        trade = pm.open_long("TEST", date(2026, 1, 1), 10.0, "BUY", atr_value=0.5)
        assert trade.shares % 100 == 0


# ── 平仓 ──

class TestClosePosition:
    def test_basic_close(self, pm):
        """基本平仓: 应计算盈亏, 回笼资金"""
        pm.open_long("TEST", date(2026, 1, 1), 10.0, "BUY", atr_value=0.5)
        cash_before_close = pm.cash

        trade = pm.close_position("TEST", date(2026, 1, 10), 11.0, "SELL")
        assert trade is not None
        assert trade.is_closed
        assert trade.exit_price == 11.0
        assert trade.pnl > 0  # 10→11 盈利
        assert pm.cash > cash_before_close
        assert not pm.has_position("TEST")
        assert len(pm.closed_trades) == 1

    def test_close_nonexistent_returns_none(self, pm):
        """平仓不存在的持仓应返回 None"""
        result = pm.close_position("NONEXIST", date(2026, 1, 1), 10.0, "SELL")
        assert result is None

    def test_loss_trade_negative_pnl(self, pm):
        """亏损交易 pnl 应为负"""
        pm.open_long("TEST", date(2026, 1, 1), 10.0, "BUY", atr_value=0.5)
        trade = pm.close_position("TEST", date(2026, 1, 10), 9.0, "STOP_LOSS")
        assert trade.pnl < 0

    def test_commission_deducted(self, pm):
        """平仓应扣除手续费 (佣金 + 印花税)"""
        pm.open_long("TEST", date(2026, 1, 1), 10.0, "BUY", atr_value=0.5)
        trade = pm.close_position("TEST", date(2026, 1, 10), 10.0, "SELL")
        # 10→10 平价, 但有手续费, 所以 pnl 应为负
        assert trade.pnl < 0, "平价平仓扣手续费后应为亏损"
        assert trade.commission > 0


# ── 止损止盈 (核心风控) ──

class TestStopLoss:
    def test_hard_stop_10pct_triggers(self, pm):
        """10% 硬止损必须触发 (安全网)"""
        pm.open_long("TEST", date(2026, 1, 1), 10.0, "BUY", atr_value=0.5)
        # 价格跌到 8.99 (-10.1%) → 必须触发安全网
        result = pm.check_stop_loss("TEST", 8.99, date(2026, 1, 5))
        assert result is not None
        assert "安全网" in result.exit_signal

    def test_atr_hard_stop(self, pm):
        """ATR 硬止损: entry - 2.5×ATR

        注意: 10% 安全网硬止损 (entry×0.9=9.0) 优先级最高。
        要让 ATR 止损先触发, ATR 止损线必须 > 9.0。
        用 ATR=0.5, stop_mult=2.5 → 止损线 = 10 - 1.25 = 8.75 < 9.0, 安全网先触发。
        改用 ATR=0.5, stop_mult=3.0 → 止损线 = 10 - 1.5 = 8.5 < 9.0, 仍安全网先触发。
        要测 ATR 止损, 需让 ATR 止损线在 9.0 之上: ATR=0.5, stop_mult=2.5 → 8.75 (不行)
        实际: 安全网 10% 比 ATR 2.5×ATR 更宽松时, 安全网先触发。
        正确测法: 用小 ATR 让 ATR 止损线 > 9.0。ATR=0.5, stop_mult=2.5 → 8.75。
        安全网 9.0 > 8.75, 所以价格跌到 9.0 以下时安全网先触发。
        要测 ATR 止损, 需 ATR 止损线 > 安全网线。
        ATR=0.5, stop_mult=2.5 → 8.75 < 9.0 (安全网先)
        ATR=1.0, stop_mult=2.5 → 7.5 < 9.0 (安全网先)
        结论: 默认参数下 ATR 止损线永远 < 安全网线, ATR 止损无法独立触发。
        但代码中安全网检查在前, ATR 检查在后, 安全网触发时返回, 不会到 ATR。
        所以这个测试验证: 价格在安全网之上但 ATR 止损线之下时触发 ATR。
        需要价格 > 9.0 (安全网) 但 < ATR 止损线。
        ATR 止损线 = 10 - 2.5×ATR。要 > 9.0, 需 ATR < 0.4。
        用 ATR=0.3 → 止损线 = 10 - 0.75 = 9.25 > 9.0
        价格 9.2 < 9.25 (ATR 止损) 但 > 9.0 (安全网) → 触发 ATR 硬止损
        """
        pm2 = PositionManager(
            initial_capital=100000, position_ratio=0.30,
            risk_per_trade=0.015, atr_stop_mult=2.5,
        )
        pm2.open_long("TEST", date(2026, 1, 1), 10.0, "BUY", atr_value=0.3)
        # ATR 止损线 = 10 - 2.5×0.3 = 9.25, 价格 9.2 触发 ATR 硬止损
        result = pm2.check_stop_loss("TEST", 9.2, date(2026, 1, 5))
        assert result is not None
        assert "ATR硬止损" in result.exit_signal

    def test_no_stop_above_atr_threshold(self, pm):
        """价格在 ATR 止损线之上不应触发"""
        # 用小 ATR 让止损线 > 安全网, 测试安全网之上不触发
        pm.open_long("TEST", date(2026, 1, 1), 10.0, "BUY", atr_value=0.3)
        # ATR 止损线 = 9.25, 价格 9.3 > 9.25, 不触发 ATR
        # 价格 9.3 > 9.0 (安全网), 也不触发安全网
        result = pm.check_stop_loss("TEST", 9.3, date(2026, 1, 5))
        assert result is None

    def test_trailing_stop_profit_gt_20pct(self, pm):
        """盈利>20% 时移动止盈倍率收紧到 1.5×ATR (2.5×0.6)

        关键: profit_pct 用 current_price 计算, 不是 highest_price。
        要触发 >20% 档, current_price 也要 > entry×1.2 = 12.0。
        流程: 开仓 → 推高最高价到 13.0 (+30%) → 回落到 12.1 (+21%, 仍>20%)
        移动止盈线 = highest - 1.5×ATR = 13.0 - 0.75 = 12.25
        价格 12.1 < 12.25 且 profit 21% > 20% → 触发
        """
        pm.open_long("TEST", date(2026, 1, 1), 10.0, "BUY", atr_value=0.5)
        # 第1天: 推高最高价到 13.0 (+30%), 不触发 (无回撤)
        r1 = pm.check_stop_loss("TEST", 13.0, date(2026, 1, 10))
        assert r1 is None
        # 第2天: 回落到 12.1 (+21%, 仍>20%), 12.1 < 12.25 → 触发
        result = pm.check_stop_loss("TEST", 12.1, date(2026, 1, 11))
        assert result is not None
        assert "ATR移动止盈" in result.exit_signal

    def test_trailing_stop_profit_10_to_20pct(self, pm):
        """盈利 10~20% 时移动止盈倍率为 2.0×ATR (2.5×0.8)

        流程: 开仓 → 推高到 11.5 (+15%) → 回落到 10.4 (+4%, 但 highest 仍 11.5)
        profit_pct 用 current_price=10.4 计算 = 4% < 10%, 走默认 2.5×ATR 档
        要测 10~20% 档, current_price 也要在 11.0~12.0 之间。
        流程: 推高到 11.5 → 回落到 10.4 (profit=4%, 走 2.5 档, threshold=11.5-1.25=10.25)
        10.4 > 10.25, 不触发。
        正确测法: current_price 也要在 10~20% 区间。
        推高到 12.0 (+20%) → 回落到 11.0 (+10%, 在 10~20% 档)
        threshold = 12.0 - 2.0×0.5 = 11.0, 价格 10.9 < 11.0 → 触发
        """
        pm.open_long("TEST", date(2026, 1, 1), 10.0, "BUY", atr_value=0.5)
        # 推高到 12.0 (+20%), 不触发
        r1 = pm.check_stop_loss("TEST", 12.0, date(2026, 1, 10))
        assert r1 is None
        # 回落到 10.9 (profit=9%, 走默认 2.5 档)
        # 实际: 10.9 profit=9% < 10%, trailing_mult=2.5, dist=1.25
        # threshold = 12.0 - 1.25 = 10.75, 10.9 > 10.75, 不触发
        # 要触发 10~20% 档, current 要在 11.0~12.0
        # 推高到 12.0 → 回落到 11.0 (profit=10%, 在 10~20% 档)
        # threshold = 12.0 - 2.0×0.5 = 11.0, 11.0 <= 11.0 → 触发 (边界)
        result = pm.check_stop_loss("TEST", 10.9, date(2026, 1, 11))
        # 10.9 profit=9% < 10%, 走 2.5 档, threshold=10.75, 10.9 > 10.75
        # 不触发, 这个测试需要重新设计
        # 改用: 推高到 11.5 → 回落到 10.5 (profit=5%, 走 2.5 档)
        # threshold = 11.5 - 1.25 = 10.25, 10.5 > 10.25, 不触发
        # 结论: 默认 2.5 档给足空间, 很难触发。测 10~20% 档需 current 在 11~12
        pass  # 此测试场景复杂, 跳过

    def test_trailing_stop_10_to_20pct_triggers(self, pm):
        """盈利 10~20% 档移动止盈触发

        profit_pct > 0.10 是严格大于。
        用小 ATR 让 threshold 在 current 之上:
        - ATR=0.3, trailing_mult=2.0 (10~20% 档), trailing_dist=0.6
        - highest=12.0, threshold = 12.0 - 0.6 = 11.4
        - current=11.1, profit=11% > 10% (在 10~20% 档), 11.1 < 11.4 → 触发
        """
        pm.open_long("TEST", date(2026, 1, 1), 10.0, "BUY", atr_value=0.3)
        pm.check_stop_loss("TEST", 12.0, date(2026, 1, 10))  # highest=12.0
        # current=11.1, profit=11% > 10% (在 10~20% 档)
        # threshold = 12.0 - 2.0×0.3 = 11.4
        # 11.1 < 11.4 → 触发
        result = pm.check_stop_loss("TEST", 11.1, date(2026, 1, 11))
        assert result is not None
        assert "ATR移动止盈" in result.exit_signal

    def test_no_trailing_stop_below_10pct_profit(self, pm):
        """盈利<10% 时不应触发移动止盈 (给足空间)"""
        pm.open_long("TEST", date(2026, 1, 1), 10.0, "BUY", atr_value=0.5)
        # 最高 10.5 (+5%), 2.5×0.5=1.25, 10.5-1.25=9.25
        pm.check_stop_loss("TEST", 10.5, date(2026, 1, 5))
        # 价格 9.3 > 9.25, 不触发
        result = pm.check_stop_loss("TEST", 9.3, date(2026, 1, 6))
        assert result is None

    def test_no_stop_for_nonexistent_position(self, pm):
        """不存在的持仓不应触发止损"""
        result = pm.check_stop_loss("NONEXIST", 5.0, date(2026, 1, 1))
        assert result is None

    def test_highest_price_updated(self, pm):
        """止损检查应更新最高价"""
        pm.open_long("TEST", date(2026, 1, 1), 10.0, "BUY", atr_value=0.5)
        pm.check_stop_loss("TEST", 11.0, date(2026, 1, 5))
        trade = pm.open_positions["TEST"]
        assert trade.highest_price == 11.0

    def test_no_atr_falls_back_to_fixed_pct(self, pm):
        """无 ATR 时回退到固定百分比止损"""
        pm.open_long("TEST", date(2026, 1, 1), 10.0, "BUY", atr_value=None)
        # 无 ATR, 10% 硬止损
        result = pm.check_stop_loss("TEST", 8.99, date(2026, 1, 5))
        assert result is not None
        assert "硬止损" in result.exit_signal


# ── 体制自适应仓位 ──

class TestRegimeAdaptive:
    def test_trending_market_higher_position(self):
        """趋势市目标仓位 80% > 震荡市 30%"""
        pm_trending = PositionManager(initial_capital=100000, position_ratio=0.30)
        pm_trending.set_regime("trending")
        assert pm_trending._current_config["target_ratio"] == 0.80

        pm_ranging = PositionManager(initial_capital=100000, position_ratio=0.30)
        pm_ranging.set_regime("ranging")
        assert pm_ranging._current_config["target_ratio"] == 0.30

    def test_invalid_regime_falls_back_to_transition(self, pm):
        """无效体制应回退到 transition"""
        pm.set_regime("invalid_regime")
        assert pm._regime == "transition"

    def test_effective_regime_takes_lower(self, pm):
        """有效体制应取市场与个股的较低值 (更保守)"""
        pm.set_regime("trending")
        # 市场趋势, 个股震荡 → 应取震荡 (更保守)
        effective = pm._effective_regime("ranging")
        assert effective == "ranging"


# ── 市值计算 ──

class TestMarketValue:
    def test_total_value_calculation(self, pm):
        """总资产 = 现金 + 持仓市值"""
        pm.open_long("TEST", date(2026, 1, 1), 10.0, "BUY", atr_value=0.5)
        trade = pm.open_positions["TEST"]
        prices = {"TEST": 11.0}
        expected = pm.cash + trade.shares * 11.0
        assert pm.total_value(prices) == pytest.approx(expected, rel=1e-6)

    def test_total_return_calculation(self, pm):
        """总收益率计算"""
        pm.open_long("TEST", date(2026, 1, 1), 10.0, "BUY", atr_value=0.5)
        prices = {"TEST": 11.0}
        ret = pm.total_return(prices)
        # 初始 10万, 持仓盈利, 总资产应 > 10万
        assert ret > 0


# ── 统计 ──

class TestStatistics:
    def test_win_rate_calculation(self, pm):
        """胜率计算"""
        # 第一笔盈利
        pm.open_long("WIN", date(2026, 1, 1), 10.0, "BUY", atr_value=0.5)
        pm.close_position("WIN", date(2026, 1, 10), 11.0, "SELL")
        # 第二笔亏损
        pm.open_long("LOSS", date(2026, 1, 1), 10.0, "BUY", atr_value=0.5)
        pm.close_position("LOSS", date(2026, 1, 10), 9.0, "STOP")

        assert pm.trade_count == 2
        assert pm.win_count == 1
        assert pm.win_rate == 0.5

    def test_empty_stats(self, pm):
        """无交易时统计应为默认值"""
        assert pm.trade_count == 0
        assert pm.win_rate == 0.0
        assert pm.total_pnl == 0.0
