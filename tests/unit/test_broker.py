"""Broker 虚拟券商单元测试

覆盖:
  - 成交价计算 (买入加滑点, 卖出减滑点)
  - 手续费计算 (佣金 + 印花税, 最低5元)
  - 涨跌停检查 (一字板无法成交)
  - T+1 次日开盘价获取
"""
from datetime import date, timedelta

import pandas as pd
import pytest

from src.backtest.broker import Broker


@pytest.fixture
def broker():
    return Broker(commission_rate=0.00025, stamp_tax=0.001, slippage=0.0001)


# ── 成交价计算 ──

class TestPriceCalculation:
    def test_buy_price_adds_slippage(self, broker):
        """买入价 = 开盘价 × (1 + 滑点)"""
        price = broker.buy_price(10.0)
        assert price == pytest.approx(10.0 * 1.0001, rel=1e-6)

    def test_sell_price_subtracts_slippage(self, broker):
        """卖出价 = 开盘价 × (1 - 滑点)"""
        price = broker.sell_price(10.0)
        assert price == pytest.approx(10.0 * 0.9999, rel=1e-6)

    def test_zero_slippage_returns_original(self):
        """滑点为 0 时成交价等于开盘价"""
        b = Broker(slippage=0.0)
        assert b.buy_price(10.0) == 10.0
        assert b.sell_price(10.0) == 10.0


# ── 手续费计算 ──

class TestCommission:
    def test_buy_commission_only(self, broker):
        """买入仅收佣金, 无印花税"""
        amount = 10000.0
        comm = broker.buy_commission(amount)
        expected = max(amount * 0.00025, 5.0)
        assert comm == pytest.approx(expected, rel=1e-6)

    def test_sell_commission_plus_stamp_tax(self, broker):
        """卖出收佣金 + 印花税"""
        amount = 10000.0
        comm = broker.sell_commission(amount)
        expected_comm = max(amount * 0.00025, 5.0)
        expected_stamp = amount * 0.001
        assert comm == pytest.approx(expected_comm + expected_stamp, rel=1e-6)

    def test_min_commission_5_yuan(self, broker):
        """小金额交易应收取最低 5 元佣金"""
        # 1000 元 × 0.025% = 0.25 元 < 5 元
        comm = broker.buy_commission(1000.0)
        assert comm == 5.0

    def test_large_amount_commission(self, broker):
        """大金额交易佣金按比例计算"""
        amount = 1000000.0  # 100万
        comm = broker.buy_commission(amount)
        expected = amount * 0.00025  # 250 元
        assert comm == pytest.approx(expected, rel=1e-6)


# ── 涨跌停检查 ──

class TestLimitUpDown:
    def test_normal_price_can_trade(self, broker):
        """正常价格可交易"""
        assert broker.can_trade(10.5, 10.0) is True
        assert broker.can_trade(9.5, 10.0) is True

    def test_limit_up_cannot_trade(self, broker):
        """一字涨停无法成交"""
        # 10.0 × 1.10 = 11.0 涨停
        assert broker.can_trade(11.0, 10.0) is False

    def test_limit_down_cannot_trade(self, broker):
        """一字跌停无法成交"""
        # 10.0 × 0.90 = 9.0 跌停
        assert broker.can_trade(9.0, 10.0) is False

    def test_near_limit_up_can_trade(self, broker):
        """接近涨停但未涨停可交易"""
        # 10.0 × 1.095 = 10.95, 未到 11.0
        assert broker.can_trade(10.95, 10.0) is True

    def test_custom_limit_pct_20(self, broker):
        """创业板/科创板 20% 涨跌停"""
        # 10.0 × 1.20 = 12.0 涨停
        assert broker.can_trade(12.0, 10.0, limit_pct=0.20) is False
        assert broker.can_trade(11.5, 10.0, limit_pct=0.20) is True


# ── T+1 次日开盘价 ──

class TestGetNextOpen:
    def test_get_next_open_normal(self, broker):
        """正常获取次日开盘价"""
        df = pd.DataFrame({
            "date": ["2026-01-01", "2026-01-02", "2026-01-03"],
            "open": [10.0, 10.5, 11.0],
            "close": [10.2, 10.8, 10.9],
        })
        result = broker.get_next_open(df, 0)
        assert result is not None
        assert result["open"] == 10.5
        assert result["prev_close"] == 10.2

    def test_get_next_open_last_day_returns_none(self, broker):
        """最后一天无次日数据, 应返回 None"""
        df = pd.DataFrame({
            "date": ["2026-01-01", "2026-01-02"],
            "open": [10.0, 10.5],
            "close": [10.2, 10.8],
        })
        result = broker.get_next_open(df, 1)  # 最后一天
        assert result is None

    def test_get_next_open_invalid_idx(self, broker):
        """无效索引应返回 None"""
        df = pd.DataFrame({
            "date": ["2026-01-01"],
            "open": [10.0],
            "close": [10.2],
        })
        # 超出范围的索引
        assert broker.get_next_open(df, 5) is None
        assert broker.get_next_open(df, 100) is None
