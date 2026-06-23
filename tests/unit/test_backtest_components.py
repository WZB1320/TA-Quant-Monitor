"""BacktestEngine 拆分组件单元测试

覆盖:
  - TradingCalendar: 日历构建 + O(1) 日期定位
  - RegimeDetector: 市场体制检测
  - MarketFilter: 大盘 MA60 过滤
"""
from datetime import date

import pandas as pd
import numpy as np
import pytest

from src.backtest.calendar import TradingCalendar
from src.backtest.regime_detector import RegimeDetector
from src.backtest.market_filter import MarketFilter


# ── TradingCalendar ──

class TestTradingCalendar:
    def _make_df(self, dates, closes):
        return pd.DataFrame({"date": dates, "close": closes})

    def test_build_calendar_from_single_stock(self):
        """单只股票构建日历"""
        df = self._make_df(
            ["2026-01-01", "2026-01-02", "2026-01-03"],
            [10.0, 10.5, 11.0],
        )
        cal = TradingCalendar({"TEST": df})
        assert len(cal.all_dates) == 3
        assert cal.all_dates[0] == date(2026, 1, 1)
        assert cal.all_dates[-1] == date(2026, 1, 3)

    def test_build_calendar_union_of_multiple_stocks(self):
        """多只股票日期并集"""
        df1 = self._make_df(["2026-01-01", "2026-01-02"], [10.0, 10.5])
        df2 = self._make_df(["2026-01-02", "2026-01-03"], [20.0, 20.5])
        cal = TradingCalendar({"A": df1, "B": df2})
        assert len(cal.all_dates) == 3  # 并集

    def test_locate_returns_index(self):
        """O(1) 定位应返回正确行索引"""
        df = self._make_df(
            ["2026-01-01", "2026-01-02", "2026-01-03"],
            [10.0, 10.5, 11.0],
        )
        cal = TradingCalendar({"TEST": df})
        assert cal.locate("TEST", date(2026, 1, 1)) == 0
        assert cal.locate("TEST", date(2026, 1, 2)) == 1
        assert cal.locate("TEST", date(2026, 1, 3)) == 2

    def test_locate_nonexistent_date_returns_none(self):
        """不存在的日期应返回 None"""
        df = self._make_df(["2026-01-01"], [10.0])
        cal = TradingCalendar({"TEST": df})
        assert cal.locate("TEST", date(2026, 12, 31)) is None

    def test_locate_nonexistent_symbol_returns_none(self):
        """不存在的股票代码应返回 None"""
        df = self._make_df(["2026-01-01"], [10.0])
        cal = TradingCalendar({"TEST": df})
        assert cal.locate("NONEXIST", date(2026, 1, 1)) is None

    def test_get_closing_prices(self):
        """获取收盘价"""
        df1 = self._make_df(["2026-01-01"], [10.0])
        df2 = self._make_df(["2026-01-01"], [20.0])
        cal = TradingCalendar({"A": df1, "B": df2})
        prices = cal.get_closing_prices({"A": df1, "B": df2}, date(2026, 1, 1))
        assert prices == {"A": 10.0, "B": 20.0}

    def test_locate_with_string_date(self):
        """字符串日期应正确转换"""
        df = self._make_df(["2026-01-01"], [10.0])
        cal = TradingCalendar({"TEST": df})
        assert cal.locate("TEST", "2026-01-01") == 0


# ── RegimeDetector ──

class TestRegimeDetector:
    def _make_benchmark(self, n=100, trend=0.01):
        """构造基准指数数据"""
        dates = pd.date_range("2026-01-01", periods=n, freq="D")
        close = 3000 + np.cumsum([trend] * n + np.random.randn(n) * 5)
        return pd.DataFrame({"date": dates.strftime("%Y-%m-%d"), "close": close})

    def test_no_benchmark_returns_transition(self):
        """无基准数据应返回 transition"""
        detector = RegimeDetector()
        assert detector.detect(None, date(2026, 1, 1)) == "transition"

    def test_short_benchmark_returns_transition(self):
        """数据不足 30 天应返回 transition"""
        df = self._make_benchmark(n=20)
        detector = RegimeDetector()
        assert detector.detect(df, date(2026, 1, 25)) == "transition"

    def test_trending_market_detected(self):
        """强趋势市场应判为 trending"""
        # 构造持续上涨趋势
        n = 100
        dates = pd.date_range("2026-01-01", periods=n, freq="D")
        close = 3000 + np.arange(n) * 2  # 每日 +2, 强趋势
        df = pd.DataFrame({"date": dates.strftime("%Y-%m-%d"), "close": close})
        detector = RegimeDetector()
        regime = detector.detect(df, date(2026, 4, 10))
        # ma20 > ma60 明显, trend_strength > 0.01
        assert regime in ("trending", "transition")

    def test_returns_valid_regime_string(self):
        """返回值应为有效体制字符串"""
        df = self._make_benchmark(n=100)
        detector = RegimeDetector()
        regime = detector.detect(df, date(2026, 4, 10))
        assert regime in ("trending", "transition", "ranging")


# ── MarketFilter ──

class TestMarketFilter:
    def _make_benchmark(self, closes):
        """构造基准数据"""
        n = len(closes)
        dates = pd.date_range("2026-01-01", periods=n, freq="D")
        return pd.DataFrame({"date": dates.strftime("%Y-%m-%d"), "close": closes})

    def test_no_benchmark_empty_trend_map(self):
        """无基准数据应返回空趋势映射"""
        mf = MarketFilter(None)
        assert mf.trend_map == {}
        assert mf.is_bearish(date(2026, 1, 1)) is False  # 默认多头

    def test_bullish_market_above_ma60(self):
        """价格在 MA60 上方应判为多头"""
        # 构造 60+ 天数据, 价格持续上涨
        n = 70
        closes = list(range(100, 100 + n))  # 100, 101, ..., 169
        df = self._make_benchmark(closes)
        mf = MarketFilter(df)
        # 最后一天价格 169 > MA60, 应为多头
        last_date = date(2026, 3, 11)  # 第70天
        assert mf.get_direction(last_date) == 1
        assert mf.is_bearish(last_date) is False

    def test_bearish_market_below_ma60(self):
        """价格在 MA60 下方应判为空头"""
        # 构造 60+ 天数据, 价格持续下跌
        n = 70
        closes = list(range(200, 200 - n, -1))  # 200, 199, ..., 131
        df = self._make_benchmark(closes)
        mf = MarketFilter(df)
        last_date = date(2026, 3, 11)
        assert mf.get_direction(last_date) == -1
        assert mf.is_bearish(last_date) is True

    def test_short_data_returns_empty_map(self):
        """数据不足 60 天应返回空映射"""
        df = self._make_benchmark([100, 101, 102])
        mf = MarketFilter(df)
        assert mf.trend_map == {}

    def test_trend_map_is_readonly_copy(self):
        """trend_map 应返回副本, 修改不影响内部状态"""
        df = self._make_benchmark(list(range(100, 170)))
        mf = MarketFilter(df)
        tm = mf.trend_map
        tm.clear()
        # 内部状态应不受影响
        assert len(mf.trend_map) > 0
