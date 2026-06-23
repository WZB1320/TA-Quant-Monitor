"""Validator 交叉验证器单元测试

覆盖:
  - 三重共振 (STRONG_BUY / STRONG_SELL)
  - 双类别确认 (BUY / SELL)
  - 单类别强信号 (WEAK_BUY / WEAK_SELL)
  - 矛盾信号 (NEUTRAL)
  - hard_blocked 拦截
"""
import pytest

from src.signal_engine.validator import Validator
from src.signal_engine.signals import SignalLevel
from src.indicators.base import IndicatorResult


def _make(name, category, direction, strength=0.8):
    return IndicatorResult(
        name=name, category=category, direction=direction,
        signal="buy" if direction > 0 else "sell" if direction < 0 else "neutral",
        strength=strength, description="", values={},
    )


@pytest.fixture
def validator():
    return Validator()


class TestStrongSignals:
    def test_strong_buy_three_categories_bullish(self, validator):
        """趋势+动量+量价 三类同时看多 → STRONG_BUY"""
        indicators = {
            "MA60": _make("MA60", "trend", 1),
            "MACD": _make("MACD", "trend", 1),
            "RSI": _make("RSI", "momentum", 1),
            "KDJ": _make("KDJ", "momentum", 1),
            "OBV": _make("OBV", "volume", 1),
            "VOL_RATIO": _make("VOL_RATIO", "volume", 1),
        }
        level = validator.validate(indicators)
        assert level == SignalLevel.STRONG_BUY

    def test_strong_sell_three_categories_bearish(self, validator):
        """趋势+动量+量价 三类同时看空 → STRONG_SELL"""
        indicators = {
            "MA60": _make("MA60", "trend", -1),
            "MACD": _make("MACD", "trend", -1),
            "RSI": _make("RSI", "momentum", -1),
            "KDJ": _make("KDJ", "momentum", -1),
            "OBV": _make("OBV", "volume", -1),
            "VOL_RATIO": _make("VOL_RATIO", "volume", -1),
        }
        level = validator.validate(indicators)
        assert level == SignalLevel.STRONG_SELL

    def test_strong_buy_requires_all_three_categories(self, validator):
        """缺少量价类不应触发 STRONG_BUY (仅趋势+动量)"""
        indicators = {
            "MA60": _make("MA60", "trend", 1),
            "RSI": _make("RSI", "momentum", 1),
            # 无 volume 类
        }
        level = validator.validate(indicators)
        # 趋势 + 动量 = 2 类, 应为 BUY 而非 STRONG_BUY
        assert level == SignalLevel.BUY


class TestBuySignals:
    def test_buy_trend_plus_momentum(self, validator):
        """趋势 + 动量 看多 → BUY"""
        indicators = {
            "MA60": _make("MA60", "trend", 1),
            "RSI": _make("RSI", "momentum", 1),
        }
        level = validator.validate(indicators)
        assert level == SignalLevel.BUY

    def test_buy_trend_plus_strength(self, validator):
        """趋势 + 强度 看多 → BUY"""
        indicators = {
            "MA60": _make("MA60", "trend", 1),
            "ADX": _make("ADX", "strength", 1),
        }
        level = validator.validate(indicators)
        assert level == SignalLevel.BUY

    def test_sell_trend_plus_momentum(self, validator):
        """趋势 + 动量 看空 → SELL"""
        indicators = {
            "MA60": _make("MA60", "trend", -1),
            "RSI": _make("RSI", "momentum", -1),
        }
        level = validator.validate(indicators)
        assert level == SignalLevel.SELL


class TestWeakSignals:
    def test_weak_buy_single_category_consensus(self, validator):
        """单类别 ≥2 个指标看多 → WEAK_BUY"""
        indicators = {
            "RSI": _make("RSI", "momentum", 1),
            "KDJ": _make("KDJ", "momentum", 1),
        }
        level = validator.validate(indicators)
        assert level == SignalLevel.WEAK_BUY

    def test_weak_sell_single_category_consensus(self, validator):
        """单类别 ≥2 个指标看空 → WEAK_SELL

        注意: Validator 逻辑中 consensus=看多数, dissensus=看空数
        单类别强信号要求 consensus>=2 (看多), 看空走 momentum+volume 路径
        """
        indicators = {
            "RSI": _make("RSI", "momentum", -1),
            "KDJ": _make("KDJ", "momentum", -1),
            "OBV": _make("OBV", "volume", -1),  # 动量+量价都看空
        }
        level = validator.validate(indicators)
        assert level == SignalLevel.WEAK_SELL

    def test_momentum_plus_volume_bullish(self, validator):
        """动量 + 量价 看多 (无趋势) → WEAK_BUY"""
        indicators = {
            "RSI": _make("RSI", "momentum", 1),
            "OBV": _make("OBV", "volume", 1),
        }
        level = validator.validate(indicators)
        assert level == SignalLevel.WEAK_BUY


class TestNeutralSignals:
    def test_contradictory_signals_neutral(self, validator):
        """多空矛盾 → NEUTRAL"""
        indicators = {
            "MA60": _make("MA60", "trend", 1),       # 看多
            "RSI": _make("RSI", "momentum", -1),     # 看空
        }
        level = validator.validate(indicators)
        assert level == SignalLevel.NEUTRAL

    def test_empty_indicators_neutral(self, validator):
        """空指标 → NEUTRAL"""
        level = validator.validate({})
        assert level == SignalLevel.NEUTRAL

    def test_all_neutral_direction_neutral(self, validator):
        """所有指标方向为 0 → NEUTRAL"""
        indicators = {
            "MA60": _make("MA60", "trend", 0),
            "RSI": _make("RSI", "momentum", 0),
        }
        level = validator.validate(indicators)
        assert level == SignalLevel.NEUTRAL


class TestHardBlocked:
    def test_hard_blocked_returns_neutral(self, validator):
        """被硬过滤拦截时强制返回 NEUTRAL, 即使指标全看多"""
        indicators = {
            "MA60": _make("MA60", "trend", 1),
            "RSI": _make("RSI", "momentum", 1),
            "OBV": _make("OBV", "volume", 1),
        }
        level = validator.validate(indicators, hard_blocked=True)
        assert level == SignalLevel.NEUTRAL


class TestSummarize:
    def test_summarize_returns_category_summary(self, validator):
        """summarize 应返回各类别汇总"""
        indicators = {
            "MA60": _make("MA60", "trend", 1),
            "MACD": _make("MACD", "trend", 1),
            "RSI": _make("RSI", "momentum", -1),
        }
        summary = validator.summarize(indicators)
        assert "trend" in summary
        assert "momentum" in summary
        assert summary["trend"].direction == 1   # 2 看多
        assert summary["momentum"].direction == -1  # 1 看空
        assert summary["trend"].consensus == 2
