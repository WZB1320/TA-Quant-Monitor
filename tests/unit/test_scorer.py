"""Scorer 加权评分器单元测试

覆盖:
  - 体制自动检测 (ADX 阈值)
  - 体制自适应权重切换
  - 强制体制覆盖
  - 分组专属权重
  - 信号强度修正
  - 边界条件 (空输入/零权重)
"""
import pytest

from src.signal_engine.scorer import Scorer, REGIME_WEIGHTS, CATEGORY_WEIGHTS
from src.indicators.base import IndicatorResult


def _make_indicator(name, category, direction, strength=0.8, **values):
    """构造 IndicatorResult 辅助函数"""
    return IndicatorResult(
        name=name, category=category, direction=direction,
        signal="buy" if direction > 0 else "sell" if direction < 0 else "neutral",
        strength=strength, description="", values=values,
    )


# ── 体制自动检测 ──

class TestRegimeDetection:
    def test_adx_gt_25_is_trending(self):
        """ADX > 25 应判为 trending"""
        indicators = {
            "ADX": _make_indicator("ADX", "strength", 1, 0.8, adx=30, adx_prev=28),
        }
        scorer = Scorer()
        regime = scorer._get_regime(indicators)
        assert regime == "trending"

    def test_adx_lt_20_is_ranging(self):
        """ADX < 20 应判为 ranging"""
        indicators = {
            "ADX": _make_indicator("ADX", "strength", -1, 0.7, adx=15, adx_prev=18),
        }
        scorer = Scorer()
        regime = scorer._get_regime(indicators)
        assert regime == "ranging"

    def test_adx_between_20_25_is_transition(self):
        """20 ≤ ADX ≤ 25 应判为 transition"""
        indicators = {
            "ADX": _make_indicator("ADX", "strength", 1, 0.6, adx=22, adx_prev=21),
        }
        scorer = Scorer()
        regime = scorer._get_regime(indicators)
        assert regime == "transition"

    def test_adx_gt_30_declining_is_trend_fading(self):
        """ADX > 30 但下降中应判为 trend_fading"""
        indicators = {
            "ADX": _make_indicator("ADX", "strength", 1, 0.7, adx=32, adx_prev=35),
        }
        scorer = Scorer()
        regime = scorer._get_regime(indicators)
        assert regime == "trend_fading"

    def test_no_adx_defaults_to_transition(self):
        """无 ADX 指标时应默认 transition"""
        indicators = {"MA60": _make_indicator("MA60", "trend", 1, 0.8)}
        scorer = Scorer()
        regime = scorer._get_regime(indicators)
        assert regime == "transition"


# ── 体制自适应权重 ──

class TestRegimeAdaptiveWeights:
    def test_trending_market_boosts_trend_weight(self):
        """趋势市应提升趋势类权重, 全看多时得分为正"""
        indicators = {
            "MA60": _make_indicator("MA60", "trend", 1, 0.9),
            "EMA_DUAL": _make_indicator("EMA_DUAL", "trend", 1, 0.8),
            "MACD": _make_indicator("MACD", "trend", 1, 0.85),
            "ADX": _make_indicator("ADX", "strength", 1, 0.8, adx=30, adx_prev=28),
            "RSI": _make_indicator("RSI", "momentum", 1, 0.7),
            "KDJ": _make_indicator("KDJ", "momentum", 1, 0.6),
            "OBV": _make_indicator("OBV", "volume", 1, 0.75),
            "VOL_RATIO": _make_indicator("VOL_RATIO", "volume", 1, 0.65),
        }
        scorer = Scorer()
        score = scorer.score(indicators)
        assert score > 50, f"趋势市全看多得分应 > 50, 实际: {score}"

    def test_ranging_market_reduces_trend_weight(self):
        """震荡市趋势权重降低, 即使趋势看空动量看多, 得分仍偏正"""
        indicators = {
            "MA60": _make_indicator("MA60", "trend", -1, 0.8),
            "EMA_DUAL": _make_indicator("EMA_DUAL", "trend", -1, 0.7),
            "MACD": _make_indicator("MACD", "trend", -1, 0.75),
            "ADX": _make_indicator("ADX", "strength", -1, 0.6, adx=15, adx_prev=18),
            "RSI": _make_indicator("RSI", "momentum", 1, 0.9),
            "KDJ": _make_indicator("KDJ", "momentum", 1, 0.85),
            "OBV": _make_indicator("OBV", "volume", 1, 0.7),
            "VOL_RATIO": _make_indicator("VOL_RATIO", "volume", 1, 0.6),
        }
        scorer = Scorer()
        score = scorer.score(indicators)
        # 震荡市动量权重 0.50, 趋势权重仅 0.15
        # 动量看多 +0.50×1, 趋势看空 -0.15×1 → 净正
        assert score > 0, f"震荡市动量看多应使得分偏正, 实际: {score}"

    def test_forced_regime_overrides_auto_detection(self):
        """forced_regime 应覆盖 ADX 自动检测"""
        indicators = {
            "MA60": _make_indicator("MA60", "trend", 1, 0.9),
            "ADX": _make_indicator("ADX", "strength", 1, 0.8, adx=15, adx_prev=18),  # 自动判为 ranging
            "RSI": _make_indicator("RSI", "momentum", 1, 0.7),
        }
        scorer = Scorer()
        # 强制 trending, 应使用 trending 权重
        score_forced = scorer.score(indicators, forced_regime="trending")
        score_auto = scorer.score(indicators)  # 自动判为 ranging
        # 同样指标, 趋势市权重对趋势类更敏感
        assert score_forced != score_auto, "强制体制应改变得分"

    def test_all_bearish_yields_negative_score(self):
        """全部看空时得分应为负"""
        indicators = {
            "MA60": _make_indicator("MA60", "trend", -1, 0.9),
            "MACD": _make_indicator("MACD", "trend", -1, 0.85),
            "ADX": _make_indicator("ADX", "strength", -1, 0.8, adx=30, adx_prev=28),
            "RSI": _make_indicator("RSI", "momentum", -1, 0.7),
            "OBV": _make_indicator("OBV", "volume", -1, 0.75),
        }
        scorer = Scorer()
        score = scorer.score(indicators)
        assert score < -30, f"全看空得分应 < -30, 实际: {score}"


# ── 分组差异化 ──

class TestGroupCustomization:
    def test_indicator_weights_override_defaults(self):
        """分组专属指标权重应覆盖默认权重"""
        indicators = {
            "MA60": _make_indicator("MA60", "trend", 1, 0.9),
            "RSI": _make_indicator("RSI", "momentum", 1, 0.8),
        }
        # 默认权重
        scorer_default = Scorer()
        score_default = scorer_default.score(indicators, forced_regime="transition")

        # 分组专属: 提升 MA60 权重, 降低 RSI 权重
        scorer_custom = Scorer(indicator_weights={"MA60": 0.5, "RSI": 0.1})
        score_custom = scorer_custom.score(indicators, forced_regime="transition")

        # MA60 强度 0.9 > RSI 强度 0.8, 提升 MA60 权重应使得分更高
        assert score_custom > score_default, "提升强指标权重应使得分更高"

    def test_strength_modifiers_adjust_score(self):
        """信号强度修正系数应调整指标贡献"""
        indicators = {
            "KDJ": _make_indicator("KDJ", "momentum", 1, 0.8),
            "RSI": _make_indicator("RSI", "momentum", 1, 0.8),
        }
        # 无修正
        scorer_plain = Scorer(indicator_weights={"KDJ": 0.5, "RSI": 0.5})
        score_plain = scorer_plain.score(indicators, forced_regime="transition")

        # KDJ 修正系数 0.5 (削弱)
        scorer_modified = Scorer(
            indicator_weights={"KDJ": 0.5, "RSI": 0.5},
            strength_modifiers={"KDJ": 0.5},
        )
        score_modified = scorer_modified.score(indicators, forced_regime="transition")

        assert score_modified < score_plain, "削弱 KDJ 应使得分降低"


# ── 边界条件 ──

class TestEdgeCases:
    def test_empty_indicators_returns_zero(self):
        """空指标输入应返回 0"""
        scorer = Scorer()
        assert scorer.score({}) == 0.0

    def test_zero_strength_returns_zero(self):
        """所有指标强度为 0 时得分应为 0"""
        indicators = {
            "MA60": _make_indicator("MA60", "trend", 1, 0.0),
            "RSI": _make_indicator("RSI", "momentum", 1, 0.0),
        }
        scorer = Scorer()
        assert scorer.score(indicators, forced_regime="transition") == 0.0

    def test_neutral_direction_returns_zero(self):
        """所有指标方向为 0 (中性) 时得分应为 0"""
        indicators = {
            "MA60": _make_indicator("MA60", "trend", 0, 0.8),
            "RSI": _make_indicator("RSI", "momentum", 0, 0.8),
        }
        scorer = Scorer()
        assert scorer.score(indicators, forced_regime="transition") == 0.0

    def test_score_range_within_100(self):
        """得分应在 -100 ~ +100 范围内"""
        indicators = {
            "MA60": _make_indicator("MA60", "trend", 1, 1.0),
            "EMA_DUAL": _make_indicator("EMA_DUAL", "trend", 1, 1.0),
            "MACD": _make_indicator("MACD", "trend", 1, 1.0),
            "ADX": _make_indicator("ADX", "strength", 1, 1.0, adx=30, adx_prev=28),
            "RSI": _make_indicator("RSI", "momentum", 1, 1.0),
            "KDJ": _make_indicator("KDJ", "momentum", 1, 1.0),
            "OBV": _make_indicator("OBV", "volume", 1, 1.0),
            "VOL_RATIO": _make_indicator("VOL_RATIO", "volume", 1, 1.0),
        }
        scorer = Scorer()
        score = scorer.score(indicators)
        assert -100 <= score <= 100, f"得分超出范围: {score}"


# ── 按类别评分 ──

class TestScoreByCategory:
    def test_category_scores_returned(self):
        """score_by_category 应返回各类别得分"""
        indicators = {
            "MA60": _make_indicator("MA60", "trend", 1, 0.9),
            "RSI": _make_indicator("RSI", "momentum", 1, 0.8),
        }
        scorer = Scorer()
        cat_scores = scorer.score_by_category(indicators)
        assert "trend" in cat_scores
        assert "momentum" in cat_scores
        assert cat_scores["trend"] > 0
        assert cat_scores["momentum"] > 0
