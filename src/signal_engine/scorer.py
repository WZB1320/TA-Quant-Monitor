"""
加权评分模型 (Scorer) — 体制自适应权重 + 分组差异化

三层差异化:
  Layer 1: 指标参数差异化 (在pipeline中实现)
  Layer 2: 指标权重差异化 (indicator_weights + regime_weights)
  Layer 3: 信号强度修正 (strength_modifiers)

基础权重设计:
  - 趋势类权重最高 (35%): MA60(10%) + EMA_DUAL(10%) + MACD(15%)
  - 强度类 (20%): ADX(20%)
  - 动量类 (25%): RSI(15%) + KDJ(10%)
  - 量价类 (20%): OBV(10%) + VOL_RATIO(10%)

体制自适应 (Regime-Aware):
  - ADX > 25 (趋势市): 趋势权重↑, 动量权重↓
  - ADX < 20 (震荡市): 动量权重↑, 趋势权重↓ → 切换为均值回归策略
  - 20 ≤ ADX ≤ 25: 基础权重

综合得分 = Σ(权重 × 方向 × 指标强度 × 修正系数) × 100
"""
from typing import Dict
from src.indicators.base import IndicatorResult


# 基础权重配置 {指标名: (类别, 权重)}
WEIGHTS = {
    "MA60":       ("trend",    0.10),
    "EMA_DUAL":   ("trend",    0.10),
    "MACD":       ("trend",    0.15),
    "ADX":        ("strength", 0.20),
    "RSI":        ("momentum", 0.15),
    "KDJ":        ("momentum", 0.10),
    "OBV":        ("volume",   0.10),
    "VOL_RATIO":  ("volume",   0.10),
}

# 类别权重汇总 (基础)
CATEGORY_WEIGHTS = {"trend": 0.35, "strength": 0.20, "momentum": 0.25, "volume": 0.20}

# 体制自适应权重
REGIME_WEIGHTS = {
    "trending":   {"trend": 0.40, "strength": 0.25, "momentum": 0.15, "volume": 0.20},
    "ranging":    {"trend": 0.15, "strength": 0.10, "momentum": 0.50, "volume": 0.25},
    "transition": {"trend": 0.28, "strength": 0.18, "momentum": 0.32, "volume": 0.22},
    "trend_fading": {"trend": 0.22, "strength": 0.15, "momentum": 0.38, "volume": 0.25},
}


class Scorer:
    """加权评分器 — 体制自适应 + 分组差异化"""

    def __init__(self, weights: dict = None,
                 indicator_weights: dict = None,
                 strength_modifiers: dict = None):
        """
        Args:
            weights: 旧式权重配置 (向后兼容)
            indicator_weights: 分组专属指标权重 {指标名: 权重}, 如 {"MA60": 0.15, "RSI": 0.18}
            strength_modifiers: 分组专属信号强度修正 {指标名: 修正系数}, 如 {"KDJ": 0.7, "OBV": 1.3}
        """
        self.weights = weights or WEIGHTS
        self.indicator_weights = indicator_weights
        self.strength_modifiers = strength_modifiers or {}

    def _get_regime(self, indicator_results: Dict[str, IndicatorResult]) -> str:
        """根据 ADX 判断当前市场体制, 包含趋势衰减检测"""
        adx = indicator_results.get("ADX")
        if adx is not None:
            adx_val = adx.values.get("adx")
            if adx_val is not None:
                adx_prev = adx.values.get("adx_prev")
                if adx_val > 30 and adx_prev is not None and adx_val < adx_prev:
                    return "trend_fading"
                if adx_val > 25:
                    return "trending"
                elif adx_val < 20:
                    return "ranging"
                else:
                    return "transition"
        return "transition"

    def _get_indicator_weight(self, name: str) -> float:
        """获取指标权重 (优先使用分组专属权重)"""
        if self.indicator_weights and name in self.indicator_weights:
            return self.indicator_weights[name]
        # 回退到旧式权重
        if name in self.weights:
            return self.weights[name][1]
        return 0.0

    def _get_indicator_category(self, name: str) -> str:
        """获取指标类别"""
        if name in self.weights:
            return self.weights[name][0]
        # 从indicator_weights推断类别
        for wname, (cat, _) in WEIGHTS.items():
            if wname == name:
                return cat
        return "unknown"

    def _apply_strength_modifier(self, name: str, strength: float) -> float:
        """应用信号强度修正系数"""
        modifier = self.strength_modifiers.get(name, 1.0)
        return strength * modifier

    def score(self, indicator_results: Dict[str, IndicatorResult],
              regime_weights: dict = None, forced_regime: str = None) -> float:
        """
        计算综合加权得分 (体制自适应 + 分组差异化)

        Args:
            indicator_results: {指标名: IndicatorResult}
            regime_weights: 分组专属体制权重 (可选), 不传则用默认
            forced_regime: 手动强制体制 (trending/ranging), 覆盖ADX自动检测

        Returns:
            综合得分 (-100 ~ +100)
        """
        regime = forced_regime if forced_regime else self._get_regime(indicator_results)
        if regime_weights:
            rw = regime_weights.get(regime, REGIME_WEIGHTS.get(regime, REGIME_WEIGHTS["transition"]))
        else:
            rw = REGIME_WEIGHTS.get(regime, REGIME_WEIGHTS["transition"])

        total_score = 0.0
        total_weight = 0.0

        # 确定要评分的指标列表
        if self.indicator_weights:
            indicator_names = set(self.indicator_weights.keys())
        else:
            indicator_names = set(self.weights.keys())

        for name in indicator_names:
            if name not in indicator_results:
                continue
            r = indicator_results[name]

            # 获取指标权重
            weight = self._get_indicator_weight(name)
            category = self._get_indicator_category(name)

            # 体制自适应: 调整每个类别的权重
            cat_weight = rw.get(category, 0.0)
            base_cat_weight = CATEGORY_WEIGHTS.get(category, 0.0)
            if base_cat_weight > 0 and self.indicator_weights:
                # 使用分组专属权重时: 计算该指标在分组中的类别内占比
                # 先算该类别下所有指标的权重总和
                cat_total = sum(
                    self.indicator_weights.get(n, 0.0)
                    for n in indicator_names
                    if self._get_indicator_category(n) == category
                )
                if cat_total > 0:
                    effective_weight = (weight / cat_total) * cat_weight
                else:
                    effective_weight = weight
            elif base_cat_weight > 0:
                effective_weight = (weight / base_cat_weight) * cat_weight
            else:
                effective_weight = weight

            # 应用信号强度修正
            modified_strength = self._apply_strength_modifier(name, r.strength)

            total_score += effective_weight * r.direction * modified_strength
            total_weight += effective_weight

        if total_weight == 0:
            return 0.0

        return (total_score / total_weight) * 100

    def score_by_category(self, indicator_results: Dict[str, IndicatorResult]
                          ) -> Dict[str, float]:
        """
        按类别分别计算得分

        Returns:
            {类别: 类别得分}
        """
        cat_scores = {}
        cat_weights = {}

        for name, weight_config in self.weights.items():
            if name not in indicator_results:
                continue
            category, weight = weight_config
            r = indicator_results[name]
            score_contrib = weight * r.direction * r.strength

            cat_scores.setdefault(category, 0.0)
            cat_weights.setdefault(category, 0.0)
            cat_scores[category] += score_contrib
            cat_weights[category] += weight

        result = {}
        for cat in cat_scores:
            result[cat] = (cat_scores[cat] / cat_weights[cat]) * 100 if cat_weights[cat] > 0 else 0.0
        return result