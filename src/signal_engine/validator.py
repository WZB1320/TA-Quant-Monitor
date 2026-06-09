"""
多维度交叉验证器 (Validator)

验证规则:
  Strrong Buy:  趋势类 + 动量类 + 量价类 三类同时看多 → 三重共振
  Buy:         趋势类 + 至少一个其他类别看多
  Weak Buy:    单个类别 ≥2 个指标看多
  Neutral:     多空矛盾，无明确倾向
  Sell family: 对称反向

核心逻辑: 不同维度的指标指向同一个方向，才产生可靠信号
"""
from typing import Dict
from src.indicators.base import IndicatorResult
from src.signal_engine.signals import SignalLevel, CategorySummary


class Validator:
    """交叉验证器 — 确保多个维度同时指向同一方向"""

    def __init__(self):
        self._categories = ["trend", "strength", "momentum", "volume"]

    def validate(self, indicator_results: Dict[str, IndicatorResult],
                 hard_blocked: bool = False) -> SignalLevel:
        """
        多维度交叉验证

        Args:
            indicator_results: 所有指标结果
            hard_blocked: 是否被硬过滤拦截

        Returns:
            最终信号级别
        """
        if hard_blocked:
            return SignalLevel.NEUTRAL

        # 按类别汇总
        categories = self._summarize(indicator_results)
        cat_dirs = {k: v.direction for k, v in categories.items()}

        # 统计各类别方向
        bull_cats = [k for k, v in cat_dirs.items() if v > 0]
        bear_cats = [k for k, v in cat_dirs.items() if v < 0]
        neutral_cats = [k for k, v in cat_dirs.items() if v == 0]

        nb = len(bull_cats)
        nber = len(bear_cats)

        # 三重共振: 趋势 + 动量 + 量价 同时看多/看空
        if nb >= 3 and "trend" in bull_cats and "momentum" in bull_cats and "volume" in bull_cats:
            return SignalLevel.STRONG_BUY
        if nber >= 3 and "trend" in bear_cats and "momentum" in bear_cats and "volume" in bear_cats:
            return SignalLevel.STRONG_SELL

        # 趋势 + 至少一个其他类别
        if "trend" in bull_cats and nb >= 2:
            return SignalLevel.BUY
        if "trend" in bear_cats and nber >= 2:
            return SignalLevel.SELL

        # 强度 + 动量或量价
        if "strength" in bull_cats and nb >= 2:
            return SignalLevel.BUY
        if "strength" in bear_cats and nber >= 2:
            return SignalLevel.SELL

        # 单类别强信号 (≥2个指标看多/看空)
        for cat, summary in categories.items():
            if summary.consensus >= 2 and summary.dissensus == 0:
                if summary.direction > 0:
                    return SignalLevel.WEAK_BUY
                else:
                    return SignalLevel.WEAK_SELL

        # 动量 + 量价 (无趋势但短期信号一致)
        if "momentum" in bull_cats and "volume" in bull_cats:
            return SignalLevel.WEAK_BUY
        if "momentum" in bear_cats and "volume" in bear_cats:
            return SignalLevel.WEAK_SELL

        return SignalLevel.NEUTRAL

    def summarize(self, indicator_results: Dict[str, IndicatorResult]
                  ) -> Dict[str, CategorySummary]:
        """获取分类汇总"""
        return self._summarize(indicator_results)

    def _summarize(self, indicator_results: Dict[str, IndicatorResult]
                   ) -> Dict[str, CategorySummary]:
        categories = {cat: {"bull": 0, "bear": 0, "neut": 0, "indicators": {}}
                      for cat in self._categories}

        for name, r in indicator_results.items():
            if not isinstance(r, IndicatorResult):
                continue  # skip non-indicator items (e.g. SCORE float)
            cat = r.category
            if cat not in categories:
                continue
            categories[cat]["indicators"][name] = r
            if r.direction > 0:
                categories[cat]["bull"] += 1
            elif r.direction < 0:
                categories[cat]["bear"] += 1
            else:
                categories[cat]["neut"] += 1

        result = {}
        for cat, data in categories.items():
            b, br, n = data["bull"], data["bear"], data["neut"]
            if b > br:
                direction = 1
            elif br > b:
                direction = -1
            else:
                direction = 0
            result[cat] = CategorySummary(
                category=cat,
                direction=direction,
                consensus=b,
                dissensus=br,
                indicators=data["indicators"],
            )
        return result