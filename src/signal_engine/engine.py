"""
信号引擎 (SignalEngine) — 核心决策层

处理流程:
  指标结果 → 硬过滤(MA60+ADX) → 加权评分 → 交叉验证 → 信号去重 → 最终信号

职责:
  1. 接收 IndicatorPipeline 的产出
  2. 按优先级执行过滤和验证
  3. 产出结构化 SignalResult
  4. 生成可读的分析报告
"""
from typing import Dict, List, Optional
from datetime import date
import pandas as pd

from src.indicators import IndicatorPipeline
from src.indicators.base import IndicatorResult
from src.signal_engine.signals import SignalLevel, SignalResult, CategorySummary
from src.signal_engine.scorer import Scorer, CATEGORY_WEIGHTS
from src.signal_engine.validator import Validator
from src.signal_engine.filter import SignalFilter


class SignalEngine:
    """信号引擎 — 多指标交叉验证 → 最终买卖信号"""

    def __init__(self, dedup_days: int = 5, group_config=None):
        self.pipeline = IndicatorPipeline()
        self.scorer = Scorer()
        self.validator = Validator()
        self.filter = SignalFilter(dedup_days=dedup_days)
        self.group_config = group_config  # GroupConfig 实例, 可选

    def analyze(self, symbol: str, df: pd.DataFrame,
                analysis_date: "date | None" = None) -> SignalResult:
        """
        对单只股票执行完整分析

        Args:
            symbol: 股票代码
            df: 标准化日线数据
            analysis_date: 分析日期 (回测时传入, 用于去重和记录)
        """
        # 获取分组专属参数
        group_params = {}
        indicator_params = None
        indicator_weights = None
        strength_modifiers = None
        regime_weights = None
        forced_regime = None

        if self.group_config:
            group_params = self.group_config.get_all_group_params(symbol)
            indicator_params = group_params.get("indicator_params")
            indicator_weights = group_params.get("indicator_weights")
            strength_modifiers = group_params.get("strength_modifiers")
            regime_weights = group_params.get("regime_weights")
            forced_regime = group_params.get("forced_regime")

        # Step 1: 指标计算 (传入分组专属指标参数)
        indicator_results = self.pipeline.run(df, indicator_params=indicator_params)

        # Step 2: 硬过滤检查
        blocked, block_reason = self.filter.hard_filter(indicator_results)

        # Step 3: 加权评分 (使用分组专属权重+强度修正)
        scorer = Scorer(indicator_weights=indicator_weights,
                        strength_modifiers=strength_modifiers)
        score = scorer.score(indicator_results, regime_weights=regime_weights, forced_regime=forced_regime)
        cat_scores = scorer.score_by_category(indicator_results)

        # Step 3.1: 将得分注入 indicator_results (供 filter 使用)
        indicator_results["SCORE"] = score

        # Step 4: 交叉验证 → 信号级别
        level = self.validator.validate(indicator_results, hard_blocked=blocked)

        # Step 5: 硬过滤方向约束 + 分组专属过滤参数 (含ADX体制自适应覆盖)
        if not blocked:
            level = self.filter.apply_hard_constraint(level, indicator_results,
                                                      score_threshold=25,
                                                      group_params=group_params,
                                                      df=df)

        # Step 5.1: 获取ADX覆盖后的参数 (供Step6/7使用)
        if group_params:
            effective_params = self.filter._apply_regime_filter_overrides(group_params, indicator_results)
        else:
            effective_params = {}

        # Step 6: 冷却期检查 (ADX体制自适应冷却天数)
        if level.is_actionable and level.is_bullish:
            gc_cooldown = effective_params.get("cooldown_days", None) if effective_params else None
            if self.filter.is_in_cooldown(symbol, analysis_date, True, group_cooldown=gc_cooldown):
                level = SignalLevel.NEUTRAL

        # Step 7: 连亏暂停检查 (ADX体制自适应连亏阈值)
        if level.is_actionable and level.is_bullish:
            max_cl = effective_params.get("max_consecutive_losses", 0) if effective_params else 0
            suspend_d = effective_params.get("consecutive_loss_suspend", 0) if effective_params else 0
            if self.filter.is_suspended(symbol, analysis_date, max_cl, suspend_d):
                level = SignalLevel.NEUTRAL

        # Step 8: 信号去重检查 (使用实际分析日期)
        is_dup = self.filter.is_duplicate(symbol, level, analysis_date=analysis_date)
        if is_dup:
            level = SignalLevel.NEUTRAL

        # Step 9: 记录信号 (使用实际分析日期)
        self.filter.record(symbol, level, analysis_date=analysis_date)

        # Step 10: 计算置信度
        confidence = self._calc_confidence(level, indicator_results)

        # Step 11: 构建结果
        category_summary = self.validator.summarize(indicator_results)
        reason, details = self._build_reason(symbol, level, score, confidence,
                                             indicator_results, category_summary,
                                             cat_scores, blocked, block_reason)

        return SignalResult(
            symbol=symbol,
            level=level,
            score=score,
            confidence=confidence,
            reason=reason,
            details=details,
            category_summary=category_summary,
            hard_filter_blocked=blocked,
            block_reason=block_reason,
        )

    def analyze_batch(self, stock_data: Dict[str, pd.DataFrame]) -> List[SignalResult]:
        """批量分析多只股票，返回有操作信号的排序列表"""
        results = []
        for symbol, df in stock_data.items():
            if df is None or df.empty:
                continue
            try:
                result = self.analyze(symbol, df)
                results.append(result)
            except Exception as e:
                results.append(SignalResult(
                    symbol=symbol,
                    level=SignalLevel.NEUTRAL,
                    score=0.0,
                    confidence=0.0,
                    reason=f"分析异常: {e}",
                    details=str(e),
                ))

        # 按置信度降序排列
        results.sort(key=lambda r: abs(r.score) * r.confidence, reverse=True)
        return results

    def get_actionable_signals(self, stock_data: Dict[str, pd.DataFrame]
                               ) -> List[SignalResult]:
        """获取需要操作的信号 (买入/卖出级别以上)"""
        all_results = self.analyze_batch(stock_data)
        return [r for r in all_results if r.level.is_actionable]

    # ── 内部方法 ──

    def _calc_confidence(self, level: SignalLevel,
                         indicator_results: Dict[str, IndicatorResult]) -> float:
        """计算信号置信度"""
        # 计算指标一致性
        directions = [r.direction for r in indicator_results.values()
                      if isinstance(r, IndicatorResult)]

        if not directions:
            return 0.0

        # NEUTRAL: 看多/看空哪方占优就用哪方算一致性
        if level == SignalLevel.NEUTRAL:
            bullish = sum(1 for d in directions if d > 0)
            bearish = sum(1 for d in directions if d < 0)
            dominant = max(bullish, bearish)
            total = bullish + bearish
            consensus = dominant / total if total > 0 else 0.0
            # NEUTRAL 置信度上限 0.5（指标矛盾或被降级，不应高置信）
            return round(min(consensus * 0.5, 0.5), 2)

        target_dir = 1 if level.is_bullish else -1

        aligned = sum(1 for d in directions if d == target_dir)
        opposed = sum(1 for d in directions if d == -target_dir)
        total = aligned + opposed
        consensus = aligned / total if total > 0 else 0.5

        # 级别越强, 基础置信度越高
        base_conf = abs(level.value) / 3.0

        return round(min(consensus * 0.6 + base_conf * 0.4, 1.0), 2)

    def _build_reason(self, symbol: str, level: SignalLevel, score: float,
                      confidence: float, indicators: Dict[str, IndicatorResult],
                      categories: Dict[str, CategorySummary],
                      cat_scores: Dict[str, float],
                      blocked: bool, block_reason: str) -> tuple[str, str]:
        """生成可读的分析报告"""
        if blocked:
            return f"硬过滤拦截: {block_reason}", f"{symbol} 被硬过滤拦截: {block_reason}"

        lines = [f"{symbol} 综合得分: {score:+.1f}, 置信度: {confidence:.0%}"]
        lines.append(f"信号: {level.label}")

        # 各类别得分
        lines.append("\n--- 类别得分 ---")
        cat_names = {"trend": "趋势", "strength": "强度", "momentum": "动量", "volume": "量价"}
        for cat, cs in cat_scores.items():
            bar = self._score_bar(cs)
            lines.append(f"  {cat_names.get(cat, cat):4s}: {cs:+6.1f} {bar}")

        # 各指标明细
        lines.append("\n--- 指标明细 ---")
        for name, r in indicators.items():
            if not isinstance(r, IndicatorResult):
                continue
            dir_label = {1: "+", 0: "0", -1: "-"}[r.direction]
            lines.append(f"  {dir_label} {r.name:<10} [{r.strength:.1f}] {r.description}")

        # 交叉验证结论
        lines.append("\n--- 交叉验证 ---")
        for cat, cs in categories.items():
            dn = {1: "偏多", 0: "中性", -1: "偏空"}[cs.direction]
            lines.append(
                f"  {cat_names.get(cat, cat):4s}: {dn} "
                f"(看多{cs.consensus} vs 看空{cs.dissensus})"
            )

        reason = f"{level.label} (得分{score:+.0f}, 置信度{confidence:.0%})"
        details = "\n".join(lines)
        return reason, details

    @staticmethod
    def _score_bar(score: float, max_len: int = 15) -> str:
        """得分可视化条"""
        n = min(int(abs(score) / 100 * max_len), max_len)
        if score >= 0:
            return "#" * n + "." * (max_len - n)
        else:
            return "." * (max_len - n) + "#" * n