"""
信号引擎 (SignalEngine) — 核心决策层

处理流程 (v2 — classifier 统一定级):
  指标结果 → 硬过滤(数据检查) → 加权评分 → classifier定级 → filter执行约束 → 最终信号

职责:
  1. 接收 IndicatorPipeline 的产出
  2. 调用 SignalClassifier 统一定级 (共振+得分+背离+方向+过热)
  3. 调用 SignalFilter 产出执行约束 (冷却/连亏/去重, 不改 level)
  4. 产出结构化 SignalResult
  5. 生成可读的分析报告

变更说明 (v2):
  - 定级权收归 SignalClassifier, validator 退化为共识计算器
  - filter 不再通过 apply_hard_constraint 改 level, 改为产出 ExecutionConstraint
  - MACD背离等结构问题由 classifier 平滑降级(压一档), 非直接砍观望
  - 展示层应展示 level.label + execution, 不再造 action 第二套标签
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
from src.signal_engine.classifier import (
    SignalClassifier, ClassificationInput, ExecutionConstraint,
)


class SignalEngine:
    """信号引擎 — 多指标交叉验证 → 最终买卖信号"""

    def __init__(self, dedup_days: int = 5, group_config=None,
                 forced_regime: Optional[str] = None):
        """
        Args:
            dedup_days: 信号去重天数
            group_config: GroupConfig 实例, 可选
            forced_regime: 请求级 regime 覆盖, 不写盘, 不影响其他请求
                None  → 不覆盖, 用 group_config.get_all_group_params 返回的 forced_regime (来自 user_preferences)
                "auto" → 强制 auto (ADX 自动判断), 覆盖 user_preferences
                "trending" / "ranging" → 强制该模式, 覆盖 user_preferences
        """
        self.pipeline = IndicatorPipeline()
        self.scorer = Scorer()
        self.validator = Validator()       # 退化为共识计算器 (summarize)
        self.classifier = SignalClassifier()  # 唯一定级器 (v2 新增)
        self.filter = SignalFilter(dedup_days=dedup_days)
        self.group_config = group_config  # GroupConfig 实例, 可选
        self.forced_regime = forced_regime  # 请求级 regime 覆盖

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

        # 请求级 regime 覆盖: 路由层传入的 forced_regime 优先于 group_config (user_preferences)
        # "auto" → None (强制 ADX 自动判断); "trending"/"ranging" → 该模式; None → 不覆盖
        if self.forced_regime is not None:
            forced_regime = None if self.forced_regime == "auto" else self.forced_regime

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

        # Step 4: 类别共识计算 (validator 退化为共识计算器)
        category_summary = self.validator.summarize(indicator_results)

        # Step 5: classifier 统一定级 (共振+得分门槛+背离+方向+过热)
        clf_result = self.classifier.classify(ClassificationInput(
            score=score,
            category_consensus=category_summary,
            indicator_results=indicator_results,
            group_params=group_params,
            hard_blocked=blocked,
            block_reason=block_reason,
        ))
        level = clf_result.level
        block_detail = clf_result.demotion_reason

        # Step 5.1: 若 classifier 未降级且仍为 NEUTRAL, 补充共识分析说明
        if not blocked and level == SignalLevel.NEUTRAL and not block_detail:
            cat_names = {"trend": "趋势", "strength": "强度", "momentum": "动量", "volume": "量价"}
            bull_cats = [cat_names.get(k, k) for k, v in category_summary.items() if v.direction > 0]
            bear_cats = [cat_names.get(k, k) for k, v in category_summary.items() if v.direction < 0]
            if bull_cats and not bear_cats:
                block_detail = f"类别共识不足: 仅{'+'.join(bull_cats)}看多，需趋势+至少1类共振"
            elif bear_cats and not bull_cats:
                block_detail = f"类别共识不足: 仅{'+'.join(bear_cats)}看空"
            elif bull_cats and bear_cats:
                block_detail = f"多空矛盾: 看多[{'+'.join(bull_cats)}] vs 看空[{'+'.join(bear_cats)}]"
            else:
                block_detail = "所有类别均为中性，无明确方向"

        # Step 6: filter 细节过滤 (量比/ATR/RSI超买等, 仍由 filter 负责)
        #         这些过滤将级别降为 NEUTRAL, 原因记入 block_detail
        if not blocked and level.is_bullish:
            level, filter_reason = self.filter.apply_hard_constraint(
                level, indicator_results,
                score_threshold=25,
                group_params=group_params,
                df=df,
            )
            if filter_reason:
                block_detail = filter_reason

        # Step 7: 获取ADX覆盖后的参数 (供执行约束使用)
        if group_params:
            effective_params = self.filter._apply_regime_filter_overrides(group_params, indicator_results)
        else:
            effective_params = {}

        # Step 8: 执行约束检查 (冷却/连亏/去重) — 产出 ExecutionConstraint
        #         约束不改 level 本身, 但为向后兼容, 受约束的信号仍降为 NEUTRAL
        execution = ExecutionConstraint()

        if level.is_actionable and level.is_bullish:
            gc_cooldown = effective_params.get("cooldown_days", None) if effective_params else None
            if self.filter.is_in_cooldown(symbol, analysis_date, True, group_cooldown=gc_cooldown):
                execution.in_cooldown = True
                execution.cooldown_reason = f"冷却期内(卖出后{gc_cooldown or self.filter.cooldown_days}天禁止开仓)"

        if level.is_actionable and level.is_bullish:
            max_cl = effective_params.get("max_consecutive_losses", 0) if effective_params else 0
            suspend_d = effective_params.get("consecutive_loss_suspend", 0) if effective_params else 0
            if self.filter.is_suspended(symbol, analysis_date, max_cl, suspend_d):
                execution.suspended = True
                execution.suspend_reason = f"连续亏损{max_cl}次，暂停{suspend_d}天"

        # 信号去重检查
        is_dup = self.filter.is_duplicate(symbol, level, analysis_date=analysis_date)
        if is_dup:
            execution.is_duplicate = True
            execution.duplicate_reason = f"信号去重({self.filter.dedup_days}天内已发同方向信号)"

        # 得分是否达标 (供展示层判断"得分不达标")
        sc_threshold = effective_params.get("score_threshold", 25) if effective_params else 25
        sc_ceiling = effective_params.get("score_ceiling", 0) if effective_params else 0
        execution.score_passes = sc_threshold <= abs(score) <= (sc_ceiling if sc_ceiling > 0 else 100)
        if not execution.score_passes:
            if abs(score) < sc_threshold:
                execution.score_reason = f"得分{score:+.1f}低于阈值{sc_threshold}"
            else:
                execution.score_reason = f"得分{score:+.1f}超过上限{sc_ceiling:.0f}"

        # 向后兼容: 执行约束阻断时, level 降为 NEUTRAL, 原因记入 block_detail
        # (后续展示层改造后, 可保留原 level 仅在 execution 中标记不可执行)
        if not execution.is_executable and level.is_bullish:
            level = SignalLevel.NEUTRAL
            block_detail = execution.blocking_reason

        # Step 9: 记录信号 (使用实际分析日期)
        self.filter.record(symbol, level, analysis_date=analysis_date)

        # Step 10: 计算置信度
        confidence = self._calc_confidence(level, indicator_results)

        # Step 11: 构建结果
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
            block_detail=block_detail,
            initial_level=clf_result.initial_level,
            demotion_chain=clf_result.demotion_chain,
            execution=execution,
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