"""
信号定级器 (SignalClassifier) — 7级信号的唯一定级权威

设计原则:
  1. 唯一定级权: 只有本模块有权决定 SignalLevel，其他模块(validator/filter)只产出"材料"
  2. 规则集中: 共振判定 + 得分门槛 + 降级规则全部在此，不在多处分散
  3. 递进而非跳变: 强买入必须先满足买入条件，避免"观望→强买入"跳跃
  4. 平滑降级: 降级支持完整链 STRONG_BUY→BUY→WEAK_BUY→NEUTRAL，而非只砍到观望
  5. 可测试: 纯函数式定级(给定材料→级别)，无副作用，不碰磁盘/历史

定级材料 (ClassificationInput):
  - score: 综合得分 (-100~+100)
  - category_consensus: 各类别方向汇总 {trend/strength/momentum/volume: CategorySummary}
  - indicator_results: 原始指标结果 (用于背离/超买等细节判定)
  - group_params: 分组参数 (score_threshold/ceiling 等)

定级流程:
  Step 1: 基础定级 — 按共振结构判定初始级别 (from validator 逻辑)
  Step 2: 得分门槛 — 低于 score_threshold 降级，超过 ceiling 按过热处理
  Step 3: 结构修正 — MACD背离/RSI超买等，按规则平滑降级(压一档)
  Step 4: 方向约束 — MA60空头区域不出看多信号，多头区域不出看空信号
  Step 5: 返回 (level, demotion_reason) — 降级原因独立返回，不污染 level

与现有模块的关系:
  - Validator: 退化为"共识计算器"，只返回 category_consensus，不再返回 level
  - SignalFilter: 不再接收/修改 level，只产出执行约束(冷却/连亏/去重) → ExecutionConstraint
  - SignalEngine: 编排流程，调用 classifier 得到 level，调用 filter 得到约束，组装 SignalResult
  - 展示层: 直接展示 level.label，执行约束单独展示，不再造 action 第二套标签

扩展点 (当前预留接口，后续实现):
  - 递进定级: 强买入需"已是买入 + 量价确认维持N天"
  - 共振确认期: 三重共振需连续维持2-3天才升强买入
  - 类别门槛提高: 某类偏多至少需2个指标看多
  - 纳入强度类: ADX突破25作为强买入必要条件
"""
from dataclasses import dataclass, field
from typing import Dict, Optional, Tuple, List

from src.indicators.base import IndicatorResult
from src.signal_engine.signals import SignalLevel, CategorySummary


@dataclass
class ClassificationInput:
    """定级材料 — 交给 classifier 的全部输入"""
    score: float                                              # 综合得分 (-100~+100)
    category_consensus: Dict[str, CategorySummary]            # 类别共识 {trend/strength/momentum/volume}
    indicator_results: Dict[str, IndicatorResult]             # 原始指标结果
    group_params: dict = field(default_factory=dict)          # 分组参数
    hard_blocked: bool = False                                # 是否被硬过滤(ADX/数据不足)
    block_reason: str = ""                                    # 硬过滤原因


@dataclass
class ExecutionConstraint:
    """执行约束 — filter 产出，不影响 level，只影响"能否操作"

    与 level 解耦: 一只股票可以是"强买入"但"冷却中不可执行"
    展示时 level 和 constraint 分开呈现，不再合并成 action 标签
    """
    in_cooldown: bool = False            # 冷却期内
    cooldown_reason: str = ""            # 冷却原因
    suspended: bool = False              # 连亏暂停
    suspend_reason: str = ""             # 暂停原因
    is_duplicate: bool = False           # 信号去重(5天内已发同方向)
    duplicate_reason: str = ""           # 去重原因
    score_passes: bool = True            # 得分是否在 [threshold, ceiling] 区间
    score_reason: str = ""               # 得分不通过原因

    @property
    def is_executable(self) -> bool:
        """当前信号是否可执行(无任何约束阻断)"""
        return not (self.in_cooldown or self.suspended or self.is_duplicate) and self.score_passes

    @property
    def blocking_reason(self) -> str:
        """阻断执行的首要原因(空字符串=可执行)"""
        if not self.score_passes:
            return self.score_reason
        if self.in_cooldown:
            return self.cooldown_reason
        if self.suspended:
            return self.suspend_reason
        if self.is_duplicate:
            return self.duplicate_reason
        return ""


@dataclass
class ClassificationResult:
    """定级结果"""
    level: SignalLevel                 # 最终信号级别(7级之一)
    initial_level: SignalLevel         # 基础定级(共振结构判定，未经修正)
    demotion_reason: str = ""          # 降级原因(空字符串=未降级)
    demotion_chain: List[str] = field(default_factory=list)  # 降级路径记录(调试用)


class SignalClassifier:
    """信号定级器 — 7级信号的唯一判定权威

    使用方式:
        classifier = SignalClassifier()
        result = classifier.classify(ClassificationInput(
            score=48.7,
            category_consensus={...},
            indicator_results={...},
            group_params={"score_threshold": 25, "score_ceiling": 52},
        ))
        print(result.level.label)  # "强买入"
    """

    def classify(self, material: ClassificationInput) -> ClassificationResult:
        """
        执行完整定级流程

        Args:
            material: 定级材料

        Returns:
            ClassificationResult 含最终级别 + 降级记录
        """
        chain: List[str] = []

        # ── Step 0: 硬过滤拦截 → 直接观望 ──
        if material.hard_blocked:
            return ClassificationResult(
                level=SignalLevel.NEUTRAL,
                initial_level=SignalLevel.NEUTRAL,
                demotion_reason=f"硬过滤拦截: {material.block_reason}",
                demotion_chain=["硬过滤→观望"],
            )

        # ── Step 1: 基础定级(共振结构) ──
        initial = self._classify_by_consensus(material.category_consensus)
        level = initial
        if level != initial:
            chain.append(f"{initial.label}→{level.label}")

        # ── Step 2: 得分门槛 ──
        level, reason = self._apply_score_gate(level, material)
        if reason:
            chain.append(reason)

        # ── Step 3: 结构修正(MACD背离/RSI超买等平滑降级) ──
        level, reason = self._apply_structure_corrections(level, material)
        if reason:
            chain.append(reason)

        # ── Step 4: 方向约束(MA60多空区域) ──
        level, reason = self._apply_direction_constraint(level, material)
        if reason:
            chain.append(reason)

        # ── Step 5: 过热信号处理(得分超过ceiling) ──
        level, reason = self._apply_overheat_gate(level, material)
        if reason:
            chain.append(reason)

        demotion_reason = " | ".join(chain) if chain else ""
        return ClassificationResult(
            level=level,
            initial_level=initial,
            demotion_reason=demotion_reason,
            demotion_chain=chain,
        )

    # ──────────────────────────────────────────────────────
    #  Step 1: 基础定级 — 共振结构判定
    # ──────────────────────────────────────────────────────
    def _classify_by_consensus(self, consensus: Dict[str, CategorySummary]) -> SignalLevel:
        """
        按类别共振结构判定初始级别

        规则(从 validator.py 迁移，保持向后兼容):
          强买入: 趋势+动量+量价 三类同时偏多 (三重共振)
          买入:   趋势 + 至少1类其他偏多，或 强度 + 至少1类其他偏多
          弱买入: 单类别≥2指标看多，或 动量+量价偏多(无趋势)
          观望:   多空矛盾或无方向
          (卖出侧对称)

        TODO 扩展点(后续实现，当前保持兼容):
          - 强买入需"已是买入 + 量价确认"的递进关系，而非并列条件
          - 三重共振需连续维持N天(需传入历史材料)
          - 某类偏多至少需2个指标看多(当前1个即可)
        """
        cat_dirs = {k: v.direction for k, v in consensus.items()}
        bull_cats = [k for k, v in cat_dirs.items() if v > 0]
        bear_cats = [k for k, v in cat_dirs.items() if v < 0]
        nb, nber = len(bull_cats), len(bear_cats)

        # 三重共振
        if nb >= 3 and "trend" in bull_cats and "momentum" in bull_cats and "volume" in bull_cats:
            return SignalLevel.STRONG_BUY
        if nber >= 3 and "trend" in bear_cats and "momentum" in bear_cats and "volume" in bear_cats:
            return SignalLevel.STRONG_SELL

        # 趋势 + 至少1类其他
        if "trend" in bull_cats and nb >= 2:
            return SignalLevel.BUY
        if "trend" in bear_cats and nber >= 2:
            return SignalLevel.SELL

        # 强度 + 至少1类其他
        if "strength" in bull_cats and nb >= 2:
            return SignalLevel.BUY
        if "strength" in bear_cats and nber >= 2:
            return SignalLevel.SELL

        # 单类别强信号(≥2指标看多)
        for cat, summary in consensus.items():
            if summary.consensus >= 2 and summary.dissensus == 0:
                if summary.direction > 0:
                    return SignalLevel.WEAK_BUY
                else:
                    return SignalLevel.WEAK_SELL

        # 动量+量价(无趋势)
        if "momentum" in bull_cats and "volume" in bull_cats:
            return SignalLevel.WEAK_BUY
        if "momentum" in bear_cats and "volume" in bear_cats:
            return SignalLevel.WEAK_SELL

        return SignalLevel.NEUTRAL

    # ──────────────────────────────────────────────────────
    #  Step 2: 得分门槛
    # ──────────────────────────────────────────────────────
    def _apply_score_gate(self, level: SignalLevel,
                          material: ClassificationInput) -> Tuple[SignalLevel, str]:
        """
        得分低于 score_threshold → 降级处理

        优化逻辑 (v2):
          - BUY 及以上 (含强买入): 结构共振已足够强, 得分门槛不砍, 保留级别
          - WEAK_BUY: 得分低于阈值 → 降为观望 (弱信号需得分确认)
          - 卖出侧对称: SELL 及以上保留, WEAK_SELL 受门槛约束

        理由: 得分受量价类单指标偏空等因素拖累, 可能低于门槛,
              但只要多类共振结构成立(趋势+至少1类), 信号本身是可靠的。
              得分门槛应只过滤"单类别弱信号", 不应误杀"多类共振达标"的信号。
        """
        # BUY/STRONG_BUY/SELL/STRONG_SELL: 结构共振已达标, 不受得分门槛约束
        if abs(level.value) >= 2:
            return level, ""

        # WEAK_BUY/WEAK_SELL: 弱信号需得分确认
        threshold = material.group_params.get("score_threshold", 25)
        if abs(material.score) < threshold:
            return SignalLevel.NEUTRAL, f"得分{material.score:+.1f}低于阈值{threshold}→观望"
        return level, ""

    # ──────────────────────────────────────────────────────
    #  Step 3: 结构修正 — 平滑降级(压一档而非砍观望)
    # ──────────────────────────────────────────────────────
    def _apply_structure_corrections(self, level: SignalLevel,
                                     material: ClassificationInput) -> Tuple[SignalLevel, str]:
        """
        MACD顶背离/RSI超买等结构问题 → 压一档降级

        与原 filter.apply_hard_constraint 的区别:
          - 原逻辑: 强买入→观望, 买入→观望 (只能砍观望)
          - 新逻辑: 强买入→买入, 买入→弱买入 (平滑降级链)
        """
        if not level.is_bullish:
            return level, ""

        corrections = []
        result_level = level

        # MACD顶背离: 强买入→买入, 买入→弱买入
        macd = material.indicator_results.get("MACD")
        if macd is not None and macd.values.get("bearish_divergence", False):
            if result_level == SignalLevel.STRONG_BUY:
                result_level = SignalLevel.BUY
                corrections.append("MACD顶背离:强买入→买入")
            elif result_level == SignalLevel.BUY:
                result_level = SignalLevel.WEAK_BUY
                corrections.append("MACD顶背离:买入→弱买入")

        # TODO 扩展点: 量价未确认时强买入→买入
        # TODO 扩展点: 共振未维持N天时强买入→买入

        return result_level, " | ".join(corrections)

    # ──────────────────────────────────────────────────────
    #  Step 4: 方向约束 — MA60多空区域
    # ──────────────────────────────────────────────────────
    def _apply_direction_constraint(self, level: SignalLevel,
                                    material: ClassificationInput) -> Tuple[SignalLevel, str]:
        """
        MA60方向约束 — 多空区域与信号方向相反时直接降为观望

        原逻辑恢复:
          - 看多信号在MA60空头区域 → 直接降为观望(不买入)
          - 看空信号在MA60多头区域 → 直接降为观望(不卖出)
          - MA60为滞后指标, 方向相反时代表趋势不支撑, 应硬性过滤
        """
        ma60 = material.indicator_results.get("MA60")
        if ma60 is None:
            return level, ""

        # 看多信号在MA60空头区域 → 直接降为观望
        if ma60.direction == -1 and level.is_bullish:
            return SignalLevel.NEUTRAL, "价格在MA60下方(空头区域)→观望"

        # 看空信号在MA60多头区域 → 直接降为观望
        if ma60.direction == 1 and level.is_bearish:
            return SignalLevel.NEUTRAL, "价格在MA60上方(多头区域)→观望"

        return level, ""

    # ──────────────────────────────────────────────────────
    #  Step 5: 过热信号处理
    # ──────────────────────────────────────────────────────
    def _apply_overheat_gate(self, level: SignalLevel,
                             material: ClassificationInput) -> Tuple[SignalLevel, str]:
        """
        得分超过 score_ceiling → 视为过热，降为观望

        保留原 filter 的动态ceiling逻辑(突破确认+均线排列加成)
        TODO: 动态ceiling计算需迁移 filter.calc_dynamic_ceiling (需 df，当前材料不含)
        """
        if not level.is_bullish:
            return level, ""

        ceiling = material.group_params.get("score_ceiling", 0)
        if ceiling <= 0:
            return level, ""

        if material.score > ceiling:
            # TODO: 接入动态ceiling(突破确认可放宽)，当前用静态ceiling
            return SignalLevel.NEUTRAL, f"得分{material.score:+.1f}超过上限{ceiling:.0f}(过热)→观望"

        return level, ""

    # ──────────────────────────────────────────────────────
    #  辅助: 级别降一档(平滑降级链)
    # ──────────────────────────────────────────────────────
    @staticmethod
    def _demote_one_step(level: SignalLevel) -> SignalLevel:
        """级别降一档(用于平滑降级)

        强买入→买入→弱买入→观望→弱卖出→卖出→强卖出
        """
        mapping = {
            SignalLevel.STRONG_BUY: SignalLevel.BUY,
            SignalLevel.BUY: SignalLevel.WEAK_BUY,
            SignalLevel.WEAK_BUY: SignalLevel.NEUTRAL,
            SignalLevel.NEUTRAL: SignalLevel.NEUTRAL,  # 观望不降
            SignalLevel.WEAK_SELL: SignalLevel.NEUTRAL,
            SignalLevel.SELL: SignalLevel.WEAK_SELL,
            SignalLevel.STRONG_SELL: SignalLevel.SELL,
        }
        return mapping.get(level, SignalLevel.NEUTRAL)
