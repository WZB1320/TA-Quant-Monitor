"""策略自进化 — P8 自动应用器 (dry-run A/B)

定位 (第一性原理补齐):
  P1-P7 建成了记忆层 (观察→提炼→验证→谱系), 但记忆无法反向驱动参数变异。
  本模块补上最后一环: 把 validated 教训的 rule_action 程序化地应用为候选参数,
  与基线同窗口 A/B 双跑, 用真实回测结果做选择。

dry-run 边界 (当前版本):
  - 只消费 validation_status=validated 的教训 (pending/applied 不动)
  - 候选参数只在本进程内临时生效 (config_override + finally 恢复), 不写回
    strategy_config.json
  - 候选跑不写记忆层 (record_memory=False), 避免污染 L1/正样本/运行基线
  - 双跑结果均写入版本谱系 (candidate 以 baseline 为 parent), 供追溯

胜出判定 (与 P4 验证器同阈值风格):
  alpha_delta = 候选alpha - 基线alpha
  >= +alpha_improve  → win
  <= alpha_worsen    → lose
  其余               → inconclusive

用法:
    from src.evolution.auto_applier import EvolutionAB

    ab = EvolutionAB()
    report = ab.run_ab(start_date="2026-01-01", end_date="2026-06-30")
    # report: {"baseline": {...}, "candidate": {...}, "verdict": "win|lose|inconclusive",
    #          "changes": [...], "skipped": [...], ...}
"""
import copy
import json
import logging
import os
from typing import Dict, List, Optional

from src.config.group_config import GroupConfig
from src.memory.lesson_library import LessonLibrary
from src.memory.rule_action import apply_rule_actions, validate_rule_action
from src.memory.version_lineage import VersionLineage
from src.memory.strategy_memory import compute_strategy_version

logger = logging.getLogger(__name__)

# 与 P4 SuggestionValidator 保持一致的判定阈值风格
_DEFAULT_THRESHOLDS = {"alpha_improve": 0.01, "alpha_worsen": -0.01}


def compare_runs(baseline_metrics: dict,
                 candidate_metrics: dict,
                 thresholds: Optional[dict] = None) -> dict:
    """纯函数: 对比基线与候选的运行指标, 给出选择判定

    Args:
        baseline_metrics / candidate_metrics: run_portfolio_backtest 返回的
            result["metrics"] (至少含 alpha; 缺失字段按 None 处理)

    Returns:
        {"alpha_delta": float|None, "verdict": "win"|"lose"|"inconclusive"|"error",
         "reason": str}
    """
    t = {**_DEFAULT_THRESHOLDS, **(thresholds or {})}
    base_alpha = baseline_metrics.get("alpha")
    cand_alpha = candidate_metrics.get("alpha")
    if base_alpha is None or cand_alpha is None:
        return {"alpha_delta": None, "verdict": "error",
                "reason": "缺少 alpha 指标, 无法比较"}

    delta = round(cand_alpha - base_alpha, 6)
    if delta >= t["alpha_improve"]:
        verdict = "win"
        reason = f"候选 alpha 较基线提升 {delta:+.4f} (>= {t['alpha_improve']})"
    elif delta <= t["alpha_worsen"]:
        verdict = "lose"
        reason = f"候选 alpha 较基线下降 {delta:+.4f} (<= {t['alpha_worsen']})"
    else:
        verdict = "inconclusive"
        reason = f"候选 alpha 变化 {delta:+.4f}, 在判定区间内, 无法得出结论"
    return {"alpha_delta": delta, "verdict": verdict, "reason": reason}


def build_candidate(groups_cfg: dict,
                    active_groups: List[str],
                    lesson_library: Optional[LessonLibrary] = None,
                    statuses: tuple = ("validated",),
                    scope: Optional[dict] = None,
                    weights: Optional[dict] = None,
                    context: Optional[dict] = None) -> dict:
    """从教训库构建候选参数覆盖 (纯函数, 不修改入参)

    动作路由:
      - type == "rebalance_weight" → 组合级 weights (差额转给 compensate 组)
      - target == "atr_stop_mult" → 组合级 atr_override (引擎实际读这里, P5)
      - target == "trade_regimes" → 组合级 regimes_cfg
      - 其余 target 在组配置 (含一层嵌套如 mean_reversion_exit) 中查找修改
      - lesson.scope.group 为 None (通配) 时应用到全部活跃组;
        指定了 group 但不在活跃组中则跳过

    缺口 4: context 非空时, 传给 lesson_library.get_applicable_actions
    前置检测 invalidation_criteria (前提失效的教训自动 invalidated 并跳过)。
    context 结构参见 LessonLibrary.check_invalidation_criteria docstring。

    Returns:
        {"groups": {组名: 修改后的完整组配置},   # 传给 config_override["groups"]
         "atr_override": {组名: 新值},
         "regimes_cfg": {组名: 体制列表},
         "weights": {组名: 新权重},              # 传给 config_override["weights"]
         "changes": [...], "skipped": [...],
         "lesson_ids": [...]}                     # 贡献动作的教训
        无任何可用动作时各覆盖为空 dict
    """
    lib = lesson_library or LessonLibrary()
    actions = lib.get_applicable_actions(
        scope or {}, statuses=statuses, context=context)

    candidate: Dict[str, dict] = {"groups": {}, "atr_override": {},
                                  "regimes_cfg": {}, "weights": {},
                                  "changes": [], "skipped": [], "lesson_ids": []}
    if not actions:
        return candidate

    # lesson_id → 应用到哪些组
    # working_weights: 循环累计当前候选权重, 解决多条 rebalance_weight 同时生效时
    # 各自独立读原始 weights 导致总权重超额 (缺口 1)
    working_weights = dict(weights or {})

    for item in actions:
        lesson_id = item["lesson_id"]
        action = item["rule_action"]
        lesson_scope = item.get("scope") or {}
        target_group = lesson_scope.get("group")

        # ── rebalance_weight → 组合级 weights (与组配置无关, 先于组过滤处理) ──
        if action.get("type") == "rebalance_weight":
            try:
                validate_rule_action(action)
            except ValueError as e:
                candidate["skipped"].append({
                    "lesson_id": lesson_id, "action": action, "reason": str(e)})
                continue
            tgt, comp = action["target"], action["compensate"]

            # 缺口 2 修复: rebalance_weight 分支之前跳过所有 scope 校验,
            # 教训 scope 写了"消费组"但 action target 写"医药组"会静默通过.
            # 现在要求: scope.group 指定了组时, 必须等于 action.target
            # (target 是"权重释放方", 应该就是 scope 指向的问题组;
            #  compensate 是让渡方, 不在 scope 约束内)
            if target_group and tgt != target_group:
                candidate["skipped"].append({
                    "lesson_id": lesson_id, "action": action,
                    "reason": f"scope.group={target_group!r} 与 rebalance_weight.target={tgt!r} 不匹配",
                })
                continue

            base = working_weights
            if tgt not in base or comp not in base:
                candidate["skipped"].append({
                    "lesson_id": lesson_id, "action": action,
                    "reason": f"权重基线缺失 ({tgt}/{comp} 不在组合权重中)"})
                continue
            new_w = float(action["value"])
            # 让渡量 = target 释放出的权重 (new_w < base 时为正, 表示要补给别人)
            transferred = base[tgt] - new_w
            comp_new = base[comp] + transferred
            if comp_new < 0:
                candidate["skipped"].append({
                    "lesson_id": lesson_id, "action": action,
                    "reason": f"承接组 {comp} 权重将变为负值 ({comp_new:.4f}), 无法 rebalance"})
                continue
            candidate["weights"][tgt] = new_w
            candidate["weights"][comp] = round(comp_new, 4)
            # 缺口 1 修复: 把本次改动写回 working_weights, 后续教训基于累计值计算
            working_weights[tgt] = new_w
            working_weights[comp] = comp_new
            candidate["lesson_ids"].append(lesson_id)
            candidate["changes"].append({
                "lesson_id": lesson_id, "group": tgt, "target": "weight",
                "via": "portfolio_weights",
                "before": base[tgt], "after": round(new_w, 4),
            })
            candidate["changes"].append({
                "lesson_id": lesson_id, "group": comp, "target": "weight",
                "via": "portfolio_weights",
                "before": base[comp], "after": round(comp_new, 4),
            })
            continue

        if target_group is None:
            groups = list(active_groups)
        elif target_group in active_groups:
            groups = [target_group]
        else:
            candidate["skipped"].append({
                "lesson_id": lesson_id, "action": action,
                "reason": f"scope.group={target_group!r} 不在活跃组中",
            })
            continue

        # scope.strategy_mode 非空时, 跨组通配也要按策略模式过滤,
        # 避免 mean_reversion 教训误伤 trend_following 组
        scope_mode = lesson_scope.get("strategy_mode")
        if scope_mode:
            filtered = [g for g in groups
                        if (groups_cfg.get(g) or {}).get(
                            "strategy_mode", "trend_following") == scope_mode]
            dropped = [g for g in groups if g not in filtered]
            if dropped:
                candidate["skipped"].append({
                    "lesson_id": lesson_id, "action": action,
                    "reason": f"strategy_mode 不匹配 (需 {scope_mode!r}), 跳过组: {dropped}",
                })
            groups = filtered
        if not groups:
            continue

        candidate["lesson_ids"].append(lesson_id)
        for g in groups:
            # ── trade_regimes → 组合级 regimes_cfg 路径 ──
            if action.get("target") == "trade_regimes":
                val = action.get("value")
                if action.get("type") != "set_param":
                    candidate["skipped"].append({
                        "lesson_id": lesson_id, "action": action, "group": g,
                        "reason": "regimes_cfg 路径只支持 set_param (值是列表, 无法 adjust)",
                    })
                    continue
                candidate["regimes_cfg"][g] = list(val) if val else None
                candidate["changes"].append({
                    "lesson_id": lesson_id, "group": g,
                    "target": "trade_regimes", "via": "regimes_cfg",
                    "before": None, "after": val,
                })
                continue

            # ── atr_stop_mult → 组合级 atr_override (P5) ──
            if action.get("target") == "atr_stop_mult":
                # atr_stop_mult 的真实生效路径是组合级 atr_override (P5)
                before = None  # 由调用方对照基线; 这里只表达"设为该值"
                if action.get("type") == "set_param":
                    new_val = action.get("value")
                else:
                    # adjust_param 需要当前值; 组合级覆盖无基线上下文, 跳过
                    candidate["skipped"].append({
                        "lesson_id": lesson_id, "action": action,
                        "reason": "atr_override 路径不支持 adjust_param (无基线值)",
                    })
                    continue
                candidate["atr_override"][g] = new_val
                candidate["changes"].append({
                    "lesson_id": lesson_id, "group": g,
                    "target": "atr_stop_mult", "via": "atr_override",
                    "before": before, "after": new_val,
                })
                continue

            base_cfg = groups_cfg.get(g)
            if not isinstance(base_cfg, dict):
                candidate["skipped"].append({
                    "lesson_id": lesson_id, "action": action,
                    "reason": f"组 {g} 无配置, 无法应用",
                })
                continue
            result = apply_rule_actions(copy.deepcopy(base_cfg), [action])
            if result["changes"]:
                candidate["groups"][g] = result["params"]
                for ch in result["changes"]:
                    candidate["changes"].append({
                        "lesson_id": lesson_id, "group": g,
                        "target": ch["target"], "via": "group_config",
                        "before": ch["before"], "after": ch["after"],
                    })
            for sk in result["skipped"]:
                candidate["skipped"].append({
                    "lesson_id": lesson_id, "group": g,
                    "action": action, "reason": sk["reason"],
                })
    return candidate


class EvolutionAB:
    """教训驱动的 A/B 双跑 (dry-run: 不写回配置)"""

    def __init__(self,
                 lesson_library: Optional[LessonLibrary] = None,
                 lineage: Optional[VersionLineage] = None,
                 thresholds: Optional[dict] = None):
        self.lesson_library = lesson_library or LessonLibrary()
        self.lineage = lineage or VersionLineage()
        self.thresholds = {**_DEFAULT_THRESHOLDS, **(thresholds or {})}

    def run_ab(self,
               start_date: Optional[str] = None,
               end_date: Optional[str] = None,
               mode: str = "base",
               initial_capital: float = 100000,
               statuses: tuple = ("validated", "applied")) -> dict:
        """基线 vs 候选 同窗口双跑

        基线跑正常记录记忆层 (它就是生产运行); 候选跑 record_memory=False。

        Returns:
            {"has_candidate": bool, "baseline": {...}, "candidate": {...},
             "verdict": ..., "changes": [...], "skipped": [...],
             "lesson_ids": [...], "lineage_recorded": bool}
        """
        # 延迟导入: 避免循环依赖 (portfolio_backtest 也 import 记忆层)
        from web.backend.services.portfolio_backtest import (
            BacktestRequest, run_portfolio_backtest, _load_portfolio_config,
        )

        req = BacktestRequest(mode=mode, start_date=start_date, end_date=end_date,
                              initial_capital=initial_capital)

        # 1. 基线跑 (生产参数, 正常记录)
        baseline_result = run_portfolio_backtest(req, record_memory=True)
        if baseline_result.get("status") != "done":
            return {"has_candidate": False, "verdict": "error",
                    "reason": f"基线回测失败: {baseline_result.get('message')}"}

        # 2. 构建候选 (用基线跑结束后的磁盘配置; A/B 期间配置页不改动的假设下等价)
        gc = GroupConfig()
        gc.reload()
        pc = _load_portfolio_config()
        active_groups = [g for g, w in pc["weights"].items() if w > 0]

        # 缺口 4: 构造 context 传给 build_candidate, 前置检测 invalidation_criteria
        # (本金/权重/策略模式变了, 教训前提可能已失效)
        context = {
            "capital_per_group": {
                g: initial_capital * w for g, w in pc["weights"].items()
            },
            "portfolio_weight": dict(pc["weights"]),
            "position_ratio": 0.3,  # 引擎硬编码值 (portfolio_backtest.py)
            "total_initial_capital": initial_capital,
            "strategy_mode": {
                g: (gc._groups.get(g) or {}).get("strategy_mode", "trend_following")
                for g in pc["weights"].keys()
            },
        }

        candidate = build_candidate(
            dict(gc._groups), active_groups,
            lesson_library=self.lesson_library, statuses=statuses,
            weights=pc["weights"], context=context)

        if not candidate["groups"] and not candidate["atr_override"] \
                and not candidate["regimes_cfg"] and not candidate["weights"]:
            return {
                "has_candidate": False, "verdict": "no_candidate",
                "reason": "没有可执行的 validated/applied 教训动作",
                "baseline": self._summarize(baseline_result),
                "changes": [], "skipped": candidate["skipped"],
                "lesson_ids": [],
            }

        config_override = {}
        if candidate["groups"]:
            config_override["groups"] = candidate["groups"]
        if candidate["atr_override"]:
            config_override["atr_override"] = candidate["atr_override"]
        if candidate["regimes_cfg"]:
            config_override["regimes_cfg"] = candidate["regimes_cfg"]
        if candidate["weights"]:
            config_override["weights"] = candidate["weights"]

        # 3. 候选跑 (不写记忆层)
        candidate_result = run_portfolio_backtest(
            req, config_override=config_override, record_memory=False)
        if candidate_result.get("status") != "done":
            return {"has_candidate": True, "verdict": "error",
                    "reason": f"候选回测失败: {candidate_result.get('message')}",
                    "baseline": self._summarize(baseline_result),
                    "changes": candidate["changes"], "skipped": candidate["skipped"],
                    "lesson_ids": candidate["lesson_ids"]}

        # 4. 选择判定: 组合级 + 受影响组组级双视角
        #    组合级 alpha 会被大权重组(科技/周期)淹没, 必须额外看受影响组的组级 alpha
        comparison = compare_runs(
            baseline_result["metrics"], candidate_result["metrics"], self.thresholds)
        group_verdict = self._group_level_verdict(
            baseline_result, candidate_result, candidate,
            weights=pc["weights"], thresholds=self.thresholds,
            portfolio_comparison=comparison)

        # 5. 谱系: 候选版本以基线版本为 parent (dry-run 也留痕, 供追溯)
        lineage_recorded = self._record_lineage(
            baseline_result, candidate_result, candidate, req, comparison)

        # 6. 验证闭环: A/B 判定直接回写教训状态 (applied → validated / invalidated)
        #    以组级判定为准, 因为教训假设针对的是受影响组
        verdict = group_verdict["verdict"]
        lesson_status_updates = self._apply_verdict_to_lessons(
            verdict=verdict,
            lesson_ids=candidate["lesson_ids"],
            run_id=candidate_result.get("run_id"),
            alpha_delta=group_verdict["alpha_delta"],
            reason=group_verdict["reason"],
        )

        return {
            "has_candidate": True,
            "baseline": self._summarize(baseline_result),
            "candidate": self._summarize(candidate_result),
            "verdict": verdict,
            "portfolio_verdict": comparison["verdict"],
            "portfolio_alpha_delta": comparison["alpha_delta"],
            "alpha_delta": group_verdict["alpha_delta"],
            "group_alpha_detail": group_verdict["detail"],
            "reason": group_verdict["reason"],
            "changes": candidate["changes"],
            "skipped": candidate["skipped"],
            "lesson_ids": candidate["lesson_ids"],
            "lesson_status_updates": lesson_status_updates,
            "lineage_recorded": lineage_recorded,
            "note": "dry-run: 候选参数未写回 strategy_config.json",
        }

    @staticmethod
    def _summarize(result: dict) -> dict:
        return {
            "run_id": result.get("run_id"),
            "strategy_version": result.get("strategy_version"),
            "metrics": result.get("metrics"),
        }

    @staticmethod
    def _group_level_verdict(baseline_result: dict, candidate_result: dict,
                             candidate: dict, weights: dict, thresholds: dict,
                             portfolio_comparison: Optional[dict] = None) -> dict:
        """只看受影响组(候选改动涉及的组)的组级 alpha 变化, 给出判定.

        组合级 alpha 会被大权重组淹没, 无法检出小权重组(消费/医药各5%)的改进.
        这里取受影响组的 alpha 变化均值, 与阈值比较.

        例外: 权重再平衡 (rebalance_weight) 改变的是资金分配而非组内策略,
        组级 alpha 是比率指标、对权重不敏感, 组级视角必然得到 0 变化;
        此时必须以组合级 alpha 为准 (这正是权重调整要优化的目标).
        """
        if any(c.get("via") == "portfolio_weights"
               for c in candidate.get("changes", [])) and portfolio_comparison:
            return {
                "verdict": portfolio_comparison["verdict"],
                "alpha_delta": portfolio_comparison["alpha_delta"],
                "reason": f"权重再平衡, 以组合级 alpha 判定: "
                          f"{portfolio_comparison['reason']}",
                "detail": {},
            }
        affected = {c.get("group") for c in candidate.get("changes", [])
                    if c.get("group")}
        base_gm = baseline_result.get("group_metrics") or {}
        cand_gm = candidate_result.get("group_metrics") or {}
        detail = {}
        deltas = []
        for g in affected:
            b = (base_gm.get(g) or {}).get("alpha")
            c = (cand_gm.get(g) or {}).get("alpha")
            if b is None or c is None:
                continue
            d = c - b
            deltas.append(d)
            detail[g] = {"before": b, "after": c, "delta": round(d, 6),
                         "weight": weights.get(g, 0)}
        if not deltas:
            return {"verdict": "inconclusive", "alpha_delta": 0.0,
                    "reason": "受影响组无有效组级 alpha", "detail": detail}
        mean_delta = sum(deltas) / len(deltas)
        if mean_delta > thresholds["alpha_improve"]:
            verdict = "win"
        elif mean_delta < thresholds["alpha_worsen"]:
            verdict = "lose"
        else:
            verdict = "inconclusive"
        return {
            "verdict": verdict,
            "alpha_delta": round(mean_delta, 6),
            "reason": f"受影响组({len(deltas)}个)平均 alpha 变化 {mean_delta:+.4f}",
            "detail": detail,
        }

    def _apply_verdict_to_lessons(self, *, verdict: str, lesson_ids: list,
                                  run_id: Optional[str],
                                  alpha_delta: float, reason: str) -> list:
        """A/B 判定回写教训状态 (缺口 3)

        状态转移:
          - win  → applied → validated, 重置 inconclusive_count
          - lose → applied → invalidated, 重置 inconclusive_count
          - inconclusive → 调 mark_inconclusive 递增计数, 连续 >= 3 次自动 invalidated
        """
        rid = run_id or "ab_unknown"
        updates = []
        for lid in lesson_ids:
            try:
                lesson = self.lesson_library.get_lesson(lid)
                if lesson is None:
                    continue
                status = lesson.get("validation_status")
                notes = f"A/B verdict={verdict}, group_alpha_delta={alpha_delta:+.4f}, {reason}"

                if verdict == "win":
                    if status == "applied":
                        self.lesson_library.mark_validated(lid, rid, notes=notes)
                        updates.append({"lesson_id": lid, "status": "validated"})
                elif verdict == "lose":
                    if status == "applied":
                        self.lesson_library.mark_invalidated(lid, rid, notes=notes)
                        updates.append({"lesson_id": lid, "status": "invalidated"})
                else:  # inconclusive
                    # 缺口 3: 记录 inconclusive, 连续 >= 阈值自动降级
                    result = self.lesson_library.mark_inconclusive(
                        lid, rid, notes=notes)
                    if result.get("auto_invalidated"):
                        updates.append({
                            "lesson_id": lid,
                            "status": "invalidated",
                            "reason": f"连续 {result['count']} 次 inconclusive 自动失效",
                        })
                    # 否则 count 递增但状态保持 applied, 不写 updates
            except Exception as e:
                logger.warning("教训 %s 状态回写失败: %s", lid, e)
        return updates

    def _record_lineage(self, baseline_result: dict, candidate_result: dict,
                        candidate: dict, req, comparison: dict) -> bool:
        """把 A/B 双跑写入版本谱系; 失败只告警不抛出"""
        try:
            base_version = baseline_result.get("strategy_version")
            cand_metrics = candidate_result.get("metrics") or {}

            # 候选版本参数 = 基线参数 + 覆盖摘要 (保证与基线可 diff)
            version_params = {
                "mode": req.mode,
                "derived_from": base_version,
                "overrides": {
                    "groups": sorted(candidate["groups"].keys()),
                    "weights": dict(candidate.get("weights") or {}),
                    "atr_override": dict(candidate["atr_override"]),
                },
                "changes": candidate["changes"],
            }
            cand_version = candidate_result.get("strategy_version") \
                or compute_strategy_version(version_params)

            self.lineage.record_version(
                version=cand_version,
                params=version_params,
                parent_version=base_version,
                run_id=candidate_result.get("run_id"),
                mode=req.mode,
                metrics=cand_metrics,
            )
            # 教训应用链: 候选版本应用了哪些教训
            for lesson_id in dict.fromkeys(candidate["lesson_ids"]):
                self.lineage.record_lesson_applied(
                    version=cand_version,
                    lesson_id=lesson_id,
                    notes="P8 A/B dry-run candidate",
                )
            # 选择结论落谱系: win=validated / lose=invalidated, inconclusive 不记
            verdict = comparison.get("verdict")
            if verdict in ("win", "lose"):
                self.lineage.record_lesson_verdict(
                    version=cand_version,
                    lesson_id=candidate["lesson_ids"][0] if candidate["lesson_ids"]
                    else "unknown",
                    verdict="validated" if verdict == "win" else "invalidated",
                    notes=f"P8 A/B: {comparison.get('reason')}",
                    run_id=candidate_result.get("run_id"),
                    alpha_delta=comparison.get("alpha_delta"),
                )
            return True
        except Exception as e:
            logger.warning("A/B 谱系记录失败: %s", e)
            return False

    @staticmethod
    def _config_path() -> str:
        return os.path.abspath(os.path.join(
            os.path.dirname(__file__), "..", "..", "config", "strategy_config.json",
        ))

    def promote(self, statuses: tuple = ("validated",)) -> dict:
        """固化: 把已验证教训的 rule_action 写回 strategy_config.json

        步骤:
          1. build_candidate 从 validated 教训构建候选 (已固化 promoted_at 的会被跳过)
          2. 无可用动作直接返回
          3. 原子写 strategy_config.json (组配置 / atr_override / regimes_filter)
          4. GroupConfig.reload()
          5. 相关教训 mark_promoted

        Returns:
            {"promoted": [{lesson_id, changes...}], "written_config": bool}
        """
        gc = GroupConfig()
        gc.reload()
        active = list(gc._groups.keys())
        # 延迟导入: 避免循环依赖
        from web.backend.services.portfolio_backtest import _load_portfolio_config
        base_weights = _load_portfolio_config()["weights"]

        candidate = build_candidate(
            gc._groups, active,
            lesson_library=self.lesson_library, statuses=statuses,
            weights=base_weights,
        )
        if not candidate["groups"] and not candidate["atr_override"] \
                and not candidate["regimes_cfg"] and not candidate["weights"]:
            return {"promoted": [], "written_config": False,
                    "reason": "无可固化的已验证教训"}

        path = self._config_path()
        with open(path, "r", encoding="utf-8") as f:
            sc = json.load(f)

        groups_node = sc.setdefault("strategy_config", {}).setdefault(
            "group_config", {}).setdefault("groups", {})
        p5 = sc.setdefault("strategy_config", {}).setdefault(
            "portfolio_config_p5", {})

        promoted = []
        for g, cfg in candidate["groups"].items():
            groups_node[g] = cfg
            promoted.append({"lesson_ids": candidate["lesson_ids"],
                             "group": g, "kind": "group_config"})
        if candidate["atr_override"]:
            atr = p5.setdefault("atr_override", {})
            atr.update(candidate["atr_override"])
            promoted.append({"lesson_ids": candidate["lesson_ids"],
                             "kind": "atr_override",
                             "values": dict(candidate["atr_override"])})
        if candidate["regimes_cfg"]:
            reg = p5.setdefault("regimes_filter", {})
            reg.update(candidate["regimes_cfg"])
            promoted.append({"lesson_ids": candidate["lesson_ids"],
                             "kind": "regimes_filter",
                             "values": dict(candidate["regimes_cfg"])})
        if candidate["weights"]:
            w_node = p5.setdefault("weights", {})
            w_node.update(candidate["weights"])
            promoted.append({"lesson_ids": candidate["lesson_ids"],
                             "kind": "weights",
                             "values": dict(candidate["weights"])})

        # 原子写 + 热更新
        from web.backend.routers.storage import atomic_write_json
        atomic_write_json(path, sc)
        try:
            gc.reload()
        except Exception as e:
            logger.warning("GroupConfig.reload() 失败: %s", e)

        # 标记已固化
        run_id = f"promote_{candidate['lesson_ids'][0] if candidate['lesson_ids'] else 'unknown'}"
        for lid in candidate["lesson_ids"]:
            try:
                self.lesson_library.mark_promoted(lid, run_id)
            except Exception as e:
                logger.warning("教训 %s mark_promoted 失败: %s", lid, e)

        return {"promoted": promoted, "written_config": True,
                "lesson_ids": candidate["lesson_ids"]}
