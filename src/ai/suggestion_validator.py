"""建议自动验证器 (P4)

定位:
  建议被应用后, 之前没有任何生产代码调用 mark_validated/mark_invalidated,
  导致"选择"环节断裂 —— 建议永远停在 applied, 教训永远停在 pending。
  本模块在每次回测后自动对比"应用前 vs 应用后"的同窗口指标, 自动判定
  建议是否有效, 并把结论回写到 L2 教训。

数据来源:
  SuggestionTracker  data/report_suggestions.jsonl   (applied 建议 + applied_at)
  RunOutcomeStore    data/failure_cases/run_outcomes.jsonl (每次回测的组合级指标)
  LessonLibrary      data/failure_cases/lessons/*.json     (待回写的教训)

判定规则 (主指标 alpha):
  after.alpha - before.alpha >= alpha_improve  → validated
  after.alpha - before.alpha <= alpha_worsen   → invalidated
  其余 → inconclusive (保持 applied, 等下一次回测)

局限:
  v1 采用"时间顺序 + 同窗口"匹配, 属于观察性验证, 不排除同期其他变量干扰;
  窗口不一致或缺少 before/after 数据时不判定。
"""
import logging
import re
from typing import Dict, List, Optional

from .suggestion_tracker import SuggestionTracker
from src.memory.run_outcome import RunOutcomeStore

logger = logging.getLogger(__name__)

_DEFAULT_THRESHOLDS = {
    "alpha_improve": 0.01,    # alpha 改善 >= 1pp 判定为有效
    "alpha_worsen": -0.01,    # alpha 恶化 <= -1pp 判定为证伪
}

_DELTA_KEYS = ("alpha", "total_return", "max_drawdown", "sharpe_ratio", "win_rate")


def _delta(before: dict, after: dict) -> Dict[str, Optional[float]]:
    """计算关键指标的 after - before"""
    result = {}
    for key in _DELTA_KEYS:
        b, a = (before or {}).get(key), (after or {}).get(key)
        if b is None or a is None:
            result[key] = None
        else:
            try:
                result[key] = round(float(a) - float(b), 6)
            except (TypeError, ValueError):
                result[key] = None
    return result


def _lesson_mentions_target(lesson: dict, target: str) -> bool:
    """判断 lesson 是否作用于该参数名 (结构化 rule_action 优先, 再退文本匹配)"""
    if not target:
        return False
    action = lesson.get("rule_action") or {}
    if action.get("target") == target:
        return True
    pattern = re.compile(r"(?<![A-Za-z0-9_])" + re.escape(target) + r"(?![A-Za-z0-9_])")
    texts = [lesson.get("rule") or ""]
    texts.extend(str(x) for x in (lesson.get("do_not_repeat") or []))
    return any(pattern.search(t) for t in texts)


class SuggestionValidator:
    """自动验证已应用建议的效果, 并回写教训库"""

    def __init__(self, tracker: Optional[SuggestionTracker] = None,
                 outcome_store: Optional[RunOutcomeStore] = None,
                 lesson_library=None,
                 version_lineage=None,
                 thresholds: Optional[dict] = None):
        self.tracker = tracker or SuggestionTracker()
        self.outcome_store = outcome_store or RunOutcomeStore()
        self.lesson_library = lesson_library
        self.version_lineage = version_lineage
        self.thresholds = {**_DEFAULT_THRESHOLDS, **(thresholds or {})}

    # ── 主入口 ──

    def validate(self, dry_run: bool = True) -> dict:
        """验证所有 applied 状态的建议

        Args:
            dry_run: True 只返回结果不写状态 (默认, 便于人工预览)

        Returns:
            {checked, validated, invalidated, inconclusive, items, dry_run}
        """
        applied = self.tracker.load_applied_suggestions()
        items = []
        for suggestion in applied:
            items.append(self._validate_one(suggestion, dry_run))

        summary = {
            "checked": len(items),
            "validated": sum(1 for i in items if i["verdict"] == "validated"),
            "invalidated": sum(1 for i in items if i["verdict"] == "invalidated"),
            "inconclusive": sum(1 for i in items if i["verdict"] == "inconclusive"),
            "items": items,
            "dry_run": dry_run,
        }
        if not dry_run and (summary["validated"] or summary["invalidated"]):
            logger.info(
                "建议验证完成: validated=%d invalidated=%d inconclusive=%d",
                summary["validated"], summary["invalidated"], summary["inconclusive"])
        return summary

    # ── 单条验证 ──

    def _validate_one(self, suggestion: dict, dry_run: bool) -> dict:
        sid = suggestion.get("suggestion_id")
        target = suggestion.get("target", "")
        base = {
            "suggestion_id": sid,
            "target": target,
            "verdict": "inconclusive",
            "reason": "",
            "delta": {},
            "before_run_id": None,
            "after_run_id": None,
            "lessons_updated": [],
        }

        applied_at = suggestion.get("applied_at")
        if not applied_at:
            base["reason"] = "缺少 applied_at, 无法确定应用时间"
            return base

        # 1. 应用后最早的一次回测
        after = self.outcome_store.find_nearest_after(applied_at)
        if not after:
            base["reason"] = "应用后尚无新回测, 等待数据"
            return base
        base["after_run_id"] = after.get("run_id")

        # 2. 应用前同窗口的最近一次回测
        before = self.outcome_store.find_nearest_before(
            applied_at, window=after.get("window"), mode=after.get("mode"))
        if not before:
            base["reason"] = "应用前缺少同窗口基线回测"
            return base
        base["before_run_id"] = before.get("run_id")

        # 3. 对比
        delta = _delta(before.get("metrics") or {}, after.get("metrics") or {})
        base["delta"] = delta
        alpha_delta = delta.get("alpha")
        if alpha_delta is None:
            base["reason"] = "缺少 alpha 指标, 无法判定"
            return base

        if alpha_delta >= self.thresholds["alpha_improve"]:
            verdict = "validated"
            reason = f"alpha 改善 {alpha_delta:+.2%} ≥ {self.thresholds['alpha_improve']:.0%}"
        elif alpha_delta <= self.thresholds["alpha_worsen"]:
            verdict = "invalidated"
            reason = f"alpha 恶化 {alpha_delta:+.2%} ≤ {self.thresholds['alpha_worsen']:.0%}"
        else:
            verdict = "inconclusive"
            reason = f"alpha 变化 {alpha_delta:+.2%}, 未达判定阈值"

        base["verdict"] = verdict
        base["reason"] = reason

        validation = {
            "before_run_id": before.get("run_id"),
            "after_run_id": after.get("run_id"),
            "before_at": before.get("recorded_at"),
            "after_at": after.get("recorded_at"),
            "delta": delta,
            "verdict": verdict,
            "reason": reason,
            "thresholds": dict(self.thresholds),
        }

        if dry_run or verdict == "inconclusive":
            return base

        # 4. 写回建议状态
        if verdict == "validated":
            self.tracker.mark_validated(sid, validation)
        else:
            self.tracker.mark_invalidated(sid, validation)

        # 5. 回写匹配的 L2 教训
        base["lessons_updated"] = self._update_lessons(
            suggestion, verdict, after.get("run_id"), reason,
            version=after.get("strategy_version"), alpha_delta=alpha_delta)
        return base

    # ── L2 回写 ──

    def _update_lessons(self, suggestion: dict, verdict: str,
                        run_id: Optional[str], notes: str,
                        version: Optional[str] = None,
                        alpha_delta: Optional[float] = None) -> List[str]:
        if self.lesson_library is None:
            return []
        target = suggestion.get("target", "")
        updated = []
        try:
            lessons = self.lesson_library.list_lessons()
        except Exception as e:
            logger.warning("加载 L2 教训失败, 跳过回写: %s", e)
            return []

        for lesson in lessons:
            if not _lesson_mentions_target(lesson, target):
                continue
            status = lesson.get("validation_status")
            if status not in ("pending", "applied"):
                continue
            lid = lesson["lesson_id"]
            try:
                if status == "pending":
                    # 建议已应用即视为教训进入实践
                    self.lesson_library.mark_applied(lid, run_id or "auto")
                    self._record_lineage_applied(lesson, version)
                if verdict == "validated":
                    self.lesson_library.mark_validated(lid, run_id or "auto", notes=notes)
                else:
                    self.lesson_library.mark_invalidated(lid, run_id or "auto", notes=notes)
                self._record_lineage_verdict(
                    lid, verdict, version, run_id, notes, alpha_delta)
                updated.append(lid)
            except Exception as e:
                logger.warning("回写教训 %s 失败: %s", lid, e)
        return updated

    # ── 版本谱系回写 (P7) ──

    def _record_lineage_applied(self, lesson: dict, version: Optional[str]) -> None:
        if self.version_lineage is None or not version:
            return
        try:
            self.version_lineage.record_lesson_applied(
                version=version,
                lesson_id=lesson["lesson_id"],
                rule_action=lesson.get("rule_action"),
                notes="auto: suggestion applied",
            )
        except Exception as e:
            logger.warning("谱系记录教训应用失败: %s", e)

    def _record_lineage_verdict(self, lesson_id: str, verdict: str,
                                version: Optional[str], run_id: Optional[str],
                                notes: Optional[str],
                                alpha_delta: Optional[float]) -> None:
        if self.version_lineage is None or not version:
            return
        try:
            self.version_lineage.record_lesson_verdict(
                version=version,
                lesson_id=lesson_id,
                verdict=verdict,
                notes=notes,
                run_id=run_id,
                alpha_delta=alpha_delta,
            )
        except Exception as e:
            logger.warning("谱系记录验证结论失败: %s", e)
