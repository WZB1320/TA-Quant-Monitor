"""失效案例库 L2 — 教训条目 (curated lessons)

定位:
  L1 只记客观事实; L2 是从 >=2 条相似 L1 记录中提炼出的可执行教训,
  服务于"策略自进化" (告诉进化器哪些方向不要再试) 和"版本记忆"
  (记录已验证的适配特征)。

存储:
  data/failure_cases/lessons/{lesson_id}.json — 一案一文件 (与 backtest_memory 同范式)。
  单文件全量重写即可完成状态更新, 无需 JSONL rewrite。

状态机 (镜像 SuggestionTracker):
  pending   — 已提炼, 等待应用到策略参数
  applied   — rule 已应用到某个 strategy_version (run_id 记录于 applied_by)
  validated — applied 后, 后续 run 验证该教训有效 (alpha 改善 / 不再踩坑)
  invalidated — applied 后, 后续 run 证明无效或反而恶化

最小支持度:
  创建 lesson 时 linked_cases 必须 >= 2 (单条失败可能是噪声, 不构成教训)。
  confidence 由 len(linked_cases) 自动推导: 2=low, 3-4=med, 5+=high。

scope:
  {group, regime, strategy_mode} — 进化器按当前上下文检索相关教训;
  检索时 lesson.scope 中非 null 字段必须与查询值匹配 (null 表示通配)。

用法:
    from src.memory.lesson_library import LessonLibrary, load_l1_records

    lib = LessonLibrary()
    # 1. 加载相关 L1 记录
    cases = load_l1_records(case_ids=["fc_xxx", "fc_yyy"])
    # 2. AI 起草 root_cause (LLM 不可用时回退模板)
    root_cause = lib.draft_root_cause(cases)
    # 3. 创建教训
    lesson = lib.create_lesson(
        linked_cases=["fc_xxx", "fc_yyy"],
        scope={"group": "医药创新型", "regime": None, "strategy_mode": "mean_reversion"},
        failure_level="param",
        root_cause=root_cause,
        rule="score_threshold 从 30 调到 38",
        rule_action={"type": "set_param", "target": "score_threshold", "value": 38},
        do_not_repeat=["score_threshold <= 30 在医药组会触发大量止损"],
    )
    # 4. 应用 + 验证
    lib.mark_applied(lesson["lesson_id"], run_id="bt_2025...")
    lib.mark_validated(lesson["lesson_id"], run_id="bt_2025...", notes="alpha +2.1%")
    # 5. 进化器检索
    relevant = lib.find_relevant_lessons(
        scope={"group": "医药创新型", "strategy_mode": "mean_reversion"})
"""
import json
import logging
import os
import uuid
from datetime import datetime
from typing import Dict, List, Optional, Sequence

from .failure_case import FAILURE_FLAGS, FAILURE_LEVELS, _DEFAULT_STORE
from .regime_slices import dominant_regime
from .rule_action import validate_rule_action

logger = logging.getLogger(__name__)

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
_DEFAULT_LESSONS_DIR = os.path.join(_ROOT, "data", "failure_cases", "lessons")

LESSON_STATUSES = ("pending", "applied", "validated", "invalidated")
_CONFIDENCE_BY_SUPPORT = {2: "low", 3: "med", 4: "med"}

# 允许更新的字段 (防止随意修改不可变字段)
_MUTABLE_FIELDS = (
    "root_cause", "rule", "rule_action", "do_not_repeat", "scope",
    "failure_level", "failure_flags_summary",
)


def _confidence_for(support: int) -> str:
    if support >= 5:
        return "high"
    return _CONFIDENCE_BY_SUPPORT.get(support, "low")


# ─────────────────────────────────────────────────────────────
# L1 记录加载 (供 L2 起草时读取原料)
# ─────────────────────────────────────────────────────────────

def load_l1_records(case_ids: Optional[Sequence[str]] = None,
                    store_path: Optional[str] = None) -> List[dict]:
    """从 L1 JSONL 加载记录

    Args:
        case_ids: 指定 case_id 列表; None 表示加载全部
        store_path: L1 文件路径; 默认 data/failure_cases/l1_failures.jsonl

    Returns:
        L1 记录字典列表
    """
    path = store_path or _DEFAULT_STORE
    if not os.path.exists(path):
        return []
    wanted = set(case_ids) if case_ids else None
    records = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            if wanted is None or rec.get("case_id") in wanted:
                records.append(rec)
    return records


def _record_regime_slices(rec: dict) -> Optional[dict]:
    """从 L1 记录取 regime_slices (优先顶层字段, 回退 metrics 内嵌)"""
    slices = rec.get("regime_slices")
    if slices:
        return slices
    metrics = rec.get("metrics")
    if isinstance(metrics, dict):
        return metrics.get("regime_slices")
    return None


def infer_scope_from_records(records: Sequence[dict],
                             thresholds: Optional[dict] = None) -> dict:
    """从一组 L1 记录推断 lesson 的 scope

    推断规则:
      group / strategy_mode — 所有记录取值唯一时才确定, 否则 None (通配)
      regime — 仅当多数记录的主导体制属于"体制失配"(is_mismatch) 时,
               才取该主导体制名; 否则 None (教训不绑定具体体制)

    Args:
        records: L1 记录列表
        thresholds: 覆盖 regime 判定阈值 (与 failure_rules 键名一致)

    Returns:
        {"group": str|None, "regime": str|None, "strategy_mode": str|None}
    """
    if not records:
        return {"group": None, "regime": None, "strategy_mode": None}

    groups = {r.get("group_name") for r in records if r.get("group_name")}
    modes = {r.get("strategy_mode") for r in records if r.get("strategy_mode")}

    # regime: 统计"主导体制失配"的票数
    votes: Dict[str, int] = {}
    mismatch_total = 0
    for rec in records:
        slices = _record_regime_slices(rec)
        if not slices:
            continue
        try:
            dom = dominant_regime(slices, thresholds)
        except Exception:
            continue
        if dom and dom.get("is_mismatch"):
            votes[dom["regime"]] = votes.get(dom["regime"], 0) + 1
            mismatch_total += 1

    regime = None
    if mismatch_total > 0:
        top_regime, count = max(votes.items(), key=lambda kv: kv[1])
        if count / mismatch_total >= 0.5:
            regime = top_regime

    return {
        "group": next(iter(groups)) if len(groups) == 1 else None,
        "regime": regime,
        "strategy_mode": next(iter(modes)) if len(modes) == 1 else None,
    }


# ─────────────────────────────────────────────────────────────
# L2 教训库
# ─────────────────────────────────────────────────────────────

# ── 缺口 3 常量: 连续 N 次 inconclusive 自动降级 ──
_INCONCLUSIVE_THRESHOLD = 3

# ── 缺口 4 白名单: invalidation_criteria 支持的 field / op ──
_INVALIDATION_FIELDS = frozenset({
    "capital_per_group",      # 教训 scope.group 对应的组资金 (float)
    "portfolio_weight",       # 教训 scope.group 对应的组权重 (float)
    "position_ratio",         # 引擎 position_ratio (float)
    "total_initial_capital",  # 组合回测总本金 (float)
    "strategy_mode",          # 教训 scope.group 对应的组策略模式 (str)
})
_INVALIDATION_OPS = frozenset({">=", "<=", "==", "!="})


class LessonLibrary:
    """L2 教训条目管理器 — 一案一文件存储 + 状态机 + scope 检索"""

    def __init__(self, lessons_dir: Optional[str] = None,
                 l1_store_path: Optional[str] = None):
        self.lessons_dir = lessons_dir or _DEFAULT_LESSONS_DIR
        self._l1_store_path = l1_store_path

    # ── 路径 / IO ──

    def _lesson_path(self, lesson_id: str) -> str:
        return os.path.join(self.lessons_dir, f"{lesson_id}.json")

    def _ensure_dir(self) -> None:
        os.makedirs(self.lessons_dir, exist_ok=True)

    def _read(self, lesson_id: str) -> Optional[dict]:
        path = self._lesson_path(lesson_id)
        if not os.path.exists(path):
            return None
        try:
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
        except (json.JSONDecodeError, OSError) as e:
            logger.warning("读取 lesson %s 失败: %s", lesson_id, e)
            return None

    def _write(self, lesson: dict) -> None:
        self._ensure_dir()
        path = self._lesson_path(lesson["lesson_id"])
        with open(path, "w", encoding="utf-8") as f:
            json.dump(lesson, f, ensure_ascii=False, indent=2, default=str)

    # ── CRUD ──

    def create_lesson(self,
                      *,
                      linked_cases: Sequence[str],
                      scope: dict,
                      failure_level: str,
                      root_cause: str,
                      rule: str,
                      rule_action: Optional[dict] = None,
                      do_not_repeat: Optional[Sequence[str]] = None,
                      failure_flags_summary: Optional[Sequence[str]] = None,
                      ai_drafted: bool = False,
                      invalidation_criteria: Optional[Sequence[dict]] = None) -> dict:
        """创建一条 L2 教训

        强制约束:
          - linked_cases 长度 >= 2 (最小支持度)
          - failure_level 必须在 FAILURE_LEVELS 中
          - scope 至少包含 group / regime / strategy_mode 三个键
          - rule_action 若提供, 必须是合法结构化动作 (见 rule_action 模块)
          - invalidation_criteria 每条 dict 必须符合白名单 field/op (缺口 4)

        invalidation_criteria (缺口 4):
          [{"field": "capital_per_group", "op": ">=", "value": 10000, "scope": {"group": "消费稳健型"}}]
          含义: 当消费组资金 ≥ ¥10000 时, 教训根因(资金约束)失效, 自动 invalidated。
          判定: 任一 criteria 被违反 → 教训失效。

        Returns:
            新建的 lesson 字典

        Raises:
            ValueError: 约束不满足
        """
        linked = list(dict.fromkeys(linked_cases or []))  # 去重保序
        if len(linked) < 2:
            raise ValueError(
                f"linked_cases 至少需要 2 条 L1 记录 (当前 {len(linked)} 条), "
                "单条失败不构成教训"
            )
        if failure_level not in FAILURE_LEVELS:
            raise ValueError(
                f"failure_level 必须是 {FAILURE_LEVELS} 之一, 收到 {failure_level!r}"
            )
        if rule_action is not None:
            validate_rule_action(rule_action)

        # 缺口 4: 校验 invalidation_criteria
        criteria_list = list(invalidation_criteria or [])
        for i, c in enumerate(criteria_list):
            if not isinstance(c, dict):
                raise ValueError(f"invalidation_criteria[{i}] 必须是 dict, 收到 {type(c).__name__}")
            if c.get("field") not in _INVALIDATION_FIELDS:
                raise ValueError(
                    f"invalidation_criteria[{i}].field 必须在 {_INVALIDATION_FIELDS} 中, "
                    f"收到 {c.get('field')!r}"
                )
            if c.get("op") not in _INVALIDATION_OPS:
                raise ValueError(
                    f"invalidation_criteria[{i}].op 必须在 {_INVALIDATION_OPS} 中, "
                    f"收到 {c.get('op')!r}"
                )
            if "value" not in c:
                raise ValueError(f"invalidation_criteria[{i}] 缺少 value")

        scope = self._normalize_scope(scope)
        flags = self._normalize_flags(failure_flags_summary)

        now = datetime.now().isoformat(timespec="seconds")
        lesson = {
            "lesson_id": f"lsn_{uuid.uuid4().hex[:12]}",
            "created_at": now,
            "updated_at": now,
            "linked_cases": linked,
            "support_count": len(linked),
            "scope": scope,
            "failure_level": failure_level,
            "failure_flags_summary": flags,
            "root_cause": root_cause,
            "rule": rule,
            "rule_action": rule_action or None,
            "do_not_repeat": list(do_not_repeat or []),
            "confidence": _confidence_for(len(linked)),
            "validation_status": "pending",
            "applied_by": None,
            "applied_at": None,
            "validated_by": None,
            "validated_at": None,
            "validation_notes": None,
            "ai_drafted": bool(ai_drafted),
            # 缺口 3: inconclusive 计数器
            "inconclusive_count": 0,
            "last_inconclusive_at": None,
            # 缺口 4: 前提失效条件
            "invalidation_criteria": criteria_list,
        }
        self._write(lesson)
        logger.info(
            "创建 L2 教训 %s (level=%s, support=%d, confidence=%s, criteria=%d)",
            lesson["lesson_id"], failure_level, len(linked), lesson["confidence"],
            len(criteria_list),
        )
        return lesson

    def get_lesson(self, lesson_id: str) -> Optional[dict]:
        return self._read(lesson_id)

    def list_lessons(self,
                     status: Optional[str] = None,
                     failure_level: Optional[str] = None,
                     scope: Optional[dict] = None) -> List[dict]:
        """列出教训, 支持按 status / failure_level / scope 过滤

        scope 过滤: lesson.scope 中非 null 字段必须与查询 scope 对应值相等。
        """
        lessons = []
        if not os.path.isdir(self.lessons_dir):
            return lessons
        for name in os.listdir(self.lessons_dir):
            if not name.endswith(".json"):
                continue
            lesson_id = name[:-5]
            lesson = self._read(lesson_id)
            if lesson is None:
                continue
            if status and lesson.get("validation_status") != status:
                continue
            if failure_level and lesson.get("failure_level") != failure_level:
                continue
            if scope and not self._scope_matches(lesson.get("scope", {}), scope):
                continue
            lessons.append(lesson)
        lessons.sort(key=lambda l: l.get("created_at", ""), reverse=True)
        return lessons

    def update_lesson(self, lesson_id: str, **fields) -> bool:
        """更新可变字段 (root_cause / rule / rule_action / do_not_repeat / scope /
        failure_level / failure_flags_summary)

        linked_cases 变更会同步重算 support_count 与 confidence。
        返回是否成功更新。
        """
        lesson = self._read(lesson_id)
        if lesson is None:
            return False

        changed = False
        for key, value in fields.items():
            if key == "linked_cases":
                linked = list(dict.fromkeys(value or []))
                if len(linked) < 2:
                    raise ValueError(
                        f"linked_cases 更新后仍需 >= 2 条 (当前 {len(linked)} 条)"
                    )
                lesson["linked_cases"] = linked
                lesson["support_count"] = len(linked)
                lesson["confidence"] = _confidence_for(len(linked))
                changed = True
            elif key in _MUTABLE_FIELDS:
                if key == "scope":
                    lesson["scope"] = self._normalize_scope(value)
                elif key == "failure_flags_summary":
                    lesson["failure_flags_summary"] = self._normalize_flags(value)
                elif key == "failure_level":
                    if value not in FAILURE_LEVELS:
                        raise ValueError(
                            f"failure_level 必须是 {FAILURE_LEVELS} 之一"
                        )
                    lesson["failure_level"] = value
                elif key == "rule_action":
                    if value is not None:
                        validate_rule_action(value)
                    lesson["rule_action"] = value
                else:
                    lesson[key] = value
                changed = True
            else:
                # 忽略不可变字段 (状态/审计字段走专用方法)
                logger.debug("update_lesson 忽略不可变字段: %s", key)

        if changed:
            lesson["updated_at"] = datetime.now().isoformat(timespec="seconds")
            self._write(lesson)
        return changed

    def delete_lesson(self, lesson_id: str) -> bool:
        path = self._lesson_path(lesson_id)
        if not os.path.exists(path):
            return False
        os.remove(path)
        logger.info("删除 L2 教训 %s", lesson_id)
        return True

    # ── 状态机 ──

    def mark_applied(self, lesson_id: str, run_id: str) -> bool:
        """标记为已应用 (pending → applied)

        缺口 3: 同时重置 inconclusive_count (教训被重新应用, 之前的 inconclusive
        历史不再有意义 — 新窗口可能有新证据)。
        """
        lesson = self._read(lesson_id)
        if lesson is None:
            return False
        if lesson["validation_status"] not in ("pending", "invalidated"):
            logger.warning(
                "lesson %s 当前状态 %s, 不能标记 applied",
                lesson_id, lesson["validation_status"],
            )
            return False
        lesson["validation_status"] = "applied"
        lesson["applied_by"] = run_id
        lesson["applied_at"] = datetime.now().isoformat(timespec="seconds")
        lesson["updated_at"] = lesson["applied_at"]
        # 缺口 3: 重新应用 = 重置 inconclusive 历史
        lesson["inconclusive_count"] = 0
        lesson["last_inconclusive_at"] = None
        self._write(lesson)
        return True

    def mark_inconclusive(self, lesson_id: str, run_id: str,
                          notes: Optional[str] = None) -> dict:
        """记录一次 inconclusive 判定 (缺口 3)

        - 状态保持 applied 不变
        - inconclusive_count 递增
        - 若计数 >= _INCONCLUSIVE_THRESHOLD, 自动调用 mark_invalidated

        Returns:
            {"status": "inconclusive"|"auto_invalidated", "count": int,
             "auto_invalidated": bool}
        """
        lesson = self._read(lesson_id)
        if lesson is None:
            return {"status": "not_found", "count": 0, "auto_invalidated": False}
        if lesson.get("validation_status") != "applied":
            logger.debug("lesson %s 状态 %s 非 applied, 跳过 inconclusive 记录",
                         lesson_id, lesson.get("validation_status"))
            return {"status": "skipped", "count": 0, "auto_invalidated": False}

        now = datetime.now().isoformat(timespec="seconds")
        count = lesson.get("inconclusive_count", 0) + 1
        lesson["inconclusive_count"] = count
        lesson["last_inconclusive_at"] = now
        lesson["updated_at"] = now

        if count >= _INCONCLUSIVE_THRESHOLD:
            # 自动降级
            lesson["validation_status"] = "invalidated"
            lesson["validated_by"] = run_id
            lesson["validated_at"] = now
            lesson["validation_notes"] = (
                f"连续 {count} 次 A/B inconclusive (阈值 {_INCONCLUSIVE_THRESHOLD}), "
                f"自动失效. 最后一次: {notes or ''}"
            )
            self._write(lesson)
            logger.info(
                "lesson %s 连续 %d 次 inconclusive, 自动 invalidated",
                lesson_id, count,
            )
            return {"status": "auto_invalidated", "count": count,
                    "auto_invalidated": True}

        self._write(lesson)
        return {"status": "inconclusive", "count": count,
                "auto_invalidated": False}

    def mark_validated(self, lesson_id: str, run_id: str,
                       notes: Optional[str] = None) -> bool:
        """标记为已验证 (applied → validated)

        缺口 3: 同时重置 inconclusive_count (已有确定结论)。
        """
        lesson = self._read(lesson_id)
        if lesson is None:
            return False
        if lesson["validation_status"] != "applied":
            logger.warning(
                "lesson %s 当前状态 %s, 不能标记 validated (需先 applied)",
                lesson_id, lesson["validation_status"],
            )
            return False
        lesson["validation_status"] = "validated"
        lesson["validated_by"] = run_id
        lesson["validated_at"] = datetime.now().isoformat(timespec="seconds")
        lesson["validation_notes"] = notes
        lesson["updated_at"] = lesson["validated_at"]
        lesson["inconclusive_count"] = 0
        lesson["last_inconclusive_at"] = None
        self._write(lesson)
        return True

    def mark_invalidated(self, lesson_id: str, run_id: str,
                         notes: Optional[str] = None) -> bool:
        """标记为已失效 (applied → invalidated)

        缺口 3: 同时重置 inconclusive_count (已有确定结论)。
        """
        lesson = self._read(lesson_id)
        if lesson is None:
            return False
        if lesson["validation_status"] != "applied":
            logger.warning(
                "lesson %s 当前状态 %s, 不能标记 invalidated (需先 applied)",
                lesson_id, lesson["validation_status"],
            )
            return False
        lesson["validation_status"] = "invalidated"
        lesson["validated_by"] = run_id
        lesson["validated_at"] = datetime.now().isoformat(timespec="seconds")
        lesson["validation_notes"] = notes
        lesson["updated_at"] = lesson["validated_at"]
        lesson["inconclusive_count"] = 0
        lesson["last_inconclusive_at"] = None
        self._write(lesson)
        return True

    def mark_promoted(self, lesson_id: str, run_id: str, notes: Optional[str] = None) -> bool:
        """标记已固化到生产配置 (validated → promoted_at 落时间戳, 状态不变)

        固化后的教训不再被 build_candidate 重复取用, 避免对同一参数反复 apply。
        """
        lesson = self._read(lesson_id)
        if lesson is None:
            return False
        now = datetime.now().isoformat(timespec="seconds")
        lesson["promoted_at"] = now
        lesson["promoted_by"] = run_id
        if notes:
            lesson["promotion_notes"] = notes
        lesson["updated_at"] = now
        self._write(lesson)
        return True

    # ── 缺口 4: 前提失效检测 ──

    @staticmethod
    def check_invalidation_criteria(lesson: dict, context: dict) -> Optional[dict]:
        """检查教训的 invalidation_criteria 是否被违反 (缺口 4)

        Args:
            lesson: L2 教训 dict (必须含 scope + invalidation_criteria)
            context: 运行时值 dict, 结构:
                {
                    "capital_per_group": {"消费稳健型": 10000, ...},   # 各组资金
                    "portfolio_weight":  {"消费稳健型": 0.05, ...},   # 各组权重
                    "position_ratio":    0.3,                         # 引擎参数
                    "total_initial_capital": 100000,                  # 总本金
                    "strategy_mode":     {"消费稳健型": "mean_reversion", ...},
                }

        Returns:
            None — 未违反
            {"field": ..., "op": ..., "value": ..., "actual": ..., "scope": ...} — 被违反的 criteria
        """
        criteria_list = lesson.get("invalidation_criteria") or []
        if not criteria_list:
            return None
        lesson_scope = lesson.get("scope") or {}
        lesson_group = lesson_scope.get("group")

        for c in criteria_list:
            field = c.get("field")
            op = c.get("op")
            threshold = c.get("value")

            # 从 context 提取实际值
            actual = None
            group_for_lookup = c.get("scope", {}).get("group") or lesson_group
            if field in ("capital_per_group", "portfolio_weight", "strategy_mode"):
                table = context.get(field, {})
                if isinstance(table, dict) and group_for_lookup:
                    actual = table.get(group_for_lookup)
                else:
                    # context 结构不对, 跳过该 criteria (避免误判)
                    continue
            elif field in ("position_ratio", "total_initial_capital"):
                actual = context.get(field)
            else:
                continue  # 未知 field, 跳过

            if actual is None:
                continue  # 运行时没有这个值, 无法判断, 保守跳过

            # 比较: criteria 本身就是"失效条件"
            # op >= / <= / == / != 是"失效发生在 actual op threshold"
            # 条件满足 → 返回 violation (教训失效)
            try:
                if op == ">=" and (actual >= threshold):
                    return {"field": field, "op": op, "value": threshold,
                            "actual": actual, "scope": c.get("scope")}
                elif op == "<=" and (actual <= threshold):
                    return {"field": field, "op": op, "value": threshold,
                            "actual": actual, "scope": c.get("scope")}
                elif op == "==" and (actual == threshold):
                    return {"field": field, "op": op, "value": threshold,
                            "actual": actual, "scope": c.get("scope")}
                elif op == "!=" and (actual != threshold):
                    return {"field": field, "op": op, "value": threshold,
                            "actual": actual, "scope": c.get("scope")}
            except TypeError:
                # actual 和 threshold 类型不可比较 (比如 str vs float), 跳过
                continue

        return None

    def auto_invalidate(self, lesson_id: str, context: dict,
                        run_id: str = "auto_invalidation") -> Optional[dict]:
        """检测并自动失效前提已变的教训 (缺口 4)

        Returns:
            None — 教训不存在
            {"action": "invalidated", "violation": {...}} — 被违反并自动 invalidated
            {"action": "ok"} — 未违反
        """
        lesson = self._read(lesson_id)
        if lesson is None:
            return None

        violation = self.check_invalidation_criteria(lesson, context)
        if violation is None:
            return {"action": "ok"}

        # 只有 applied / validated 状态才自动失效 (pending 还没被测试,
        # invalidated 已失效)
        if lesson.get("validation_status") not in ("applied", "validated"):
            return {"action": "ignored", "violation": violation,
                    "reason": f"status={lesson.get('validation_status')} 非 applied/validated"}

        now = datetime.now().isoformat(timespec="seconds")
        lesson["validation_status"] = "invalidated"
        lesson["validated_by"] = run_id
        lesson["validated_at"] = now
        lesson["validation_notes"] = (
            f"前提失效自动检测: field={violation['field']}, "
            f"op={violation['op']}, 阈值={violation['value']}, "
            f"实际={violation['actual']}"
        )
        lesson["updated_at"] = now
        self._write(lesson)
        logger.info(
            "lesson %s 前提失效自动 invalidated: %s",
            lesson_id, lesson["validation_notes"],
        )
        return {"action": "invalidated", "violation": violation}

    # ── 进化器检索 ──

    def find_relevant_lessons(self,
                              scope: dict,
                              failure_level: Optional[str] = None,
                              status: Optional[str] = None) -> List[dict]:
        """按 scope 检索相关教训 (进化器入口)

        优先返回 validated, 其次 applied, 最后 pending;
        同状态内按 confidence (high→med→low) 与 support_count 降序。
        """
        candidates = self.list_lessons(
            status=status, failure_level=failure_level, scope=scope)

        status_rank = {"validated": 0, "applied": 1, "pending": 2, "invalidated": 3}
        conf_rank = {"high": 0, "med": 1, "low": 2}
        candidates.sort(key=lambda l: (
            status_rank.get(l.get("validation_status"), 9),
            conf_rank.get(l.get("confidence"), 9),
            -l.get("support_count", 0),
        ))
        return candidates

    def get_applicable_actions(self,
                               scope: dict,
                               failure_level: Optional[str] = None,
                               statuses: Sequence[str] = ("validated", "applied"),
                               context: Optional[dict] = None) -> List[dict]:
        """返回可被程序执行的结构化动作 (P6)

        只取带 rule_action 且 validation_status 在 statuses 中的教训,
        排序与 find_relevant_lessons 一致 (validated > applied > pending)。

        缺口 4: context 非空时, 前置检测 invalidation_criteria,
        把前提已失效的教训自动标记 invalidated 并跳过。

        Returns:
            [{"lesson_id", "rule_action", "confidence", "validation_status", ...,
              "invalidation_criteria": [...]}]
        """
        allowed = set(statuses)
        out = []
        for lesson in self.find_relevant_lessons(scope, failure_level=failure_level):
            if lesson.get("validation_status") not in allowed:
                continue
            action = lesson.get("rule_action")
            if not action:
                continue
            if lesson.get("promoted_at"):
                # 已固化到生产配置的教训不再重复取用
                continue
            # 缺口 4: 前置前提失效检测
            if context is not None and lesson.get("invalidation_criteria"):
                violation = self.check_invalidation_criteria(lesson, context)
                if violation is not None:
                    # 自动失效 (只对 applied/validated 状态)
                    result = self.auto_invalidate(
                        lesson["lesson_id"], context,
                        run_id="auto_invalidation_from_get_applicable",
                    )
                    if result and result.get("action") == "invalidated":
                        logger.info(
                            "get_applicable_actions 跳过教训 %s: 前提失效 %s",
                            lesson["lesson_id"], violation,
                        )
                    continue
            out.append({
                "lesson_id": lesson["lesson_id"],
                "rule_action": action,
                "confidence": lesson.get("confidence"),
                "validation_status": lesson.get("validation_status"),
                "support_count": lesson.get("support_count"),
                "scope": lesson.get("scope"),
                "invalidation_criteria": lesson.get("invalidation_criteria") or [],
            })
        return out

    # ── AI 起草 root_cause ──

    def draft_root_cause(self, l1_records: Sequence[dict]) -> str:
        """基于一组 L1 记录起草 root_cause

        优先调用 LLMClient; 不可用时从结构化字段生成模板草稿。
        返回的文本供人工确认/修改后写入 lesson.root_cause。
        """
        if not l1_records:
            return ""
        try:
            from src.ai.llm_client import LLMClient
            client = LLMClient()
            if client.is_available:
                return self._draft_with_llm(client, l1_records)
        except Exception as e:
            logger.warning("LLM 起草 root_cause 失败, 回退模板: %s", e)
        return self._draft_template(l1_records)

    def _draft_with_llm(self, client, records: Sequence[dict]) -> str:
        system_prompt = (
            "你是量化策略复盘专家。给定多条失效案例(L1), 请提炼出一条"
            "可执行的根因分析(root_cause), 要求:\n"
            "1. 聚焦共性, 不罗列单条细节\n"
            "2. 指出失败层级 (选股域/结构/参数/过拟合) 与证据\n"
            "3. 给出一条明确的改进方向 rule\n"
            "4. 给出 1-3 条 do_not_repeat (不要再试的方向)\n"
            "输出纯文本, 分三段: 根因 / 改进 / 禁忌。"
        )
        digest = self._records_digest(records)
        user_prompt = f"以下是 {len(records)} 条失效案例摘要:\n{digest}"
        try:
            return client.chat(system_prompt, user_prompt, temperature=0.3)
        except Exception as e:
            logger.warning("LLM 调用失败, 回退模板: %s", e)
            return self._draft_template(records)

    @staticmethod
    def _records_digest(records: Sequence[dict]) -> str:
        lines = []
        for i, rec in enumerate(records, 1):
            flags = rec.get("failure_flags", [])
            level = rec.get("failure_level", "unknown")
            metrics = rec.get("metrics", {}) or {}
            alpha = metrics.get("alpha")
            dd = metrics.get("max_drawdown")
            stop_rate = metrics.get("stop_loss_rate")
            scan = rec.get("param_scan") or {}
            scan_dir = ""
            if scan:
                mac = scan.get("max_abs_corr")
                if mac is not None:
                    scan_dir = f" 扫描方向性|r|={mac:.2f}"
            lines.append(
                f"[{i}] group={rec.get('group_name')} mode={rec.get('strategy_mode')} "
                f"level={level} flags={flags} alpha={alpha} dd={dd} "
                f"stop_loss_rate={stop_rate}{scan_dir}"
            )
        return "\n".join(lines)

    @staticmethod
    def _draft_template(records: Sequence[dict]) -> str:
        """无 LLM 时从结构化字段生成模板草稿 (人工确认起点)"""
        n = len(records)
        levels = {}
        flags_counter: Dict[str, int] = {}
        groups = set()
        modes = set()
        for rec in records:
            lv = rec.get("failure_level", "unknown")
            levels[lv] = levels.get(lv, 0) + 1
            for f in rec.get("failure_flags", []):
                flags_counter[f] = flags_counter.get(f, 0) + 1
            if rec.get("group_name"):
                groups.add(rec["group_name"])
            if rec.get("strategy_mode"):
                modes.add(rec["strategy_mode"])

        top_level = max(levels, key=levels.get) if levels else "unknown"
        top_flags = sorted(flags_counter, key=lambda f: -flags_counter[f])[:3]

        lines = [
            f"【根因】共 {n} 条案例, 主要失败层级={top_level} "
            f"({levels.get(top_level, 0)}/{n})。",
            f"  高频 failure_flags: {top_flags}。",
            f"  涉及组: {', '.join(sorted(groups)) or '未记录'}。",
            f"  策略模式: {', '.join(sorted(modes)) or '未记录'}。",
            "",
            "【改进】(待人工填写) 针对上述层级的可执行调优动作。",
            "",
            "【禁忌】(待人工填写) 不要再尝试的方向。",
        ]
        return "\n".join(lines)

    # ── 辅助 ──

    @staticmethod
    def _normalize_scope(scope: Optional[dict]) -> dict:
        scope = scope or {}
        return {
            "group": scope.get("group"),
            "regime": scope.get("regime"),
            "strategy_mode": scope.get("strategy_mode"),
        }

    @staticmethod
    def _normalize_flags(flags: Optional[Sequence[str]]) -> List[str]:
        return [f for f in (flags or []) if f in FAILURE_FLAGS]

    @staticmethod
    def _scope_matches(lesson_scope: dict, query_scope: dict) -> bool:
        """lesson.scope 中非 null 字段必须与 query 对应值相等; query 非 null 才约束"""
        for key in ("group", "regime", "strategy_mode"):
            qv = (query_scope or {}).get(key)
            if qv is None:
                continue
            lv = lesson_scope.get(key)
            if lv is None:
                continue  # lesson 该字段通配
            if lv != qv:
                return False
        return True
