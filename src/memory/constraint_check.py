"""失效案例库 — do_not_repeat 硬约束校验

定位:
  周报生成时先把 L2 教训的 do_not_repeat 文本注入 prompt (软提示),
  但 LLM 仍可能给出违反禁忌的参数建议。本模块在建议落库/追踪前做硬校验,
  把与禁忌冲突的建议剔除, 从"软提示"升级为"硬拦截"。

禁忌格式 (可解析才硬拦截):
  "<参数名> <比较符> <数值>"  例: score_threshold <= 30
  支持比较符: <= >= < > == 以及中文 ≤ ≥

  自然语言禁忌 (如 "震荡市不要满仓") 无法解析, 不参与硬校验,
  仍由 prompt 注入的软提示约束。

用法:
    from src.memory.constraint_check import filter_suggestions

    kept, rejected = filter_suggestions(suggestions, lessons)
    # kept: 通过校验的建议
    # rejected: [{"suggestion": {...}, "conflicts": [{...}]}]
"""
import re
from typing import Dict, List, Optional, Tuple

_PATTERN = re.compile(
    r"(?P<target>[A-Za-z_][A-Za-z0-9_.]*)\s*"
    r"(?P<op><=|>=|==|≤|≥|<|>)\s*"
    r"(?P<value>-?\d+(?:\.\d+)?)"
)

_OP_NORMALIZE = {"≤": "<=", "≥": ">="}


def parse_constraint(text: str) -> Optional[dict]:
    """解析禁忌文本为 {target, op, value}

    仅解析 "参数名 比较符 数值" 形式; 无法解析返回 None。
    """
    if not text or not isinstance(text, str):
        return None
    m = _PATTERN.search(text)
    if not m:
        return None
    op = _OP_NORMALIZE.get(m.group("op"), m.group("op"))
    try:
        value = float(m.group("value"))
    except ValueError:
        return None
    return {"target": m.group("target"), "op": op, "value": value}


def _violates(value: float, op: str, limit: float) -> bool:
    if op == "<=":
        return value <= limit
    if op == "<":
        return value < limit
    if op == ">=":
        return value >= limit
    if op == ">":
        return value > limit
    if op == "==":
        return value == limit
    return False


def _as_float(value) -> Optional[float]:
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        try:
            return float(value.strip())
        except ValueError:
            return None
    return None


def check_suggestion_conflicts(suggestion: dict,
                               lessons: List[dict]) -> List[dict]:
    """检查一条建议是否违反某条 lesson 的 do_not_repeat

    Returns:
        冲突列表 [{"lesson_id", "constraint", "reason"}]; 无冲突返回 []
    """
    suggested = _as_float(suggestion.get("suggested_value"))
    if suggested is None:
        return []  # 非数值建议 (如 add_filter) 无法硬校验

    target = suggestion.get("target")
    if not target:
        return []

    conflicts = []
    for lesson in lessons:
        for text in lesson.get("do_not_repeat") or []:
            parsed = parse_constraint(text)
            if not parsed or parsed["target"] != target:
                continue
            if _violates(suggested, parsed["op"], parsed["value"]):
                conflicts.append({
                    "lesson_id": lesson.get("lesson_id"),
                    "constraint": text,
                    "reason": (
                        f"教训 {lesson.get('lesson_id')} 禁止 「{text}」, "
                        f"建议值 {suggested:g} 仍落在禁忌区间"
                    ),
                })
    return conflicts


def filter_suggestions(suggestions: List[dict],
                       lessons: List[dict]) -> Tuple[List[dict], List[dict]]:
    """按 do_not_repeat 硬约束过滤建议

    Returns:
        (kept, rejected)
        rejected 每项: {"suggestion": 原建议, "conflicts": [冲突详情]}
    """
    kept, rejected = [], []
    for suggestion in suggestions or []:
        conflicts = check_suggestion_conflicts(suggestion, lessons or [])
        if conflicts:
            rejected.append({"suggestion": suggestion, "conflicts": conflicts})
        else:
            kept.append(suggestion)
    return kept, rejected


def summarize_conflicts(rejected: List[dict]) -> Dict[str, int]:
    """统计被拦截建议涉及的 lesson_id → 拦截条数 (便于日志/监控)"""
    counter: Dict[str, int] = {}
    for item in rejected or []:
        for c in item.get("conflicts", []):
            lid = c.get("lesson_id") or "unknown"
            counter[lid] = counter.get(lid, 0) + 1
    return counter
