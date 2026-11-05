"""结构化 rule_action — 让 L2 教训可被程序直接执行 (P6)

定位:
  自然语言 rule 只能喂给 LLM, 无法被程序稳定地"遗传"到下一版参数。
  本模块定义结构化动作, 支持:
    set_param        — 把某参数设为指定值
    adjust_param     — 在某参数现有值上增减
    rebalance_weight — 调整某组权重, 差额转移给另一组 (总仓位不变)

  lesson 同时保留自然语言 rule (给人/LLM 看) 与 rule_action (给程序执行),
  二者互补。

动作 schema:
  {"type": "set_param",    "target": "score_threshold", "value": 38}
  {"type": "adjust_param", "target": "score_threshold", "delta": 8}
  {"type": "rebalance_weight", "target": "消费稳健型", "value": 0.0,
   "compensate": "科技成长型"}

  rebalance_weight 说明: 把 target 组的组合权重设为 value, 差额 (±) 转给
  compensate 组, 保证总仓位不变。用于"某组结构性无法交易/负 alpha 时把
  资金让渡给有效组"这类失败, 这是参数调整无法表达的。

用法:
    from src.memory.rule_action import apply_rule_actions

    result = apply_rule_actions(params_snapshot, [action])
    # result: {"params": 新快照, "changes": [...], "skipped": [...]}
"""
import copy
from typing import Dict, List, Optional, Sequence

ACTION_TYPES = ("set_param", "adjust_param", "rebalance_weight")

_MISSING = object()


def validate_rule_action(action: dict) -> None:
    """校验动作合法性, 非法时抛 ValueError"""
    if not isinstance(action, dict):
        raise ValueError(f"rule_action 必须是 dict, 收到 {type(action).__name__}")

    atype = action.get("type")
    if atype not in ACTION_TYPES:
        raise ValueError(f"rule_action.type 必须是 {ACTION_TYPES} 之一, 收到 {atype!r}")

    target = action.get("target")
    if not isinstance(target, str) or not target.strip():
        raise ValueError("rule_action.target 必须是非空字符串")

    if atype == "set_param":
        if "value" not in action:
            raise ValueError("set_param 缺少 value")
        v = action["value"]
        if isinstance(v, bool) or not isinstance(v, (int, float, str, list, dict)):
            raise ValueError(
                f"set_param.value 必须是数值/字符串/列表/dict, 收到 {type(v).__name__}")
    elif atype == "adjust_param":
        delta = action.get("delta", _MISSING)
        if delta is _MISSING:
            raise ValueError("adjust_param 缺少 delta")
        if isinstance(delta, bool) or not isinstance(delta, (int, float)):
            raise ValueError("adjust_param.delta 必须是数值")
    elif atype == "rebalance_weight":
        v = action.get("value", _MISSING)
        if v is _MISSING:
            raise ValueError("rebalance_weight 缺少 value (目标权重)")
        if isinstance(v, bool) or not isinstance(v, (int, float)):
            raise ValueError("rebalance_weight.value 必须是数值")
        if not 0 <= v <= 1:
            raise ValueError(f"rebalance_weight.value 必须在 [0,1], 收到 {v}")
        compensate = action.get("compensate")
        if not isinstance(compensate, str) or not compensate.strip():
            raise ValueError("rebalance_weight 缺少 compensate (承接差额的组名)")
        if compensate == target:
            raise ValueError("rebalance_weight.compensate 不能与 target 相同")


def _find_section(params: dict, target: str) -> Optional[dict]:
    """定位 target 所在的参数段 (顶层或一层嵌套 dict)"""
    if target in params and not isinstance(params[target], dict):
        return params
    for value in params.values():
        if isinstance(value, dict) and target in value:
            return value
    return None


def apply_rule_actions(params: dict,
                       actions: Sequence[dict]) -> Dict[str, list]:
    """把结构化动作应用到参数快照 (纯函数, 不修改入参)

    Args:
        params: 参数快照 (可嵌套, 如 {engine: {...}, portfolio: {...}})
        actions: rule_action 列表

    Returns:
        {"params": 新快照,
         "changes": [{target, action, before, after}],
         "skipped": [{action, reason}]}
    """
    new_params = copy.deepcopy(params or {})
    changes: List[dict] = []
    skipped: List[dict] = []

    for action in actions or []:
        try:
            validate_rule_action(action)
        except ValueError as e:
            skipped.append({"action": action, "reason": str(e)})
            continue

        target = action["target"]
        section = _find_section(new_params, target)
        if section is None:
            skipped.append({"action": action, "reason": f"参数 {target} 不存在"})
            continue

        before = section[target]
        if action["type"] == "set_param":
            after = action["value"]
        else:  # adjust_param
            if isinstance(before, bool) or not isinstance(before, (int, float)):
                skipped.append({
                    "action": action,
                    "reason": f"参数 {target} 当前值非数值, 无法 adjust",
                })
                continue
            after = before + action["delta"]

        section[target] = after
        changes.append({
            "target": target,
            "action": action["type"],
            "before": before,
            "after": after,
        })

    return {"params": new_params, "changes": changes, "skipped": skipped}


def parse_action_spec(spec: str) -> dict:
    """解析 CLI 动作字符串

    支持:
      set_param:score_threshold=38
      adjust_param:score_threshold=+8
      adjust_param:cooldown_days=-1
    """
    if not spec or ":" not in spec or "=" not in spec:
        raise ValueError(
            f"动作格式错误 {spec!r}, 示例: set_param:score_threshold=38")

    kind, rest = spec.split(":", 1)
    target, raw_value = rest.split("=", 1)
    kind, target = kind.strip(), target.strip()

    if kind not in ACTION_TYPES:
        raise ValueError(f"动作类型必须是 {ACTION_TYPES} 之一, 收到 {kind!r}")

    try:
        number = float(raw_value)
    except ValueError:
        raise ValueError(f"动作值必须是数值, 收到 {raw_value!r}")

    if number.is_integer():
        number = int(number)

    if kind == "set_param":
        return {"type": kind, "target": target, "value": number}
    return {"type": kind, "target": target, "delta": number}
