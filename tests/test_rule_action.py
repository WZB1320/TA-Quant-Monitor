"""结构化 rule_action — 单元测试 (P6)

覆盖:
  - validate_rule_action 合法/非法动作
  - parse_action_spec CLI 解析 (set_param / adjust_param / 负数 / 非法)
  - apply_rule_actions 顶层与嵌套参数段修改, 不修改入参
  - apply_rule_actions 缺失 target / 非数值 adjust / 非法动作 -> skipped
  - LessonLibrary.create_lesson 携带并校验 rule_action
  - get_applicable_actions 仅返回带动作且状态允许的教训, 排序正确
  - update_lesson 更新 rule_action 时校验
  - SuggestionValidator 匹配优先使用 rule_action.target
"""
import pytest

from src.memory.lesson_library import LessonLibrary
from src.memory.rule_action import (
    ACTION_TYPES,
    apply_rule_actions,
    parse_action_spec,
    validate_rule_action,
)
from src.ai.suggestion_validator import _lesson_mentions_target


# ─────────────────────────────────────────────────────────────
# validate_rule_action
# ─────────────────────────────────────────────────────────────

def test_validate_set_param_ok():
    validate_rule_action({"type": "set_param", "target": "score_threshold", "value": 38})


def test_validate_adjust_param_ok():
    validate_rule_action({"type": "adjust_param", "target": "atr_stop_mult", "delta": -0.2})


@pytest.mark.parametrize("action, match", [
    ("not a dict", "必须是 dict"),
    ({"type": "noop", "target": "x", "value": 1}, "type"),
    ({"type": "set_param", "target": "", "value": 1}, "target"),
    ({"type": "set_param", "target": "x"}, "缺少 value"),
    ({"type": "set_param", "target": "x", "value": True}, "数值/字符串/列表/dict"),
    ({"type": "adjust_param", "target": "x"}, "缺少 delta"),
    ({"type": "adjust_param", "target": "x", "delta": "8"}, "必须是数值"),
])
def test_validate_rule_action_rejects(action, match):
    with pytest.raises(ValueError, match=match):
        validate_rule_action(action)


# ─────────────────────────────────────────────────────────────
# parse_action_spec
# ─────────────────────────────────────────────────────────────

def test_parse_set_param():
    assert parse_action_spec("set_param:score_threshold=38") == {
        "type": "set_param", "target": "score_threshold", "value": 38}


def test_parse_adjust_param_positive_delta():
    assert parse_action_spec("adjust_param:cooldown_days=+8") == {
        "type": "adjust_param", "target": "cooldown_days", "delta": 8}


def test_parse_adjust_param_negative_delta():
    assert parse_action_spec("adjust_param:atr_stop_mult=-0.2") == {
        "type": "adjust_param", "target": "atr_stop_mult", "delta": -0.2}


def test_parse_keeps_float_when_not_integer():
    assert parse_action_spec("set_param:risk_per_trade=0.03")["value"] == 0.03


@pytest.mark.parametrize("spec", [
    "score_threshold=38",          # 缺类型
    "set_param:score_threshold",   # 缺值
    "noop:score_threshold=38",     # 非法类型
    "set_param:score_threshold=abc",  # 非数值
])
def test_parse_action_spec_rejects(spec):
    with pytest.raises(ValueError):
        parse_action_spec(spec)


# ─────────────────────────────────────────────────────────────
# apply_rule_actions
# ─────────────────────────────────────────────────────────────

PARAMS = {
    "group_name": "医药创新型",
    "engine": {"score_threshold": 30, "atr_stop_mult": 2.0, "position_ratio": 0.3},
    "portfolio": {"weight": 0.05},
    "strategy_mode": "mean_reversion",
}


def test_apply_set_param_nested():
    result = apply_rule_actions(
        PARAMS, [{"type": "set_param", "target": "score_threshold", "value": 38}])
    assert result["params"]["engine"]["score_threshold"] == 38
    assert result["changes"] == [
        {"target": "score_threshold", "action": "set_param", "before": 30, "after": 38}]
    assert result["skipped"] == []


def test_apply_adjust_param_nested():
    result = apply_rule_actions(
        PARAMS, [{"type": "adjust_param", "target": "atr_stop_mult", "delta": -0.2}])
    assert result["params"]["engine"]["atr_stop_mult"] == pytest.approx(1.8)
    assert result["changes"][0]["before"] == 2.0


def test_apply_top_level_target():
    result = apply_rule_actions(
        PARAMS, [{"type": "set_param", "target": "strategy_mode", "value": "momentum"}])
    assert result["params"]["strategy_mode"] == "momentum"


def test_apply_does_not_mutate_input():
    apply_rule_actions(
        PARAMS, [{"type": "set_param", "target": "score_threshold", "value": 38}])
    assert PARAMS["engine"]["score_threshold"] == 30


def test_apply_multiple_actions():
    result = apply_rule_actions(PARAMS, [
        {"type": "set_param", "target": "score_threshold", "value": 38},
        {"type": "adjust_param", "target": "weight", "delta": 0.05},
    ])
    assert result["params"]["engine"]["score_threshold"] == 38
    assert result["params"]["portfolio"]["weight"] == pytest.approx(0.10)
    assert len(result["changes"]) == 2


def test_apply_missing_target_skipped():
    result = apply_rule_actions(
        PARAMS, [{"type": "set_param", "target": "unknown_param", "value": 1}])
    assert len(result["skipped"]) == 1
    assert "不存在" in result["skipped"][0]["reason"]
    assert result["changes"] == []


def test_apply_adjust_non_numeric_skipped():
    params = {"engine": {"flag": "fast"}}
    result = apply_rule_actions(
        params, [{"type": "adjust_param", "target": "flag", "delta": 1}])
    assert len(result["skipped"]) == 1
    assert "非数值" in result["skipped"][0]["reason"]


def test_apply_invalid_action_skipped():
    result = apply_rule_actions(PARAMS, [{"type": "noop", "target": "score_threshold"}])
    assert len(result["skipped"]) == 1


def test_apply_empty_actions_returns_copy():
    result = apply_rule_actions(PARAMS, [])
    assert result["params"] == PARAMS
    assert result["params"] is not PARAMS


# ─────────────────────────────────────────────────────────────
# LessonLibrary 集成
# ─────────────────────────────────────────────────────────────

@pytest.fixture
def lib(tmp_path):
    return LessonLibrary(lessons_dir=str(tmp_path / "lessons"))


def _make(lib, **overrides):
    base = dict(
        linked_cases=["fc_a", "fc_b"],
        scope={"group": "医药创新型", "regime": None, "strategy_mode": "mean_reversion"},
        failure_level="param",
        root_cause="score_threshold 偏低",
        rule="score_threshold 30 -> 38",
        do_not_repeat=["score_threshold <= 30"],
    )
    base.update(overrides)
    return lib.create_lesson(**base)


def test_create_lesson_stores_rule_action(lib):
    action = {"type": "set_param", "target": "score_threshold", "value": 38}
    lesson = _make(lib, rule_action=action)
    assert lesson["rule_action"] == action
    assert lib.get_lesson(lesson["lesson_id"])["rule_action"] == action


def test_create_lesson_without_rule_action_is_none(lib):
    lesson = _make(lib)
    assert lesson["rule_action"] is None


def test_create_lesson_rejects_invalid_rule_action(lib):
    with pytest.raises(ValueError, match="type"):
        _make(lib, rule_action={"type": "noop", "target": "x"})


def test_update_lesson_rule_action_validated(lib):
    lesson = _make(lib)
    new_action = {"type": "adjust_param", "target": "atr_stop_mult", "delta": -0.2}
    assert lib.update_lesson(lesson["lesson_id"], rule_action=new_action)
    assert lib.get_lesson(lesson["lesson_id"])["rule_action"] == new_action


def test_update_lesson_invalid_rule_action_raises(lib):
    lesson = _make(lib, rule_action={"type": "set_param", "target": "x", "value": 1})
    with pytest.raises(ValueError):
        lib.update_lesson(lesson["lesson_id"], rule_action={"type": "bad", "target": "x"})


def test_get_applicable_actions_filters_status_and_action(lib):
    no_action = _make(lib)  # pending, 无动作
    pending_action = _make(
        lib, rule_action={"type": "set_param", "target": "score_threshold", "value": 38})
    applied_action = _make(
        lib, rule_action={"type": "adjust_param", "target": "atr_stop_mult", "delta": -0.2})
    lib.mark_applied(applied_action["lesson_id"], "run_1")

    out = lib.get_applicable_actions(
        scope={"group": "医药创新型", "strategy_mode": "mean_reversion"})
    ids = {x["lesson_id"] for x in out}
    assert no_action["lesson_id"] not in ids  # 无 rule_action
    assert pending_action["lesson_id"] not in ids  # 默认只取 validated/applied
    assert applied_action["lesson_id"] in ids
    assert out[0]["rule_action"]["target"] == "atr_stop_mult"
    assert out[0]["validation_status"] == "applied"


def test_get_applicable_actions_ordering_validated_first(lib):
    a = _make(lib, rule_action={"type": "set_param", "target": "p", "value": 1})
    b = _make(lib, rule_action={"type": "set_param", "target": "q", "value": 2})
    lib.mark_applied(b["lesson_id"], "run_1")
    lib.mark_validated(b["lesson_id"], "run_2", notes="ok")

    out = lib.get_applicable_actions(
        scope={"group": "医药创新型"}, statuses=("validated", "applied", "pending"))
    assert [x["lesson_id"] for x in out][0] == b["lesson_id"]
    # validated 优先于 pending
    assert a["lesson_id"] in {x["lesson_id"] for x in out}


def test_get_applicable_actions_respects_statuses(lib):
    a = _make(lib, rule_action={"type": "set_param", "target": "p", "value": 1})
    out = lib.get_applicable_actions(
        scope={"group": "医药创新型"}, statuses=("pending",))
    assert [x["lesson_id"] for x in out] == [a["lesson_id"]]


# ─────────────────────────────────────────────────────────────
# SuggestionValidator 匹配
# ─────────────────────────────────────────────────────────────

def test_lesson_mentions_target_prefers_rule_action():
    lesson = {
        "rule": "调高某个阈值",
        "rule_action": {"type": "set_param", "target": "score_threshold", "value": 38},
    }
    assert _lesson_mentions_target(lesson, "score_threshold") is True
    assert _lesson_mentions_target(lesson, "atr_stop_mult") is False


def test_lesson_mentions_target_falls_back_to_text():
    lesson = {"rule": "score_threshold 从 30 调到 38", "rule_action": None}
    assert _lesson_mentions_target(lesson, "score_threshold") is True


def test_lesson_mentions_target_no_partial_match():
    lesson = {"rule": "my_score_threshold 偏低", "rule_action": None}
    assert _lesson_mentions_target(lesson, "score_threshold") is False
