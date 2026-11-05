"""do_not_repeat 硬约束 — 单元测试

覆盖:
  - parse_constraint: 标准/中文符号/负数小数/自然语言不可解析
  - check_suggestion_conflicts: 命中/不命中/target 不同/非数值建议
  - filter_suggestions: 拆分、顺序保持
  - summarize_conflicts
  - report_generator._apply_constraints + _format_rejected_note 集成
"""
import pytest

from src.ai.report_generator import WeeklyReportGenerator
from src.memory.constraint_check import (
    check_suggestion_conflicts,
    filter_suggestions,
    parse_constraint,
    summarize_conflicts,
)
from src.memory.lesson_library import LessonLibrary


# ─────────────────────────────────────────────────────────────
# parse_constraint
# ─────────────────────────────────────────────────────────────

@pytest.mark.parametrize("text,expected", [
    ("score_threshold <= 30", ("score_threshold", "<=", 30.0)),
    ("score_threshold<=30", ("score_threshold", "<=", 30.0)),
    ("score_threshold ≥ 40", ("score_threshold", ">=", 40.0)),
    ("score_threshold ≤ 25", ("score_threshold", "<=", 25.0)),
    ("cooldown_days < 3", ("cooldown_days", "<", 3.0)),
    ("score_threshold == 30", ("score_threshold", "==", 30.0)),
    ("threshold > -5.5", ("threshold", ">", -5.5)),
    ("在医药组 score_threshold <= 30 会持续亏损", ("score_threshold", "<=", 30.0)),
])
def test_parse_constraint_ok(text, expected):
    parsed = parse_constraint(text)
    assert parsed == {"target": expected[0], "op": expected[1], "value": expected[2]}


@pytest.mark.parametrize("text", [
    "震荡市不要满仓",
    "不要追高",
    "",
    None,
    "评分阈值低于30会亏损",   # 中文参数名, 不可解析
])
def test_parse_constraint_unparseable(text):
    assert parse_constraint(text) is None


# ─────────────────────────────────────────────────────────────
# check_suggestion_conflicts
# ─────────────────────────────────────────────────────────────

def _lesson(dnr, lesson_id="lsn_test"):
    return {"lesson_id": lesson_id, "do_not_repeat": dnr}


def test_conflict_hit_lower_than_forbidden():
    s = {"target": "score_threshold", "suggested_value": 28}
    conflicts = check_suggestion_conflicts(s, [_lesson(["score_threshold <= 30"])])
    assert len(conflicts) == 1
    assert conflicts[0]["lesson_id"] == "lsn_test"
    assert "score_threshold <= 30" in conflicts[0]["reason"]


def test_conflict_no_hit_when_value_outside():
    s = {"target": "score_threshold", "suggested_value": 35}
    assert check_suggestion_conflicts(
        s, [_lesson(["score_threshold <= 30"])]) == []


def test_conflict_boundary_inclusive():
    s = {"target": "score_threshold", "suggested_value": 30}
    # <= 30 含边界
    assert len(check_suggestion_conflicts(
        s, [_lesson(["score_threshold <= 30"])])) == 1


def test_conflict_target_mismatch():
    s = {"target": "cooldown_days", "suggested_value": 28}
    assert check_suggestion_conflicts(
        s, [_lesson(["score_threshold <= 30"])]) == []


def test_conflict_non_numeric_suggestion_skipped():
    s = {"target": "score_threshold", "suggested_value": "auto"}
    assert check_suggestion_conflicts(
        s, [_lesson(["score_threshold <= 30"])]) == []


def test_conflict_natural_language_ignored():
    """自然语言禁忌不可解析, 不参与硬拦截"""
    s = {"target": "score_threshold", "suggested_value": 20}
    assert check_suggestion_conflicts(
        s, [_lesson(["震荡市不要满仓"])]) == []


def test_conflict_multiple_lessons():
    s = {"target": "score_threshold", "suggested_value": 22}
    lessons = [
        _lesson(["score_threshold <= 30"], lesson_id="lsn_a"),
        _lesson([], lesson_id="lsn_b"),
        _lesson(["score_threshold <= 25"], lesson_id="lsn_c"),
    ]
    conflicts = check_suggestion_conflicts(s, lessons)
    assert {c["lesson_id"] for c in conflicts} == {"lsn_a", "lsn_c"}


def test_conflict_string_numeric_value():
    s = {"target": "score_threshold", "suggested_value": "28"}
    assert len(check_suggestion_conflicts(
        s, [_lesson(["score_threshold <= 30"])])) == 1


def test_conflict_greater_op():
    s = {"target": "max_consecutive_losses", "suggested_value": 6}
    assert len(check_suggestion_conflicts(
        s, [_lesson(["max_consecutive_losses >= 5"])])) == 1


# ─────────────────────────────────────────────────────────────
# filter_suggestions / summarize_conflicts
# ─────────────────────────────────────────────────────────────

def test_filter_split_and_order():
    lessons = [_lesson(["score_threshold <= 30"])]
    suggestions = [
        {"target": "score_threshold", "suggested_value": 28},
        {"target": "cooldown_days", "suggested_value": 2},
        {"target": "score_threshold", "suggested_value": 38},
    ]
    kept, rejected = filter_suggestions(suggestions, lessons)
    assert [s["suggested_value"] for s in kept] == [2, 38]
    assert len(rejected) == 1
    assert rejected[0]["suggestion"]["suggested_value"] == 28


def test_filter_empty_inputs():
    kept, rejected = filter_suggestions([], [])
    assert kept == [] and rejected == []


def test_summarize_conflicts():
    rejected = [
        {"suggestion": {}, "conflicts": [{"lesson_id": "lsn_a"}]},
        {"suggestion": {}, "conflicts": [{"lesson_id": "lsn_a"}, {"lesson_id": "lsn_b"}]},
    ]
    assert summarize_conflicts(rejected) == {"lsn_a": 2, "lsn_b": 1}


# ─────────────────────────────────────────────────────────────
# report_generator 集成
# ─────────────────────────────────────────────────────────────

@pytest.fixture
def lib(tmp_path):
    return LessonLibrary(lessons_dir=str(tmp_path / "lessons"))


@pytest.fixture
def gen(lib):
    return WeeklyReportGenerator(llm=object(), lesson_library=lib)


def _create_lesson(lib, dnr):
    return lib.create_lesson(
        linked_cases=["fc_a", "fc_b"],
        scope={"group": None, "regime": None, "strategy_mode": None},
        failure_level="param",
        root_cause="阈值偏低导致亏损",
        rule="阈值 30 → 38",
        do_not_repeat=dnr,
    )


def test_apply_constraints_filters_conflicts(gen, lib):
    _create_lesson(lib, ["score_threshold <= 30"])
    stats = {"by_regime": [{"name": "ranging", "count": 5}]}
    suggestions = [
        {"category": "param_adjust", "target": "score_threshold",
         "current_value": 35, "suggested_value": 28, "rationale": "提高交易频次"},
        {"category": "param_adjust", "target": "score_threshold",
         "current_value": 35, "suggested_value": 38, "rationale": "提高质量"},
    ]
    kept, rejected = gen._apply_constraints(suggestions, stats)
    assert len(kept) == 1
    assert kept[0]["suggested_value"] == 38
    assert len(rejected) == 1
    assert gen._last_suggestions_rejected == 1


def test_apply_constraints_no_lessons_passes_all(gen):
    stats = {"by_regime": [{"name": "ranging", "count": 5}]}
    suggestions = [{"target": "score_threshold", "suggested_value": 20}]
    kept, rejected = gen._apply_constraints(suggestions, stats)
    assert len(kept) == 1 and rejected == []


def test_apply_constraints_ignores_invalidated_lesson(gen, lib):
    lesson = _create_lesson(lib, ["score_threshold <= 30"])
    lib.mark_applied(lesson["lesson_id"], "run_a")
    lib.mark_invalidated(lesson["lesson_id"], "run_b", notes="证伪")
    stats = {"by_regime": [{"name": "ranging", "count": 5}]}
    kept, rejected = gen._apply_constraints(
        [{"target": "score_threshold", "suggested_value": 28}], stats)
    assert len(kept) == 1 and rejected == []


def test_format_rejected_note(gen):
    rejected = [{
        "suggestion": {"target": "score_threshold", "current_value": 35,
                       "suggested_value": 28},
        "conflicts": [{"reason": "教训 lsn_x 禁止 「score_threshold <= 30」"}],
    }]
    note = gen._format_rejected_note(rejected)
    assert "硬约束拦截" in note
    assert "score_threshold" in note
    assert "lsn_x" in note


def test_format_rejected_note_empty(gen):
    assert gen._format_rejected_note([]) == ""
