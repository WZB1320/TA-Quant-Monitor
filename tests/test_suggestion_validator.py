"""P4 自动验证器 — 单元测试

覆盖:
  - RunOutcomeStore 写入/加载/前后查找 (窗口与 mode 过滤)
  - alpha 改善 → validated; 恶化 → invalidated; 小幅变化 → inconclusive
  - 缺 applied_at / after / before → inconclusive
  - dry_run 不写状态
  - 建议状态写回 (mark_validated / mark_invalidated)
  - L2 教训回写 (pending → applied → validated/invalidated)
"""
import pytest

from src.ai.suggestion_tracker import SuggestionTracker
from src.ai.suggestion_validator import SuggestionValidator
from src.memory.lesson_library import LessonLibrary
from src.memory.run_outcome import RunOutcomeStore


WINDOW = {"start": "2025-07-01", "end": "2025-12-31"}


@pytest.fixture
def tracker(tmp_path):
    return SuggestionTracker(file_path=str(tmp_path / "suggestions.jsonl"))


@pytest.fixture
def store(tmp_path):
    return RunOutcomeStore(store_path=str(tmp_path / "run_outcomes.jsonl"))


@pytest.fixture
def lib(tmp_path):
    return LessonLibrary(lessons_dir=str(tmp_path / "lessons"))


def _seed_applied(tracker, *, target="score_threshold", suggested=38,
                  applied_at="2026-01-10T00:00:00"):
    tracker.save_suggestions(
        [{"category": "param_adjust", "target": target,
          "current_value": 30, "suggested_value": suggested,
          "rationale": "test"}],
        "weekly_test", "2026-01-01", "2026-01-07",
    )
    record = tracker.load_all_suggestions()[-1]
    sid = record["suggestion_id"]
    tracker.mark_applied(sid, "v_after")
    records = tracker.load_all_suggestions()
    for r in records:
        if r["suggestion_id"] == sid:
            r["applied_at"] = applied_at
    tracker._rewrite_file(records)
    return sid


def _add_outcome(store, run_id, recorded_at, alpha, window=WINDOW, mode="balanced"):
    store.record(
        run_id=run_id, window=window, mode=mode,
        metrics={"alpha": alpha, "total_return": alpha + 0.05},
        recorded_at=recorded_at,
    )


# ─────────────────────────────────────────────────────────────
# RunOutcomeStore
# ─────────────────────────────────────────────────────────────

def test_store_record_and_load(store):
    rec = store.record(run_id="bt_1", window=WINDOW,
                       metrics={"alpha": 0.02}, mode="balanced")
    assert rec["outcome_id"].startswith("oc_")
    loaded = store.load_all()
    assert len(loaded) == 1
    assert loaded[0]["run_id"] == "bt_1"
    assert loaded[0]["window"] == WINDOW


def test_store_find_nearest_after(store):
    _add_outcome(store, "bt_a", "2026-01-01T00:00:00", 0.01)
    _add_outcome(store, "bt_b", "2026-01-05T00:00:00", 0.02)
    _add_outcome(store, "bt_c", "2026-01-20T00:00:00", 0.03)
    a = store.find_nearest_after("2026-01-02T00:00:00")
    assert a["run_id"] == "bt_b"


def test_store_find_nearest_before(store):
    _add_outcome(store, "bt_a", "2026-01-01T00:00:00", 0.01)
    _add_outcome(store, "bt_b", "2026-01-05T00:00:00", 0.02)
    b = store.find_nearest_before("2026-01-10T00:00:00")
    assert b["run_id"] == "bt_b"


def test_store_find_before_requires_same_window(store):
    _add_outcome(store, "bt_old_win", "2026-01-01T00:00:00", 0.01,
                 window={"start": "2025-01-01", "end": "2025-06-30"})
    _add_outcome(store, "bt_same_win", "2026-01-05T00:00:00", 0.02, window=WINDOW)
    b = store.find_nearest_before("2026-01-10T00:00:00", window=WINDOW)
    assert b["run_id"] == "bt_same_win"


def test_store_find_before_requires_same_mode(store):
    _add_outcome(store, "bt_conservative", "2026-01-05T00:00:00", 0.02, mode="conservative")
    _add_outcome(store, "bt_balanced", "2026-01-04T00:00:00", 0.03, mode="balanced")
    b = store.find_nearest_before("2026-01-10T00:00:00", window=WINDOW, mode="balanced")
    assert b["run_id"] == "bt_balanced"


# ─────────────────────────────────────────────────────────────
# verdict 判定
# ─────────────────────────────────────────────────────────────

def test_validate_improved(tracker, store):
    _seed_applied(tracker, applied_at="2026-01-10T00:00:00")
    _add_outcome(store, "bt_before", "2026-01-05T00:00:00", 0.02)
    _add_outcome(store, "bt_after", "2026-01-15T00:00:00", 0.05)

    result = SuggestionValidator(tracker=tracker, outcome_store=store).validate(dry_run=False)
    item = result["items"][0]
    assert item["verdict"] == "validated"
    assert item["delta"]["alpha"] == pytest.approx(0.03, abs=1e-6)
    assert result["validated"] == 1


def test_validate_worsened(tracker, store):
    _seed_applied(tracker, applied_at="2026-01-10T00:00:00")
    _add_outcome(store, "bt_before", "2026-01-05T00:00:00", 0.05)
    _add_outcome(store, "bt_after", "2026-01-15T00:00:00", 0.01)

    result = SuggestionValidator(tracker=tracker, outcome_store=store).validate(dry_run=False)
    assert result["items"][0]["verdict"] == "invalidated"
    assert result["invalidated"] == 1


def test_validate_small_change_inconclusive(tracker, store):
    _seed_applied(tracker, applied_at="2026-01-10T00:00:00")
    _add_outcome(store, "bt_before", "2026-01-05T00:00:00", 0.050)
    _add_outcome(store, "bt_after", "2026-01-15T00:00:00", 0.055)

    result = SuggestionValidator(tracker=tracker, outcome_store=store).validate(dry_run=False)
    assert result["items"][0]["verdict"] == "inconclusive"
    assert result["inconclusive"] == 1


def test_validate_missing_after(tracker, store):
    _seed_applied(tracker, applied_at="2026-01-10T00:00:00")
    _add_outcome(store, "bt_before", "2026-01-05T00:00:00", 0.02)

    result = SuggestionValidator(tracker=tracker, outcome_store=store).validate(dry_run=False)
    assert result["items"][0]["verdict"] == "inconclusive"
    assert "尚无新回测" in result["items"][0]["reason"]


def test_validate_missing_before_same_window(tracker, store):
    _seed_applied(tracker, applied_at="2026-01-10T00:00:00")
    # 应用前只有不同窗口的回测
    _add_outcome(store, "bt_before_other_win", "2026-01-05T00:00:00", 0.02,
                 window={"start": "2025-01-01", "end": "2025-06-30"})
    _add_outcome(store, "bt_after", "2026-01-15T00:00:00", 0.05)

    result = SuggestionValidator(tracker=tracker, outcome_store=store).validate(dry_run=False)
    assert result["items"][0]["verdict"] == "inconclusive"
    assert "基线" in result["items"][0]["reason"]


def test_validate_missing_applied_at(tracker, store):
    tracker.save_suggestions(
        [{"category": "param_adjust", "target": "score_threshold",
          "current_value": 30, "suggested_value": 38, "rationale": "x"}],
        "weekly_test", "2026-01-01", "2026-01-07",
    )
    sid = tracker.load_all_suggestions()[-1]["suggestion_id"]
    tracker.mark_applied(sid, "v")
    records = tracker.load_all_suggestions()
    for r in records:
        if r["suggestion_id"] == sid:
            r["applied_at"] = None
    tracker._rewrite_file(records)

    result = SuggestionValidator(tracker=tracker, outcome_store=store).validate(dry_run=False)
    assert result["items"][0]["verdict"] == "inconclusive"
    assert "applied_at" in result["items"][0]["reason"]


# ─────────────────────────────────────────────────────────────
# 状态写回
# ─────────────────────────────────────────────────────────────

def test_dry_run_does_not_write(tracker, store):
    _seed_applied(tracker, applied_at="2026-01-10T00:00:00")
    _add_outcome(store, "bt_before", "2026-01-05T00:00:00", 0.02)
    _add_outcome(store, "bt_after", "2026-01-15T00:00:00", 0.05)

    SuggestionValidator(tracker=tracker, outcome_store=store).validate(dry_run=True)
    statuses = {r["status"] for r in tracker.load_all_suggestions()}
    assert statuses == {"applied"}  # 未写入 validated


def test_apply_writes_suggestion_status(tracker, store):
    sid = _seed_applied(tracker, applied_at="2026-01-10T00:00:00")
    _add_outcome(store, "bt_before", "2026-01-05T00:00:00", 0.02)
    _add_outcome(store, "bt_after", "2026-01-15T00:00:00", 0.05)

    SuggestionValidator(tracker=tracker, outcome_store=store).validate(dry_run=False)
    rec = next(r for r in tracker.load_all_suggestions() if r["suggestion_id"] == sid)
    assert rec["status"] == "validated"
    assert rec["validation"]["verdict"] == "validated"
    assert rec["validation"]["after_run_id"] == "bt_after"


# ─────────────────────────────────────────────────────────────
# L2 回写
# ─────────────────────────────────────────────────────────────

def _make_lesson(lib, rule):
    return lib.create_lesson(
        linked_cases=["fc_a", "fc_b"],
        scope={"group": None, "regime": None, "strategy_mode": None},
        failure_level="param", root_cause="阈值问题", rule=rule,
    )


def test_lesson_backfill_validated(tracker, store, lib):
    sid = _seed_applied(tracker, target="score_threshold", applied_at="2026-01-10T00:00:00")
    _add_outcome(store, "bt_before", "2026-01-05T00:00:00", 0.02)
    _add_outcome(store, "bt_after", "2026-01-15T00:00:00", 0.05)
    lesson = _make_lesson(lib, "score_threshold 30 → 38")

    result = SuggestionValidator(
        tracker=tracker, outcome_store=store, lesson_library=lib
    ).validate(dry_run=False)

    assert result["items"][0]["lessons_updated"] == [lesson["lesson_id"]]
    fetched = lib.get_lesson(lesson["lesson_id"])
    assert fetched["validation_status"] == "validated"


def test_lesson_backfill_invalidated(tracker, store, lib):
    _seed_applied(tracker, target="score_threshold", applied_at="2026-01-10T00:00:00")
    _add_outcome(store, "bt_before", "2026-01-05T00:00:00", 0.05)
    _add_outcome(store, "bt_after", "2026-01-15T00:00:00", 0.01)
    lesson = _make_lesson(lib, "把 score_threshold 调到 38")

    result = SuggestionValidator(
        tracker=tracker, outcome_store=store, lesson_library=lib
    ).validate(dry_run=False)

    assert result["items"][0]["lessons_updated"] == [lesson["lesson_id"]]
    assert lib.get_lesson(lesson["lesson_id"])["validation_status"] == "invalidated"


def test_lesson_backfill_skips_unrelated(tracker, store, lib):
    _seed_applied(tracker, target="score_threshold", applied_at="2026-01-10T00:00:00")
    _add_outcome(store, "bt_before", "2026-01-05T00:00:00", 0.02)
    _add_outcome(store, "bt_after", "2026-01-15T00:00:00", 0.05)
    unrelated = _make_lesson(lib, "cooldown_days 调大")

    result = SuggestionValidator(
        tracker=tracker, outcome_store=store, lesson_library=lib
    ).validate(dry_run=False)

    assert result["items"][0]["lessons_updated"] == []
    assert lib.get_lesson(unrelated["lesson_id"])["validation_status"] == "pending"


def test_lesson_backfill_ignores_inconclusive(tracker, store, lib):
    _seed_applied(tracker, target="score_threshold", applied_at="2026-01-10T00:00:00")
    _add_outcome(store, "bt_before", "2026-01-05T00:00:00", 0.050)
    _add_outcome(store, "bt_after", "2026-01-15T00:00:00", 0.055)
    lesson = _make_lesson(lib, "score_threshold 30 → 38")

    SuggestionValidator(
        tracker=tracker, outcome_store=store, lesson_library=lib
    ).validate(dry_run=False)

    assert lib.get_lesson(lesson["lesson_id"])["validation_status"] == "pending"
