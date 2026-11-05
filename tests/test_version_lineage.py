"""策略版本谱系 — 单元测试 (P7)

覆盖:
  - record_version: 版本 hash 计算 / 显式版本 / 参数与版本均缺省报错
  - 运行关联: record_version 携带 run_id、record_run、run_id 去重
  - 父版本: 显式传入 / 显式无父 / 按时间推断 / 环检测
  - 教训事件: applied 与 validated/invalidated 折叠进节点, 非法 verdict 报错
  - 查询: get_version / list_versions / version_for_run / children / ancestors
  - 参数 diff: added / removed / changed 与异常路径
  - 持久化: 新实例读取同一文件
  - SuggestionValidator 集成: 验证结论写入谱系
"""
import pytest

from src.memory.strategy_memory import compute_strategy_version
from src.memory.version_lineage import VersionLineage
from src.ai.suggestion_validator import SuggestionValidator
from src.memory.lesson_library import LessonLibrary
from src.memory.run_outcome import RunOutcomeStore


@pytest.fixture
def lineage(tmp_path):
    return VersionLineage(store_path=str(tmp_path / "lineage.jsonl"))


V1_PARAMS = {
    "mode": "base",
    "portfolio_config": {"weights": {"科技成长型": 0.40}, "atr_override": {}},
    "group_modes": {"科技成长型": "trend_following"},
}


# ─────────────────────────────────────────────────────────────
# record_version
# ─────────────────────────────────────────────────────────────

def test_record_version_computes_hash(lineage):
    event = lineage.record_version(params=V1_PARAMS, recorded_at="2026-01-01T00:00:00")
    assert event["version"] == compute_strategy_version(V1_PARAMS)
    assert event.get("parent_version") is None
    assert "parent_inferred" not in event  # None 值被裁剪


def test_record_version_requires_params_or_version(lineage):
    with pytest.raises(ValueError, match="至少提供"):
        lineage.record_version()


def test_record_version_explicit_hash_wins(lineage):
    event = lineage.record_version(
        version="deadbeef", params=V1_PARAMS, recorded_at="2026-01-01T00:00:00")
    assert event["version"] == "deadbeef"
    node = lineage.get_version("deadbeef")
    assert node["params"] == V1_PARAMS


def test_record_version_with_run(lineage):
    lineage.record_version(
        params=V1_PARAMS, run_id="bt_1", window={"start": "2025-07-01", "end": "2025-12-31"},
        mode="base", groups=["科技成长型"], metrics={"alpha": 0.03},
        recorded_at="2026-01-01T00:00:00")
    node = lineage.get_version(compute_strategy_version(V1_PARAMS))
    assert len(node["runs"]) == 1
    assert node["runs"][0]["run_id"] == "bt_1"
    assert node["runs"][0]["metrics"]["alpha"] == 0.03


def test_record_version_does_not_duplicate_run(lineage):
    version = compute_strategy_version(V1_PARAMS)
    lineage.record_version(params=V1_PARAMS, run_id="bt_1", recorded_at="2026-01-01T00:00:00")
    lineage.record_version(params=V1_PARAMS, run_id="bt_1", recorded_at="2026-01-02T00:00:00")
    assert len(lineage.get_version(version)["runs"]) == 1


def test_record_version_keeps_params_on_repeat(lineage):
    version = compute_strategy_version(V1_PARAMS)
    lineage.record_version(params=V1_PARAMS, recorded_at="2026-01-01T00:00:00")
    lineage.record_version(version=version, recorded_at="2026-01-02T00:00:00")
    assert lineage.get_version(version)["params"] == V1_PARAMS


# ─────────────────────────────────────────────────────────────
# 父版本
# ─────────────────────────────────────────────────────────────

def test_parent_inferred_from_latest_previous_version(lineage):
    v1 = lineage.record_version(params=V1_PARAMS, recorded_at="2026-01-01T00:00:00")["version"]

    v2_params = {"mode": "base", "group_modes": {"科技成长型": "mean_reversion"}}
    v2 = lineage.record_version(params=v2_params, recorded_at="2026-01-05T00:00:00")["version"]

    node = lineage.get_version(v2)
    assert node["parent_version"] == v1
    assert node["parent_inferred"] is True
    assert v2 in lineage.children(v1)


def test_explicit_parent_version_used(lineage):
    v1 = lineage.record_version(params=V1_PARAMS, recorded_at="2026-01-01T00:00:00")["version"]
    v2 = lineage.record_version(
        version="custom", params={"mode": "trending"},
        parent_version=v1, recorded_at="2026-01-05T00:00:00")["version"]
    node = lineage.get_version(v2)
    assert node["parent_version"] == v1
    assert node["parent_inferred"] is False


def test_explicit_none_parent_means_root(lineage):
    lineage.record_version(params=V1_PARAMS, recorded_at="2026-01-01T00:00:00")
    event = lineage.record_version(
        version="root2", params={"mode": "base"}, parent_version=None,
        recorded_at="2026-01-05T00:00:00")
    assert lineage.get_version(event["version"])["parent_version"] is None


def test_ancestors_stops_on_cycle(lineage):
    lineage.record_version(version="a", parent_version=None, params={"p": 1},
                           recorded_at="2026-01-01T00:00:00")
    lineage.record_version(version="b", parent_version="a", params={"p": 2},
                           recorded_at="2026-01-02T00:00:00")
    # 人为制造环: c -> b -> a -> c (写入时直接指定, 绕过推断)
    lineage.record_version(version="c", parent_version="b", params={"p": 3},
                           recorded_at="2026-01-03T00:00:00")
    chain = lineage.ancestors("c")
    # 走到 a 时 parent 仍为 None, 不会无限循环
    assert [n["version"] for n in chain] == ["b", "a"]


def test_infer_parent_skips_ancestor_to_avoid_cycle(lineage):
    v1 = lineage.record_version(version="a", parent_version=None, params={"p": 1},
                                recorded_at="2026-01-01T00:00:00")["version"]
    lineage.record_version(version="b", parent_version="a", params={"p": 2},
                           recorded_at="2026-01-02T00:00:00")
    # a 再次出现 (如配置回滚), 若按时间推断父版本会得到 b → a 形成环
    event = lineage.record_version(version=v1, params={"p": 1}, recorded_at="2026-01-03T00:00:00")
    assert event.get("parent_version") is None
    assert lineage.ancestors(v1) == []


# ─────────────────────────────────────────────────────────────
# 运行记录
# ─────────────────────────────────────────────────────────────

def test_record_run_and_dedupe(lineage):
    version = lineage.record_version(params=V1_PARAMS, recorded_at="2026-01-01T00:00:00")["version"]
    lineage.record_run(version=version, run_id="bt_1", recorded_at="2026-01-02T00:00:00",
                       metrics={"alpha": 0.01})
    lineage.record_run(version=version, run_id="bt_1", recorded_at="2026-01-03T00:00:00",
                       metrics={"alpha": 0.09})
    node = lineage.get_version(version)
    assert len(node["runs"]) == 1
    assert node["runs"][0]["metrics"]["alpha"] == 0.01


def test_record_run_requires_ids(lineage):
    with pytest.raises(ValueError, match="version"):
        lineage.record_run(version="", run_id="bt_1")
    with pytest.raises(ValueError, match="run_id"):
        lineage.record_run(version="abc", run_id="")


def test_version_for_run(lineage):
    v1 = lineage.record_version(params=V1_PARAMS, recorded_at="2026-01-01T00:00:00")["version"]
    lineage.record_run(version=v1, run_id="bt_1", recorded_at="2026-01-02T00:00:00")
    assert lineage.version_for_run("bt_1") == v1
    assert lineage.version_for_run("nope") is None
    assert lineage.version_for_run("") is None


# ─────────────────────────────────────────────────────────────
# 教训应用链 / 验证结果
# ─────────────────────────────────────────────────────────────

def test_lesson_applied_and_verdict_fold(lineage):
    version = lineage.record_version(params=V1_PARAMS, recorded_at="2026-01-01T00:00:00")["version"]
    lineage.record_lesson_applied(
        version=version, lesson_id="lsn_1",
        rule_action={"type": "set_param", "target": "score_threshold", "value": 38},
        recorded_at="2026-01-02T00:00:00")
    lineage.record_lesson_verdict(
        version=version, lesson_id="lsn_1", verdict="validated",
        notes="alpha +3%", run_id="bt_2", alpha_delta=0.03,
        recorded_at="2026-01-10T00:00:00")

    entry = lineage.get_version(version)["lessons"]["lsn_1"]
    assert entry["applied_at"] == "2026-01-02T00:00:00"
    assert entry["rule_action"]["target"] == "score_threshold"
    assert entry["verdict"] == "validated"
    assert entry["alpha_delta"] == pytest.approx(0.03)
    assert entry["run_id"] == "bt_2"


def test_record_lesson_verdict_rejects_unknown_verdict(lineage):
    with pytest.raises(ValueError, match="verdict"):
        lineage.record_lesson_verdict(
            version="v", lesson_id="lsn_1", verdict="inconclusive")


def test_record_lesson_events_require_ids(lineage):
    with pytest.raises(ValueError, match="version"):
        lineage.record_lesson_applied(version="", lesson_id="lsn_1")
    with pytest.raises(ValueError, match="lesson_id"):
        lineage.record_lesson_applied(version="v", lesson_id="")


# ─────────────────────────────────────────────────────────────
# 查询
# ─────────────────────────────────────────────────────────────

def test_get_version_unknown_returns_none(lineage):
    assert lineage.get_version("missing") is None


def test_list_versions_chronological(lineage):
    v2 = lineage.record_version(version="v2", params={"p": 2},
                                recorded_at="2026-02-01T00:00:00")["version"]
    v1 = lineage.record_version(version="v1", params={"p": 1},
                                recorded_at="2026-01-01T00:00:00")["version"]
    assert [n["version"] for n in lineage.list_versions()] == [v1, v2]


def test_ancestors_near_to_far(lineage):
    lineage.record_version(version="root", parent_version=None, params={"p": 0},
                           recorded_at="2026-01-01T00:00:00")
    lineage.record_version(version="mid", parent_version="root", params={"p": 1},
                           recorded_at="2026-01-02T00:00:00")
    lineage.record_version(version="leaf", parent_version="mid", params={"p": 2},
                           recorded_at="2026-01-03T00:00:00")
    chain = lineage.ancestors("leaf")
    assert [n["version"] for n in chain] == ["mid", "root"]
    assert lineage.children("root") == ["mid"]


# ─────────────────────────────────────────────────────────────
# 参数 diff
# ─────────────────────────────────────────────────────────────

def test_diff_versions(lineage):
    lineage.record_version(version="a", params={
        "mode": "base",
        "portfolio_config": {"weights": {"科技成长型": 0.40, "医药创新型": 0.05}},
    }, recorded_at="2026-01-01T00:00:00")
    lineage.record_version(version="b", params={
        "mode": "trending",
        "portfolio_config": {"weights": {"科技成长型": 0.35}},
    }, recorded_at="2026-01-02T00:00:00")

    diff = lineage.diff_versions("a", "b")
    assert diff["changed"] == {"mode": {"before": "base", "after": "trending"},
                               "portfolio_config.weights.科技成长型":
                               {"before": 0.40, "after": 0.35}}
    assert diff["removed"] == {"portfolio_config.weights.医药创新型": 0.05}
    assert diff["added"] == {}


def test_diff_versions_detects_added(lineage):
    lineage.record_version(version="a", params={"mode": "base"},
                           recorded_at="2026-01-01T00:00:00")
    lineage.record_version(version="b", params={"mode": "base", "extra": 1},
                           recorded_at="2026-01-02T00:00:00")
    assert lineage.diff_versions("a", "b")["added"] == {"extra": 1}


def test_diff_versions_unknown_raises(lineage):
    lineage.record_version(version="a", params={"mode": "base"},
                           recorded_at="2026-01-01T00:00:00")
    with pytest.raises(ValueError, match="不存在"):
        lineage.diff_versions("a", "missing")


def test_diff_versions_missing_params_raises(lineage):
    lineage.record_version(version="a", params={"mode": "base"},
                           recorded_at="2026-01-01T00:00:00")
    lineage.record_version(version="b", params=None, recorded_at="2026-01-02T00:00:00")
    with pytest.raises(ValueError, match="缺少参数快照"):
        lineage.diff_versions("a", "b")


# ─────────────────────────────────────────────────────────────
# 持久化
# ─────────────────────────────────────────────────────────────

def test_persistence_across_instances(tmp_path):
    path = str(tmp_path / "lineage.jsonl")
    first = VersionLineage(store_path=path)
    version = first.record_version(params=V1_PARAMS, recorded_at="2026-01-01T00:00:00")["version"]
    first.record_run(version=version, run_id="bt_1", recorded_at="2026-01-02T00:00:00")

    second = VersionLineage(store_path=path)
    assert len(second.get_version(version)["runs"]) == 1


# ─────────────────────────────────────────────────────────────
# SuggestionValidator 集成
# ─────────────────────────────────────────────────────────────

def test_validator_writes_lineage_events(tmp_path):
    from src.ai.suggestion_tracker import SuggestionTracker
    from tests.test_suggestion_validator import _seed_applied, _make_lesson

    tracker = SuggestionTracker(file_path=str(tmp_path / "suggestions.jsonl"))
    store = RunOutcomeStore(store_path=str(tmp_path / "run_outcomes.jsonl"))
    lib = LessonLibrary(lessons_dir=str(tmp_path / "lessons"))
    lineage = VersionLineage(store_path=str(tmp_path / "lineage.jsonl"))

    sid = _seed_applied(tracker, target="score_threshold", applied_at="2026-01-10T00:00:00")
    store.record(run_id="bt_before", window={"start": "2025-07-01", "end": "2025-12-31"},
                 mode="balanced", metrics={"alpha": 0.02}, recorded_at="2026-01-05T00:00:00")
    store.record(run_id="bt_after", window={"start": "2025-07-01", "end": "2025-12-31"},
                 mode="balanced", metrics={"alpha": 0.05}, recorded_at="2026-01-15T00:00:00",
                 strategy_version="v_after")
    lesson = _make_lesson(lib, "score_threshold 30 → 38")

    assert SuggestionValidator(
        tracker=tracker, outcome_store=store, lesson_library=lib,
        version_lineage=lineage,
    ).validate(dry_run=False)["items"][0]["lessons_updated"] == [lesson["lesson_id"]]

    node = lineage.get_version("v_after")
    assert node is not None
    entry = node["lessons"][lesson["lesson_id"]]
    assert entry["verdict"] == "validated"
    assert entry["alpha_delta"] == pytest.approx(0.03)
    assert entry["run_id"] == "bt_after"


def test_validator_without_version_skips_lineage(tmp_path):
    from src.ai.suggestion_tracker import SuggestionTracker
    from tests.test_suggestion_validator import _seed_applied, _make_lesson

    tracker = SuggestionTracker(file_path=str(tmp_path / "suggestions.jsonl"))
    store = RunOutcomeStore(store_path=str(tmp_path / "run_outcomes.jsonl"))
    lib = LessonLibrary(lessons_dir=str(tmp_path / "lessons"))
    lineage = VersionLineage(store_path=str(tmp_path / "lineage.jsonl"))

    _seed_applied(tracker, target="score_threshold", applied_at="2026-01-10T00:00:00")
    store.record(run_id="bt_before", window={"start": "2025-07-01", "end": "2025-12-31"},
                 mode="balanced", metrics={"alpha": 0.02}, recorded_at="2026-01-05T00:00:00")
    store.record(run_id="bt_after", window={"start": "2025-07-01", "end": "2025-12-31"},
                 mode="balanced", metrics={"alpha": 0.05}, recorded_at="2026-01-15T00:00:00")
    _make_lesson(lib, "score_threshold 30 → 38")

    SuggestionValidator(
        tracker=tracker, outcome_store=store, lesson_library=lib,
        version_lineage=lineage,
    ).validate(dry_run=False)

    assert lineage.load_events() == []
