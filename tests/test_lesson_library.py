"""失效案例库 L2 — 单元测试 (P2)

覆盖:
  - min-support 约束 / failure_level 校验
  - confidence 自动推导 (2=low, 3-4=med, 5+=high)
  - CRUD (create/get/list/update/delete)
  - 状态机 pending → applied → validated|invalidated 及非法跃迁
  - scope 匹配 (非 null 字段约束)
  - find_relevant_lessons 排序 (validated > applied > pending, confidence, support)
  - update_lesson 重算 confidence
  - AI root_cause 起草回退模板 (无 LLM 环境)
  - load_l1_records 按 case_ids 过滤
"""
import json
import os

import pytest

from src.memory.lesson_library import (
    LESSON_STATUSES,
    LessonLibrary,
    infer_scope_from_records,
    load_l1_records,
)


@pytest.fixture
def lib(tmp_path):
    return LessonLibrary(lessons_dir=str(tmp_path / "lessons"))


def _make_lesson(lib, **overrides):
    base = dict(
        linked_cases=["fc_a", "fc_b"],
        scope={"group": "医药创新型", "regime": None, "strategy_mode": "mean_reversion"},
        failure_level="param",
        root_cause="score_threshold 偏低",
        rule="score_threshold 30 -> 38",
        do_not_repeat=["score_threshold <= 30"],
        failure_flags_summary=["alpha_negative", "param_scan_clear"],
    )
    base.update(overrides)
    return lib.create_lesson(**base)


# ─────────────────────────────────────────────────────────────
# 约束校验
# ─────────────────────────────────────────────────────────────

def test_min_support_required(lib):
    with pytest.raises(ValueError, match="至少需要 2"):
        lib.create_lesson(
            linked_cases=["fc_only_one"],
            scope={"group": "X"},
            failure_level="param",
            root_cause="r", rule="rule",
        )


def test_empty_linked_cases_rejected(lib):
    with pytest.raises(ValueError):
        lib.create_lesson(
            linked_cases=[], scope={}, failure_level="param",
            root_cause="r", rule="rule",
        )


def test_invalid_failure_level_rejected(lib):
    with pytest.raises(ValueError, match="failure_level"):
        lib.create_lesson(
            linked_cases=["fc_a", "fc_b"], scope={},
            failure_level="not_a_level", root_cause="r", rule="rule",
        )


def test_linked_cases_deduped(lib):
    lesson = lib.create_lesson(
        linked_cases=["fc_a", "fc_a", "fc_b", "fc_b"],
        scope={}, failure_level="param", root_cause="r", rule="rule",
    )
    assert lesson["linked_cases"] == ["fc_a", "fc_b"]
    assert lesson["support_count"] == 2


# ─────────────────────────────────────────────────────────────
# confidence 推导
# ─────────────────────────────────────────────────────────────

@pytest.mark.parametrize("n,expected", [
    (2, "low"), (3, "med"), (4, "med"), (5, "high"), (10, "high"),
])
def test_confidence_by_support(lib, n, expected):
    cases = [f"fc_{i}" for i in range(n)]
    lesson = lib.create_lesson(
        linked_cases=cases, scope={}, failure_level="param",
        root_cause="r", rule="rule",
    )
    assert lesson["confidence"] == expected


# ─────────────────────────────────────────────────────────────
# CRUD
# ─────────────────────────────────────────────────────────────

def test_create_and_get(lib):
    lesson = _make_lesson(lib)
    fetched = lib.get_lesson(lesson["lesson_id"])
    assert fetched is not None
    assert fetched["lesson_id"] == lesson["lesson_id"]
    assert fetched["validation_status"] == "pending"
    assert fetched["ai_drafted"] is False
    assert set(fetched["scope"]) == {"group", "regime", "strategy_mode"}


def test_get_missing_returns_none(lib):
    assert lib.get_lesson("lsn_nonexistent") is None


def test_list_lessons_empty(lib):
    assert lib.list_lessons() == []


def test_list_lessons_sorted_by_created_at(lib):
    import time
    l1 = _make_lesson(lib, root_cause="first")
    time.sleep(1.0)  # created_at 精度到秒, 需跨秒才能区分
    l2 = _make_lesson(lib, root_cause="second")
    listed = lib.list_lessons()
    assert [l["lesson_id"] for l in listed] == [l2["lesson_id"], l1["lesson_id"]]


def test_list_filter_by_status(lib):
    l1 = _make_lesson(lib, root_cause="a")
    l2 = _make_lesson(lib, root_cause="b")
    lib.mark_applied(l1["lesson_id"], "run_1")
    pending = lib.list_lessons(status="pending")
    applied = lib.list_lessons(status="applied")
    assert len(pending) == 1 and pending[0]["lesson_id"] == l2["lesson_id"]
    assert len(applied) == 1 and applied[0]["lesson_id"] == l1["lesson_id"]


def test_list_filter_by_failure_level(lib):
    _make_lesson(lib, failure_level="param")
    _make_lesson(lib, failure_level="overfit")
    assert len(lib.list_lessons(failure_level="param")) == 1
    assert len(lib.list_lessons(failure_level="overfit")) == 1


def test_update_mutable_fields(lib):
    lesson = _make_lesson(lib)
    changed = lib.update_lesson(
        lesson["lesson_id"],
        root_cause="新根因",
        rule="新规则",
        do_not_repeat=["禁忌1", "禁忌2"],
    )
    assert changed is True
    updated = lib.get_lesson(lesson["lesson_id"])
    assert updated["root_cause"] == "新根因"
    assert updated["rule"] == "新规则"
    assert updated["do_not_repeat"] == ["禁忌1", "禁忌2"]
    assert updated["updated_at"] >= lesson["updated_at"]


def test_update_ignores_immutable_fields(lib):
    lesson = _make_lesson(lib)
    # validation_status 是不可变字段, 走专用方法
    lib.update_lesson(lesson["lesson_id"], validation_status="validated")
    fetched = lib.get_lesson(lesson["lesson_id"])
    assert fetched["validation_status"] == "pending"  # 未变


def test_update_linked_cases_recompute_confidence(lib):
    lesson = _make_lesson(lib)  # 2 cases -> low
    assert lesson["confidence"] == "low"
    lib.update_lesson(
        lesson["lesson_id"],
        linked_cases=["fc_a", "fc_b", "fc_c", "fc_d", "fc_e"],
    )
    updated = lib.get_lesson(lesson["lesson_id"])
    assert updated["support_count"] == 5
    assert updated["confidence"] == "high"


def test_update_linked_cases_below_min_rejected(lib):
    lesson = _make_lesson(lib)
    with pytest.raises(ValueError, match="仍需 >= 2"):
        lib.update_lesson(lesson["lesson_id"], linked_cases=["fc_a"])


def test_update_invalid_failure_level_rejected(lib):
    lesson = _make_lesson(lib)
    with pytest.raises(ValueError):
        lib.update_lesson(lesson["lesson_id"], failure_level="bad")


def test_update_missing_returns_false(lib):
    assert lib.update_lesson("lsn_nope", root_cause="x") is False


def test_delete(lib):
    lesson = _make_lesson(lib)
    assert lib.delete_lesson(lesson["lesson_id"]) is True
    assert lib.get_lesson(lesson["lesson_id"]) is None
    assert lib.delete_lesson(lesson["lesson_id"]) is False  # 幂等


# ─────────────────────────────────────────────────────────────
# 状态机
# ─────────────────────────────────────────────────────────────

def test_full_status_lifecycle(lib):
    lesson = _make_lesson(lib)
    assert lesson["validation_status"] == "pending"

    assert lib.mark_applied(lesson["lesson_id"], "run_apply")
    fetched = lib.get_lesson(lesson["lesson_id"])
    assert fetched["validation_status"] == "applied"
    assert fetched["applied_by"] == "run_apply"

    assert lib.mark_validated(lesson["lesson_id"], "run_validate", notes="ok")
    fetched = lib.get_lesson(lesson["lesson_id"])
    assert fetched["validation_status"] == "validated"
    assert fetched["validated_by"] == "run_validate"
    assert fetched["validation_notes"] == "ok"


def test_applied_to_invalidated(lib):
    lesson = _make_lesson(lib)
    lib.mark_applied(lesson["lesson_id"], "run_apply")
    assert lib.mark_invalidated(lesson["lesson_id"], "run_inv", notes="worsened")
    assert lib.get_lesson(lesson["lesson_id"])["validation_status"] == "invalidated"


def test_invalid_transition_validated_directly(lib):
    """pending 不能直接 validated, 必须先 applied"""
    lesson = _make_lesson(lib)
    assert lib.mark_validated(lesson["lesson_id"], "run_x") is False
    assert lib.get_lesson(lesson["lesson_id"])["validation_status"] == "pending"


def test_invalid_transition_applied_twice(lib):
    """validated 状态不能再 applied"""
    lesson = _make_lesson(lib)
    lib.mark_applied(lesson["lesson_id"], "run_1")
    lib.mark_validated(lesson["lesson_id"], "run_2")
    assert lib.mark_applied(lesson["lesson_id"], "run_3") is False


def test_statuses_constant(lib):
    assert set(LESSON_STATUSES) == {"pending", "applied", "validated", "invalidated"}


# ─────────────────────────────────────────────────────────────
# scope 匹配 / 检索
# ─────────────────────────────────────────────────────────────

def test_scope_matches_exact(lib):
    _make_lesson(lib, scope={"group": "医药", "regime": None, "strategy_mode": "mr"})
    res = lib.list_lessons(scope={"group": "医药", "strategy_mode": "mr"})
    assert len(res) == 1


def test_scope_no_match_on_group(lib):
    _make_lesson(lib, scope={"group": "医药", "regime": None, "strategy_mode": "mr"})
    res = lib.list_lessons(scope={"group": "科技"})
    assert len(res) == 0


def test_scope_null_in_lesson_is_wildcard(lib):
    """lesson.scope.regime=None 表示通配, 任何 regime 查询都匹配"""
    _make_lesson(lib, scope={"group": "医药", "regime": None, "strategy_mode": "mr"})
    res = lib.list_lessons(scope={"group": "医药", "regime": "ranging"})
    assert len(res) == 1


def test_scope_null_in_query_no_constraint(lib):
    """query.scope 某字段为 None 表示不约束该字段"""
    _make_lesson(lib, scope={"group": "医药", "regime": None, "strategy_mode": "mr"})
    _make_lesson(lib, scope={"group": "科技", "regime": None, "strategy_mode": "tf"})
    res = lib.list_lessons(scope={"strategy_mode": "mr"})
    assert len(res) == 1


def test_find_relevant_ranking(lib):
    """validated > applied > pending; 同状态按 confidence, support"""
    scope = {"group": "医药", "strategy_mode": "mr"}
    low = _make_lesson(lib, root_cause="low", scope=scope)  # pending -> applied, low
    lib.mark_applied(low["lesson_id"], "r1")

    high = lib.create_lesson(
        linked_cases=["fc_1", "fc_2", "fc_3", "fc_4", "fc_5"],
        scope=scope,
        failure_level="param", root_cause="h", rule="r",
    )  # pending, high

    validated = lib.create_lesson(
        linked_cases=["fc_a", "fc_b"],
        scope=scope,
        failure_level="param", root_cause="v", rule="r",
    )
    lib.mark_applied(validated["lesson_id"], "r2")
    lib.mark_validated(validated["lesson_id"], "r3")

    res = lib.find_relevant_lessons(scope=scope)
    order = [r["lesson_id"] for r in res]
    assert order[0] == validated["lesson_id"]  # validated 第一
    assert order[1] == low["lesson_id"]        # applied 第二 (虽 confidence 低)
    assert order[2] == high["lesson_id"]       # pending 第三 (虽 confidence 高)


def test_find_relevant_filter_by_status(lib):
    _make_lesson(lib, root_cause="a")
    applied = _make_lesson(lib, root_cause="b")
    lib.mark_applied(applied["lesson_id"], "r1")
    res = lib.find_relevant_lessons(scope={}, status="applied")
    assert len(res) == 1
    assert res[0]["lesson_id"] == applied["lesson_id"]


# ─────────────────────────────────────────────────────────────
# AI root_cause 起草 (无 LLM 环境 → 模板回退)
# ─────────────────────────────────────────────────────────────

def _sample_l1_records():
    return [
        {
            "case_id": "fc_a", "group_name": "医药", "strategy_mode": "mean_reversion",
            "failure_level": "param",
            "failure_flags": ["alpha_negative", "param_scan_clear"],
            "metrics": {"alpha": -0.05, "max_drawdown": -0.08, "stop_loss_rate": 0.5},
        },
        {
            "case_id": "fc_b", "group_name": "医药", "strategy_mode": "mean_reversion",
            "failure_level": "param",
            "failure_flags": ["alpha_negative"],
            "metrics": {"alpha": -0.03},
        },
    ]


def test_draft_root_cause_fallback_template(lib, monkeypatch):
    # 强制 LLM 不可用, 走模板回退
    import src.ai.llm_client as llm_mod
    monkeypatch.setattr(llm_mod.LLMClient, "is_available", property(lambda self: False))

    draft = lib.draft_root_cause(_sample_l1_records())
    assert "根因" in draft
    assert "改进" in draft
    assert "禁忌" in draft
    assert "param" in draft
    assert "医药" in draft


def test_draft_root_cause_empty(lib):
    assert lib.draft_root_cause([]) == ""


def test_draft_template_summarizes_flags(lib, monkeypatch):
    import src.ai.llm_client as llm_mod
    monkeypatch.setattr(llm_mod.LLMClient, "is_available", property(lambda self: False))

    draft = lib.draft_root_cause(_sample_l1_records())
    assert "alpha_negative" in draft
    assert "param_scan_clear" in draft


def test_draft_root_cause_uses_llm_when_available(lib, monkeypatch):
    """LLM 可用时, 调用 chat 并返回其输出 (不走模板)"""
    import src.ai.llm_client as llm_mod

    class _FakeLLM:
        is_available = True

        def chat(self, system_prompt, user_prompt, temperature=0.3, max_tokens=4096):
            return "LLM生成的根因分析"

    monkeypatch.setattr(llm_mod, "LLMClient", lambda: _FakeLLM())

    draft = lib.draft_root_cause(_sample_l1_records())
    assert draft == "LLM生成的根因分析"


# ─────────────────────────────────────────────────────────────
# load_l1_records
# ─────────────────────────────────────────────────────────────

def test_load_l1_records_missing_file(tmp_path):
    assert load_l1_records(store_path=str(tmp_path / "no.jsonl")) == []


def test_load_l1_records_filter_by_ids(tmp_path):
    store = tmp_path / "l1.jsonl"
    records = [
        {"case_id": "fc_1", "failure_flags": ["alpha_negative"]},
        {"case_id": "fc_2", "failure_flags": ["drawdown_deep"]},
        {"case_id": "fc_3", "failure_flags": []},
    ]
    with open(store, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    all_recs = load_l1_records(store_path=str(store))
    assert len(all_recs) == 3

    subset = load_l1_records(case_ids=["fc_1", "fc_3"], store_path=str(store))
    assert {r["case_id"] for r in subset} == {"fc_1", "fc_3"}


def test_load_l1_records_skips_malformed_lines(tmp_path):
    store = tmp_path / "l1.jsonl"
    with open(store, "w", encoding="utf-8") as f:
        f.write(json.dumps({"case_id": "fc_good"}) + "\n")
        f.write("not json\n")
        f.write("\n")
    recs = load_l1_records(store_path=str(store))
    assert len(recs) == 1
    assert recs[0]["case_id"] == "fc_good"


# ─────────────────────────────────────────────────────────────
# 持久化: 跨实例读取
# ─────────────────────────────────────────────────────────────

def test_persistence_across_instances(tmp_path):
    lessons_dir = str(tmp_path / "lessons")
    lib1 = LessonLibrary(lessons_dir=lessons_dir)
    lesson = _make_lesson(lib1)

    lib2 = LessonLibrary(lessons_dir=lessons_dir)
    fetched = lib2.get_lesson(lesson["lesson_id"])
    assert fetched is not None
    assert fetched["root_cause"] == lesson["root_cause"]


# ─────────────────────────────────────────────────────────────
# scope.regime 推断 (P3 接入)
# ─────────────────────────────────────────────────────────────

def _l1_rec(regime="ranging", share=0.6, alpha=-0.05,
            group="医药", mode="mr", case_id=None, nested=False):
    """构造一条带 regime_slices 的 L1 记录"""
    slices = {
        regime: {"days": int(share * 100), "share": share, "alpha": alpha},
        "trending": {"days": int((1 - share) * 100), "share": round(1 - share, 2),
                     "alpha": 0.02},
    }
    rec = {
        "case_id": case_id or f"fc_{regime}_{abs(hash((regime, share, alpha))) % 9999}",
        "group_name": group,
        "strategy_mode": mode,
    }
    if nested:
        rec["metrics"] = {"regime_slices": slices}
    else:
        rec["regime_slices"] = slices
    return rec


def test_infer_scope_unique_group_mode_and_mismatch_regime():
    recs = [_l1_rec(case_id="a"), _l1_rec(case_id="b")]
    scope = infer_scope_from_records(recs)
    assert scope["group"] == "医药"
    assert scope["strategy_mode"] == "mr"
    assert scope["regime"] == "ranging"


def test_infer_scope_mixed_group_gives_none():
    recs = [_l1_rec(group="医药", case_id="a"), _l1_rec(group="科技", case_id="b")]
    scope = infer_scope_from_records(recs)
    assert scope["group"] is None
    assert scope["regime"] == "ranging"  # regime 仍可推断


def test_infer_scope_no_slices_gives_none_regime():
    recs = [
        {"case_id": "a", "group_name": "医药", "strategy_mode": "mr"},
        {"case_id": "b", "group_name": "医药", "strategy_mode": "mr"},
    ]
    scope = infer_scope_from_records(recs)
    assert scope["regime"] is None
    assert scope["group"] == "医药"


def test_infer_scope_regime_not_mismatch_gives_none():
    """主导体制 alpha 为正 → 不算失配 → 不绑定 regime"""
    recs = [_l1_rec(alpha=0.03, case_id="a"), _l1_rec(alpha=0.03, case_id="b")]
    scope = infer_scope_from_records(recs)
    assert scope["regime"] is None


def test_infer_scope_majority_regime():
    """3 条中 2 条主导体制为 ranging → 取 ranging"""
    recs = [
        _l1_rec(regime="ranging", share=0.7, case_id="a"),
        _l1_rec(regime="ranging", share=0.7, case_id="b"),
        _l1_rec(regime="trending", share=0.7, alpha=-0.05, case_id="c"),
    ]
    scope = infer_scope_from_records(recs)
    assert scope["regime"] == "ranging"


def test_infer_scope_reads_nested_metrics_slices():
    """regime_slices 内嵌在 metrics 里也能读到"""
    recs = [_l1_rec(case_id="a", nested=True), _l1_rec(case_id="b", nested=True)]
    scope = infer_scope_from_records(recs)
    assert scope["regime"] == "ranging"


def test_infer_scope_empty():
    assert infer_scope_from_records([]) == {
        "group": None, "regime": None, "strategy_mode": None,
    }


def test_infer_scope_custom_thresholds():
    """主导体制 share=0.55, 门槛提到 0.7 → 不再算主导 → regime None"""
    recs = [_l1_rec(share=0.55, case_id="a"), _l1_rec(share=0.55, case_id="b")]
    scope = infer_scope_from_records(recs, {"regime_mismatch_share": 0.7})
    assert scope["regime"] is None


def test_evolution_lookup_by_regime(lib):
    """创建带 regime 的教训后, 进化器按当前体制检索"""
    scope = {"group": "医药", "regime": "ranging", "strategy_mode": "mr"}
    lib.create_lesson(
        linked_cases=["fc_a", "fc_b"], scope=scope, failure_level="regime_mismatch",
        root_cause="震荡市均值回归失效", rule="ranging 降低仓位",
    )
    hit = lib.find_relevant_lessons(scope={"regime": "ranging", "group": "医药"})
    miss = lib.find_relevant_lessons(scope={"regime": "trending"})
    assert len(hit) == 1
    assert len(miss) == 0


def test_evolution_lookup_regime_wildcard(lib):
    """lesson.regime=None 时, 任意体制查询都命中"""
    lib.create_lesson(
        linked_cases=["fc_a", "fc_b"],
        scope={"group": "医药", "regime": None, "strategy_mode": "mr"},
        failure_level="param", root_cause="参数问题", rule="调参",
    )
    for reg in ("ranging", "trending", "transition"):
        res = lib.find_relevant_lessons(scope={"regime": reg})
        assert len(res) == 1
