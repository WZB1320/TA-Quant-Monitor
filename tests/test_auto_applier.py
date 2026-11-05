"""P8 自动应用器 — 单元测试

覆盖:
  - compare_runs 纯函数: win / lose / inconclusive / error / 自定义阈值
  - build_candidate: validated 动作路由 (组参数 / atr_override / 通配 scope /
    非活跃组跳过 / target 不存在跳过 / pending 不消费 / 空教训库)
  - build_candidate 不修改入参 groups_cfg
  - run_ab 集成 (monkeypatch run_portfolio_backtest): 双跑调用参数、
    候选不写记忆层、谱系记录 parent 链与教训事件、基线失败短路
"""
import pytest

from src.evolution.auto_applier import EvolutionAB, build_candidate, compare_runs
from src.memory.lesson_library import LessonLibrary
from src.memory.version_lineage import VersionLineage


GROUPS = {
    "医药创新型": {
        "strategy_mode": "mean_reversion",
        "score_threshold": 30,
        "mean_reversion_exit": {"target_profit_pct": 0.05},
        "atr_stop_mult": 2.0,
    },
    "科技成长型": {"strategy_mode": "trend_following", "score_threshold": 25},
}
ACTIVE = ["医药创新型", "科技成长型"]


@pytest.fixture
def lib(tmp_path):
    return LessonLibrary(lessons_dir=str(tmp_path / "lessons"))


def _add_lesson(lib, *, group, target, action_type="set_param", value=38,
                status="validated", delta=None, n_cases=2):
    action = {"type": action_type, "target": target}
    if action_type == "set_param":
        action["value"] = value
    else:
        action["delta"] = delta if delta is not None else value
    lesson = lib.create_lesson(
        linked_cases=[f"fc_{n_cases}a", f"fc_{n_cases}b"],
        scope={"group": group, "regime": None, "strategy_mode": None},
        failure_level="param",
        root_cause="测试根因",
        rule="测试规则",
        rule_action=action,
    )
    if status == "validated":
        # 状态机: pending -> applied -> validated
        lib.mark_applied(lesson["lesson_id"], "run_t")
        lib.mark_validated(lesson["lesson_id"], "run_t", notes="test")
    elif status == "applied":
        lib.mark_applied(lesson["lesson_id"], "run_t")
    return lesson


# ─────────────────────────────────────────────────────────────
# compare_runs
# ─────────────────────────────────────────────────────────────

def test_compare_win():
    out = compare_runs({"alpha": 0.02}, {"alpha": 0.05})
    assert out["verdict"] == "win"
    assert out["alpha_delta"] == pytest.approx(0.03)


def test_compare_lose():
    out = compare_runs({"alpha": 0.05}, {"alpha": 0.02})
    assert out["verdict"] == "lose"


def test_compare_inconclusive():
    out = compare_runs({"alpha": 0.03}, {"alpha": 0.035})
    assert out["verdict"] == "inconclusive"


def test_compare_missing_alpha():
    assert compare_runs({}, {"alpha": 0.05})["verdict"] == "error"


def test_compare_custom_thresholds():
    out = compare_runs({"alpha": 0.03}, {"alpha": 0.05},
                       thresholds={"alpha_improve": 0.015, "alpha_worsen": -0.015})
    assert out["verdict"] == "win"


# ─────────────────────────────────────────────────────────────
# build_candidate
# ─────────────────────────────────────────────────────────────

def test_build_candidate_empty_library(lib):
    out = build_candidate(GROUPS, ACTIVE, lesson_library=lib)
    assert out["groups"] == {} and out["atr_override"] == {}
    assert out["changes"] == [] and out["lesson_ids"] == []


def test_build_candidate_applies_group_param(lib):
    _add_lesson(lib, group="医药创新型", target="score_threshold", value=38)
    out = build_candidate(GROUPS, ACTIVE, lesson_library=lib)

    assert out["groups"]["医药创新型"]["score_threshold"] == 38
    # 未涉及的组不被修改
    assert "科技成长型" not in out["groups"]
    ch = out["changes"][0]
    assert ch["before"] == 30 and ch["after"] == 38 and ch["via"] == "group_config"


def test_build_candidate_nested_target(lib):
    _add_lesson(lib, group="医药创新型", target="target_profit_pct", value=0.08)
    out = build_candidate(GROUPS, ACTIVE, lesson_library=lib)
    assert out["groups"]["医药创新型"]["mean_reversion_exit"]["target_profit_pct"] == 0.08


def test_build_candidate_atr_routes_to_override(lib):
    _add_lesson(lib, group="科技成长型", target="atr_stop_mult", value=1.5)
    out = build_candidate(GROUPS, ACTIVE, lesson_library=lib)
    assert out["atr_override"] == {"科技成长型": 1.5}
    assert "科技成长型" not in out["groups"]


def test_build_candidate_atr_adjust_skipped(lib):
    _add_lesson(lib, group="科技成长型", target="atr_stop_mult",
                action_type="adjust_param", delta=-0.2)
    out = build_candidate(GROUPS, ACTIVE, lesson_library=lib)
    assert out["atr_override"] == {}
    assert any("不支持 adjust_param" in s["reason"] for s in out["skipped"])


def test_build_candidate_wildcard_scope_applies_to_all(lib):
    # 两组都有 score_threshold, 通配 scope 应应用到全部活跃组
    _add_lesson(lib, group=None, target="score_threshold", value=38)
    out = build_candidate(GROUPS, ACTIVE, lesson_library=lib)
    assert set(out["groups"].keys()) == set(ACTIVE)
    assert all(c["score_threshold"] == 38 for c in out["groups"].values())


def test_build_candidate_non_active_group_skipped(lib):
    _add_lesson(lib, group="消费稳健型", target="score_threshold", value=40)
    out = build_candidate(GROUPS, ACTIVE, lesson_library=lib)
    assert out["groups"] == {}
    assert any("不在活跃组中" in s["reason"] for s in out["skipped"])


def test_build_candidate_unknown_target_skipped(lib):
    _add_lesson(lib, group="医药创新型", target="no_such_param", value=1)
    out = build_candidate(GROUPS, ACTIVE, lesson_library=lib)
    assert out["groups"] == {}
    assert any("不存在" in s["reason"] for s in out["skipped"])


def test_build_candidate_ignores_non_validated(lib):
    lesson = _add_lesson(lib, group="医药创新型", target="score_threshold",
                         value=38, status="applied")
    out = build_candidate(GROUPS, ACTIVE, lesson_library=lib)
    assert out["groups"] == {}
    assert lesson["lesson_id"] not in out["lesson_ids"]


def test_build_candidate_does_not_mutate_input(lib):
    import copy
    snapshot = copy.deepcopy(GROUPS)
    _add_lesson(lib, group="医药创新型", target="score_threshold", value=38)
    build_candidate(GROUPS, ACTIVE, lesson_library=lib)
    assert GROUPS == snapshot


# ─────────────────────────────────────────────────────────────
# run_ab 集成 (mock 双跑)
# ─────────────────────────────────────────────────────────────

class _FakeReq:
    mode = "base"


def _patch_runs(monkeypatch, baseline, candidate, calls):
    import web.backend.services.portfolio_backtest as pb

    def fake_run(req, on_progress=None, config_override=None, record_memory=True):
        calls.append({"config_override": config_override,
                      "record_memory": record_memory})
        return candidate if config_override else baseline

    monkeypatch.setattr(pb, "run_portfolio_backtest", fake_run)


def _baseline_result(alpha=0.02, group_alpha=0.02):
    return {"status": "done", "run_id": "bt_base", "strategy_version": "v_base",
            "metrics": {"alpha": alpha, "max_drawdown": -0.1, "win_rate": 0.5},
            "group_metrics": {"医药创新型": {"alpha": group_alpha, "total_return": 0.01,
                                          "trade_count": 3}}}


def _candidate_result(alpha=0.05, group_alpha=0.05):
    return {"status": "done", "run_id": "bt_cand", "strategy_version": "v_cand",
            "metrics": {"alpha": alpha, "max_drawdown": -0.08, "win_rate": 0.55},
            "group_metrics": {"医药创新型": {"alpha": group_alpha, "total_return": 0.04,
                                          "trade_count": 5}}}


def test_run_ab_win_writes_lineage(monkeypatch, tmp_path, lib):
    _add_lesson(lib, group="医药创新型", target="score_threshold", value=38)
    lineage = VersionLineage(store_path=str(tmp_path / "lineage.jsonl"))
    ab = EvolutionAB(lesson_library=lib, lineage=lineage)

    calls = []
    _patch_runs(monkeypatch, _baseline_result(), _candidate_result(), calls)

    # build_candidate 内部走真实 GroupConfig? 不 — run_ab 调用 run_portfolio_backtest
    # 前会用 GroupConfig.reload(); mock 掉 _load_portfolio_config 与 gc 以隔离磁盘
    import web.backend.services.portfolio_backtest as pb
    monkeypatch.setattr(pb, "_load_portfolio_config",
                        lambda: {"weights": {g: 0.5 for g in ACTIVE},
                                 "regimes_cfg": {}, "atr_override": {},
                                 "dd_config": {}})
    from src.config.group_config import GroupConfig
    monkeypatch.setattr(GroupConfig, "reload", lambda self: None)
    # GroupConfig 是单例, 直接注入 _groups
    gc = GroupConfig()
    gc._groups = {k: dict(v) for k, v in GROUPS.items()}

    report = ab.run_ab()

    assert calls[0]["record_memory"] is True and calls[0]["config_override"] is None
    assert calls[1]["record_memory"] is False
    assert calls[1]["config_override"]["groups"]["医药创新型"]["score_threshold"] == 38

    assert report["has_candidate"] is True
    assert report["verdict"] == "win"
    assert report["alpha_delta"] == pytest.approx(0.03)
    assert report["lineage_recorded"] is True

    # 谱系: 候选版本 parent = 基线版本, 教训应用链 + validated 结论
    node = lineage.get_version("v_cand")
    assert node is not None
    assert node["parent_version"] == "v_base"
    lesson_ids = list(node["lessons"].keys())
    assert len(lesson_ids) == 1
    assert node["lessons"][lesson_ids[0]]["verdict"] == "validated"


def test_run_ab_no_candidate(monkeypatch, tmp_path, lib):
    lineage = VersionLineage(store_path=str(tmp_path / "lineage.jsonl"))
    ab = EvolutionAB(lesson_library=lib, lineage=lineage)

    calls = []
    _patch_runs(monkeypatch, _baseline_result(), _candidate_result(), calls)
    import web.backend.services.portfolio_backtest as pb
    monkeypatch.setattr(pb, "_load_portfolio_config",
                        lambda: {"weights": {g: 0.5 for g in ACTIVE},
                                 "regimes_cfg": {}, "atr_override": {},
                                 "dd_config": {}})
    from src.config.group_config import GroupConfig
    monkeypatch.setattr(GroupConfig, "reload", lambda self: None)
    gc = GroupConfig()
    gc._groups = {}

    report = ab.run_ab()
    assert report["has_candidate"] is False
    assert report["verdict"] == "no_candidate"
    # 只有基线跑了一次
    assert len(calls) == 1
    assert lineage.load_events() == []


def test_run_ab_baseline_failure_short_circuits(monkeypatch, tmp_path, lib):
    _add_lesson(lib, group="医药创新型", target="score_threshold", value=38)
    lineage = VersionLineage(store_path=str(tmp_path / "lineage.jsonl"))
    ab = EvolutionAB(lesson_library=lib, lineage=lineage)

    calls = []
    _patch_runs(monkeypatch, {"status": "error", "message": "自选股列表为空"},
                _candidate_result(), calls)
    import web.backend.services.portfolio_backtest as pb
    monkeypatch.setattr(pb, "_load_portfolio_config",
                        lambda: {"weights": {g: 0.5 for g in ACTIVE},
                                 "regimes_cfg": {}, "atr_override": {},
                                 "dd_config": {}})
    from src.config.group_config import GroupConfig
    monkeypatch.setattr(GroupConfig, "reload", lambda self: None)
    gc = GroupConfig()
    gc._groups = {k: dict(v) for k, v in GROUPS.items()}

    report = ab.run_ab()
    assert report["verdict"] == "error"
    assert len(calls) == 1
    assert lineage.load_events() == []


def test_build_candidate_rebalance_weight_preserves_total(lib):
    """rebalance_weight: 让渡组释放的权重必须转入承接组, 组合总权重不变"""
    lesson = lib.create_lesson(
        linked_cases=["fc_rw_a", "fc_rw_b"],
        scope={"group": "消费稳健型", "regime": None, "strategy_mode": None},
        failure_level="param",
        root_cause="资金配置错配",
        rule="权重让渡",
        rule_action={"type": "rebalance_weight", "target": "消费稳健型",
                     "value": 0.0, "compensate": "科技成长型"},
    )
    lib.mark_applied(lesson["lesson_id"], "run_rw")
    lib.mark_validated(lesson["lesson_id"], "run_rw", notes="test")

    weights = {"消费稳健型": 0.05, "科技成长型": 0.40, "周期资源型": 0.375}
    cand = build_candidate(
        {"消费稳健型": {"strategy_mode": "mean_reversion"},
         "科技成长型": {"strategy_mode": "trend_following"}},
        ["消费稳健型", "科技成长型"], lesson_library=lib,
        statuses=("validated",), weights=weights)

    assert cand["weights"]["消费稳健型"] == 0.0
    assert cand["weights"]["科技成长型"] == pytest.approx(0.45)
    before_total = sum(weights.values())
    after_total = (sum(weights.values())
                   - weights["消费稳健型"] - weights["科技成长型"]
                   + cand["weights"]["消费稳健型"] + cand["weights"]["科技成长型"])
    assert after_total == pytest.approx(before_total)


def test_promote_writes_validated_lesson_to_config(monkeypatch, tmp_path, lib):
    """promote: validated 教训的 rule_action 原子写回 strategy_config,
    并把教训标记 promoted_at, 避免重复取用"""
    import json as _json
    import src.evolution.auto_applier as aa

    # 构造一份临时策略配置
    cfg_path = tmp_path / "strategy_config.json"
    sc = {
        "strategy_config": {
            "group_config": {"groups": {
                "医药创新型": {"strategy_mode": "mean_reversion",
                             "score_threshold": 30, "atr_stop_mult": 2.0},
            }},
            "portfolio_config_p5": {"weights": {"医药创新型": 1.0},
                                    "atr_override": {}, "regimes_filter": {}},
        }
    }
    cfg_path.write_text(_json.dumps(sc, ensure_ascii=False), encoding="utf-8")

    # 注入一条 validated 教训: score_threshold set_param=38
    lesson = _add_lesson(lib, group="医药创新型", target="score_threshold", value=38)

    lineage = VersionLineage(store_path=str(tmp_path / "lineage.jsonl"))
    ab = EvolutionAB(lesson_library=lib, lineage=lineage)

    monkeypatch.setattr(EvolutionAB, "_config_path",
                        staticmethod(lambda: str(cfg_path)))
    from src.config.group_config import GroupConfig
    monkeypatch.setattr(GroupConfig, "reload", lambda self: None)
    gc = GroupConfig()
    gc._groups = {"医药创新型": {"strategy_mode": "mean_reversion",
                              "score_threshold": 30, "atr_stop_mult": 2.0}}

    result = ab.promote()
    assert result["written_config"] is True

    # 配置已写回
    written = _json.loads(cfg_path.read_text(encoding="utf-8"))
    assert (written["strategy_config"]["group_config"]["groups"]
            ["医药创新型"]["score_threshold"] == 38)

    # 教训已标记 promoted
    promoted = lib.get_lesson(lesson["lesson_id"])
    assert promoted.get("promoted_at") is not None

    # 再 promote 一次应跳过已固化教训
    result2 = ab.promote()
    assert result2["written_config"] is False
    assert "无可固化" in result2["reason"]

