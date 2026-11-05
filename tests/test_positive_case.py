"""正样本轨迹 — 单元测试 (P5)

覆盖:
  - derive_positive_flags 各维度阈值与 regime_favorable
  - record_backtest 门槛: alpha 不够正 / 无交易 -> 不落盘
  - 落盘字段: case_id / strategy_version / dominant_regime / positive_score
  - list_positive_cases 过滤 (group/mode/regime/min_alpha) 与 alpha 降序
  - best_configs 按 strategy_version 去重 + limit
  - 空存储返回空列表
"""
from types import SimpleNamespace

import pytest

from src.memory.positive_case import (
    POSITIVE_FLAGS,
    PositiveCaseRecorder,
    derive_positive_flags,
)
from src.memory.strategy_memory import compute_strategy_version


@pytest.fixture
def recorder(tmp_path):
    return PositiveCaseRecorder(store_path=str(tmp_path / "positive.jsonl"))


PARAMS = {"engine": {"score_threshold": 38}, "strategy_mode": "trend_following"}
WINDOW = {"start": "2025-01-01", "end": "2025-12-31"}
SLICES = {
    "trending": {"days": 120, "share": 0.6, "return": 0.12,
                 "benchmark_return": 0.08, "alpha": 0.04},
    "ranging": {"days": 80, "share": 0.4, "return": 0.01,
                "benchmark_return": 0.02, "alpha": -0.01},
}


def _metrics(alpha=0.03, dd=-0.05, win=0.55, trade_count=10):
    return {"alpha": alpha, "max_drawdown": dd, "win_rate": win,
            "trade_count": trade_count, "total_return": alpha + 0.05}


def _record(recorder, **overrides):
    base = dict(
        source_ref={"run_id": "bt_1", "group": "科技成长型", "mode": "base"},
        group_name="科技成长型",
        stock_codes=["600000"],
        window=WINDOW,
        strategy_mode="trend_following",
        params=PARAMS,
        metrics=_metrics(),
        trades=[SimpleNamespace(exit_signal="take_profit")],
        benchmark_return=0.05,
        regime_slices=SLICES,
    )
    base.update(overrides)
    return recorder.record_backtest(**base)


# ─────────────────────────────────────────────────────────────
# derive_positive_flags
# ─────────────────────────────────────────────────────────────

def test_strong_alpha_wins_over_positive():
    flags = derive_positive_flags({"alpha": 0.06})
    assert flags == ["strong_alpha"]


def test_positive_alpha_band():
    assert derive_positive_flags({"alpha": 0.02}) == ["positive_alpha"]


def test_negative_alpha_has_no_alpha_flag():
    assert derive_positive_flags({"alpha": -0.01}) == []


def test_drawdown_shallow_uses_absolute_value():
    assert derive_positive_flags({"max_drawdown": -0.05}) == ["drawdown_shallow"]
    assert derive_positive_flags({"max_drawdown": -0.15}) == []


def test_win_rate_and_stop_loss_flags():
    flags = derive_positive_flags({"win_rate": 0.55, "stop_loss_rate": 0.10})
    assert flags == ["win_rate_high", "stop_loss_low"]


def test_thresholds_override():
    flags = derive_positive_flags({"alpha": 0.03}, thresholds={"strong_alpha": 0.02})
    assert flags == ["strong_alpha"]


def test_regime_favorable_from_dominant_slice():
    flags = derive_positive_flags({"alpha": 0.02}, regime_slices=SLICES)
    assert "regime_favorable" in flags


def test_regime_favorable_requires_positive_dominant_alpha():
    slices = {"ranging": {"days": 200, "share": 0.9, "alpha": -0.05}}
    assert "regime_favorable" not in derive_positive_flags({"alpha": 0.02}, regime_slices=slices)


def test_flags_follow_constant_order():
    flags = derive_positive_flags({
        "alpha": 0.06, "max_drawdown": -0.05, "win_rate": 0.6,
        "stop_loss_rate": 0.1,
    }, regime_slices=SLICES)
    # strong_alpha 取代 positive_alpha (互斥), 其余按 POSITIVE_FLAGS 顺序
    assert flags == [f for f in POSITIVE_FLAGS if f != "positive_alpha"]


# ─────────────────────────────────────────────────────────────
# record_backtest 门槛与落盘
# ─────────────────────────────────────────────────────────────

def test_record_rejects_negative_alpha(recorder):
    assert _record(recorder, metrics=_metrics(alpha=-0.02)) is None
    assert recorder.load_all() == []


def test_record_rejects_no_trades(recorder):
    assert _record(recorder, metrics=_metrics(trade_count=0),
                   trades=[]) is None
    assert recorder.load_all() == []


def test_record_rejects_missing_alpha(recorder):
    metrics = _metrics()
    metrics.pop("alpha")
    assert _record(recorder, metrics=metrics) is None


def test_record_persists_full_case(recorder):
    record = _record(recorder)
    assert record is not None
    assert record["record_type"] == "positive_case"
    assert record["case_id"].startswith("pc_")
    assert record["strategy_version"] == compute_strategy_version(PARAMS)
    assert record["dominant_regime"] == "trending"
    assert record["positive_score"] == len(record["positive_flags"])
    assert record["metrics"]["alpha"] == pytest.approx(0.03)
    stored = recorder.load_all()
    assert len(stored) == 1
    assert stored[0]["case_id"] == record["case_id"]


def test_record_without_regime_slices(recorder):
    record = _record(recorder, regime_slices=None)
    assert record["dominant_regime"] is None
    assert "regime_favorable" not in record["positive_flags"]


# ─────────────────────────────────────────────────────────────
# 查询
# ─────────────────────────────────────────────────────────────

def test_list_empty_store(recorder):
    assert recorder.list_positive_cases() == []


def test_list_filters_by_group_and_mode(recorder):
    _record(recorder, metrics=_metrics(alpha=0.02), group_name="科技成长型",
            strategy_mode="trend_following")
    _record(recorder, metrics=_metrics(alpha=0.04), group_name="消费稳健型",
            strategy_mode="mean_reversion")

    assert recorder.list_positive_cases(group="消费稳健型")[0]["group_name"] == "消费稳健型"
    assert recorder.list_positive_cases(strategy_mode="trend_following")[0]["group_name"] == "科技成长型"


def test_list_filters_by_regime(recorder):
    _record(recorder, regime_slices=SLICES)  # 主导 trending
    other = {"ranging": {"days": 200, "share": 0.9, "alpha": 0.03}}
    _record(recorder, regime_slices=other)   # 主导 ranging

    assert recorder.list_positive_cases(regime="ranging")[0]["dominant_regime"] == "ranging"
    assert recorder.list_positive_cases(regime="trending")[0]["dominant_regime"] == "trending"


def test_list_sorted_by_alpha_desc(recorder):
    _record(recorder, metrics=_metrics(alpha=0.02))
    _record(recorder, metrics=_metrics(alpha=0.06))
    alphas = [c["metrics"]["alpha"] for c in recorder.list_positive_cases()]
    assert alphas == [0.06, 0.02]


def test_list_min_alpha(recorder):
    _record(recorder, metrics=_metrics(alpha=0.02))
    _record(recorder, metrics=_metrics(alpha=0.06))
    assert len(recorder.list_positive_cases(min_alpha=0.05)) == 1


# ─────────────────────────────────────────────────────────────
# best_configs
# ─────────────────────────────────────────────────────────────

def test_best_configs_dedupes_by_version_keeping_best(recorder):
    _record(recorder, metrics=_metrics(alpha=0.02))
    _record(recorder, metrics=_metrics(alpha=0.06))
    best = recorder.best_configs()
    assert len(best) == 1
    assert best[0]["metrics"]["alpha"] == 0.06
    assert best[0]["strategy_version"] == compute_strategy_version(PARAMS)


def test_best_configs_rank_and_limit(recorder):
    params_a = {"engine": {"score_threshold": 30}}
    params_b = {"engine": {"score_threshold": 35}}
    params_c = {"engine": {"score_threshold": 40}}
    _record(recorder, params=params_a, metrics=_metrics(alpha=0.02))
    _record(recorder, params=params_b, metrics=_metrics(alpha=0.05))
    _record(recorder, params=params_c, metrics=_metrics(alpha=0.03))

    best = recorder.best_configs(limit=2)
    assert [b["metrics"]["alpha"] for b in best] == [0.05, 0.03]
    assert best[0]["params"] == params_b


def test_best_configs_filter_by_group(recorder):
    _record(recorder, group_name="科技成长型", metrics=_metrics(alpha=0.06))
    _record(recorder, group_name="消费稳健型", params={"engine": {"score_threshold": 20}},
            metrics=_metrics(alpha=0.02))
    best = recorder.best_configs(group="消费稳健型")
    assert len(best) == 1
    assert best[0]["metrics"]["alpha"] == 0.02
