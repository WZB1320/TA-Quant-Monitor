"""失效案例库 P3 — regime_slices + R2 规则 单元测试

覆盖:
  - compute_regime_slices 基本计算 (按体制切分收益/基准/alpha)
  - 同一体制分散多段时的日收益累乘
  - 基准缺失时 alpha 为 None
  - 数据不足时返回 {}
  - dominant_regime 的 is_mismatch 判定 (share / alpha 阈值)
  - derive_failure_flags 产出 regime_alpha_negative
  - classify_failure_level R2 命中与不命中
  - R1 优先于 R2
  - record_backtest 传递 regime_slices 并落盘
"""
from datetime import date

import pandas as pd
import pytest

from src.memory.failure_case import (
    classify_failure_level,
    derive_failure_flags,
    FailureCaseRecorder,
    load_rule_config,
    metrics_to_dict,
)
from src.memory.regime_slices import compute_regime_slices, dominant_regime


def _dates(n: int, start="2025-01-01"):
    return [d.date() if hasattr(d, "date") else d
            for d in pd.date_range(start, periods=n)]


def _build_nav(patterns):
    """patterns: [(regime, daily_return, count), ...] → (nav_series, regime_by_date)

    首个净值点为起点 (100), 之后按 pattern 逐日累乘。
    """
    dates = []
    regime_by_date = {}
    values = [100.0]
    for regime, ret, count in patterns:
        for _ in range(count):
            values.append(values[-1] * (1 + ret))
    n = len(values)
    dates = _dates(n)
    v = 100.0
    for i in range(n):
        regime_by_date[dates[i]] = None
    # 逐日填充 regime: 第 0 天用第一个 pattern 的 regime, 之后按段
    idx = 0
    first_regime = patterns[0][0] if patterns else "transition"
    regime_by_date[dates[0]] = first_regime
    for regime, ret, count in patterns:
        for _ in range(count):
            idx += 1
            regime_by_date[dates[idx]] = regime
    return pd.Series(values, index=dates), regime_by_date


# ─────────────────────────────────────────────────────────────
# compute_regime_slices
# ─────────────────────────────────────────────────────────────

def test_slices_basic_split():
    # 5 天 ranging 每天 -1%, 4 天 trending 每天 +1.5%
    nav, regimes = _build_nav([
        ("ranging", -0.01, 5),
        ("trending", 0.015, 4),
    ])
    # 基准持平
    bench = pd.Series([1.0] * len(nav), index=nav.index)
    slices = compute_regime_slices(nav, bench, regimes)

    assert set(slices) == {"ranging", "trending"}
    r = slices["ranging"]
    t = slices["trending"]
    # 5 个 -1% 收益点落在 ranging, 4 个 +1.5% 落在 trending
    assert r["days"] == 5 and t["days"] == 4
    assert r["return"] == pytest.approx(0.99 ** 5 - 1, abs=1e-4)
    assert t["return"] == pytest.approx(1.015 ** 4 - 1, abs=1e-4)
    assert r["benchmark_return"] == pytest.approx(0.0, abs=1e-6)
    assert r["alpha"] == pytest.approx(0.99 ** 5 - 1, abs=1e-4)
    assert t["alpha"] == pytest.approx(1.015 ** 4 - 1, abs=1e-4)
    assert r["share"] == pytest.approx(5 / 9, abs=1e-3)
    assert t["share"] == pytest.approx(4 / 9, abs=1e-3)


def test_slices_compound_across_segments():
    """同一体制分散多段, 收益应累乘 (而非只看首尾)"""
    nav, regimes = _build_nav([
        ("trending", 0.01, 2),
        ("ranging", -0.01, 3),
        ("trending", 0.01, 2),
    ])
    slices = compute_regime_slices(nav, None, regimes)
    # trending 共 4 个 +1% 收益点 (2+2)
    assert slices["trending"]["days"] == 4
    assert slices["trending"]["return"] == pytest.approx(1.01 ** 4 - 1, abs=1e-4)
    assert slices["ranging"]["days"] == 3
    assert slices["ranging"]["return"] == pytest.approx(0.99 ** 3 - 1, abs=1e-4)


def test_slices_without_benchmark_alpha_none():
    nav, regimes = _build_nav([("ranging", -0.01, 4)])
    slices = compute_regime_slices(nav, None, regimes)
    assert slices["ranging"]["benchmark_return"] is None
    assert slices["ranging"]["alpha"] is None


def test_slices_insufficient_data():
    nav = pd.Series([100.0], index=_dates(1))
    assert compute_regime_slices(nav, None, {}) == {}
    # 有净值但无 regime 映射
    nav2 = pd.Series([100.0, 101.0], index=_dates(2))
    assert compute_regime_slices(nav2, None, {}) == {}


def test_slices_ignores_unmapped_dates():
    nav, regimes = _build_nav([("ranging", -0.01, 4)])
    # 去掉部分日期映射
    regimes = {k: v for i, (k, v) in enumerate(sorted(regimes.items())) if i < 3}
    slices = compute_regime_slices(nav, None, regimes)
    assert slices["ranging"]["days"] == 2  # 3 天映射 → 2 个收益点


# ─────────────────────────────────────────────────────────────
# dominant_regime
# ─────────────────────────────────────────────────────────────

def test_dominant_regime_mismatch():
    slices = {
        "ranging": {"days": 60, "share": 0.6, "alpha": -0.05, "return": -0.04,
                    "benchmark_return": 0.01},
        "trending": {"days": 40, "share": 0.4, "alpha": 0.03, "return": 0.05,
                     "benchmark_return": 0.02},
    }
    dom = dominant_regime(slices)
    assert dom["regime"] == "ranging"
    assert dom["is_mismatch"] is True


def test_dominant_regime_share_too_low():
    slices = {
        "ranging": {"days": 40, "share": 0.4, "alpha": -0.05},
        "trending": {"days": 60, "share": 0.6, "alpha": 0.02},
    }
    dom = dominant_regime(slices)
    assert dom["regime"] == "trending"
    assert dom["is_mismatch"] is False  # 主导体制 alpha 为正


def test_dominant_regime_alpha_not_negative_enough():
    slices = {
        "ranging": {"days": 60, "share": 0.6, "alpha": -0.01},  # 未达 -2%
        "trending": {"days": 40, "share": 0.4, "alpha": 0.02},
    }
    dom = dominant_regime(slices)
    assert dom["is_mismatch"] is False


def test_dominant_regime_empty():
    assert dominant_regime({}) is None


def test_dominant_regime_custom_thresholds():
    slices = {"ranging": {"days": 60, "share": 0.6, "alpha": -0.05}}
    dom = dominant_regime(slices, {"regime_mismatch_share": 0.7})
    assert dom["is_mismatch"] is False  # 0.6 < 0.7


# ─────────────────────────────────────────────────────────────
# derive_failure_flags / R2
# ─────────────────────────────────────────────────────────────

def _mismatch_slices():
    return {
        "ranging": {"days": 60, "share": 0.6, "alpha": -0.05, "return": -0.04,
                    "benchmark_return": 0.01},
        "trending": {"days": 40, "share": 0.4, "alpha": 0.03, "return": 0.05,
                     "benchmark_return": 0.02},
    }


def test_derive_flags_regime_alpha_negative():
    cfg = load_rule_config()
    flags = derive_failure_flags(
        {"alpha": -0.03},
        regime_slices=_mismatch_slices(),
        thresholds=cfg["thresholds"],
    )
    assert "regime_alpha_negative" in flags


def test_derive_flags_no_regime_flag_when_match():
    cfg = load_rule_config()
    good_slices = {
        "ranging": {"days": 60, "share": 0.6, "alpha": 0.05},
        "trending": {"days": 40, "share": 0.4, "alpha": 0.02},
    }
    flags = derive_failure_flags(
        {"alpha": 0.02},
        regime_slices=good_slices,
        thresholds=cfg["thresholds"],
    )
    assert "regime_alpha_negative" not in flags


def test_classify_regime_mismatch():
    result = classify_failure_level(
        ["alpha_negative", "regime_alpha_negative"],
        metrics={"alpha": -0.03},
        regime_slices=_mismatch_slices(),
    )
    assert result["level"] == "regime_mismatch"
    assert result["rule_id"] == "R2"
    assert "ranging" in result["reason"]


def test_classify_no_regime_mismatch_when_alpha_positive():
    result = classify_failure_level(
        ["alpha_negative"],
        metrics={"alpha": -0.03},
        regime_slices={
            "ranging": {"days": 60, "share": 0.6, "alpha": 0.02},
        },
    )
    assert result["level"] != "regime_mismatch"


def test_classify_r1_priority_over_r2():
    """同时满足 R1(选股域) 与 R2(体制失配) 时, R1 优先"""
    flags = ["alpha_negative", "train_test_dual_negative", "param_scan_flat",
             "regime_alpha_negative"]
    result = classify_failure_level(
        flags,
        metrics={"alpha": -0.10},
        train_metrics={"alpha": -0.04},
        test_metrics={"alpha": -0.22},
        scan={
            "all_combo_negative": True,
            "direction_flat": True,
            "direction_clear": False,
            "max_abs_corr": 0.1,
            "combo_count": 36,
        },
        stock_pool_underperform_ratio=0.8,
        regime_slices=_mismatch_slices(),
    )
    assert result["level"] == "stock_pool"
    assert result["rule_id"] == "R1"


def test_classify_r2_disabled_when_config_says_so():
    cfg = load_rule_config()
    cfg["rules"] = [
        {**r, "enabled": False} if r["id"] == "R2" else r for r in cfg["rules"]
    ]
    result = classify_failure_level(
        ["alpha_negative", "regime_alpha_negative"],
        metrics={"alpha": -0.03},
        regime_slices=_mismatch_slices(),
        config=cfg,
    )
    assert result["level"] != "regime_mismatch"


# ─────────────────────────────────────────────────────────────
# record_backtest 传递 regime_slices
# ─────────────────────────────────────────────────────────────

def test_record_backtest_persists_regime_slices(tmp_path):
    store = tmp_path / "cases" / "l1.jsonl"
    recorder = FailureCaseRecorder(store_path=str(store))
    record = recorder.record_backtest(
        source_ref={"run_id": "bt_p3", "group": "科技成长型"},
        group_name="科技成长型",
        stock_codes=["A", "B"],
        window={"start": "2025-01-01", "end": "2025-06-30"},
        strategy_mode="trend_following",
        params={"score_threshold": 30},
        metrics={"alpha": -0.03, "max_drawdown": -0.05},
        trades=[],
        regime_slices=_mismatch_slices(),
    )
    assert record is not None
    assert record["regime_slices"]["ranging"]["share"] == pytest.approx(0.6)
    assert record["failure_level"] == "regime_mismatch"
    assert record["failure_level_rule"] == "R2"
    assert "regime_alpha_negative" in record["failure_flags"]

    # 已落盘
    import json
    saved = json.loads(store.read_text(encoding="utf-8").strip())
    assert saved["regime_slices"]["ranging"]["alpha"] == pytest.approx(-0.05)


def test_record_backtest_no_slices_still_works(tmp_path):
    """无 regime_slices 时行为与 P1/P2 一致 (不因新增字段报错)"""
    store = tmp_path / "cases" / "l1.jsonl"
    recorder = FailureCaseRecorder(store_path=str(store))
    record = recorder.record_backtest(
        source_ref={"run_id": "bt_p3b", "group": "医药创新型"},
        group_name="医药创新型",
        stock_codes=["A"],
        window={"start": "2025-01-01", "end": "2025-06-30"},
        strategy_mode="mean_reversion",
        params={"score_threshold": 30},
        metrics={"alpha": -0.08},
        trades=[],
    )
    assert record is not None
    assert record["regime_slices"] is None
    assert "regime_alpha_negative" not in record["failure_flags"]
