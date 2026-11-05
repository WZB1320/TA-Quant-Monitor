"""失效案例库 L1 — 单元测试 (P1)

覆盖:
  - 退出原因归因 / 止损率
  - failure_flags 推导
  - failure_level 规则 R1/R3/R4/R5/R6
  - L1 JSONL 写入与入库门槛
  - 参数扫描方向性 / 股票池跑输比例
"""
import json

import pandas as pd
import pytest

from src.memory.failure_case import (
    FAILURE_LEVELS,
    FailureCaseRecorder,
    classify_failure_level,
    compute_exit_distribution,
    compute_param_scan_direction,
    compute_stock_pool_snapshot,
    compute_window_return,
    derive_failure_flags,
    load_rule_config,
    metrics_to_dict,
)


class _Trade:
    def __init__(self, exit_signal: str, pnl: float = 1.0, pnl_pct: float = 0.05):
        self.exit_signal = exit_signal
        self.pnl = pnl
        self.pnl_pct = pnl_pct
        self.is_closed = True


# ─────────────────────────────────────────────────────────────
# 退出分布 / flags
# ─────────────────────────────────────────────────────────────

def test_exit_distribution_and_stop_loss_rate():
    trades = (
        [_Trade("ATR硬止损 (-8.0%, ATR=1.0)")] * 5
        + [_Trade("ATR移动止盈 (最高10.0)")] * 2
        + [_Trade("均值回归止盈 (6.0%, 目标6%)")] * 2
        + [_Trade("强买入 score=+35.0")]
    )
    exits = compute_exit_distribution(trades)
    assert exits["total"] == 10
    assert exits["counts"]["hard_stop"] == 5
    assert exits["counts"]["trailing_take_profit"] == 2
    assert exits["counts"]["mean_reversion_take_profit"] == 2
    assert exits["counts"]["signal_exit"] == 1
    assert exits["stop_loss_rate"] == pytest.approx(0.5)


def test_derive_failure_flags():
    cfg = load_rule_config()
    metrics = {"alpha": -0.05, "max_drawdown": -0.15}
    exits = {"stop_loss_rate": 0.5, "ma_breakdown_rate": 0.0}
    flags = derive_failure_flags(metrics, exits=exits, thresholds=cfg["thresholds"])
    assert "alpha_negative" in flags
    assert "stop_loss_high" in flags
    assert "drawdown_deep" in flags
    assert flags == [f for f in flags]  # 稳定顺序


def test_derive_dual_negative_and_overfit():
    cfg = load_rule_config()
    flags = derive_failure_flags(
        {"alpha": -0.02},
        train_metrics={"alpha": -0.01, "win_rate": 0.3},
        test_metrics={"alpha": -0.20, "win_rate": 0.2},
        thresholds=cfg["thresholds"],
    )
    assert "train_test_dual_negative" in flags


# ─────────────────────────────────────────────────────────────
# failure_level 规则
# ─────────────────────────────────────────────────────────────

def test_classify_stock_pool():
    flags = ["alpha_negative", "train_test_dual_negative", "param_scan_flat"]
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
    )
    assert result["level"] == "stock_pool"
    assert result["rule_id"] == "R1"


def test_classify_structure():
    flags = ["alpha_negative", "param_scan_clear"]
    result = classify_failure_level(
        flags,
        metrics={"alpha": -0.02},
        train_metrics={"alpha": 0.03},
        test_metrics={"alpha": -0.02},
        exits={"stop_loss_rate": 0.6, "ma_breakdown_rate": 0.0},
        scan={
            "all_combo_negative": False,
            "direction_flat": False,
            "direction_clear": True,
            "max_abs_corr": 0.85,
            "best_test_alpha": -0.02,
            "current_test_alpha": None,
        },
    )
    assert result["level"] == "structure"
    assert result["rule_id"] == "R3"


def test_classify_param():
    flags = ["param_scan_clear"]
    result = classify_failure_level(
        flags,
        metrics={"alpha": 0.01},
        train_metrics={"alpha": 0.05},
        test_metrics={"alpha": 0.01},
        scan={
            "all_combo_negative": False,
            "direction_flat": False,
            "direction_clear": True,
            "max_abs_corr": 0.90,
            "best_test_alpha": 0.06,
            "current_test_alpha": 0.01,
        },
    )
    assert result["level"] == "param"
    assert result["rule_id"] == "R4"


def test_classify_overfit():
    flags = ["alpha_negative", "overfit_gap"]
    result = classify_failure_level(
        flags,
        metrics={"alpha": -0.03},
        train_metrics={"alpha": 0.12, "win_rate": 0.7},
        test_metrics={"alpha": -0.03, "win_rate": 0.3},
    )
    assert result["level"] == "overfit"
    assert result["rule_id"] == "R5"


def test_classify_unknown_without_scan():
    result = classify_failure_level(["alpha_negative"], metrics={"alpha": -0.05})
    assert result["level"] == "unknown"
    assert result["rule_id"] == "R6"
    assert result["level"] in FAILURE_LEVELS


# ─────────────────────────────────────────────────────────────
# 参数扫描方向性
# ─────────────────────────────────────────────────────────────

def test_param_scan_direction_clear():
    combos = [
        {"params": {"score_threshold": 30}, "train": {"alpha": 0.01},
         "test": {"alpha": 0.0}, "combined_alpha": 0.01},
        {"params": {"score_threshold": 35}, "train": {"alpha": 0.03},
         "test": {"alpha": 0.0}, "combined_alpha": 0.03},
        {"params": {"score_threshold": 40}, "train": {"alpha": 0.05},
         "test": {"alpha": 0.0}, "combined_alpha": 0.05},
    ]
    direction = compute_param_scan_direction(combos, {"score_threshold": [30, 35, 40]})
    assert direction["max_abs_corr"] == pytest.approx(1.0)
    assert direction["direction_clear"] is True
    assert direction["direction_flat"] is False


# ─────────────────────────────────────────────────────────────
# 股票池 / 基准
# ─────────────────────────────────────────────────────────────

def _price_df(window_return: float) -> pd.DataFrame:
    return pd.DataFrame({
        "date": pd.to_datetime(["2025-07-01", "2025-07-31"]),
        "close": [100.0, 100.0 * (1 + window_return)],
    })


def test_compute_window_return_and_stock_pool():
    bench = _price_df(0.10)
    data_map = {
        "A": _price_df(0.20),   # 跑赢
        "B": _price_df(0.05),   # 跑输
        "C": _price_df(-0.10),  # 跑输
    }
    assert compute_window_return(bench, "2025-07-01", "2025-07-31") == pytest.approx(0.10)

    snapshot = compute_stock_pool_snapshot(
        data_map, bench, "2025-07-01", "2025-07-31", stock_codes=["A", "B", "C"])
    assert snapshot["total"] == 3
    assert snapshot["underperform_count"] == 2
    assert snapshot["underperform_ratio"] == pytest.approx(2 / 3, abs=1e-4)


# ─────────────────────────────────────────────────────────────
# L1 写入 / 入库门槛
# ─────────────────────────────────────────────────────────────

def test_recorder_skips_successful_run(tmp_path):
    store = tmp_path / "cases" / "l1.jsonl"
    recorder = FailureCaseRecorder(store_path=str(store))
    record = recorder.record_backtest(
        source_ref={"run_id": "bt_test", "group": "科技成长型"},
        group_name="科技成长型",
        stock_codes=["A"],
        window={"start": "2025-01-01", "end": "2025-06-30"},
        strategy_mode="trend_following",
        params={"score_threshold": 25},
        metrics={"alpha": 0.05, "max_drawdown": -0.03, "total_return": 0.10},
        trades=[],
    )
    assert record is None
    assert not store.exists()


def test_recorder_writes_failed_run(tmp_path):
    store = tmp_path / "cases" / "l1.jsonl"
    recorder = FailureCaseRecorder(store_path=str(store))
    record = recorder.record_backtest(
        source_ref={"run_id": "bt_test", "group": "医药创新型"},
        group_name="医药创新型",
        stock_codes=["A", "B"],
        window={"start": "2025-01-01", "end": "2025-06-30"},
        strategy_mode="mean_reversion",
        params={"score_threshold": 30},
        metrics={"alpha": -0.08, "max_drawdown": -0.05, "total_return": -0.02},
        trades=[_Trade("ATR硬止损 (-8%)")],
        regime_dist={"ranging": 5, "trending": 1},
        stock_pool_snapshot={"underperform_ratio": 0.6},
    )
    assert record is not None
    assert record["failure_level"] in FAILURE_LEVELS
    assert "alpha_negative" in record["failure_flags"]
    assert record["strategy_version"]
    assert record["record_type"] == "failure_case"

    lines = store.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 1
    saved = json.loads(lines[0])
    assert saved["case_id"] == record["case_id"]
    assert saved["window"]["start"] == "2025-01-01"


def test_recorder_dedups_same_case(tmp_path):
    """同组+同策略+同窗口+同参数版本重复回测, 不应重复落盘"""
    store = tmp_path / "cases" / "l1.jsonl"
    recorder = FailureCaseRecorder(store_path=str(store))
    kwargs = dict(
        group_name="医药创新型",
        stock_codes=["A", "B"],
        window={"start": "2026-01-05", "end": "2026-06-30"},
        strategy_mode="mean_reversion",
        params={"score_threshold": 30},
        metrics={"alpha": -0.08, "max_drawdown": -0.05, "total_return": -0.02},
        trades=[_Trade("ATR硬止损 (-8%)")],
        regime_dist={"ranging": 5, "trending": 1},
        stock_pool_snapshot={"underperform_ratio": 0.6},
    )
    first = recorder.record_backtest(source_ref={"run_id": "bt_a"}, **kwargs)
    second = recorder.record_backtest(
        source_ref={"run_id": "bt_b"},
        **{**kwargs, "window": {"start": "2025-01-05", "end": "2025-06-30"}})
    third = recorder.record_backtest(source_ref={"run_id": "bt_c"}, **kwargs)

    assert first is not None
    assert second is not None            # 窗口不同 → 新案例
    assert third is None                 # 完全相同 → 去重
    lines = store.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 2


def test_recorder_param_scan_always_writes(tmp_path):
    store = tmp_path / "cases" / "l1.jsonl"
    recorder = FailureCaseRecorder(store_path=str(store))
    combos = [
        {"params": {"score_threshold": v}, "train": {"alpha": 0.01 * v},
         "test": {"alpha": 0.01 * v}, "combined_alpha": 0.02 * v}
        for v in (1, 2, 3)
    ]
    record = recorder.record_param_scan(
        scan_id="scan_test",
        group_name="测试组",
        stock_codes=["A"],
        combos=combos,
        grid={"score_threshold": [1, 2, 3]},
        window={"train_start": "2024-07-01", "train_end": "2025-06-30",
                "test_start": "2025-07-01", "test_end": "2026-06-30"},
        params={"scan_grid": {"score_threshold": [1, 2, 3]}},
    )
    assert record is not None
    assert record["source"] == "param_scan"
    assert record["param_scan"]["combo_count"] == 3
    assert len(record["param_scan"]["combos"]) == 3
    assert record["failure_level"] in FAILURE_LEVELS
    assert store.exists()


def test_metrics_to_dict_inf_handling():
    class _M:
        alpha = -0.05
        total_return = -0.02
        annual_return = -0.01
        sharpe_ratio = -0.3
        win_rate = 0.3
        profit_factor = float("inf")
        trade_count = 4
        win_count = 1
        max_drawdown = -0.08
        max_drawdown_days = 20
        avg_holding_days = 12.0
        benchmark_return = 0.03
        volatility = 0.2

    out = metrics_to_dict(_M())
    assert out["profit_loss_ratio"] is None
    assert out["alpha"] == pytest.approx(-0.05)
    assert out["recovery_days"] == 20
