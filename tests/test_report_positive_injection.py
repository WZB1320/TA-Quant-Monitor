"""P5 正样本基线注入周报 prompt — 单元测试

覆盖:
  - 空正样本库返回空串
  - 按主导体制检索并注入 alpha/回撤/胜率/关键参数
  - 其它体制的正样本不混入
  - 主导体制无正样本时回退全局最佳
  - max_items 限制
  - 无主导体制时使用全局最佳
"""
from types import SimpleNamespace

import pytest

from src.ai.report_generator import WeeklyReportGenerator
from src.memory.positive_case import PositiveCaseRecorder


@pytest.fixture
def recorder(tmp_path):
    return PositiveCaseRecorder(store_path=str(tmp_path / "positive.jsonl"))


@pytest.fixture
def gen(recorder):
    # llm 用占位对象: 本测试只验证 prompt 构建, 不触发真实调用
    return WeeklyReportGenerator(llm=object(), positive_recorder=recorder)


def _seed(recorder, *, regime="trending", group="科技成长型", alpha=0.03,
          params=None, window=None):
    slices = {regime: {"days": 120, "share": 0.8, "return": alpha + 0.04,
                       "benchmark_return": 0.04, "alpha": alpha}}
    return recorder.record_backtest(
        source_ref={"run_id": f"bt_{alpha}", "group": group, "mode": "base"},
        group_name=group,
        stock_codes=["600000"],
        window=window or {"start": "2025-01-01", "end": "2025-12-31"},
        strategy_mode="trend_following",
        params=params or {"engine": {"score_threshold": 30, "atr_stop_mult": 2.0}},
        metrics={"alpha": alpha, "max_drawdown": -0.05, "win_rate": 0.55,
                 "trade_count": 10},
        trades=[SimpleNamespace(exit_signal="take_profit")],
        benchmark_return=0.04,
        regime_slices=slices,
    )


STATS_TRENDING = {"by_regime": [{"name": "trending", "count": 10},
                                {"name": "ranging", "count": 2}]}


# ─────────────────────────────────────────────────────────────

def test_empty_store_returns_empty(gen):
    assert gen._build_positive_context({}) == ""


def test_context_includes_baseline_fields(gen, recorder):
    _seed(recorder, alpha=0.03, params={"engine": {"score_threshold": 30}})
    ctx = gen._build_positive_context(STATS_TRENDING)

    assert "【历史有效配置 (正样本基线)】" in ctx
    assert "version=" in ctx
    assert "科技成长型/trend_following" in ctx
    assert "alpha=0.03" in ctx
    assert "回撤=-0.05" in ctx
    assert "胜率=0.55" in ctx
    assert "score_threshold" in ctx
    assert "回测窗口: 2025-01-01 ~ 2025-12-31" in ctx
    assert "优点:" in ctx


def test_context_prefers_dominant_regime(gen, recorder):
    _seed(recorder, regime="trending", alpha=0.03, params={"engine": {"k": 1}})
    _seed(recorder, regime="ranging", alpha=0.06, params={"engine": {"k": 2}})

    ctx = gen._build_positive_context(STATS_TRENDING)
    assert "体制=trending" in ctx
    assert "体制=ranging" not in ctx


def test_context_falls_back_to_global_when_regime_has_none(gen, recorder):
    _seed(recorder, regime="ranging", alpha=0.06)
    stats = {"by_regime": [{"name": "trending", "count": 10}]}

    ctx = gen._build_positive_context(stats)
    assert ctx != ""
    assert "体制=ranging" in ctx


def test_context_uses_global_when_no_dominant_regime(gen, recorder):
    _seed(recorder, regime="trending", alpha=0.03)
    assert gen._build_positive_context({}) != ""


def test_context_respects_max_items(gen, recorder):
    _seed(recorder, alpha=0.06, params={"engine": {"v": 1}})
    _seed(recorder, alpha=0.05, params={"engine": {"v": 2}})
    _seed(recorder, alpha=0.04, params={"engine": {"v": 3}})

    ctx = gen._build_positive_context(STATS_TRENDING, max_items=1)
    assert ctx.count("version=") == 1
    assert "alpha=0.06" in ctx


def test_context_handles_recorder_errors(gen):
    class Broken:
        def best_configs(self, **kwargs):
            raise RuntimeError("boom")

    gen.positive_recorder = Broken()
    assert gen._build_positive_context(STATS_TRENDING) == ""


def test_context_without_engine_params(gen, recorder):
    _seed(recorder, alpha=0.03, params={"strategy_mode": "trend_following"})
    ctx = gen._build_positive_context(STATS_TRENDING)
    assert "关键参数" not in ctx
    assert "version=" in ctx
