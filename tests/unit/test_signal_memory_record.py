"""SignalEngine 记忆记录字段单测

针对历史缺陷: 实时分析路径 (analyze_batch) 不传 analysis_date, 导致记忆记录
analysis_date 恒为 null, report_generator 的 signal→outcome 关联键退化为
"symbol_None" 永远匹配不上, 记忆层也无法按交易日判重。
"""
import pandas as pd
import pytest

from src.memory import StrategyMemory
from src.signal_engine.classifier import ExecutionConstraint
from src.signal_engine.engine import SignalEngine
from src.signal_engine.signals import CategorySummary, SignalLevel


class _StubMemory:
    """只收集不落盘, 用于断言引擎传了什么字段"""

    def __init__(self):
        self.records = []

    def record_signal(self, data: dict) -> bool:
        self.records.append(data)
        return True


def _df(last_date="2026-09-28", last_close=12.34) -> pd.DataFrame:
    return pd.DataFrame({
        "date": ["2026-09-25", last_date],
        "close": [12.0, last_close],
    })


def _record(engine, memory, df, analysis_date):
    engine._record_signal_memory(
        symbol="600549",
        df=df,
        analysis_date=analysis_date,
        group_params={},
        forced_regime="auto",
        effective_params={},
        indicator_results={},
        score=-72.5,
        cat_scores={"trend": -80.0},
        category_summary={"trend": CategorySummary(
            category="trend", direction=-1, consensus=0, dissensus=3)},
        level=SignalLevel.STRONG_SELL,
        execution=ExecutionConstraint(),
        confidence=0.82,
    )
    return memory.records[-1] if hasattr(memory, "records") else None


class TestAnalysisDateFallback:
    def test_falls_back_to_data_last_date_when_not_provided(self):
        mem = _StubMemory()
        rec = _record(SignalEngine(memory=mem), mem, _df(), None)
        assert rec["analysis_date"] == "2026-09-28"

    def test_falls_back_to_last_row_of_arbitrary_length(self):
        mem = _StubMemory()
        df = pd.DataFrame({"date": ["2026-09-24", "2026-09-25", "2026-09-28"],
                           "close": [11.0, 12.0, 12.34]})
        rec = _record(SignalEngine(memory=mem), mem, df, None)
        assert rec["analysis_date"] == "2026-09-28"

    def test_explicit_analysis_date_wins(self):
        mem = _StubMemory()
        rec = _record(SignalEngine(memory=mem), mem, _df(), "2026-09-21")
        assert rec["analysis_date"] == "2026-09-21"

    def test_explicit_datetime_is_truncated_to_day(self):
        mem = _StubMemory()
        rec = _record(SignalEngine(memory=mem), mem, _df(),
                      pd.Timestamp("2026-09-21 00:00:00"))
        assert rec["analysis_date"] == "2026-09-21"

    def test_no_date_column_leaves_record_date_none_without_raising(self):
        mem = _StubMemory()
        rec = _record(SignalEngine(memory=mem), mem,
                      pd.DataFrame({"close": [12.0, 12.34]}), None)
        assert rec["analysis_date"] is None

    def test_price_and_key_fields_present(self):
        mem = _StubMemory()
        rec = _record(SignalEngine(memory=mem), mem, _df(), None)
        assert rec["price_at_signal"] == pytest.approx(12.34)
        assert rec["level"] == "强卖出"
        assert rec["symbol"] == "600549"
        assert rec["strategy_version"]


class TestEndToEndWithRealMemory:
    def test_repeated_analysis_of_same_day_lands_one_record(self, tmp_path):
        """引擎 + 真实记忆层: 同一交易日重复分析只落一条"""
        mem = StrategyMemory(source="live", data_dir=str(tmp_path))
        engine = SignalEngine(memory=mem)
        _record(engine, mem, _df(), None)
        _record(engine, mem, _df(), None)

        with open(mem.file_path, "r", encoding="utf-8") as f:
            lines = [ln for ln in f if ln.strip()]
        assert len(lines) == 1
        assert '"analysis_date": "2026-09-28"' in lines[0]
