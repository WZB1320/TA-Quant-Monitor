"""策略记忆层幂等写入单测

覆盖: 去重键语义 / 索引跨进程重建 / 关闭去重 / 禁用写盘 / 月份切分路径 / 回测分文件
"""
import json
import os
from datetime import datetime

import pytest

from src.memory import StrategyMemory, signal_dedupe_key


def _read(path: str) -> list:
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def _signal(symbol: str = "600549", analysis_date: str = "2026-09-28",
            level: str = "强卖出", version: str = "ea642d06",
            score: float = -72.5) -> dict:
    return {
        "symbol": symbol,
        "analysis_date": analysis_date,
        "group": "科技",
        "regime": "auto",
        "level": level,
        "score": score,
        "confidence": 0.82,
        "executable": True,
        "strategy_version": version,
        "price_at_signal": 12.34,
    }


@pytest.fixture
def sandbox(tmp_path):
    """注入临时数据目录, 不触碰真实 data/"""
    return str(tmp_path)


class TestDedupeKey:
    def test_key_normalizes_analysis_date_to_10_chars(self):
        rec = _signal(analysis_date="2026-09-28 00:00:00")
        assert signal_dedupe_key(rec)[1] == "2026-09-28"

    def test_key_falls_back_to_recorded_at_date_when_date_missing(self):
        rec = _signal(analysis_date=None)
        rec["recorded_at"] = "2026-09-29T10:02:01"
        assert signal_dedupe_key(rec)[1] == "rec:2026-09-29"

    def test_key_ignores_score_so_same_fact_stays_one_record(self):
        assert signal_dedupe_key(_signal(score=-72.5)) == signal_dedupe_key(_signal(score=-70.1))


class TestSignalDedupe:
    def test_same_fact_written_twice_lands_one_line(self, sandbox):
        mem = StrategyMemory(source="live", data_dir=sandbox, run_id="run1")
        assert mem.record_signal(_signal()) is True
        assert mem.record_signal(_signal()) is False
        assert len(_read(mem.file_path)) == 1

    def test_distinct_level_is_a_new_record(self, sandbox):
        mem = StrategyMemory(source="live", data_dir=sandbox)
        mem.record_signal(_signal(level="强卖出"))
        mem.record_signal(_signal(level="卖出"))
        assert len(_read(mem.file_path)) == 2

    def test_distinct_strategy_version_is_a_new_record(self, sandbox):
        mem = StrategyMemory(source="live", data_dir=sandbox)
        mem.record_signal(_signal(version="ea642d06"))
        mem.record_signal(_signal(version="0c909465"))
        assert len(_read(mem.file_path)) == 2

    def test_distinct_analysis_date_is_a_new_record(self, sandbox):
        mem = StrategyMemory(source="live", data_dir=sandbox)
        mem.record_signal(_signal(analysis_date="2026-09-25"))
        mem.record_signal(_signal(analysis_date="2026-09-28"))
        assert len(_read(mem.file_path)) == 2

    def test_index_survives_process_restart(self, sandbox):
        """后端热重载 = 新进程新实例, 索引必须从磁盘重建, 否则判重失效"""
        first = StrategyMemory(source="live", data_dir=sandbox, run_id="run_old")
        assert first.record_signal(_signal()) is True

        restarted = StrategyMemory(source="live", data_dir=sandbox, run_id="run_new")
        assert restarted.record_signal(_signal()) is False
        assert len(_read(restarted.file_path)) == 1

    def test_missing_analysis_date_still_deduped_within_same_write_day(self, sandbox):
        """analysis_date 缺失时退化为写入日, 同日重跑仍判重"""
        mem = StrategyMemory(source="live", data_dir=sandbox)
        assert mem.record_signal(_signal(analysis_date=None)) is True
        assert mem.record_signal(_signal(analysis_date=None)) is False
        assert len(_read(mem.file_path)) == 1

    def test_missing_analysis_date_not_merged_across_write_days(self, sandbox):
        """不同写入日的无日期记录不得被误并成一条"""
        rec_yesterday = _signal(analysis_date=None)
        rec_yesterday["recorded_at"] = "2026-09-28T15:00:00"
        rec_today = _signal(analysis_date=None)
        rec_today["recorded_at"] = "2026-09-29T15:00:00"
        assert signal_dedupe_key(rec_yesterday) != signal_dedupe_key(rec_today)

    def test_dedupe_disabled_appends_duplicates(self, sandbox):
        mem = StrategyMemory(source="live", data_dir=sandbox, dedupe=False)
        mem.record_signal(_signal())
        mem.record_signal(_signal())
        assert len(_read(mem.file_path)) == 2

    def test_legacy_null_date_records_do_not_block_new_records(self, sandbox):
        """存量无日期记录与带日期记录键不同, 不会互相吞掉"""
        mem = StrategyMemory(source="live", data_dir=sandbox)
        legacy = _signal(analysis_date=None)
        legacy["recorded_at"] = "2026-08-01T09:00:00"
        mem.record_signal(legacy)
        assert mem.record_signal(_signal(analysis_date="2026-09-28")) is True
        assert len(_read(mem.file_path)) == 2


class TestDisabled:
    def test_enabled_false_writes_nothing(self, sandbox):
        mem = StrategyMemory(source="live", data_dir=sandbox, enabled=False)
        assert mem.record_signal(_signal()) is False
        assert not os.path.exists(mem.file_path)

    def test_outcome_record_written_once_per_call(self, sandbox):
        """结果记录不做去重 (同一标的可多次进出场), 保持原有行为"""
        mem = StrategyMemory(source="backtest", data_dir=sandbox)
        mem.record_outcome({"symbol": "600549", "entry_date": "2026-09-01",
                            "exit_date": "2026-09-10", "pnl": 100.0})
        mem.record_outcome({"symbol": "600549", "entry_date": "2026-09-01",
                            "exit_date": "2026-09-10", "pnl": 100.0})
        assert len(_read(mem.file_path)) == 2


class TestPaths:
    def test_live_path_is_monthly_split_under_data_dir(self, sandbox):
        mem = StrategyMemory(source="live", data_dir=sandbox, run_id="r1")
        expect_name = f"strategy_memory_{datetime.now().strftime('%Y-%m')}.jsonl"
        assert os.path.basename(mem.file_path) == expect_name
        assert os.path.dirname(mem.file_path) == sandbox

    def test_month_rollover_targets_different_file(self, sandbox):
        mem = StrategyMemory(source="live", data_dir=sandbox)
        aug = mem._get_monthly_file_path(datetime(2026, 8, 31, 23, 0))
        sep = mem._get_monthly_file_path(datetime(2026, 9, 1, 0, 30))
        assert os.path.basename(aug) == "strategy_memory_2026-08.jsonl"
        assert os.path.basename(sep) == "strategy_memory_2026-09.jsonl"

    def test_backtest_path_is_one_file_per_run(self, sandbox):
        mem = StrategyMemory(source="backtest", data_dir=sandbox, run_id="bt_x")
        mem.record_signal(_signal())
        assert mem.file_path == os.path.join(sandbox, "backtest_memory", "bt_x.jsonl")
        assert os.path.exists(mem.file_path)

    def test_default_data_dir_points_to_project_data(self):
        mem = StrategyMemory(source="live")
        assert os.path.basename(mem.file_path).startswith("strategy_memory_")
        assert os.path.basename(os.path.dirname(mem.file_path)) == "data"
