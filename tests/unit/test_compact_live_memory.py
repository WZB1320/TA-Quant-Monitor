"""记忆文件压缩脚本的纯函数单测

重点保护两条容易踩错的地方:
  1. 副本判定必须用"整条记录指纹", 不能用"价格+级别+版本" —— 不同交易日可能恰好同价位
  2. 日期推断必须取 ≤ 写入日 的最大同价位交易日, 且歧义时宁缺勿错
"""
import json
import sys
import os

import pytest

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)
if os.path.join(_ROOT, "scripts") not in sys.path:
    sys.path.insert(0, os.path.join(_ROOT, "scripts"))

from compact_live_memory import (  # noqa: E402
    compact_records, content_fingerprint, infer_trade_date,
)


def _rec(recorded_at, *, symbol="600549", price=12.34, level="强卖出",
         version="ea642d06", score=-72.5, analysis_date=None,
         indicators=None, record_id="sig_x", run_id="run1"):
    return {
        "record_type": "signal",
        "record_id": record_id,
        "source": "live",
        "run_id": run_id,
        "recorded_at": recorded_at,
        "symbol": symbol,
        "analysis_date": analysis_date,
        "level": level,
        "score": score,
        "strategy_version": version,
        "price_at_signal": price,
        "indicators": indicators if indicators is not None else {"MA60": {"dir": 1, "strength": 0.5}},
    }


CLOSE_INDEX = {"600549": [("2026-09-24", 12.00), ("2026-09-25", 12.10), ("2026-09-28", 12.34)]}


class TestContentFingerprint:
    def test_write_metadata_does_not_change_fingerprint(self):
        a = _rec("2026-09-28T12:38:05", record_id="sig_a", run_id="run_a")
        b = _rec("2026-09-28T16:20:40", record_id="sig_b", run_id="run_b")
        assert content_fingerprint(a) == content_fingerprint(b)

    def test_indicator_snapshot_difference_changes_fingerprint(self):
        """两个交易日恰好同价位时, 指标快照不同 → 必须是两条事实"""
        a = _rec("2026-09-28T12:38:05", indicators={"MA60": {"dir": 1, "strength": 0.5}})
        b = _rec("2026-09-28T12:38:05", indicators={"MA60": {"dir": 1, "strength": 0.9}})
        assert content_fingerprint(a) != content_fingerprint(b)

    def test_score_difference_changes_fingerprint(self):
        assert content_fingerprint(_rec("2026-09-28T12:38:05", score=-72.5)) != \
            content_fingerprint(_rec("2026-09-28T12:38:05", score=-70.0))


class TestDedupe:
    def test_same_day_reruns_collapse_to_one_fact(self):
        recs = [_rec(f"2026-09-28T{h}:{m:02d}:00") for h, m in
                ((12, 38), (12, 41), (12, 46), (16, 20))]
        kept, stat = compact_records(recs, CLOSE_INDEX)
        assert len(kept) == 1
        assert stat["duplicates_removed"] == 3

    def test_same_price_different_indicators_keeps_both(self):
        recs = [
            _rec("2026-09-25T22:00:00", indicators={"MA60": {"dir": 1, "strength": 0.5}}),
            _rec("2026-09-28T22:00:00", indicators={"MA60": {"dir": 1, "strength": 0.9}}),
        ]
        kept, stat = compact_records(recs, CLOSE_INDEX)
        assert len(kept) == 2
        assert stat["duplicates_removed"] == 0

    def test_outcome_records_pass_through_untouched(self):
        outcomes = [{"record_type": "outcome", "record_id": f"out_{i}",
                     "symbol": "600549", "recorded_at": "2026-09-28T22:00:00"}
                    for i in range(2)]
        kept, _ = compact_records(outcomes, CLOSE_INDEX)
        assert len(kept) == 2

    def test_keeps_first_occurrence(self):
        recs = [_rec("2026-09-28T12:38:00", record_id="sig_first", run_id="r1"),
                _rec("2026-09-28T16:20:00", record_id="sig_second", run_id="r2")]
        kept, _ = compact_records(recs, CLOSE_INDEX)
        assert kept[0]["record_id"] == "sig_first"


class TestDateBackfill:
    def test_picks_latest_matching_trade_date_not_after_write(self):
        date, is_current = infer_trade_date(_rec("2026-09-28T22:00:00"), CLOSE_INDEX, "2026-09-28")
        assert (date, is_current) == ("2026-09-28", True)

    def test_morning_run_maps_to_previous_session(self):
        """早上 09:10 分析用的是上一交易日的收盘数据"""
        date, is_current = infer_trade_date(_rec("2026-09-28T09:10:00", price=12.10),
                                            CLOSE_INDEX, "2026-09-28")
        assert (date, is_current) == ("2026-09-25", False)

    def test_no_price_match_leaves_date_none(self):
        date, _ = infer_trade_date(_rec("2026-09-28T22:00:00", price=99.99),
                                   CLOSE_INDEX, "2026-09-28")
        assert date is None

    def test_unknown_symbol_leaves_date_none(self):
        date, _ = infer_trade_date(_rec("2026-09-28T22:00:00", symbol="000000"),
                                   CLOSE_INDEX, "2026-09-28")
        assert date is None

    def test_price_match_far_in_history_is_rejected(self):
        """缓存覆盖多年, 历史同价位是巧合, 不能当成交交易日"""
        index = {"600549": [("2025-01-24", 12.34), ("2026-09-28", 12.90)]}
        date, _ = infer_trade_date(_rec("2026-09-28T22:00:00"), index, "2026-09-28")
        assert date is None

    def test_lag_window_boundary_is_inclusive(self):
        index = {"600549": [("2026-09-21", 12.34)]}
        date, _ = infer_trade_date(_rec("2026-09-28T22:00:00"), index, "2026-09-28")
        assert date == "2026-09-21"

    def test_just_outside_lag_window_is_rejected(self):
        index = {"600549": [("2026-09-20", 12.34)]}
        date, _ = infer_trade_date(_rec("2026-09-28T22:00:00"), index, "2026-09-28")
        assert date is None

    def test_backfill_written_into_kept_record(self):
        recs = [_rec("2026-09-28T12:38:00", analysis_date=None),
                _rec("2026-09-28T16:20:00", analysis_date=None)]
        kept, stat = compact_records(recs, CLOSE_INDEX)
        assert kept[0]["analysis_date"] == "2026-09-28"
        assert stat["date_backfilled"] == 1

    def test_existing_date_is_never_overwritten(self):
        recs = [_rec("2026-09-28T22:00:00", analysis_date="2026-09-21")]
        kept, stat = compact_records(recs, CLOSE_INDEX)
        assert kept[0]["analysis_date"] == "2026-09-21"
        assert stat["date_already_present"] == 1

    def test_backfill_disabled_keeps_date_none(self):
        kept, stat = compact_records([_rec("2026-09-28T22:00:00")], CLOSE_INDEX,
                                     backfill=False)
        assert kept[0]["analysis_date"] is None
        assert stat["date_unresolved"] == 1

    def test_conflicting_dates_on_same_key_keep_earliest_and_null_the_rest(self):
        """两条真事实推出同一个去重键时, 宁可缺日期也不给错日期"""
        recs = [
            _rec("2026-09-28T12:38:00", price=12.34, indicators={"MA60": {"s": 1}}),
            _rec("2026-09-28T16:20:00", price=12.34, indicators={"MA60": {"s": 2}}),
        ]
        kept, stat = compact_records(recs, CLOSE_INDEX)
        assert len(kept) == 2
        assert stat["date_nulled_on_conflict"] == 1
        assert [r["analysis_date"] for r in kept] == ["2026-09-28", None]
