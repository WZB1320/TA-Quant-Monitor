"""收盘复盘大屏 — builder 服务单测

覆盖:
  - 数据新鲜度判定（工作日近似 / 节假日探明值绕过 / 滞后交易日数）
  - 原子写与失败不覆盖
  - 快照 schema 与曲线对齐
  - 并发互斥
"""
import json
import os
import sys
from datetime import datetime

import pytest

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_BACKEND_ROOT = os.path.join(_PROJECT_ROOT, "web", "backend")
for _p in (_PROJECT_ROOT, _BACKEND_ROOT):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from services import dashboard_builder as db  # noqa: E402


# ── 新鲜度 ──

def _snap(data_date, generated_at, probed=None):
    return {
        "data_date": data_date,
        "generated_at": generated_at,
        "probed_latest_trade_date": probed or data_date,
    }


def test_freshness_no_snapshot():
    fr = db.compute_freshness(None)
    assert fr["has_snapshot"] is False
    assert fr["is_fresh"] is False
    assert fr["data_date"] is None


def test_freshness_fresh_same_as_expected():
    # 2026-09-29 10:00（盘中）→ 预期最新交易日回退到 2026-09-28
    now = datetime(2026, 9, 29, 10, 0)
    fr = db.compute_freshness(_snap("2026-09-28", "2026-09-29 09:30:00"), now=now)
    assert fr["expected_last_trade_date"] == "2026-09-28"
    assert fr["is_fresh"] is True
    assert fr["stale_trade_days"] == 0


def test_freshness_stale_counts_business_days():
    now = datetime(2026, 9, 29, 18, 0)
    fr = db.compute_freshness(_snap("2026-09-24", "2026-09-25 09:30:00"), now=now)
    # 预期 = 2026-09-29；滞后交易日 = 9/25, 9/28, 9/29
    assert fr["expected_last_trade_date"] == "2026-09-29"
    assert fr["is_fresh"] is False
    assert fr["stale_trade_days"] == 3


def test_freshness_holiday_probe_bypasses_workday_approx():
    """节假日场景: 工作日近似把 9/25 当作预期交易日，但当日已探明源端最新=9/24 → 视为新鲜。

    这是「节假日每次打开都白等一次全量重下」的修复点。
    """
    now = datetime(2026, 9, 25, 10, 0)   # 中秋休市
    fr = db.compute_freshness(_snap("2026-09-24", "2026-09-25 09:30:00", probed="2026-09-24"),
                              now=now)
    assert fr["expected_last_trade_date"] == "2026-09-24"
    assert fr["is_fresh"] is True


def test_freshness_probe_bypass_requires_same_day():
    """探明值绕过仅对「当日生成」有效：隔天必须重新判定为过期。"""
    snap = _snap("2026-09-24", "2026-09-25 09:30:00", probed="2026-09-24")
    now = datetime(2026, 9, 28, 18, 0)
    fr = db.compute_freshness(snap, now=now)
    assert fr["is_fresh"] is False
    assert fr["stale_trade_days"] == 2   # 9/25(近似), 9/28


# ── 原子写 ──

def test_atomic_write_and_load(tmp_path):
    path = str(tmp_path / "dashboard.json")
    snap = {"data_date": "2026-09-24", "nav": {"latest": 1.295}}
    db._atomic_write(snap, path)
    assert db.load_snapshot(path) == snap
    assert not os.path.exists(path + ".tmp")   # 临时文件已清理


def test_atomic_write_failure_keeps_old_snapshot(tmp_path):
    path = str(tmp_path / "dashboard.json")
    good = {"data_date": "2026-09-24", "nav": {"latest": 1.295}}
    db._atomic_write(good, path)

    with pytest.raises(TypeError):
        db._atomic_write({"bad": {1, 2, 3}}, path)   # set 不可 JSON 序列化

    assert db.load_snapshot(path) == good    # 旧快照完好


def test_load_snapshot_corrupt_returns_none(tmp_path):
    path = str(tmp_path / "dashboard.json")
    with open(path, "w", encoding="utf-8") as f:
        f.write("{ this is not json")
    assert db.load_snapshot(path) is None


# ── 快照组装 ──

class _FakeDM:
    """替代 DataManager: 不触网，仅返回 None（用于验证 chg_pct 缺数据兜底）"""
    def get_daily_kline(self, *a, **kw):
        return None


# 与 _fake_sig_res() 完全对齐的自选股清单 → 无缺失行
WL_FULL = [
    {"code": "600392", "name": "盛和资源", "market": "sh"},
    {"code": "603002", "name": "宏昌电子", "market": "sh"},
]


def _fake_bt_res():
    return {
        "status": "done",
        "trades": [
            {"symbol": "603002", "name": "宏昌电子", "group": "科技成长型", "status": "open",
             "entry_date": "2026-09-17", "entry_price": 19.382, "exit_price": 19.19,
             "shares": 800, "cost": 15505.6, "pnl": -153.6, "pnl_pct": -0.99},
            {"symbol": "600000", "name": "浦发银行", "group": "消费稳健型", "status": "closed",
             "entry_date": "2026-02-02", "entry_price": 10.0, "exit_price": 11.0,
             "shares": 100, "cost": 1000.0, "pnl": 100.0, "pnl_pct": 10.0},
        ],
        "daily_values": {"dates": ["2026-01-05", "2026-01-06", "2026-03-02", "2026-09-24"],
                         "values": [100000.0, 101000.0, 95000.0, 129500.71]},
        "benchmark_values": {"dates": ["2026-01-05", "2026-01-06", "2026-03-02", "2026-09-24"],
                             "values": [1.0, 1.01, 0.97, 0.940946]},
    }


def _fake_sig_res():
    return {
        "summary": {"bullish": 0, "neutral": 1, "bearish": 1, "actionable": 1, "total": 2},
        "results": [
            {"code": "600392", "name": "盛和资源", "group": "周期资源型", "close": 20.1,
             "level": "STRONG_SELL", "label": "强卖出", "score": -80.0,
             "execution": {"executable": False, "status": "无需操作", "reason": ""},
             "indicators": {}},
            {"code": "603002", "name": "宏昌电子", "group": "科技成长型", "close": 19.19,
             "level": "NEUTRAL", "label": "观望", "score": 5.0,
             "execution": {"executable": False, "status": "无需操作", "reason": ""},
             "indicators": {}},
        ],
    }


def test_build_snapshot_schema_and_curves():
    snap = db.build_snapshot(
        data_date="2026-09-24", dm=_FakeDM(),
        indices=[{"code": "sh.000300", "name": "沪深300", "close": 4439.14,
                  "chg": -78.0, "chg_pct": -1.73, "spark": [4400.0, 4439.14]}],
        sig_res=_fake_sig_res(), bt_res=_fake_bt_res(), t_start=0.0,
        watchlist_stocks=WL_FULL,
    )

    # 口径字段
    assert snap["data_date"] == "2026-09-24"
    assert snap["probed_latest_trade_date"] == "2026-09-24"
    assert snap["regime_mode"] == "auto"
    assert snap["initial_capital"] == 100000.0
    assert snap["nav_start"] == "2026-01-01"
    assert snap["param_fixed_date"] == "2026-08-07"

    # 信号排序: 按等级降序（NEUTRAL 在 STRONG_SELL 之前）
    assert [r["code"] for r in snap["signals"]["rows"]] == ["603002", "600392"]
    # 自选股全覆盖 → 无缺失行
    assert snap["signals"]["summary"]["missing"] == 0
    assert snap["signals"]["summary"]["total"] == 2

    # 持仓: 仅未平仓，权重 100%，现价取末日常规收盘
    pos = snap["positions"]
    assert pos["count"] == 1
    assert pos["rows"][0]["code"] == "603002"
    assert pos["rows"][0]["weight_pct"] == 100.0
    assert pos["position_pct"] == round(15505.6 / 100000 * 100, 1)

    # 净值: 金额 → 归一化；基准保持 1.0 起点；两线等长对齐
    nav = snap["nav"]
    assert nav["latest"] == 1.295            # 大屏展示用 4 位小数
    assert nav["latest_value"] == 129500.71
    assert nav["strategy_curve"]["values"][-1] == 1.295007   # 曲线保留 6 位
    assert nav["strategy_curve"]["values"][0] == 1.0
    assert len(nav["strategy_curve"]["dates"]) == len(nav["strategy_curve"]["values"])
    assert len(nav["benchmark_curve"]["dates"]) == len(nav["benchmark_curve"]["values"])
    assert len(nav["strategy_curve"]["dates"]) == len(nav["benchmark_curve"]["dates"])
    assert nav["alpha"] == round(nav["total_return"] - nav["benchmark_return"], 4)
    assert nav["max_drawdown"] == -0.0594     # 101000 → 95000 的回撤

    # 可 JSON 序列化（原子写前提）
    json.dumps(snap, ensure_ascii=False)


def test_snapshot_chg_pct_none_when_unavailable():
    """信号行取不到两日收盘时 chg_pct 必须为 None（前端渲染「—」），不得伪装成 0。"""
    snap = db.build_snapshot(
        data_date="2026-09-24", dm=_FakeDM(), indices=[],
        sig_res=_fake_sig_res(), bt_res=_fake_bt_res(), t_start=0.0,
        watchlist_stocks=WL_FULL,
    )
    for r in snap["signals"]["rows"]:
        assert r["chg_pct"] is None
    json.dumps(snap, ensure_ascii=False)   # None 可序列化


# ── 数据缺失占位行（需求: 35 只不漏不折叠 + 数据缺失不隐藏） ──

def test_missing_rows_appended_at_tail_and_marked():
    """自选股中有票未产出信号 → 补占位行（保留行、标数据缺失、沉到最末）。"""
    wl = WL_FULL + [{"code": "999999", "name": "样本不足股", "market": "sz"}]
    snap = db.build_snapshot(
        data_date="2026-09-24", dm=_FakeDM(), indices=[],
        sig_res=_fake_sig_res(), bt_res=_fake_bt_res(), t_start=0.0,
        watchlist_stocks=wl,
    )
    rows = snap["signals"]["rows"]
    assert len(rows) == 3                       # 不漏行
    assert rows[-1]["code"] == "999999"         # 沉到最末
    m = rows[-1]
    assert m["missing"] is True
    assert m["level"] == "MISSING"
    assert m["label"] == "数据缺失"
    assert m["execution"]["status"] == "数据缺失"
    assert m["close"] is None and m["chg_pct"] is None and m["score"] is None
    assert snap["signals"]["summary"]["missing"] == 1
    assert snap["signals"]["summary"]["total"] == 3
    json.dumps(snap, ensure_ascii=False)


def test_missing_row_reason_marks_paused_group(monkeypatch):
    """缺失行若属已暂停分组，原因需写明「已暂停」而非笼统「数据缺失」。"""
    import src.config.group_config as gcm

    class _FakeGC:
        def get_group(self, code):
            return "周期资源型"

    monkeypatch.setattr(gcm, "GroupConfig", _FakeGC)

    sig = _fake_sig_res()
    sig["paused_skipped"] = {"周期资源型": 1}
    snap = db.build_snapshot(
        data_date="2026-09-24", dm=_FakeDM(), indices=[],
        sig_res=sig, bt_res=_fake_bt_res(), t_start=0.0,
        watchlist_stocks=[{"code": "777777", "name": "暂停组股", "market": "sh"}],
    )
    m = snap["signals"]["rows"][-1]
    assert m["code"] == "777777" and m["missing"] is True
    assert "已暂停" in m["execution"]["reason"]


def test_missing_row_keeps_held_flag():
    """缺失行若恰是回测持仓票，仍要打上持仓标记（避免误以为持仓丢了）。"""
    sig = _fake_sig_res()
    sig["results"] = [r for r in sig["results"] if r["code"] != "603002"]  # 603002 未产出信号
    snap = db.build_snapshot(
        data_date="2026-09-24", dm=_FakeDM(), indices=[],
        sig_res=sig, bt_res=_fake_bt_res(), t_start=0.0,
        watchlist_stocks=WL_FULL,
    )
    m = next(r for r in snap["signals"]["rows"] if r["code"] == "603002")
    assert m["missing"] is True
    assert m["held"] is True          # 回测仍有该持仓 → 标记不丢
    assert snap["signals"]["summary"]["missing"] == 1


# ── 并发互斥 ──

def test_generate_mutex_returns_running(monkeypatch):
    """已有生成在跑时，第二次调用直接返回 running，不重复启动管线。"""
    db._state["lock"].acquire()
    try:
        res = db.generate()
        assert res["status"] == "running"
    finally:
        db._state["lock"].release()
