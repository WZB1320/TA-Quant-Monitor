"""收盘复盘大屏 — 路由层单测（刷新互斥 / 响应契约）"""
import os
import sys

import pytest
from fastapi import BackgroundTasks, HTTPException

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_BACKEND_ROOT = os.path.join(_PROJECT_ROOT, "web", "backend")
for _p in (_PROJECT_ROOT, _BACKEND_ROOT):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from routers import dashboard as dr  # noqa: E402
from services import dashboard_builder as db  # noqa: E402


def test_refresh_conflicts_when_running(monkeypatch):
    """已有生成在跑 → 409（前端据此禁用按钮）"""
    monkeypatch.setattr(db, "is_running", lambda: True)
    with pytest.raises(HTTPException) as ei:
        dr.refresh(BackgroundTasks())
    assert ei.value.status_code == 409


def test_refresh_accepted_and_schedules_job(monkeypatch):
    """空闲 → 202 accepted，且生成任务被排入后台"""
    monkeypatch.setattr(db, "is_running", lambda: False)
    called = {"n": 0}

    def _fake_generate(*a, **kw):
        called["n"] += 1
        return {"status": "done"}

    monkeypatch.setattr(db, "generate", _fake_generate)
    bg = BackgroundTasks()
    res = dr.refresh(bg)

    assert res["status"] == "accepted"
    assert len(bg.tasks) == 1
    bg.tasks[0].func()          # 执行排入的任务
    assert called["n"] == 1


def test_snapshot_shape(monkeypatch):
    """快照端点返回 snapshot/freshness/status 三段（无快照时 snapshot=None）"""
    monkeypatch.setattr(db, "load_snapshot", lambda *a, **kw: None)
    res = dr.get_snapshot()
    assert set(res.keys()) == {"snapshot", "freshness", "status"}
    assert res["snapshot"] is None
    assert res["freshness"]["has_snapshot"] is False


def test_status_keys():
    st = dr.get_status()
    assert set(st.keys()) == {"state", "stage", "progress", "error", "started_at", "finished_at"}
