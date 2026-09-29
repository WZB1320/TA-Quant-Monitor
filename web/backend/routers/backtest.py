"""回测路由 — 运行回测 / 获取结果

2026-09-29 瘦身为薄壳：回测编排（P5 分组独立回测 + 权重合并 + 结果格式化）
已抽取至 services/portfolio_backtest.py::run_portfolio_backtest（纯移动，零逻辑改动）。
本路由只负责：并发互斥锁、进度状态维护、结果缓存。
"""
import logging
import os
import sys
import threading

from fastapi import APIRouter

# 确保项目根目录在 sys.path
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

# 编排实现已迁至服务层；此处重新导出 BacktestRequest 保持对外契约不变
from services.portfolio_backtest import BacktestRequest, run_portfolio_backtest  # noqa: F401

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/backtest", tags=["回测分析"])


# ── 全局状态 (单例, 同一时间只允许一个回测任务) ──
_state = {
    "running": False,
    "progress": 0,
    "total": 0,
    "progress_text": "",
    "result": None,
    "lock": threading.Lock(),
}


@router.post("/run", summary="运行回测")
def run_backtest(req: BacktestRequest):
    """同步执行回测（阻塞直到完成）"""
    if not _state["lock"].acquire(blocking=False):
        return {"status": "error", "message": "已有回测任务正在运行"}

    def _on_progress(progress: int, text: str) -> None:
        _state["progress"] = progress
        if text:
            _state["progress_text"] = text

    try:
        _state["running"] = True
        _state["progress"] = 0

        result = run_portfolio_backtest(req, on_progress=_on_progress)

        if result.get("status") == "done":
            _state["result"] = result
        return result

    finally:
        _state["running"] = False
        _state["lock"].release()


@router.get("/status", summary="查询回测状态")
def get_status():
    return {
        "running": _state["running"],
        "progress": _state["progress"],
        "progress_text": _state["progress_text"],
        "has_result": _state["result"] is not None,
    }
