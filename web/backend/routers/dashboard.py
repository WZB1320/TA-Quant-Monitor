"""收盘复盘大屏路由 — 快照读取 / 异步生成 / 生成状态

口径（需求冻结）: 持仓=策略模拟持仓 · 净值=回测口径(模拟本金 ¥100,000) · 体制=auto
刷新策略: 快照已是最近交易日 → 前端秒开不重跑；过期/缺失 → 手动触发生成。
"""
import logging
import os
import sys

from fastapi import APIRouter, BackgroundTasks, HTTPException

# 确保项目根目录与 web/backend 在 sys.path
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
_BACKEND_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for _p in (_PROJECT_ROOT, _BACKEND_ROOT):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from services import dashboard_builder as db  # noqa: E402

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/dashboard", tags=["复盘大屏"])


@router.get("/snapshot", summary="读取大屏快照")
def get_snapshot():
    """返回快照 + 数据新鲜度 + 当前生成状态。

    快照不存在时 snapshot=null（前端进入首次生成空态）。
    """
    snap = db.load_snapshot()
    return {
        "snapshot": snap,
        "freshness": db.compute_freshness(snap),
        "status": db.get_status(),
    }


@router.post("/refresh", summary="触发生成（异步）", status_code=202)
def refresh(background_tasks: BackgroundTasks):
    """异步生成快照。

    - 已有生成在跑 → 409（前端据此禁用按钮并继续轮询 status）
    - 生成失败不覆盖旧快照（原子写）
    """
    if db.is_running():
        raise HTTPException(status_code=409, detail="已有生成任务正在运行")

    def _job():
        res = db.generate()
        if res.get("status") != "done":
            logger.error("大屏生成失败: %s", res.get("message"))

    background_tasks.add_task(_job)
    return {"status": "accepted", "message": "生成任务已启动", **db.get_status()}


@router.get("/status", summary="查询生成状态")
def get_status():
    return db.get_status()
