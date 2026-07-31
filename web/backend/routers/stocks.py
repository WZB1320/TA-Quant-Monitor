"""股票知识库路由 — 联想搜索 + 手动同步"""
import logging

from fastapi import APIRouter, BackgroundTasks, HTTPException

from schemas.stock import StockSearchResponse, StockSearchItem, SyncResponse
from services.knowledge_base import KnowledgeBase

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/stocks", tags=["股票知识库"])

# 全局知识库单例
kb = KnowledgeBase()


@router.get("/search", response_model=StockSearchResponse, summary="联想搜索股票")
def search_stocks(q: str = "", limit: int = 10):
    """从本地知识库联想搜索股票

    支持输入类型：
    - 纯数字（如 000001）→ 代码前缀匹配
    - 纯英文（如 PA / pingan）→ 拼音首字母 + 全拼搜索
    - 含中文（如 平安）→ 名称模糊匹配
    """
    if not q or len(q.strip()) < 1:
        return StockSearchResponse(results=[])
    results = kb.search(q.strip(), limit)
    return StockSearchResponse(
        results=[StockSearchItem(**r) for r in results]
    )


@router.post("/sync", response_model=SyncResponse, summary="手动同步股票知识库")
def sync_stocks(background_tasks: BackgroundTasks):
    """手动触发全量同步

    从 akshare 拉取全量 A 股股票列表，同步到本地 SQLite 知识库。
    根据网络状况预计耗时 3-10 秒。
    """
    try:
        count = kb.sync_from_akshare(use_slim=True)
        return SyncResponse(ok=True, synced=count,
                            message=f"同步完成，共 {count} 只股票")
    except Exception as e:
        logger.exception("同步知识库失败")
        raise HTTPException(status_code=500, detail=f"同步失败: {e}")


@router.get("/status", summary="知识库状态")
def kb_status():
    """查看知识库状态"""
    return {
        "ok": True,
        "db_path": kb.db_path,
        "total": kb.count,
        "is_empty": kb.is_empty(),
    }