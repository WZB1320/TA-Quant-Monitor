"""FastAPI 应用入口

启动方式:
  cd web/backend
  uvicorn app:app --reload --port 8000

API 文档:
  http://localhost:8000/docs      (Swagger UI)
  http://localhost:8000/redoc     (ReDoc)
"""
import logging
import os
import sys

# 确保项目根目录在 sys.path 中, 以便 import src.*
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from src.config.runtime_mode import set_mode, RuntimeMode
# 设置实时模式: SignalFilter 读写磁盘, 用户偏好持久化
set_mode(RuntimeMode.LIVE)

from routers.watchlist import router as watchlist_router
from routers.stocks import router as stocks_router
from routers.signals import router as signals_router
from routers.config import router as config_router
from routers.backtest import router as backtest_router
from routers.stock_backtest import router as stock_backtest_router
from routers.ai_report import router as ai_report_router
from services.knowledge_base import KnowledgeBase

logging.basicConfig(level=logging.INFO, format="%(levelname)s [%(name)s] %(message)s")
logger = logging.getLogger("app")


# ── 启动与关闭 ──

@asynccontextmanager
async def lifespan(app: FastAPI):
    """应用生命周期：启动时初始化知识库"""
    logger.info("正在启动量化交易监控系统...")
    try:
        kb = KnowledgeBase()
        if kb.is_empty():
            logger.info("知识库为空，开始首次同步...")
            count = kb.sync_from_akshare(use_slim=True)
            logger.info("首次同步完成，共 %d 只股票", count)
        else:
            logger.info("知识库已有 %d 只股票，跳过同步", kb.count)
    except Exception:
        logger.exception("知识库初始化失败，搜索功能可能不可用")
    yield
    logger.info("应用关闭")


app = FastAPI(
    title="量化交易监控系统 API",
    description="自选股管理 / 策略配置 / 回测执行",
    version="0.2.0",
    lifespan=lifespan,
)

# CORS: 允许前端开发服务器访问
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],       # 开发阶段允许所有来源
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 注册路由
app.include_router(watchlist_router)
app.include_router(stocks_router)
app.include_router(signals_router)
app.include_router(config_router)
app.include_router(backtest_router)
app.include_router(stock_backtest_router)
app.include_router(ai_report_router)


@app.get("/api/health", tags=["系统"])
def health_check():
    """健康检查"""
    return {"status": "ok", "version": "0.1.0"}
