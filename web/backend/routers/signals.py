"""信号分析路由 — 运行分析 / 查询状态 / 获取结果 / 分组列表"""
import json
import logging
import os
import threading

from fastapi import APIRouter

from services.signal_analyzer import SignalAnalyzer

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/signals", tags=["信号分析"])

analyzer = SignalAnalyzer()

# strategy_config.json 路径
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
_CONFIG_FILE = os.path.join(_PROJECT_ROOT, "config", "strategy_config.json")


@router.post("/run", summary="运行信号分析")
def run_analysis(body: dict = None):
    """启动信号分析任务（后台线程执行，立即返回）

    Body: {
        "group": "科技成长型",          // 可选，分组名称
        "user_regime": "trending"       // 可选，"auto"=自动 | "trending"=趋势上涨 | "ranging"=震荡
    }
    """
    if analyzer.is_running:
        return {"status": "already_running", "message": "分析任务正在运行中"}

    group = ""
    user_regime = "auto"
    if body:
        group = body.get("group", "").strip()
        user_regime = body.get("user_regime", "auto").strip()

    def _run():
        logger.info("后台线程启动: group=%s, regime=%s", group, user_regime)
        try:
            result = analyzer.run_analysis(group=group, user_regime=user_regime)
            logger.info("后台线程完成: status=%s, summary=%s", result.get("status"), result.get("summary"))
        except Exception:
            logger.exception("后台分析任务异常")

    t = threading.Thread(target=_run, daemon=True)
    t.start()

    regime_label = {"trending": "趋势上涨", "ranging": "震荡", "auto": "自动判断"}.get(user_regime, "")
    return {"status": "started", "message": f"分析任务已启动" + (f"（分组: {group}）" if group else "") + (f"（模式: {regime_label}）" if user_regime != "auto" else "")}


@router.get("/groups", summary="获取自选股分组列表")
def get_groups():
    """返回所有分组名称及股票数量，供前端选择分析范围"""
    if not os.path.exists(_CONFIG_FILE):
        return {"groups": []}

    with open(_CONFIG_FILE, "r", encoding="utf-8") as f:
        config = json.load(f)

    watchlist = config.get("strategy_config", {}).get("watchlist", {})
    groups = []
    for name, stocks in watchlist.items():
        if name.startswith("_"):
            continue
        groups.append({"name": name, "count": len(stocks)})

    return {"groups": groups}


@router.get("/status", summary="查询分析状态")
def get_status():
    """获取分析任务状态和进度"""
    return analyzer.get_status()


@router.get("/result", summary="获取分析结果")
def get_result():
    """获取最新分析结果"""
    result = analyzer.get_result()
    if result is None:
        return {
            "status": "no_result",
            "message": "尚未运行过分析，请先点击运行分析",
            "analyzed_at": None,
            "group": "",
            "summary": {"bullish": 0, "neutral": 0, "bearish": 0, "actionable": 0, "total": 0},
            "results": [],
        }
    return result
