"""AI 周报路由 — 触发策略周报生成

端点:
  POST /api/ai/weekly-report        — 生成周报 (指定 memory 文件)
  POST /api/ai/weekly-report/auto   — 按自然周生成周报 (反馈闭环)
  POST /api/ai/weekly-report/stream — 流式生成周报 (SSE, 实时输出)
  GET  /api/ai/llm-status            — 检查 LLM 配置状态
  GET  /api/ai/suggestions           — 列出所有 AI 建议
  GET  /api/ai/suggestions/pending   — 列出待应用建议
  GET  /api/ai/reports               — 列出所有周报 (索引)
  GET  /api/ai/reports/{report_id}   — 检索单篇周报内容
"""
import os
import sys
import json
import logging

from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from typing import Optional

# 确保项目根目录在 sys.path
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from src.ai import WeeklyReportGenerator, LLMClient, WeeklyReportScheduler, SuggestionTracker

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/ai", tags=["AI 分析"])


class ReportRequest(BaseModel):
    """周报生成请求"""
    memory_file: Optional[str] = None  # 指定记忆文件路径, None 则自动查找最新


class ReportResponse(BaseModel):
    """周报生成响应"""
    status: str           # "ok" | "error"
    report: str           # 周报 Markdown 文本
    saved_to: Optional[str] = None  # 保存路径
    llm_available: bool   # LLM 是否可用
    message: Optional[str] = None   # 附加信息


@router.post("/weekly-report", response_model=ReportResponse)
def generate_weekly_report(req: ReportRequest = ReportRequest()):
    """生成策略周报

    - 自动查找最新记忆文件, 或使用指定的 memory_file
    - 若 LLM 已配置, 生成 AI 分析周报; 否则降级输出统计摘要
    """
    try:
        gen = WeeklyReportGenerator()
        report = gen.generate(memory_file=req.memory_file, save=True)

        # 查找保存的文件路径
        saved_to = None
        if gen.llm.is_available or "统计摘要" in report:
            reports_dir = os.path.join(_PROJECT_ROOT, "data", "reports")
            if os.path.isdir(reports_dir):
                files = sorted(os.listdir(reports_dir), reverse=True)
                if files:
                    saved_to = os.path.join(reports_dir, files[0])

        return ReportResponse(
            status="ok",
            report=report,
            saved_to=saved_to,
            llm_available=gen.llm.is_available,
            message=None if gen.llm.is_available else "LLM 未配置, 输出统计摘要",
        )
    except FileNotFoundError as e:
        return ReportResponse(
            status="error",
            report="",
            llm_available=False,
            message=str(e),
        )
    except Exception as e:
        logger.exception("周报生成失败")
        return ReportResponse(
            status="error",
            report="",
            llm_available=LLMClient().is_available,
            message=f"生成失败: {e}",
        )


@router.get("/llm-status")
def llm_status():
    """检查 LLM 配置状态"""
    client = LLMClient()
    return {
        "available": client.is_available,
        "model": client.model,
        "base_url": client.base_url,
        "has_api_key": bool(client.api_key and client.api_key != "sk-your-api-key-here"),
    }


# ── 按周生成周报 (反馈闭环) ──

class AutoReportRequest(BaseModel):
    """按周生成周报请求"""
    weeks_ago: int = 0  # 0=本周, 1=上周


@router.post("/weekly-report/auto")
def generate_auto_report(req: AutoReportRequest = AutoReportRequest()):
    """按自然周生成周报 (启用反馈闭环)

    - 按 recorded_at 筛选指定周的 live memory 记录
    - 自动加载历史建议注入 LLM prompt
    - LLM 输出的结构化建议被解析并保存
    - 下次生成时引用上次建议
    """
    try:
        scheduler = WeeklyReportScheduler()
        result = scheduler.generate_week(weeks_ago=req.weeks_ago, save=True)

        return {
            "status": result["status"],
            "report": result["report"],
            "week_start": result["week_start"],
            "week_end": result["week_end"],
            "record_count": result["record_count"],
            "saved_to": result["saved_to"],
            "suggestions_saved": result["suggestions_saved"],
            "llm_available": LLMClient().is_available,
            "message": result["message"],
        }
    except Exception as e:
        logger.exception("按周生成周报失败")
        return {
            "status": "error",
            "report": "",
            "message": f"生成失败: {e}",
        }


# ── 流式生成周报 (SSE) ──

@router.post("/weekly-report/stream")
def stream_weekly_report(req: ReportRequest = ReportRequest()):
    """流式生成周报 (Server-Sent Events)

    与 /weekly-report 功能相同, 但 LLM 输出逐块推送到前端,
    用户无需等待完整报告生成即可看到内容逐步出现.
    同时启用反馈闭环: 加载历史建议 + 保存本次建议.

    SSE 事件格式:
      data: {"type": "delta", "content": "..."}\\n\\n     — 文本块 (前端拼接显示)
      data: {"type": "done", "suggestions_saved": N, ...}\\n\\n  — 完成
      data: {"type": "error", "message": "..."}\\n\\n     — 错误

    前端示例 (fetch + ReadableStream, 因 EventSource 不支持 POST):
      const resp = await fetch("/api/ai/weekly-report/stream", {{
          method: "POST",
          headers: {{ "Content-Type": "application/json" }},
          body: JSON.stringify({{}})
      }});
      const reader = resp.body.getReader();
      const decoder = new TextDecoder();
      let buffer = "";
      while (true) {{
          const {{ done, value }} = await reader.read();
          if (done) break;
          buffer += decoder.decode(value, {{ stream: true }});
          const lines = buffer.split("\\n\\n");
          buffer = lines.pop();
          for (const line of lines) {{
              if (line.startsWith("data: ")) {{
                  const data = JSON.parse(line.slice(6));
                  if (data.type === "delta") appendToUI(data.content);
                  if (data.type === "done") return;
              }}
          }}
      }}
    """
    from datetime import datetime as _dt
    gen = WeeklyReportGenerator()

    # 构造 report_meta 以启用反馈闭环 (建议保存)
    today_str = _dt.now().strftime("%Y-%m-%d")
    report_meta = {"week_start": today_str, "week_end": today_str}

    def event_stream():
        try:
            for event in gen.generate_stream(
                memory_file=req.memory_file,
                save=True,
                report_meta=report_meta,
            ):
                yield f"data: {json.dumps(event, ensure_ascii=False)}\n\n"
        except Exception as e:
            logger.exception("SSE 流式生成异常")
            error_event = {"type": "error", "message": f"生成失败: {e}"}
            yield f"data: {json.dumps(error_event, ensure_ascii=False)}\n\n"

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",  # 禁用 Nginx 缓冲
        },
    )


# ── 建议查询 ──

@router.get("/suggestions")
def list_suggestions(status: Optional[str] = None):
    """列出 AI 建议

    Args:
        status: 可选过滤状态 (pending/applied/validated/superseded).
                不传则返回全部.
    """
    tracker = SuggestionTracker()
    all_suggestions = tracker.load_all_suggestions()

    if status:
        all_suggestions = [s for s in all_suggestions if s.get("status") == status]

    # 按生成时间降序
    all_suggestions.sort(key=lambda s: s.get("generated_at", ""), reverse=True)

    return {
        "total": len(all_suggestions),
        "suggestions": all_suggestions,
    }


@router.get("/suggestions/pending")
def list_pending_suggestions():
    """列出待应用建议"""
    tracker = SuggestionTracker()
    pending = tracker.load_pending_suggestions()
    pending.sort(key=lambda s: s.get("generated_at", ""), reverse=True)

    return {
        "total": len(pending),
        "suggestions": pending,
    }


# ── 周报索引与检索 ──

_REPORT_DIR = os.path.join(_PROJECT_ROOT, "data", "reports")
_REPORT_INDEX_FILE = os.path.join(_REPORT_DIR, "index.json")


def _load_report_index() -> list:
    """加载周报索引文件, 不存在则返回空列表"""
    if not os.path.exists(_REPORT_INDEX_FILE):
        return []
    try:
        with open(_REPORT_INDEX_FILE, "r", encoding="utf-8") as f:
            index = json.load(f)
            return index if isinstance(index, list) else []
    except (json.JSONDecodeError, OSError):
        return []


@router.get("/reports")
def list_reports(limit: int = 50):
    """列出所有周报 (按生成时间降序)

    Args:
        limit: 最多返回条数 (默认 50)
    """
    index = _load_report_index()
    # 按生成时间降序
    index.sort(key=lambda r: r.get("generated_at", ""), reverse=True)
    return {
        "total": len(index),
        "reports": index[:limit],
    }


@router.get("/reports/{report_id}")
def get_report(report_id: str):
    """检索单篇周报内容

    Args:
        report_id: 周报 ID (如 weekly_20260710_143000)
    """
    index = _load_report_index()
    entry = next((r for r in index if r.get("report_id") == report_id), None)

    if entry is None:
        raise HTTPException(status_code=404, detail=f"周报不存在: {report_id}")

    filepath = os.path.join(_REPORT_DIR, entry.get("filename", ""))
    if not os.path.exists(filepath):
        raise HTTPException(status_code=404, detail=f"周报文件已丢失: {entry.get('filename')}")

    with open(filepath, "r", encoding="utf-8") as f:
        content = f.read()

    return {
        "report_id": report_id,
        "content": content,
        "metadata": entry,
    }
