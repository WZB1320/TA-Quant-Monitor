"""周报调度器 — 按自然周切分记忆数据, 生成周报

职责:
  1. 计算周区间 (周一~周日)
  2. 从 live memory 文件中筛选指定周的记录
  3. 调用 WeeklyReportGenerator 生成周报

用法:
  from src.ai import WeeklyReportScheduler
  scheduler = WeeklyReportScheduler()

  # 生成本周周报
  result = scheduler.generate_week(weeks_ago=0)

  # 生成上周周报
  result = scheduler.generate_week(weeks_ago=1)

设计说明:
  - 按 recorded_at 字段筛选 (而非 analysis_date), 因为 recorded_at 是写入时间
  - live memory 按月切分 (strategy_memory_YYYY-MM.jsonl), 支持跨月读取
  - backtest memory 按 run_id 分文件, 不适合按周切分, 直接用整个文件
"""
import json
import os
import tempfile
import logging
from datetime import datetime, timedelta
from typing import Optional

from src.memory import find_live_memory_files
from .report_generator import WeeklyReportGenerator
from .suggestion_tracker import SuggestionTracker

logger = logging.getLogger(__name__)

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class WeeklyReportScheduler:
    """按自然周切分记忆数据, 生成周报"""

    def __init__(self, memory_file: Optional[str] = None):
        """
        Args:
            memory_file: 记忆文件路径. None 则自动查找所有 live memory 文件
                         (含按月切分的 strategy_memory_YYYY-MM.jsonl + 旧格式).
        """
        self._memory_file = memory_file  # None = 自动查找所有 live 文件

    # ── 周区间计算 ──

    @staticmethod
    def get_week_range(date: datetime, weeks_ago: int = 0) -> tuple:
        """返回指定日期所在周的 (周一, 周日) 日期字符串

        Args:
            date: 参考日期
            weeks_ago: 往前推几周 (0=本周, 1=上周)

        Returns:
            (week_start_str, week_end_str) 格式 "YYYY-MM-DD"
        """
        # Monday=0, Sunday=6
        monday = date.date() - timedelta(days=date.weekday()) - timedelta(weeks=weeks_ago)
        sunday = monday + timedelta(days=6)
        return monday.strftime("%Y-%m-%d"), sunday.strftime("%Y-%m-%d")

    @staticmethod
    def get_current_week_range(weeks_ago: int = 0) -> tuple:
        """获取当前周的区间"""
        return WeeklyReportScheduler.get_week_range(datetime.now(), weeks_ago)

    # ── 按周筛选记录 ──

    def filter_records_by_week(self, week_start: str, week_end: str) -> Optional[str]:
        """从 memory 文件中筛选指定周的记录, 写入临时文件

        按 recorded_at 字段过滤 (recorded_at 是 ISO 格式, 含时间部分).
        周区间按日期比较 (取前 10 字符).

        支持跨月读取: 当 self._memory_file 为 None 时, 自动查找所有 live memory 文件
        (含按月切分的 strategy_memory_YYYY-MM.jsonl + 旧格式),
        确保跨月周边间 (如 7/30~8/5) 的数据不会丢失.

        Args:
            week_start: "YYYY-MM-DD"
            week_end: "YYYY-MM-DD"

        Returns:
            临时文件路径, 若无匹配记录则返回 None
        """
        # 确定要读取的文件列表
        if self._memory_file:
            files = [self._memory_file] if os.path.exists(self._memory_file) else []
        else:
            files = find_live_memory_files()

        if not files:
            logger.warning("无 live memory 文件可读")
            return None

        # 从所有文件中筛选匹配记录
        matched = []
        for file_path in files:
            with open(file_path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        record = json.loads(line)
                        recorded_at = record.get("recorded_at", "")
                        # recorded_at 格式: "2025-07-07T09:00:00", 取前 10 字符比较
                        record_date = recorded_at[:10]
                        if week_start <= record_date <= week_end:
                            matched.append(line)
                    except json.JSONDecodeError:
                        continue

        if not matched:
            logger.info("周区间 %s ~ %s 无匹配记录 (扫描 %d 个文件)",
                        week_start, week_end, len(files))
            return None

        # 写入临时文件
        temp_fd, temp_path = tempfile.mkstemp(
            suffix=".jsonl", prefix=f"week_{week_start}_", dir=_PROJECT_ROOT
        )
        with os.fdopen(temp_fd, "w", encoding="utf-8") as f:
            for line in matched:
                f.write(line + "\n")

        logger.info("周区间 %s ~ %s 筛选完成: %d 条记录 (扫描 %d 个文件) → %s",
                    week_start, week_end, len(matched), len(files), temp_path)
        return temp_path

    # ── 生成周报 ──

    def generate_week(self, weeks_ago: int = 0, save: bool = True) -> dict:
        """生成指定周的周报

        Args:
            weeks_ago: 0=本周, 1=上周, 2=上上周...
            save: 是否保存到 data/reports/

        Returns:
            {
                "status": "ok" | "error",
                "report": str,          # 周报 Markdown
                "week_start": str,
                "week_end": str,
                "record_count": int,    # 筛选到的记录数
                "saved_to": str | None,
                "suggestions_saved": int, # 本次保存的建议数
                "message": str | None,
            }
        """
        week_start, week_end = self.get_current_week_range(weeks_ago)
        logger.info("生成周报: weeks_ago=%d, 区间 %s ~ %s", weeks_ago, week_start, week_end)

        # 1. 按周筛选记录
        temp_file = self.filter_records_by_week(week_start, week_end)
        if temp_file is None:
            return {
                "status": "error",
                "report": "",
                "week_start": week_start,
                "week_end": week_end,
                "record_count": 0,
                "saved_to": None,
                "suggestions_saved": 0,
                "message": f"周区间 {week_start} ~ {week_end} 无记忆数据",
            }

        # 统计记录数
        with open(temp_file, "r", encoding="utf-8") as f:
            record_count = sum(1 for line in f if line.strip())

        # 2. 生成周报 (传入周报元信息); try/finally 确保临时文件被清理
        gen = WeeklyReportGenerator()
        try:
            report = gen.generate(
                memory_file=temp_file,
                save=save,
                report_meta={
                    "week_start": week_start,
                    "week_end": week_end,
                    "weeks_ago": weeks_ago,
                },
            )

            # 3. 查找保存路径
            saved_to = None
            if save:
                reports_dir = os.path.join(_PROJECT_ROOT, "data", "reports")
                if os.path.isdir(reports_dir):
                    files = sorted(os.listdir(reports_dir), reverse=True)
                    if files:
                        saved_to = os.path.join(reports_dir, files[0])
        finally:
            # 4. 清理临时文件 (无论成功或异常都必须执行)
            try:
                os.unlink(temp_file)
            except OSError:
                pass

        return {
            "status": "ok",
            "report": report,
            "week_start": week_start,
            "week_end": week_end,
            "record_count": record_count,
            "saved_to": saved_to,
            "suggestions_saved": getattr(gen, "_last_suggestions_saved", 0),
            "message": None,
        }
