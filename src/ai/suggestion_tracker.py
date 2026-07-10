"""建议追踪器 — 持久化 AI 建议, 追踪应用状态, 验证效果

数据流:
  LLM 周报 → 解析结构化建议 → 保存到 report_suggestions.jsonl
  下次生成 → 加载 pending 建议 → 注入 prompt → 检查是否被应用

存储: data/report_suggestions.jsonl (append-only, 每行一条建议)

建议状态流转:
  pending → applied (params 已变更, 匹配建议) → validated (前后对比完成)
  pending → superseded (超过 4 周未应用, 被新建议取代)
  pending → rejected (用户手动拒绝, 暂未实现)
"""
import json
import os
import uuid
from datetime import datetime
from typing import Optional

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
_SUGGESTIONS_FILE = os.path.join(_PROJECT_ROOT, "data", "report_suggestions.jsonl")


class SuggestionTracker:
    """AI 建议追踪器 — 保存/加载/验证策略优化建议"""

    def __init__(self, file_path: Optional[str] = None):
        self._file_path = file_path or _SUGGESTIONS_FILE

    @property
    def file_path(self) -> str:
        return self._file_path

    # ── 保存 ──

    def save_suggestions(self, suggestions: list, report_id: str,
                         week_start: str, week_end: str) -> int:
        """保存建议列表到 JSONL

        Args:
            suggestions: 建议字典列表 (从 LLM 输出解析)
            report_id: 来源周报 ID
            week_start: 周报数据区间开始
            week_end: 周报数据区间结束

        Returns:
            保存的条数
        """
        if not suggestions:
            return 0

        os.makedirs(os.path.dirname(self._file_path), exist_ok=True)
        count = 0
        for sug in suggestions:
            record = {
                "suggestion_id": f"sug_{uuid.uuid4().hex[:12]}",
                "report_id": report_id,
                "generated_at": datetime.now().isoformat(timespec="seconds"),
                "week_start": week_start,
                "week_end": week_end,
                "category": sug.get("category", "unknown"),
                "target": sug.get("target", ""),
                "current_value": sug.get("current_value"),
                "suggested_value": sug.get("suggested_value"),
                "rationale": sug.get("rationale", ""),
                "status": "pending",
                "applied_at": None,
                "applied_version": None,
                "validation": None,
            }
            with open(self._file_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")
            count += 1

        return count

    # ── 加载 ──

    def load_all_suggestions(self) -> list:
        """加载所有建议"""
        if not os.path.exists(self._file_path):
            return []
        records = []
        with open(self._file_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    records.append(json.loads(line))
        return records

    def load_pending_suggestions(self) -> list:
        """加载所有 status=pending 的建议"""
        return [s for s in self.load_all_suggestions() if s.get("status") == "pending"]

    def load_applied_suggestions(self) -> list:
        """加载所有 status=applied 的建议 (待验证)"""
        return [s for s in self.load_all_suggestions() if s.get("status") == "applied"]

    # ── 状态更新 ──

    def _rewrite_file(self, records: list) -> None:
        """全量重写文件 (状态更新时使用)"""
        with open(self._file_path, "w", encoding="utf-8") as f:
            for r in records:
                f.write(json.dumps(r, ensure_ascii=False, default=str) + "\n")

    def mark_applied(self, suggestion_id: str, applied_version: str) -> bool:
        """标记建议为已应用

        Args:
            suggestion_id: 建议 ID
            applied_version: 应用后的新 strategy_version

        Returns:
            是否成功更新
        """
        records = self.load_all_suggestions()
        updated = False
        for r in records:
            if r.get("suggestion_id") == suggestion_id:
                r["status"] = "applied"
                r["applied_at"] = datetime.now().isoformat(timespec="seconds")
                r["applied_version"] = applied_version
                updated = True
                break
        if updated:
            self._rewrite_file(records)
        return updated

    def mark_validated(self, suggestion_id: str, validation: dict) -> bool:
        """标记建议为已验证

        Args:
            suggestion_id: 建议 ID
            validation: 验证结果 {before_stats, after_stats, verdict}

        Returns:
            是否成功更新
        """
        records = self.load_all_suggestions()
        updated = False
        for r in records:
            if r.get("suggestion_id") == suggestion_id:
                r["status"] = "validated"
                r["validation"] = validation
                updated = True
                break
        if updated:
            self._rewrite_file(records)
        return updated

    def mark_superseded(self, suggestion_id: str) -> bool:
        """标记建议为已过期 (超时未应用)"""
        records = self.load_all_suggestions()
        updated = False
        for r in records:
            if r.get("suggestion_id") == suggestion_id:
                r["status"] = "superseded"
                updated = True
                break
        if updated:
            self._rewrite_file(records)
        return updated

    # ── 应用检测 ──

    def check_applied(self, current_params: dict,
                      current_strategy_version: str) -> list:
        """检查 pending 建议是否已被应用

        对比 current_params 中 target 字段的值是否与 suggested_value 匹配.
        若匹配则标记为 applied.

        Args:
            current_params: 当前参数快照 (params_snapshot)
            current_strategy_version: 当前 strategy_version 哈希

        Returns:
            新标记为 applied 的建议列表
        """
        pending = self.load_pending_suggestions()
        if not pending:
            return []

        # 从 params_snapshot 中提取可比较的参数值
        # params_snapshot 结构: {execution_params: {...}, indicator_params: {...}, ...}
        flat_params = self._flatten_params(current_params)

        newly_applied = []
        for sug in pending:
            target = sug.get("target", "")
            suggested = sug.get("suggested_value")
            current_val = flat_params.get(target)

            if current_val is not None and suggested is not None:
                # 类型安全的比较
                try:
                    if self._values_match(current_val, suggested):
                        self.mark_applied(sug["suggestion_id"], current_strategy_version)
                        newly_applied.append(sug)
                except (TypeError, ValueError):
                    continue

        return newly_applied

    @staticmethod
    def _flatten_params(params_snapshot: dict) -> dict:
        """将嵌套的 params_snapshot 展平为 {key: value} 字典

        execution_params.score_threshold → score_threshold
        indicator_params.ma_short → ma_short
        """
        flat = {}
        for section in params_snapshot.values():
            if isinstance(section, dict):
                for k, v in section.items():
                    flat[k] = v
            else:
                # 顶层非嵌套字段
                pass
        return flat

    @staticmethod
    def _values_match(current, suggested) -> bool:
        """比较两个值是否匹配 (容忍 int/float 类型差异)"""
        try:
            return float(current) == float(suggested)
        except (TypeError, ValueError):
            return str(current) == str(suggested)
