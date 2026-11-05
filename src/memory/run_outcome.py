"""运行结果基线存储 (P4 自动验证器的输入)

定位:
  L1 失效案例库只在命中失败规则时写入, 没有"每次运行都落一条"的基线,
  导致无法对比"建议应用前 vs 应用后"。本模块每次回测完成写一条 outcome
  (成功/失败都写), 为 SuggestionValidator 提供 before/after 数据。

存储: data/failure_cases/run_outcomes.jsonl (append-only, 线程安全)

一条记录:
  {
    "outcome_id": "oc_xxx",
    "run_id": "bt_xxx",
    "recorded_at": "2026-11-02T10:00:00",
    "mode": "balanced",
    "groups": ["科技成长型", ...],
    "window": {"start": "2024-07-01", "end": "2024-12-31"},
    "strategy_version": "abcdef12",
    "metrics": {"alpha": 0.03, "total_return": 0.12, ...}
  }
"""
import json
import os
import threading
import uuid
from datetime import datetime
from typing import Dict, List, Optional

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
_DEFAULT_OUTCOME_STORE = os.path.join(
    _PROJECT_ROOT, "data", "failure_cases", "run_outcomes.jsonl")

_WRITE_LOCK = threading.Lock()


class RunOutcomeStore:
    """回测运行结果基线 (组合级, 每次 run 一条)"""

    def __init__(self, store_path: Optional[str] = None):
        self.store_path = store_path or _DEFAULT_OUTCOME_STORE

    def record(self, *, run_id: str,
               window: Dict[str, str],
               metrics: dict,
               mode: Optional[str] = None,
               groups: Optional[List[str]] = None,
               strategy_version: Optional[str] = None,
               recorded_at: Optional[str] = None) -> dict:
        """写入一条运行结果

        Args:
            run_id: 本次运行 ID
            window: {"start": "YYYY-MM-DD", "end": "YYYY-MM-DD"}
            metrics: 组合级指标 (alpha/total_return/...)
            mode: 运行模式
            groups: 参与组
            strategy_version: 参数版本 (可选)
            recorded_at: 覆盖记录时间 (测试用)

        Returns:
            写入的记录
        """
        record = {
            "outcome_id": f"oc_{uuid.uuid4().hex[:12]}",
            "run_id": run_id,
            "recorded_at": recorded_at or datetime.now().isoformat(timespec="seconds"),
            "mode": mode,
            "groups": list(groups or []),
            "window": {
                "start": (window or {}).get("start"),
                "end": (window or {}).get("end"),
            },
            "strategy_version": strategy_version,
            "metrics": dict(metrics or {}),
        }
        parent = os.path.dirname(self.store_path)
        if parent:
            os.makedirs(parent, exist_ok=True)
        with _WRITE_LOCK:
            with open(self.store_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")
        return record

    def load_all(self) -> List[dict]:
        """加载全部运行结果"""
        if not os.path.exists(self.store_path):
            return []
        records = []
        with open(self.store_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
        return records

    @staticmethod
    def _same_window(wa: Optional[dict], wb: Optional[dict]) -> bool:
        """比较两个 window 字典是否一致"""
        wa, wb = wa or {}, wb or {}
        return (wa.get("start") == wb.get("start")
                and wa.get("end") == wb.get("end"))

    def find_nearest_after(self, timestamp: str,
                           window: Optional[dict] = None) -> Optional[dict]:
        """找出指定时间之后最早的一条运行结果

        Args:
            timestamp: ISO 时间字符串 (通常是建议的 applied_at)
            window: 若提供, 只匹配同窗口
        """
        if not timestamp:
            return None
        candidates = [
            r for r in self.load_all()
            if str(r.get("recorded_at", "")) >= timestamp
            and (window is None or self._same_window(r.get("window"), window))
        ]
        if not candidates:
            return None
        return min(candidates, key=lambda r: str(r.get("recorded_at", "")))

    def find_nearest_before(self, timestamp: str,
                            window: Optional[dict] = None,
                            mode: Optional[str] = None) -> Optional[dict]:
        """找出指定时间之前最晚的一条运行结果

        Args:
            timestamp: ISO 时间字符串
            window: 若提供, 只匹配同窗口
            mode: 若提供, 只匹配同 mode
        """
        if not timestamp:
            return None
        candidates = [
            r for r in self.load_all()
            if str(r.get("recorded_at", "")) <= timestamp
            and (window is None or self._same_window(r.get("window"), window))
            and (mode is None or r.get("mode") == mode)
        ]
        if not candidates:
            return None
        return max(candidates, key=lambda r: str(r.get("recorded_at", "")))
