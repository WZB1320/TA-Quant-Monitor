"""策略版本谱系 (P7) — 版本节点 + 派生关系 + 教训应用链 + 验证结果

定位:
  L1 记录了"哪次运行失败", L2 记录了"从失败中提炼了什么教训", P4 把教训回写为
  validated/invalidated。但缺少一条主线: 哪个策略版本由哪些教训派生而来、它跑出了
  什么结果、哪条教训在哪个版本上被验证。本模块以 append-only 事件流补齐版本记忆。

事件类型 (均带 version):
  version          — 首次见到某版本, 携带参数快照 (可用于 diff) 与父版本
  run              — 该版本的一次运行结果 (窗口/模式/指标)
  lesson_applied   — 某条 L2 教训被应用进该版本 (含结构化 rule_action)
  lesson_validated — 某条教训在该版本上被验证有效
  lesson_invalidated — 某条教训在该版本上被证伪

存储: data/failure_cases/version_lineage.jsonl (append-only, 线程安全)

版本定义:
  由 compute_strategy_version(params) 得到的 8 位 hash, 与 L1 / run_outcome 同源。
  组合级版本参数建议包含 mode / portfolio 配置 / 各组 strategy_mode (见
  web/backend/services/portfolio_backtest.py 的调用)。

父版本:
  - 显式传入 parent_version 时以传入值为准;
  - 未传入时按"本事件之前最近一次出现过的其他版本"推断, 并标记 parent_inferred。
    推断会跳过祖先节点, 避免环。

用法:
    from src.memory.version_lineage import VersionLineage

    lineage = VersionLineage()
    lineage.record_version(params=params, run_id=run_id, window=window,
                           mode="base", metrics={"alpha": 0.02})
    lineage.record_lesson_event(version="abcdef12", lesson_id="lsn_xxx",
                                verdict="validated", alpha_delta=0.03)

    node = lineage.get_version("abcdef12")
    chain = lineage.ancestors("abcdef12")     # 由近及远的父版本链
    delta = lineage.diff_versions("abcdef12", "12345678")
"""
import json
import os
import threading
from datetime import datetime
from typing import Dict, List, Optional

from .strategy_memory import compute_strategy_version

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
_DEFAULT_STORE = os.path.join(
    _PROJECT_ROOT, "data", "failure_cases", "version_lineage.jsonl")

_WRITE_LOCK = threading.Lock()

_EVENT_VERSION = "version"
_EVENT_RUN = "run"
_EVENT_LESSON_APPLIED = "lesson_applied"
_EVENT_VERDICTS = {"validated": "lesson_validated", "invalidated": "lesson_invalidated"}

_UNSET = object()


def _flatten_params(params: Optional[dict], prefix: str = "") -> Dict[str, object]:
    """把嵌套参数快照拍平成点分键, 便于逐项比较"""
    out: Dict[str, object] = {}
    for key, value in (params or {}).items():
        name = f"{prefix}.{key}" if prefix else str(key)
        if isinstance(value, dict):
            out.update(_flatten_params(value, name))
        else:
            out[name] = value
    return out


class VersionLineage:
    """策略版本谱系 — 事件流写入 + 节点折叠查询"""

    def __init__(self, store_path: Optional[str] = None):
        self.store_path = store_path or _DEFAULT_STORE

    # ── IO ──

    def load_events(self) -> List[dict]:
        """加载全部谱系事件"""
        if not os.path.exists(self.store_path):
            return []
        events = []
        with open(self.store_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    events.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
        return events

    def _append(self, event: dict) -> None:
        parent = os.path.dirname(self.store_path)
        if parent:
            os.makedirs(parent, exist_ok=True)
        line = json.dumps(event, ensure_ascii=False, default=str)
        with _WRITE_LOCK:
            with open(self.store_path, "a", encoding="utf-8") as f:
                f.write(line + "\n")

    @staticmethod
    def _clean(d: dict) -> dict:
        """去掉 None 值, 保持事件文件紧凑"""
        return {k: v for k, v in d.items() if v is not None}

    # ── 写入 ──

    def record_version(self, *,
                       params: Optional[dict] = None,
                       version: Optional[str] = None,
                       parent_version: Optional[str] = _UNSET,
                       run_id: Optional[str] = None,
                       window: Optional[dict] = None,
                       mode: Optional[str] = None,
                       groups: Optional[List[str]] = None,
                       metrics: Optional[dict] = None,
                       recorded_at: Optional[str] = None) -> dict:
        """记录一个版本 (可同时携带该版本的一次运行结果)

        Args:
            params: 参数快照; 与 version 至少给一个 (两者都给时以 version 为准)
            version: 显式版本 hash
            parent_version: 父版本; 不传则按时间推断, None 表示明确无父版本
            run_id / window / mode / groups / metrics: 可选, 本次运行信息

        Returns:
            写入的事件
        """
        if version is None:
            if not params:
                raise ValueError("params 与 version 至少提供一个")
            version = compute_strategy_version(params)

        timestamp = recorded_at or datetime.now().isoformat(timespec="seconds")

        inferred = False
        if parent_version is _UNSET:
            parent_version = self._infer_parent(version, timestamp)
            inferred = parent_version is not None

        event = self._clean({
            "event": _EVENT_VERSION,
            "version": version,
            "recorded_at": timestamp,
            "params": params,
            "parent_version": parent_version,
            "parent_inferred": True if inferred else None,
            "run_id": run_id,
            "window": window,
            "mode": mode,
            "groups": list(groups) if groups else None,
            "metrics": metrics,
        })
        self._append(event)
        return event

    def record_run(self, *,
                   version: str,
                   run_id: str,
                   window: Optional[dict] = None,
                   mode: Optional[str] = None,
                   groups: Optional[List[str]] = None,
                   metrics: Optional[dict] = None,
                   recorded_at: Optional[str] = None) -> dict:
        """记录某已有版本的一次运行"""
        if not version:
            raise ValueError("record_run 需要 version")
        if not run_id:
            raise ValueError("record_run 需要 run_id")
        event = self._clean({
            "event": _EVENT_RUN,
            "version": version,
            "recorded_at": recorded_at or datetime.now().isoformat(timespec="seconds"),
            "run_id": run_id,
            "window": window,
            "mode": mode,
            "groups": list(groups) if groups else None,
            "metrics": metrics,
        })
        self._append(event)
        return event

    def record_lesson_applied(self, *,
                              version: str,
                              lesson_id: str,
                              rule_action: Optional[dict] = None,
                              notes: Optional[str] = None,
                              recorded_at: Optional[str] = None) -> dict:
        """记录一条 L2 教训被应用进某个版本"""
        if not version or not lesson_id:
            raise ValueError("record_lesson_applied 需要 version 与 lesson_id")
        event = self._clean({
            "event": _EVENT_LESSON_APPLIED,
            "version": version,
            "recorded_at": recorded_at or datetime.now().isoformat(timespec="seconds"),
            "lesson_id": lesson_id,
            "rule_action": rule_action,
            "notes": notes,
        })
        self._append(event)
        return event

    def record_lesson_verdict(self, *,
                              version: str,
                              lesson_id: str,
                              verdict: str,
                              notes: Optional[str] = None,
                              run_id: Optional[str] = None,
                              alpha_delta: Optional[float] = None,
                              recorded_at: Optional[str] = None) -> dict:
        """记录一条教训在某版本上的验证结论 (validated / invalidated)"""
        if verdict not in _EVENT_VERDICTS:
            raise ValueError(
                f"verdict 必须是 {tuple(_EVENT_VERDICTS)} 之一, 收到 {verdict!r}")
        if not version or not lesson_id:
            raise ValueError("record_lesson_verdict 需要 version 与 lesson_id")
        event = self._clean({
            "event": _EVENT_VERDICTS[verdict],
            "version": version,
            "recorded_at": recorded_at or datetime.now().isoformat(timespec="seconds"),
            "lesson_id": lesson_id,
            "notes": notes,
            "run_id": run_id,
            "alpha_delta": alpha_delta,
            "verdict": verdict,
        })
        self._append(event)
        return event

    # ── 读取 / 折叠 ──

    def _new_node(self, version: str, first_seen: Optional[str]) -> dict:
        return {
            "version": version,
            "params": None,
            "parent_version": None,
            "parent_inferred": False,
            "first_seen": first_seen,
            "last_seen": first_seen,
            "runs": [],
            "lessons": {},
            "children": [],
        }

    def _fold(self, events: Optional[List[dict]] = None) -> Dict[str, dict]:
        """把事件流折叠为 version → node"""
        events = self.load_events() if events is None else events
        nodes: Dict[str, dict] = {}

        for ev in events:
            version = ev.get("version")
            if not version:
                continue
            ts = ev.get("recorded_at")
            node = nodes.setdefault(version, self._new_node(version, ts))
            if ts:
                if not node["first_seen"] or ts < node["first_seen"]:
                    node["first_seen"] = ts
                if not node["last_seen"] or ts > node["last_seen"]:
                    node["last_seen"] = ts

            kind = ev.get("event")

            if kind == _EVENT_VERSION:
                if ev.get("params") is not None:
                    node["params"] = ev["params"]
                if node["parent_version"] is None and ev.get("parent_version"):
                    node["parent_version"] = ev["parent_version"]
                    node["parent_inferred"] = bool(ev.get("parent_inferred"))
                self._add_run(node, ev)

            elif kind == _EVENT_RUN:
                self._add_run(node, ev)

            elif kind == _EVENT_LESSON_APPLIED:
                lesson_id = ev.get("lesson_id")
                if lesson_id:
                    entry = node["lessons"].setdefault(
                        lesson_id, {"applied_at": None, "rule_action": None, "verdict": None})
                    entry["applied_at"] = ev.get("recorded_at")
                    if ev.get("rule_action") is not None:
                        entry["rule_action"] = ev["rule_action"]

            elif kind in _EVENT_VERDICTS.values():
                lesson_id = ev.get("lesson_id")
                if lesson_id:
                    entry = node["lessons"].setdefault(
                        lesson_id, {"applied_at": None, "rule_action": None, "verdict": None})
                    entry["verdict"] = ev.get("verdict")
                    entry["verdict_at"] = ev.get("recorded_at")
                    entry["notes"] = ev.get("notes")
                    entry["run_id"] = ev.get("run_id")
                    entry["alpha_delta"] = ev.get("alpha_delta")

        # children: 由 parent_version 反推
        for node in nodes.values():
            parent = node["parent_version"]
            if parent and parent in nodes and parent != node["version"]:
                nodes[parent]["children"].append(node["version"])

        for node in nodes.values():
            node["children"].sort()
            node["runs"].sort(key=lambda r: str(r.get("recorded_at") or ""))
        return nodes

    @staticmethod
    def _add_run(node: dict, ev: dict) -> None:
        run_id = ev.get("run_id")
        if not run_id:
            return
        if any(r.get("run_id") == run_id for r in node["runs"]):
            return
        node["runs"].append({
            "run_id": run_id,
            "recorded_at": ev.get("recorded_at"),
            "window": ev.get("window"),
            "mode": ev.get("mode"),
            "groups": ev.get("groups"),
            "metrics": ev.get("metrics"),
        })

    def get_version(self, version: str) -> Optional[dict]:
        """返回单版本节点 (含 runs / lessons / children)"""
        return self._fold().get(version)

    def list_versions(self) -> List[dict]:
        """按首次出现时间升序返回全部版本节点"""
        nodes = self._fold().values()
        return sorted(nodes, key=lambda n: str(n.get("first_seen") or ""))

    def version_for_run(self, run_id: str) -> Optional[str]:
        """通过 run_id 反查版本"""
        if not run_id:
            return None
        for node in self._fold().values():
            if any(r.get("run_id") == run_id for r in node["runs"]):
                return node["version"]
        return None

    def children(self, version: str) -> List[str]:
        """直接子版本 (由该版本派生的版本)"""
        node = self.get_version(version)
        return list(node["children"]) if node else []

    def ancestors(self, version: str, max_depth: int = 20) -> List[dict]:
        """父版本链 (由近及远), 遇到未知父版本或环时停止"""
        nodes = self._fold()
        chain: List[dict] = []
        seen = {version}
        current = nodes.get(version)
        depth = 0
        while current and current.get("parent_version") and depth < max_depth:
            parent = current["parent_version"]
            if parent in seen:
                break
            seen.add(parent)
            parent_node = nodes.get(parent)
            if parent_node is None:
                break
            chain.append(parent_node)
            current = parent_node
            depth += 1
        return chain

    def diff_versions(self, before: str, after: str) -> dict:
        """比较两个版本的参数差异

        Returns:
            {"added": {...}, "removed": {...},
             "changed": {key: {"before": ..., "after": ...}}}

        Raises:
            ValueError: 任一版本不存在或缺少 params
        """
        nodes = self._fold()
        a, b = nodes.get(before), nodes.get(after)
        if a is None or b is None:
            raise ValueError(f"版本不存在: {before!r} / {after!r}")
        if a.get("params") is None or b.get("params") is None:
            raise ValueError(f"版本缺少参数快照, 无法比较: {before!r} / {after!r}")

        flat_a = _flatten_params(a["params"])
        flat_b = _flatten_params(b["params"])

        added = {k: flat_b[k] for k in flat_b.keys() - flat_a.keys()}
        removed = {k: flat_a[k] for k in flat_a.keys() - flat_b.keys()}
        changed = {
            k: {"before": flat_a[k], "after": flat_b[k]}
            for k in flat_a.keys() & flat_b.keys()
            if flat_a[k] != flat_b[k]
        }
        return {"added": added, "removed": removed, "changed": changed}

    # ── 父版本推断 ──

    def _parent_map(self, events: List[dict]) -> Dict[str, str]:
        parents: Dict[str, str] = {}
        for ev in events:
            if ev.get("event") != _EVENT_VERSION:
                continue
            version, parent = ev.get("version"), ev.get("parent_version")
            if version and parent and version not in parents:
                parents[version] = parent
        return parents

    @staticmethod
    def _would_cycle(child: str, parent: str,
                     parents: Dict[str, str]) -> bool:
        seen, current = set(), parent
        while current and current not in seen:
            if current == child:
                return True
            seen.add(current)
            current = parents.get(current)
        return False

    def _infer_parent(self, version: str, timestamp: str) -> Optional[str]:
        """取本事件之前最近出现过的其他版本作为父版本"""
        events = self.load_events()
        parents = self._parent_map(events)

        candidates = [
            ev for ev in events
            if ev.get("version") and ev["version"] != version
            and str(ev.get("recorded_at") or "") <= timestamp
        ]
        if not candidates:
            return None

        latest = max(candidates, key=lambda ev: str(ev.get("recorded_at") or ""))
        parent = latest["version"]
        if self._would_cycle(version, parent, parents):
            return None
        return parent
