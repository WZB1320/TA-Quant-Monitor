"""正样本轨迹 (P5) — 记录"什么参数组合有效"

定位:
  L1 只记录命中失败规则的运行, 缺少成功侧的基线, 进化器只能看到"不要做什么",
  看不到"什么做对了"。本模块在每组回测完成后按正样本规则打标, 仅当 alpha >=
  positive_alpha 且 trade_count > 0 时落一条正样本轨迹, 与 L1 失效案例共享同一份
  参数快照 / strategy_version, 便于横向对照。

存储:
  data/failure_cases/positive_cases.jsonl — append-only, 一行一条完整 JSON。

正样本标签 (positive_flags):
  strong_alpha       — alpha >= strong_alpha 阈值
  positive_alpha     — alpha >= positive_alpha 阈值 (未达 strong)
  drawdown_shallow   — 最大回撤 <= drawdown_shallow 阈值
  win_rate_high      — 胜率 >= win_rate_high 阈值
  stop_loss_low      — 止损率 <= stop_loss_low 阈值
  regime_favorable   — 主导体制下 alpha >= regime_favorable_alpha 阈值

用法:
    from src.memory.positive_case import PositiveCaseRecorder

    recorder = PositiveCaseRecorder()
    record = recorder.record_backtest(
        source_ref={"run_id": run_id, "group": "科技成长型", "mode": "base"},
        group_name="科技成长型",
        stock_codes=[...],
        window={"start": "2025-01-01", "end": "2025-12-31"},
        strategy_mode="trend_following",
        params=params_snapshot,
        metrics=metrics,
        trades=trades,
        benchmark_return=metrics.benchmark_return,
        regime_slices=engine.regime_slices,
    )
    # 未达门槛 (alpha 不够正或无交易) 时返回 None (不落盘)

    cases = recorder.list_positive_cases(group="科技成长型", min_alpha=0.02)
    best = recorder.best_configs(group="科技成长型", limit=5)
"""
import json
import os
import threading
import uuid
from datetime import datetime
from typing import Dict, List, Optional, Sequence

from .failure_case import (
    compute_exit_distribution,
    metrics_to_dict,
)
from .regime_slices import dominant_regime
from .strategy_memory import compute_strategy_version

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
_DEFAULT_STORE = os.path.join(
    _PROJECT_ROOT, "data", "failure_cases", "positive_cases.jsonl")

_WRITE_LOCK = threading.Lock()

POSITIVE_FLAGS = (
    "strong_alpha",
    "positive_alpha",
    "drawdown_shallow",
    "win_rate_high",
    "stop_loss_low",
    "regime_favorable",
)

_DEFAULT_THRESHOLDS = {
    "strong_alpha": 0.05,
    "positive_alpha": 0.01,
    "drawdown_shallow": 0.08,
    "win_rate_high": 0.50,
    "stop_loss_low": 0.20,
    "regime_favorable_alpha": 0.02,
}


def derive_positive_flags(metrics: dict,
                          regime_slices: Optional[dict] = None,
                          thresholds: Optional[dict] = None) -> List[str]:
    """按正样本阈值给一次运行打标

    Args:
        metrics: 归一化后的指标 dict (alpha / max_drawdown / win_rate / stop_loss_rate)
        regime_slices: 体制分片 (可选, 用于判定 regime_favorable)
        thresholds: 覆盖默认阈值

    Returns:
        positive_flags 列表 (按 POSITIVE_FLAGS 顺序), 可能为空
    """
    t = {**_DEFAULT_THRESHOLDS, **(thresholds or {})}
    flags: List[str] = []

    alpha = _as_float(metrics.get("alpha"))
    if alpha is not None:
        if alpha >= t["strong_alpha"]:
            flags.append("strong_alpha")
        elif alpha >= t["positive_alpha"]:
            flags.append("positive_alpha")

    dd = _as_float(metrics.get("max_drawdown"))
    if dd is not None and abs(dd) <= t["drawdown_shallow"]:
        flags.append("drawdown_shallow")

    win_rate = _as_float(metrics.get("win_rate"))
    if win_rate is not None and win_rate >= t["win_rate_high"]:
        flags.append("win_rate_high")

    stop_rate = _as_float(metrics.get("stop_loss_rate"))
    if stop_rate is not None and stop_rate <= t["stop_loss_low"]:
        flags.append("stop_loss_low")

    if regime_slices:
        try:
            dom = dominant_regime(regime_slices)
        except Exception:
            dom = None
        if dom and _as_float(dom.get("alpha")) is not None \
                and dom["alpha"] >= t["regime_favorable_alpha"]:
            flags.append("regime_favorable")

    return [f for f in POSITIVE_FLAGS if f in flags]


def _as_float(value) -> Optional[float]:
    if value is None or isinstance(value, bool):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _alpha_of(case: dict) -> float:
    value = _as_float((case.get("metrics") or {}).get("alpha"))
    return value if value is not None else float("-inf")


class PositiveCaseRecorder:
    """正样本轨迹写入器 (append-only JSONL)"""

    def __init__(self, store_path: Optional[str] = None,
                 thresholds: Optional[dict] = None):
        self.store_path = store_path or _DEFAULT_STORE
        self.thresholds = {**_DEFAULT_THRESHOLDS, **(thresholds or {})}

    def record_backtest(self, *,
                        source_ref: dict,
                        group_name: str,
                        stock_codes: Sequence[str],
                        window: dict,
                        strategy_mode: str,
                        params: dict,
                        metrics,
                        trades: Sequence = (),
                        benchmark_return: Optional[float] = None,
                        regime_dist: Optional[dict] = None,
                        regime_slices: Optional[dict] = None,
                        granularity: str = "group") -> Optional[dict]:
        """记录一次正样本回测

        仅当 alpha >= positive_alpha 且 trade_count > 0 时才落盘:
        没有交易的"零回撤"不算成功, 负 alpha 的运行也不构成有效基线。
        未通过门槛时返回 None。
        """
        exits = compute_exit_distribution(trades)
        metric_dict = metrics_to_dict(metrics)
        metric_dict["exit_reason_dist"] = exits["counts"]
        metric_dict["stop_loss_rate"] = exits["stop_loss_rate"]
        metric_dict["ma_breakdown_rate"] = exits["ma_breakdown_rate"]
        if benchmark_return is not None:
            metric_dict["benchmark_return"] = benchmark_return
        if regime_slices:
            metric_dict["regime_slices"] = regime_slices

        alpha = _as_float(metric_dict.get("alpha"))
        trade_count = _as_float(metric_dict.get("trade_count")) or 0
        if alpha is None or alpha < self.thresholds["positive_alpha"] or trade_count <= 0:
            return None

        flags = derive_positive_flags(
            metric_dict, regime_slices=regime_slices, thresholds=self.thresholds)
        if not flags:
            return None

        dom = None
        if regime_slices:
            try:
                dom = dominant_regime(regime_slices)
            except Exception:
                dom = None

        params = params or {}
        record = {
            "record_type": "positive_case",
            "case_id": f"pc_{uuid.uuid4().hex[:12]}",
            "recorded_at": datetime.now().isoformat(timespec="seconds"),
            "source": "backtest",
            "source_ref": source_ref,
            "window": window,
            "granularity": granularity,
            "group_name": group_name,
            "stock_codes": list(stock_codes or []),
            "strategy_version": compute_strategy_version(params),
            "strategy_mode": strategy_mode,
            "params": params,
            "regime_dist": regime_dist,
            "regime_slices": regime_slices,
            "dominant_regime": (dom or {}).get("regime"),
            "benchmark_return": benchmark_return,
            "metrics": metric_dict,
            "positive_flags": flags,
            "positive_score": len(flags),
        }
        self._append(record)
        return record

    def _append(self, record: dict) -> None:
        parent = os.path.dirname(self.store_path)
        if parent:
            os.makedirs(parent, exist_ok=True)
        line = json.dumps(record, ensure_ascii=False, default=str)
        with _WRITE_LOCK:
            with open(self.store_path, "a", encoding="utf-8") as f:
                f.write(line + "\n")

    # ── 查询 (进化器入口) ──

    def load_all(self) -> List[dict]:
        """加载全部正样本记录"""
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

    def list_positive_cases(self, *,
                            group: Optional[str] = None,
                            strategy_mode: Optional[str] = None,
                            regime: Optional[str] = None,
                            min_alpha: Optional[float] = None) -> List[dict]:
        """按条件检索正样本, 按 alpha 降序"""
        cases = self.load_all()
        result = []
        for case in cases:
            if group and case.get("group_name") != group:
                continue
            if strategy_mode and case.get("strategy_mode") != strategy_mode:
                continue
            if regime and case.get("dominant_regime") != regime:
                continue
            if min_alpha is not None and _alpha_of(case) < min_alpha:
                continue
            result.append(case)
        result.sort(key=_alpha_of, reverse=True)
        return result

    def best_configs(self, *,
                     group: Optional[str] = None,
                     strategy_mode: Optional[str] = None,
                     regime: Optional[str] = None,
                     limit: int = 5) -> List[dict]:
        """按 strategy_version 去重, 返回表现最好的若干参数组合

        同一参数版本可能被多次运行, 这里只保留 alpha 最高的一条;
        结果按 alpha 降序, 最多 limit 条。
        """
        best: Dict[str, dict] = {}
        for case in self.list_positive_cases(
                group=group, strategy_mode=strategy_mode, regime=regime):
            version = case.get("strategy_version")
            current = best.get(version)
            if current is None or _alpha_of(case) > _alpha_of(current):
                best[version] = case

        ranked = sorted(best.values(), key=_alpha_of, reverse=True)[:max(limit, 0)]
        return [{
            "strategy_version": c.get("strategy_version"),
            "params": c.get("params"),
            "metrics": c.get("metrics"),
            "positive_flags": c.get("positive_flags"),
            "positive_score": c.get("positive_score"),
            "window": c.get("window"),
            "recorded_at": c.get("recorded_at"),
            "group_name": c.get("group_name"),
            "strategy_mode": c.get("strategy_mode"),
            "dominant_regime": c.get("dominant_regime"),
        } for c in ranked]
