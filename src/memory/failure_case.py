"""失效案例库 L1 — 原始失败记录

定位:
  为"策略自进化 + 版本记忆"提供故障诊断数据。L1 只记录客观事实(只写不改),
  L2 教训条目由人工复盘后从多条 L1 记录提炼(见 P2, 本模块暂不涉及)。

存储:
  data/failure_cases/l1_failures.jsonl — append-only, 一行一条完整 JSON。
  与 strategy_memory.jsonl 同一存储范式: L1 是历史事实, 不需要修改。

记录来源 (P1):
  - backtest   : 回测/组合回测每组跑完后的结果
  - param_scan : 参数扫描(每次扫描一条, 内嵌全部组合明细; 不论结果好坏都记录)

P1 判定能力:
  failure_flags      自动打标 (alpha_negative / stop_loss_high / drawdown_deep ...)
  failure_level      按 config/failure_rules.json 的优先级规则判定
                      R1 stock_pool / R3 structure / R4 param / R5 overfit / R6 unknown
                      (R2 regime_mismatch 依赖 regime_slices, P1 不启用)

用法:
    from src.memory.failure_case import FailureCaseRecorder, compute_stock_pool_snapshot

    recorder = FailureCaseRecorder()
    recorder.record_backtest(
        source_ref={"run_id": engine.memory.run_id, "group": "科技成长型"},
        group_name="科技成长型",
        stock_codes=[...],
        window={"start": "2024-07-01", "end": "2025-06-30"},
        strategy_mode="trend_following",
        params={...},
        metrics=metrics,
        trades=engine.position_mgr.closed_trades,
        regime_dist=engine.regime_dist,
        stock_pool_snapshot=compute_stock_pool_snapshot(...),
    )
"""
import json
import os
import threading
import uuid
from datetime import datetime
from typing import Dict, List, Optional, Sequence

import pandas as pd

from .strategy_memory import compute_strategy_version, generate_run_id

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
_DEFAULT_STORE = os.path.join(_ROOT, "data", "failure_cases", "l1_failures.jsonl")
_DEFAULT_RULES = os.path.join(_ROOT, "config", "failure_rules.json")

_WRITE_LOCK = threading.Lock()

# failure_flags 全部可能取值 (便于检索与校验)
FAILURE_FLAGS = (
    "alpha_negative",
    "train_test_dual_negative",
    "stop_loss_high",
    "drawdown_deep",
    "overfit_gap",
    "param_scan_flat",
    "param_scan_clear",
    "regime_dominated",
    "regime_alpha_negative",
    "impl_gap_large",
)

FAILURE_LEVELS = ("stock_pool", "regime_mismatch", "structure", "param", "overfit", "unknown")

_DEFAULT_THRESHOLDS = {
    "alpha_negative": 0.0,
    "stop_loss_high": 0.40,
    "drawdown_deep": 0.12,
    "overfit_gap": 0.10,
    "impl_gap_large": 0.05,
    "regime_dominated": 0.70,
    "regime_mismatch_share": 0.50,
    "regime_mismatch_alpha": -0.02,
    "param_scan_flat_corr": 0.30,
    "param_scan_clear_corr": 0.70,
    "stock_pool_underperform_ratio": 0.60,
    "structure_stop_loss_ratio": 0.50,
    "structure_ma_breakdown_ratio": 0.60,
    "param_best_gap": 0.03,
    "overfit_train_alpha": 0.05,
    "overfit_train_win_rate": 0.60,
    "overfit_test_win_rate": 0.45,
}

_DEFAULT_ENTRY_FLAGS = (
    "alpha_negative",
    "train_test_dual_negative",
    "stop_loss_high",
    "drawdown_deep",
    "overfit_gap",
    "impl_gap_large",
)


# ─────────────────────────────────────────────────────────────
# 配置
# ─────────────────────────────────────────────────────────────

def load_rule_config(path: Optional[str] = None) -> dict:
    """读取规则配置; 文件缺失/损坏时回落到内置默认值(保证记录不中断)"""
    rules_path = path or _DEFAULT_RULES
    config = {}
    try:
        with open(rules_path, "r", encoding="utf-8") as f:
            config = json.load(f)
    except Exception:
        config = {}

    thresholds = {**_DEFAULT_THRESHOLDS, **(config.get("thresholds") or {})}
    entry_flags = tuple(config.get("entry_flags") or _DEFAULT_ENTRY_FLAGS)
    rules = config.get("rules") or [
        {"id": "R1", "failure_level": "stock_pool", "priority": 1, "enabled": True},
        {"id": "R2", "failure_level": "regime_mismatch", "priority": 2, "enabled": False},
        {"id": "R3", "failure_level": "structure", "priority": 3, "enabled": True},
        {"id": "R4", "failure_level": "param", "priority": 4, "enabled": True},
        {"id": "R5", "failure_level": "overfit", "priority": 5, "enabled": True},
        {"id": "R6", "failure_level": "unknown", "priority": 6, "enabled": True},
    ]
    return {
        "version": config.get("version", 1),
        "thresholds": thresholds,
        "entry_flags": entry_flags,
        "rules": sorted(rules, key=lambda r: r.get("priority", 999)),
    }


# ─────────────────────────────────────────────────────────────
# 指标 / 退出归因
# ─────────────────────────────────────────────────────────────

def categorize_exit_signal(signal: Optional[str]) -> str:
    """把平仓信号文本归入稳定类别

    类别命名与 L1 字段约定一致, 供 exit_reason_dist / 规则判定使用。
    """
    s = str(signal or "")
    if "回撤保护" in s:
        return "drawdown_reduce"
    if "均值回归止盈" in s:
        return "mean_reversion_take_profit"
    if "安全网" in s:
        return "safety_net_hard_stop"
    if "ATR硬止损" in s or "硬止损" in s or "固定止损" in s:
        return "hard_stop"
    if "ATR移动止盈" in s or "移动止盈" in s:
        return "trailing_take_profit"
    if "MA60" in s or "破位" in s:
        return "ma_breakdown"
    if "score=" in s:
        return "signal_exit"
    return "other"


def compute_exit_distribution(trades: Sequence) -> dict:
    """统计退出原因分布 + 止损率 / 破位率

    止损类 = hard_stop + safety_net_hard_stop
    """
    counts: Dict[str, int] = {}
    for t in trades or []:
        category = categorize_exit_signal(getattr(t, "exit_signal", None))
        counts[category] = counts.get(category, 0) + 1

    total = sum(counts.values())
    stop_loss = counts.get("hard_stop", 0) + counts.get("safety_net_hard_stop", 0)
    return {
        "counts": counts,
        "total": total,
        "stop_loss_rate": round(stop_loss / total, 4) if total else 0.0,
        "ma_breakdown_rate": round(counts.get("ma_breakdown", 0) / total, 4) if total else 0.0,
        "signal_exit_rate": round(counts.get("signal_exit", 0) / total, 4) if total else 0.0,
    }


_METRIC_FIELDS = {
    "alpha": "alpha",
    "total_return": "total_return",
    "annual_return": "annual_return",
    "sharpe": "sharpe_ratio",
    "win_rate": "win_rate",
    "profit_loss_ratio": "profit_factor",
    "trade_count": "trade_count",
    "win_count": "win_count",
    "max_drawdown": "max_drawdown",
    "recovery_days": "max_drawdown_days",
    "avg_holding_days": "avg_holding_days",
    "benchmark_return": "benchmark_return",
    "volatility": "volatility",
}


def metrics_to_dict(metrics) -> dict:
    """把 BacktestMetrics 或指标 dict 归一化成稳定字段名"""
    out = {}
    for key, attr in _METRIC_FIELDS.items():
        value = metrics.get(key) if isinstance(metrics, dict) else getattr(metrics, attr, None)
        if value is None and isinstance(metrics, dict):
            value = metrics.get(attr)
        if value is None:
            out[key] = None
            continue
        # profit_factor 可能为 inf
        if isinstance(value, float) and value == float("inf"):
            out[key] = None
        else:
            out[key] = round(float(value), 6) if isinstance(value, (int, float)) else value
    return out


def normalize_exit_reasons(raw: Optional[dict]) -> dict:
    """把扫描脚本里的中文退出原因归一为类别分布"""
    mapping = {
        "止损": "hard_stop",
        "trailing止盈": "trailing_take_profit",
        "均值回归止盈": "mean_reversion_take_profit",
        "信号退出": "signal_exit",
        "回撤保护": "drawdown_reduce",
        "其他": "other",
    }
    counts = {}
    for key, value in (raw or {}).items():
        category = mapping.get(key, key)
        counts[category] = counts.get(category, 0) + int(value)
    total = sum(counts.values())
    stop_loss = counts.get("hard_stop", 0) + counts.get("safety_net_hard_stop", 0)
    return {
        "counts": counts,
        "total": total,
        "stop_loss_rate": round(stop_loss / total, 4) if total else 0.0,
        "ma_breakdown_rate": round(counts.get("ma_breakdown", 0) / total, 4) if total else 0.0,
        "signal_exit_rate": round(counts.get("signal_exit", 0) / total, 4) if total else 0.0,
    }


# ─────────────────────────────────────────────────────────────
# 股票池 / 基准
# ─────────────────────────────────────────────────────────────

def compute_window_return(df: Optional[pd.DataFrame], start, end) -> Optional[float]:
    """计算单只股票/指数在 [start, end] 的区间收益 (qfq close)"""
    if df is None or len(df) == 0 or "date" not in df.columns or "close" not in df.columns:
        return None
    try:
        dates = pd.to_datetime(df["date"])
        mask = (dates >= pd.Timestamp(start)) & (dates <= pd.Timestamp(end))
        closes = df.loc[mask, "close"].dropna()
        if len(closes) < 2:
            return None
        first, last = float(closes.iloc[0]), float(closes.iloc[-1])
        if first <= 0:
            return None
        return round(last / first - 1.0, 6)
    except Exception:
        return None


def compute_stock_pool_snapshot(data_map: Dict[str, pd.DataFrame],
                                benchmark_df: Optional[pd.DataFrame],
                                start, end,
                                stock_codes: Optional[Sequence[str]] = None) -> dict:
    """统计股票池窗口内跑输基准的比例 (R1 选股域判定的关键证据)"""
    codes = list(stock_codes) if stock_codes is not None else list((data_map or {}).keys())
    benchmark_return = compute_window_return(benchmark_df, start, end)
    stock_returns = {}
    for code in codes:
        r = compute_window_return((data_map or {}).get(code), start, end)
        if r is not None:
            stock_returns[code] = r

    underperform = 0
    if benchmark_return is not None:
        underperform = sum(1 for r in stock_returns.values() if r < benchmark_return)

    total = len(stock_returns)
    return {
        "benchmark_return": benchmark_return,
        "stock_returns": stock_returns,
        "total": total,
        "underperform_count": underperform,
        "underperform_ratio": round(underperform / total, 4) if total else None,
    }


# ─────────────────────────────────────────────────────────────
# 参数扫描方向性
# ─────────────────────────────────────────────────────────────

def _pearson(xs: Sequence[float], ys: Sequence[float]) -> Optional[float]:
    n = len(xs)
    if n < 3:
        return None
    mx = sum(xs) / n
    my = sum(ys) / n
    num = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    den = (sum((x - mx) ** 2 for x in xs) * sum((y - my) ** 2 for y in ys)) ** 0.5
    if den == 0:
        return None
    return num / den


def _combo_metric(combo: dict, window: str, key: str) -> Optional[float]:
    metrics = (combo or {}).get(window) or {}
    value = metrics.get(key)
    if value is None:
        return None
    try:
        value = float(value)
    except (TypeError, ValueError):
        return None
    if value == float("inf"):
        return None
    return value


def _combo_combined_alpha(combo: dict) -> Optional[float]:
    combined = (combo or {}).get("combined_alpha")
    if combined is not None:
        try:
            return float(combined)
        except (TypeError, ValueError):
            pass
    train_alpha = _combo_metric(combo, "train", "alpha")
    test_alpha = _combo_metric(combo, "test", "alpha")
    if train_alpha is None or test_alpha is None:
        return None
    return train_alpha + test_alpha


def compute_param_scan_direction(combos: Sequence[dict], grid: dict,
                                 thresholds: Optional[dict] = None) -> dict:
    """计算参数扫描的方向性强度

    对每个参数维度, 计算"参数值 vs 综合Alpha"的皮尔逊相关;
    取所有维度中绝对值最大的相关系数作为方向性强度。
      |r| >= clear_corr → 方向明确 (调参知道往哪走)
      |r| <  flat_corr  → 方向不明确 (调参无效, 可能是结构/选股问题)
    """
    th = {**_DEFAULT_THRESHOLDS, **(thresholds or {})}
    per_key = {}
    for key in (grid or {}).keys():
        xs, ys = [], []
        for combo in combos or []:
            param_value = (combo.get("params") or {}).get(key)
            alpha = _combo_combined_alpha(combo)
            if param_value is None or alpha is None:
                continue
            try:
                xs.append(float(param_value))
            except (TypeError, ValueError):
                continue
            ys.append(alpha)
        if len(set(xs)) < 2:
            continue
        corr = _pearson(xs, ys)
        if corr is not None:
            per_key[key] = round(corr, 4)

    max_abs = max((abs(v) for v in per_key.values()), default=0.0)
    return {
        "per_key": per_key,
        "max_abs_corr": round(max_abs, 4),
        "direction_clear": max_abs >= th["param_scan_clear_corr"],
        "direction_flat": max_abs < th["param_scan_flat_corr"],
    }


# ─────────────────────────────────────────────────────────────
# failure_flags
# ─────────────────────────────────────────────────────────────

def derive_failure_flags(metrics: dict,
                         *,
                         exits: Optional[dict] = None,
                         train_metrics: Optional[dict] = None,
                         test_metrics: Optional[dict] = None,
                         impl_gap: Optional[float] = None,
                         regime_dist: Optional[dict] = None,
                         regime_slices: Optional[dict] = None,
                         scan_direction: Optional[dict] = None,
                         thresholds: Optional[dict] = None) -> List[str]:
    """从客观指标推导 failure_flags (多标签, 是 failure_level 的原料)"""
    th = {**_DEFAULT_THRESHOLDS, **(thresholds or {})}
    flags = []

    alpha = metrics.get("alpha") if isinstance(metrics, dict) else None
    if alpha is not None and alpha < th["alpha_negative"]:
        flags.append("alpha_negative")

    train_alpha = (train_metrics or {}).get("alpha")
    test_alpha = (test_metrics or {}).get("alpha")
    if train_alpha is not None and test_alpha is not None:
        if train_alpha < th["alpha_negative"] and test_alpha < th["alpha_negative"]:
            flags.append("train_test_dual_negative")
        if train_alpha - test_alpha > th["overfit_gap"]:
            flags.append("overfit_gap")

    stop_loss_rate = (exits or {}).get("stop_loss_rate", 0.0)
    if stop_loss_rate > th["stop_loss_high"]:
        flags.append("stop_loss_high")

    drawdown = metrics.get("max_drawdown") if isinstance(metrics, dict) else None
    if drawdown is not None and abs(drawdown) >= th["drawdown_deep"]:
        flags.append("drawdown_deep")

    if impl_gap is not None and impl_gap < -th["impl_gap_large"]:
        flags.append("impl_gap_large")

    if regime_dist and sum(regime_dist.values()) > 0:
        dominant = max(regime_dist.values()) / sum(regime_dist.values())
        if dominant >= th["regime_dominated"]:
            flags.append("regime_dominated")

    # 主导体制下 alpha 显著为负 → 体制失配信号 (P3)
    if regime_slices:
        try:
            from .regime_slices import dominant_regime
            dom = dominant_regime(regime_slices, th)
            if dom and dom.get("is_mismatch"):
                flags.append("regime_alpha_negative")
        except Exception:
            pass

    if scan_direction:
        if scan_direction.get("direction_flat"):
            flags.append("param_scan_flat")
        if scan_direction.get("direction_clear"):
            flags.append("param_scan_clear")

    # 去重且保持稳定顺序
    return [f for f in FAILURE_FLAGS if f in flags]


# ─────────────────────────────────────────────────────────────
# failure_level 判定
# ─────────────────────────────────────────────────────────────

def classify_failure_level(flags: List[str],
                          *,
                          metrics: dict,
                          exits: Optional[dict] = None,
                          train_metrics: Optional[dict] = None,
                          test_metrics: Optional[dict] = None,
                          impl_gap: Optional[float] = None,
                          scan: Optional[dict] = None,
                          stock_pool_underperform_ratio: Optional[float] = None,
                          regime_slices: Optional[dict] = None,
                          config: Optional[dict] = None) -> dict:
    """按优先级判定 failure_level, 返回 {level, rule_id, reason}

    优先级: stock_pool → regime_mismatch → structure → param → overfit → unknown
    """
    cfg = config or load_rule_config()
    th = cfg["thresholds"]
    enabled = {r["failure_level"]: r.get("enabled", True) for r in cfg["rules"]}
    scan = scan or {}

    flags = set(flags or [])

    # ── R1 选股域 ──
    if enabled.get("stock_pool", True):
        ratio = stock_pool_underperform_ratio
        if (("train_test_dual_negative" in flags)
                and scan.get("all_combo_negative")
                and scan.get("direction_flat")
                and ratio is not None
                and ratio >= th["stock_pool_underperform_ratio"]):
            return {
                "level": "stock_pool",
                "rule_id": "R1",
                "reason": (
                    f"train/test 双负且 {scan.get('combo_count', 0)} 个扫描组合全负, "
                    f"方向性不明确(|r|={scan.get('max_abs_corr', 0):.2f}), "
                    f"{ratio:.0%} 个股跑输基准 → 选股域问题"
                ),
            }

    # ── R2 体制失配 ──
    if enabled.get("regime_mismatch", False):
        slices = regime_slices or (metrics.get("regime_slices") if isinstance(metrics, dict) else None)
        if slices:
            from .regime_slices import dominant_regime
            dom = dominant_regime(slices, th)
            if dom and dom.get("is_mismatch"):
                return {
                    "level": "regime_mismatch",
                    "rule_id": "R2",
                    "reason": (
                        f"主导体制 {dom['regime']} 占比 {dom['share']:.0%}, "
                        f"该体制下 Alpha {dom['alpha']:+.2%} → 策略与市场体制不匹配"
                    ),
                }

    # ── R3 策略结构 ──
    if enabled.get("structure", True):
        best_test_alpha = scan.get("best_test_alpha")
        stop_loss_rate = (exits or {}).get("stop_loss_rate", 0.0)
        ma_rate = (exits or {}).get("ma_breakdown_rate", 0.0)
        if (("train_test_dual_negative" not in flags)
                and scan.get("direction_clear")
                and best_test_alpha is not None
                and best_test_alpha < th["alpha_negative"]
                and (stop_loss_rate >= th["structure_stop_loss_ratio"]
                     or ma_rate >= th["structure_ma_breakdown_ratio"])):
            return {
                "level": "structure",
                "rule_id": "R3",
                "reason": (
                    f"非双负但扫描方向明确(|r|={scan.get('max_abs_corr', 0):.2f}), "
                    f"最优组合测试窗 Alpha 仍为 {best_test_alpha:+.2%}, "
                    f"止损率 {stop_loss_rate:.0%}/破位率 {ma_rate:.0%} → 策略原型不适配"
                ),
            }

    # ── R4 参数 ──
    if enabled.get("param", True):
        best_test_alpha = scan.get("best_test_alpha")
        current_test_alpha = scan.get("current_test_alpha")
        if (scan.get("direction_clear")
                and best_test_alpha is not None
                and best_test_alpha >= th["alpha_negative"]
                and current_test_alpha is not None
                and current_test_alpha < best_test_alpha - th["param_best_gap"]):
            return {
                "level": "param",
                "rule_id": "R4",
                "reason": (
                    f"扫描方向明确且有正解(最优测试窗 Alpha {best_test_alpha:+.2%}), "
                    f"当前参数仅 {current_test_alpha:+.2%}, "
                    f"差 {best_test_alpha - current_test_alpha:.2%} → 调参可解决"
                ),
            }

    # ── R5 过拟合 ──
    if enabled.get("overfit", True):
        train_alpha = (train_metrics or {}).get("alpha")
        test_alpha = (test_metrics or {}).get("alpha")
        train_wr = (train_metrics or {}).get("win_rate")
        test_wr = (test_metrics or {}).get("win_rate")
        if (("overfit_gap" in flags)
                or (train_alpha is not None and test_alpha is not None
                    and train_alpha > th["overfit_train_alpha"] and test_alpha < th["alpha_negative"])
                or (train_wr is not None and test_wr is not None
                    and train_wr > th["overfit_train_win_rate"]
                    and test_wr < th["overfit_test_win_rate"])
                or ("impl_gap_large" in flags)):
            detail = ""
            if train_alpha is not None and test_alpha is not None:
                detail = f"训练窗 {train_alpha:+.2%} → 测试窗 {test_alpha:+.2%}"
            elif impl_gap is not None:
                detail = f"实盘-回测差距 {impl_gap:+.2%}"
            return {
                "level": "overfit",
                "rule_id": "R5",
                "reason": f"训练窗好/样本外崩: {detail} → 过拟合",
            }

    # ── R6 兜底 ──
    return {
        "level": "unknown",
        "rule_id": "R6",
        "reason": "现有字段不足以判定失败层级, 待人工复盘补充 root_cause",
    }


# ─────────────────────────────────────────────────────────────
# L1 记录器
# ─────────────────────────────────────────────────────────────

class FailureCaseRecorder:
    """L1 原始失败记录写入器 (append-only JSONL)"""

    def __init__(self,
                 store_path: Optional[str] = None,
                 rules_path: Optional[str] = None,
                 enabled: bool = True):
        self.store_path = store_path or _DEFAULT_STORE
        self.rules_path = rules_path or _DEFAULT_RULES
        self.enabled = enabled

    def record_backtest(self,
                        *,
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
                        stock_pool_snapshot: Optional[dict] = None,
                        train_metrics=None,
                        test_metrics=None,
                        impl_gap: Optional[float] = None,
                        parent_version: Optional[str] = None,
                        granularity: str = "group") -> Optional[dict]:
        """记录一次回测结果

        仅当命中 entry_flags 任一条件时才落盘 (成功 run 不入失效库)。
        返回写入的 record; 未写入返回 None。
        """
        if not self.enabled:
            return None

        cfg = load_rule_config(self.rules_path)
        exits = compute_exit_distribution(trades)
        metric_dict = metrics_to_dict(metrics)
        metric_dict["exit_reason_dist"] = exits["counts"]
        metric_dict["stop_loss_rate"] = exits["stop_loss_rate"]
        metric_dict["ma_breakdown_rate"] = exits["ma_breakdown_rate"]
        if benchmark_return is not None:
            metric_dict["benchmark_return"] = benchmark_return
        if regime_slices:
            metric_dict["regime_slices"] = regime_slices

        train_dict = metrics_to_dict(train_metrics) if train_metrics is not None else None
        test_dict = metrics_to_dict(test_metrics) if test_metrics is not None else None

        flags = derive_failure_flags(
            metric_dict,
            exits=exits,
            train_metrics=train_dict,
            test_metrics=test_dict,
            impl_gap=impl_gap,
            regime_dist=regime_dist,
            regime_slices=regime_slices,
            thresholds=cfg["thresholds"],
        )
        if not any(flag in cfg["entry_flags"] for flag in flags):
            return None

        ratio = (stock_pool_snapshot or {}).get("underperform_ratio")
        judgement = classify_failure_level(
            flags,
            metrics=metric_dict,
            exits=exits,
            train_metrics=train_dict,
            test_metrics=test_dict,
            impl_gap=impl_gap,
            stock_pool_underperform_ratio=ratio,
            regime_slices=regime_slices,
            config=cfg,
        )

        record = self._build_record(
            source="backtest",
            source_ref=source_ref,
            granularity=granularity,
            group_name=group_name,
            stock_codes=stock_codes,
            window=window,
            strategy_mode=strategy_mode,
            params=params,
            metrics=metric_dict,
            benchmark_return=benchmark_return,
            regime_dist=regime_dist,
            regime_slices=regime_slices,
            stock_pool_snapshot=stock_pool_snapshot,
            flags=flags,
            judgement=judgement,
            parent_version=parent_version,
        )
        key = self._case_key(
            granularity=granularity,
            group_name=group_name,
            strategy_mode=strategy_mode,
            window=window,
            params=params,
            source="backtest",
        )
        if self._is_duplicate(key):
            return None
        self._append(record)
        return record

    def record_param_scan(self,
                          *,
                          scan_id: str,
                          group_name: str,
                          stock_codes: Sequence[str],
                          combos: Sequence[dict],
                          grid: dict,
                          window: dict,
                          strategy_mode: str = "mean_reversion",
                          params: Optional[dict] = None,
                          benchmark_return_train: Optional[float] = None,
                          benchmark_return_test: Optional[float] = None,
                          stock_pool_snapshot: Optional[dict] = None,
                          current_train=None,
                          current_test=None,
                          current_params: Optional[dict] = None,
                          parent_version: Optional[str] = None,
                          granularity: str = "group") -> Optional[dict]:
        """记录一次参数扫描 (一次扫描 = 一条 L1, 组合明细内嵌)

        与回测不同: 参数扫描不论结果好坏都记录, 因为方向性规律需要全量组合数据。
        """
        if not self.enabled:
            return None

        cfg = load_rule_config(self.rules_path)
        direction = compute_param_scan_direction(combos, grid, cfg["thresholds"])

        valid_combos = [c for c in (combos or []) if _combo_combined_alpha(c) is not None]
        best_combo = max(valid_combos, key=_combo_combined_alpha) if valid_combos else {}
        best_train = (best_combo.get("train") or {}) if best_combo else {}
        best_test = (best_combo.get("test") or {}) if best_combo else {}

        all_negative = bool(valid_combos) and all(
            (_combo_metric(c, "train", "alpha") is not None and _combo_metric(c, "train", "alpha") < 0
             and _combo_metric(c, "test", "alpha") is not None and _combo_metric(c, "test", "alpha") < 0)
            for c in valid_combos
        )
        best_test_alpha = _combo_metric(best_combo, "test", "alpha")
        current_test_alpha = _combo_metric({"test": current_test}, "test", "alpha") \
            if current_test is not None else None

        scan = {
            "combo_count": len(valid_combos),
            "all_combo_negative": all_negative,
            "best_params": (best_combo.get("params") or {}) if best_combo else None,
            "best_combined_alpha": _combo_combined_alpha(best_combo) if best_combo else None,
            "best_test_alpha": best_test_alpha,
            "current_params": current_params,
            "current_test_alpha": current_test_alpha,
            **direction,
        }

        # 退出结构以最优组合的测试窗为准 (样本外)
        exits = normalize_exit_reasons(best_test.get("exit_reasons"))
        metric_dict = metrics_to_dict(best_test)
        metric_dict["exit_reason_dist"] = exits["counts"]
        metric_dict["stop_loss_rate"] = exits["stop_loss_rate"]
        metric_dict["ma_breakdown_rate"] = exits["ma_breakdown_rate"]
        metric_dict["benchmark_return"] = benchmark_return_test

        train_dict = metrics_to_dict(best_train) if best_train else None
        test_dict = metrics_to_dict(best_test) if best_test else None

        flags = derive_failure_flags(
            metric_dict,
            exits=exits,
            train_metrics=train_dict,
            test_metrics=test_dict,
            scan_direction=direction,
            thresholds=cfg["thresholds"],
        )
        judgement = classify_failure_level(
            flags,
            metrics=metric_dict,
            exits=exits,
            train_metrics=train_dict,
            test_metrics=test_dict,
            scan=scan,
            stock_pool_underperform_ratio=(stock_pool_snapshot or {}).get("underperform_ratio"),
            config=cfg,
        )

        record = self._build_record(
            source="param_scan",
            source_ref={"scan_id": scan_id, "group": group_name},
            granularity=granularity,
            group_name=group_name,
            stock_codes=stock_codes,
            window=window,
            strategy_mode=strategy_mode,
            params=params or {"scan_grid": grid},
            metrics=metric_dict,
            benchmark_return=benchmark_return_test,
            regime_dist=None,
            regime_slices=None,
            stock_pool_snapshot=stock_pool_snapshot,
            flags=flags,
            judgement=judgement,
            parent_version=parent_version,
            extra={
                "param_scan": {
                    **scan,
                    "grid": grid,
                    "combos": [
                        {
                            "params": c.get("params"),
                            "train": metrics_to_dict(c.get("train")),
                            "test": metrics_to_dict(c.get("test")),
                            "combined_alpha": _combo_combined_alpha(c),
                        }
                        for c in valid_combos
                    ],
                    "benchmark_return_train": benchmark_return_train,
                    "benchmark_return_test": benchmark_return_test,
                }
            },
        )
        self._append(record)
        return record

    def _build_record(self, *,
                      source: str,
                      source_ref: dict,
                      granularity: str,
                      group_name: str,
                      stock_codes: Sequence[str],
                      window: dict,
                      strategy_mode: str,
                      params: dict,
                      metrics: dict,
                      benchmark_return: Optional[float],
                      regime_dist: Optional[dict],
                      regime_slices: Optional[dict],
                      stock_pool_snapshot: Optional[dict],
                      flags: List[str],
                      judgement: dict,
                      parent_version: Optional[str],
                      extra: Optional[dict] = None) -> dict:
        params = params or {}
        record = {
            "record_type": "failure_case",
            "case_id": f"fc_{uuid.uuid4().hex[:12]}",
            "recorded_at": datetime.now().isoformat(timespec="seconds"),
            "source": source,
            "source_ref": source_ref,
            "window": window,
            "granularity": granularity,
            "group_name": group_name,
            "stock_codes": list(stock_codes or []),
            "strategy_version": compute_strategy_version(params),
            "parent_version": parent_version,
            "strategy_mode": strategy_mode,
            "params": params,
            "regime_dist": regime_dist,
            "regime_slices": regime_slices,
            "benchmark_return": benchmark_return,
            "metrics": metrics,
            "stock_pool_snapshot": stock_pool_snapshot,
            "failure_flags": flags,
            "failure_level": judgement["level"],
            "failure_level_rule": judgement["rule_id"],
            "failure_level_reason": judgement["reason"],
        }
        if extra:
            record.update(extra)
        return record

    @staticmethod
    def _case_key(*, granularity, group_name, strategy_mode, window, params, source):
        """L1 去重键: 同组+同策略+同窗口+同参数版本+同来源 = 同一失败案例

        重复跑同一窗口(如反复跑 evolution_ab 基线)不应重复落盘,
        否则会虚增教训支持度 (linked_cases 被同一失败刷满)。
        """
        return (
            granularity,
            group_name,
            strategy_mode,
            source,
            (window or {}).get("start"),
            (window or {}).get("end"),
            compute_strategy_version(params or {}),
        )

    def _is_duplicate(self, key) -> bool:
        if not os.path.exists(self.store_path):
            return False
        with open(self.store_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except ValueError:
                    continue
                if rec.get("record_type") != "failure_case":
                    continue
                if self._case_key(
                    granularity=rec.get("granularity"),
                    group_name=rec.get("group_name"),
                    strategy_mode=rec.get("strategy_mode"),
                    window=rec.get("window"),
                    params=rec.get("params"),
                    source=rec.get("source"),
                ) == key:
                    return True
        return False

    def _append(self, record: dict) -> None:
        parent = os.path.dirname(self.store_path)
        if parent:
            os.makedirs(parent, exist_ok=True)
        line = json.dumps(record, ensure_ascii=False, default=str)
        with _WRITE_LOCK:
            with open(self.store_path, "a", encoding="utf-8") as f:
                f.write(line + "\n")


def make_backtest_source_ref(run_id: Optional[str] = None, **extra) -> dict:
    """构造 source_ref; 无 run_id 时生成一个 (便于无记忆层的脚本使用)"""
    ref = {"run_id": run_id or generate_run_id("backtest")}
    ref.update(extra)
    return ref
