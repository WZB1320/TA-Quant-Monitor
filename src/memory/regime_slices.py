"""失效案例库 P3 — regime 分片统计

定位:
  P1 只统计了 regime_dist (各体制的天数分布), 无法回答"策略在哪个体制下亏钱"。
  regime_slices 把回测区间按市场体制切分, 统计每个体制下组合收益/基准收益/alpha,
  是 R2 regime_mismatch 规则的判定原料。

算法:
  对每个体制, 收集落在该体制的所有交易日的日收益 (nav[t]/nav[t-1]-1),
  累乘得到该体制片段收益; 基准同理; alpha = 组合收益 - 基准收益。
  同一体制可能分散在多段 (trending → ranging → trending), 用日收益累乘
  比"取首尾净值"更能反映该体制下的真实暴露。

用法:
    from src.memory.regime_slices import compute_regime_slices

    slices = compute_regime_slices(
        daily_values=engine.daily_values,
        benchmark_values=bench_series,   # 可与 daily_values 不对齐
        regime_by_date=engine.regime_by_date,  # {date: "trending"}
    )
    # -> {"trending": {"days": 120, "share": 0.5, "return": 0.08,
    #                  "benchmark_return": 0.12, "alpha": -0.04}, ...}
"""
import logging
from collections import defaultdict
from datetime import date, datetime
from typing import Dict, Optional

import pandas as pd

logger = logging.getLogger(__name__)

_DEFAULT_THRESHOLDS = {
    "regime_mismatch_share": 0.50,      # 主导体制占比 >= 该值才算"体制主导"
    "regime_mismatch_alpha": -0.02,     # 主导体制下 alpha <= 该值才算"显著跑输"
}


def _norm_date(value) -> Optional[date]:
    """把 Timestamp / datetime / date 统一成 date"""
    if value is None:
        return None
    if isinstance(value, pd.Timestamp):
        return value.date()
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    try:
        return pd.Timestamp(value).date()
    except Exception:
        return None


def _compound(returns) -> Optional[float]:
    """日收益累乘 → 区间收益"""
    if not returns:
        return None
    acc = 1.0
    for r in returns:
        acc *= (1.0 + r)
    return acc - 1.0


def compute_regime_slices(daily_values,
                          benchmark_values=None,
                          regime_by_date: Optional[Dict] = None,
                          thresholds: Optional[dict] = None) -> Dict[str, dict]:
    """按市场体制切分回测区间, 统计各体制下的组合/基准收益

    Args:
        daily_values: 组合日净值 (pd.Series, index 为日期)
        benchmark_values: 基准日净值 (可选, 允许与 daily_values 日期不完全对齐)
        regime_by_date: {date: regime}, 来自 BacktestEngine.regime_by_date
        thresholds: 覆盖默认阈值 (当前仅用于文档一致性, 不参与分片计算)

    Returns:
        {regime: {days, share, return, benchmark_return, alpha}}
        数据不足时返回 {}
    """
    if daily_values is None or len(daily_values) < 2 or not regime_by_date:
        return {}

    try:
        nav = daily_values.sort_index()
        nav_returns = nav.pct_change().dropna()
    except Exception:
        logger.warning("regime_slices: 计算净值日收益失败", exc_info=True)
        return {}
    if len(nav_returns) == 0:
        return {}

    bench_returns = None
    if benchmark_values is not None and len(benchmark_values) > 1:
        try:
            bench = benchmark_values.sort_index()
            bench_returns = bench.pct_change().dropna()
        except Exception:
            bench_returns = None

    # 统一 regime_by_date 的 key 类型
    regime_map = {}
    for k, v in (regime_by_date or {}).items():
        d = _norm_date(k)
        if d is not None and v:
            regime_map[d] = v

    regime_nav_rets: Dict[str, list] = defaultdict(list)
    regime_bench_rets: Dict[str, list] = defaultdict(list)
    regime_days: Dict[str, int] = defaultdict(int)
    matched_days = 0

    for idx in nav_returns.index:
        d = _norm_date(idx)
        if d is None:
            continue
        regime = regime_map.get(d)
        if not regime:
            continue
        regime_nav_rets[regime].append(float(nav_returns.loc[idx]))
        regime_days[regime] += 1
        matched_days += 1
        if bench_returns is not None and d in bench_returns.index:
            regime_bench_rets[regime].append(float(bench_returns.loc[d]))

    if matched_days == 0:
        return {}

    slices: Dict[str, dict] = {}
    for regime, rets in regime_nav_rets.items():
        nav_return = _compound(rets)
        bench_rets = regime_bench_rets.get(regime) or []
        bench_return = _compound(bench_rets) if bench_rets else None
        alpha = None
        if nav_return is not None and bench_return is not None:
            alpha = nav_return - bench_return
        slices[regime] = {
            "days": regime_days[regime],
            "share": round(regime_days[regime] / matched_days, 4),
            "return": round(nav_return, 6) if nav_return is not None else None,
            "benchmark_return": round(bench_return, 6) if bench_return is not None else None,
            "alpha": round(alpha, 6) if alpha is not None else None,
        }
    return slices


def dominant_regime(slices: Dict[str, dict],
                    thresholds: Optional[dict] = None) -> Optional[dict]:
    """返回主导体制信息 {regime, share, alpha, is_mismatch}

    is_mismatch 判定: 主导体制占比 >= regime_mismatch_share 且该体制 alpha
    <= regime_mismatch_alpha (显著跑输基准)。
    """
    th = {**_DEFAULT_THRESHOLDS, **(thresholds or {})}
    if not slices:
        return None
    name, sl = max(slices.items(), key=lambda kv: kv[1].get("share", 0.0))
    share = sl.get("share", 0.0) or 0.0
    alpha = sl.get("alpha")
    is_mismatch = (
        share >= th["regime_mismatch_share"]
        and alpha is not None
        and alpha <= th["regime_mismatch_alpha"]
    )
    return {
        "regime": name,
        "share": share,
        "alpha": alpha,
        "is_mismatch": is_mismatch,
    }
