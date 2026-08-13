"""
组级Alpha独立验证脚本 — quant-claim-verify skill 配套工具

用途: 在下结论"某组Alpha是X%"之前, 必须用本脚本独立验证.
严禁从组合聚合数据反推单组Alpha.

用法:
    python scripts/verify_group_alpha.py --group 医药创新型 --start 2024-07-01 --end 2026-06-30
    python scripts/verify_group_alpha.py --group 科技成长型 --start 2025-07-01 --end 2026-06-30
    python scripts/verify_group_alpha.py --all  # 验证全部5组

输出:
    - 该组独立Alpha/收益/Sharpe/回撤/交易数
    - 与历史基线对比 (data/group_alpha_baselines.json)
    - 差距告警 (符号反转 或 差距>历史绝对值50%)
"""
import sys
import os
import json
import argparse
import warnings
from datetime import datetime

import pandas as pd

warnings.filterwarnings("ignore")
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from src.config.runtime_mode import set_mode, RuntimeMode
set_mode(RuntimeMode.BACKTEST)

from src.backtest.engine import BacktestEngine
from src.backtest.rotation_strategy import RotationStrategy
from src.data_fetcher import DataManager
from src.config.group_config import GroupConfig

# ── 配置 ──
DATA_START = "2024-02-01"
DATA_END = "2026-07-13"
BENCHMARK = "sh.000300"
GROUP_CAPITAL = 100000  # 单组回测资金, 与P2/P3脚本一致

# 组合回测中的差异化配置 (与 backtest.py PORTFOLIO_WEIGHTS 一致)
ATR_OVERRIDE = {"科技成长型": 1.8}
REGIMES_CFG = {"周期资源型": {"trending"}}

BASELINE_FILE = os.path.join(PROJECT_ROOT, "data", "group_alpha_baselines.json")


def load_baselines():
    if os.path.exists(BASELINE_FILE):
        with open(BASELINE_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}


def compare_with_baseline(group, window, new_alpha):
    """对比历史基线, 返回告警信息"""
    baselines = load_baselines()
    group_bl = baselines.get(group, {})
    hist = group_bl.get(window)
    if not hist:
        return None, "无历史基线"

    hist_alpha = hist.get("alpha")
    if hist_alpha is None:
        return None, "历史基线alpha为null (未独立验证过)"

    diff = new_alpha - hist_alpha
    hist_abs = abs(hist_alpha)

    alerts = []
    # 符号反转告警
    if hist_alpha * new_alpha < 0:
        alerts.append(f"符号反转! 历史{hist_alpha*100:+.2f}% → 现在{new_alpha*100:+.2f}%")
    # 差距>历史绝对值50%告警
    if hist_abs > 0 and abs(diff) > hist_abs * 0.5:
        alerts.append(f"差距>{hist_abs*50:.1f}%历史绝对值! 差{diff*100:+.2f}%")

    alert_str = " | ".join(alerts) if alerts else "无告警"
    return hist_alpha, alert_str


def run_group_backtest(group_name, start, end):
    """独立跑单组回测, 返回metrics"""
    # 加载配置
    GroupConfig._instance = None
    GroupConfig._config = None
    gc = GroupConfig()
    gc._load()

    # 读取该组股票
    cfg_path = os.path.join(PROJECT_ROOT, "config", "strategy_config.json")
    with open(cfg_path, "r", encoding="utf-8") as f:
        cfg = json.load(f)
    stocks = cfg["strategy_config"]["watchlist"].get(group_name, [])
    codes = [s["code"] for s in stocks]

    if not codes:
        return None, f"组 {group_name} 无自选股"

    # 拉数据
    dm = DataManager()
    data_map = {}
    for c in codes:
        df = dm.get_daily_kline(c, start_date=DATA_START, end_date=DATA_END)
        if df is not None and len(df) > 80:
            data_map[c] = df
    bench = dm.get_daily_kline(BENCHMARK, start_date=DATA_START, end_date=DATA_END)

    if not data_map:
        return None, f"组 {group_name} 数据获取失败"

    group_cfg = gc._groups.get(group_name, {})
    strategy_mode = group_cfg.get("strategy_mode", "trend_following")

    if strategy_mode == "rotation":
        # 动量轮动 (机械组)
        rotation = RotationStrategy(initial_capital=GROUP_CAPITAL)
        result = rotation.run(
            {c: data_map[c] for c in codes if c in data_map},
            bench_df=bench, start_date=start, end_date=end,
        )
        # 用RotationStrategy的daily_values计算metrics
        from src.backtest.metrics import compute_metrics
        # 对齐基准净值: 用收盘价构建与daily_values同日期的基准序列
        bench_aligned = None
        if bench is not None and "close" in bench.columns and "date" in bench.columns:
            bench_copy = bench.copy()
            bench_copy["date"] = pd.to_datetime(bench_copy["date"])
            bench_close = bench_copy.set_index("date")["close"]
            bench_aligned = bench_close.reindex(result["daily_values"].index).ffill()
        m = compute_metrics(
            result["daily_values"],
            trades=result["closed_trades"],
            initial_capital=GROUP_CAPITAL,
            benchmark_values=bench_aligned,
        )
        return m, None

    # 标准趋势跟踪/均值回归
    mean_reversion_config = group_cfg.get("mean_reversion_exit") or None
    atr_mult = ATR_OVERRIDE.get(group_name, group_cfg.get("atr_stop_mult", 2.0))

    engine = BacktestEngine(
        initial_capital=GROUP_CAPITAL, lookback_days=120, position_ratio=0.3,
        commission_rate=0.00025, stamp_tax=0.001, slippage=0.0001,
        signal_dedup_days=5, risk_per_trade=0.05,
        atr_stop_mult=atr_mult,
        forced_regime=None,
        trade_regimes=REGIMES_CFG.get(group_name),
        dd_protection_config=BacktestEngine.DEFAULT_DD_PROTECTION_CONFIG,
        mean_reversion_config=mean_reversion_config,
    )

    m = engine.run(
        {c: data_map[c] for c in codes if c in data_map},
        benchmark_df=bench, start_date=start, end_date=end,
    )
    return m, None


def format_result(group, window, m, start, end):
    """格式化输出"""
    new_alpha = getattr(m, "alpha", 0) or 0
    hist_alpha, alert = compare_with_baseline(group, window, new_alpha)

    lines = []
    lines.append(f"\n{'='*70}")
    lines.append(f"  {group} - {window} ({start} ~ {end})")
    lines.append(f"{'='*70}")
    lines.append(f"  收益:     {getattr(m,'total_return',0)*100:+.2f}%")
    lines.append(f"  基准:     {getattr(m,'benchmark_return',0)*100:+.2f}%")
    lines.append(f"  Alpha:    {new_alpha*100:+.2f}%  [已验证]")
    lines.append(f"  Sharpe:   {getattr(m,'sharpe_ratio',0):.3f}")
    lines.append(f"  回撤:     {getattr(m,'max_drawdown',0)*100:.2f}%")
    lines.append(f"  交易数:   {getattr(m,'trade_count',0)}笔")
    lines.append(f"  胜率:     {getattr(m,'win_rate',0)*100:.0f}%")
    if hist_alpha is not None:
        lines.append(f"  历史基线: {hist_alpha*100:+.2f}%")
        lines.append(f"  对比告警: {alert}")
    else:
        lines.append(f"  历史基线: {alert}")
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="组级Alpha独立验证")
    parser.add_argument("--group", type=str, help="组名 (如 医药创新型)")
    parser.add_argument("--all", action="store_true", help="验证全部5组")
    parser.add_argument("--start", type=str, default="2024-07-01")
    parser.add_argument("--end", type=str, default="2026-06-30")
    parser.add_argument("--window", type=str, default="full",
                        choices=["train", "test", "full"],
                        help="train=2024-07~2025-06, test=2025-07~2026-06, full=2024-07~2026-06")
    args = parser.parse_args()

    # 根据window覆盖日期
    if args.window == "train":
        args.start, args.end = "2024-07-01", "2025-06-30"
    elif args.window == "test":
        args.start, args.end = "2025-07-01", "2026-06-30"

    ALL_GROUPS = ["科技成长型", "消费稳健型", "周期资源型", "医药创新型", "机械制造型"]

    if args.all:
        groups = ALL_GROUPS
    elif args.group:
        groups = [args.group]
    else:
        parser.error("必须指定 --group 或 --all")

    print(f"\n组级Alpha独立验证 — quant-claim-verify skill")
    print(f"窗口: {args.window} ({args.start} ~ {args.end})")
    print(f"资金: ¥{GROUP_CAPITAL:,}  基准: {BENCHMARK}")

    for group in groups:
        m, err = run_group_backtest(group, args.start, args.end)
        if err:
            print(f"\n[跳过] {group}: {err}")
            continue
        print(format_result(group, args.window, m, args.start, args.end))

    print(f"\n{'='*70}")
    print(f"  验证完成 — 所有Alpha均标注[已验证], 来自独立单组回测")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
