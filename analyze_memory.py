"""策略记忆层数据分析 — 信号等级 vs 收益率统计

用法:
  python analyze_memory.py                    # 分析最新的回测记忆文件
  python analyze_memory.py <file.jsonl>       # 分析指定文件
  python analyze_memory.py data/strategy_memory.jsonl  # 分析实盘记忆

输出:
  - 按信号等级分组的交易统计 (笔数/胜率/平均收益/平均持仓)
  - 按退出原因分组的统计
  - 按 regime 分组的统计
"""
import os
import sys
import json
import glob
from collections import defaultdict

_PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
_BACKTEST_MEM_DIR = os.path.join(_PROJECT_ROOT, "data", "backtest_memory")


def find_latest_memory_file() -> str:
    """查找最新的回测记忆文件"""
    files = glob.glob(os.path.join(_BACKTEST_MEM_DIR, "*.jsonl"))
    if not files:
        return None
    return max(files, key=os.path.getmtime)


def load_records(file_path: str) -> tuple:
    """加载 JSONL 记录, 返回 (signals, outcomes)

    signals: {run_id: {symbol_date_key: signal_record}}
    outcomes: [outcome_record, ...]
    """
    signals = defaultdict(dict)
    outcomes = []

    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            if record["record_type"] == "signal":
                key = f"{record['symbol']}_{record['analysis_date']}"
                signals[record["run_id"]][key] = record
            elif record["record_type"] == "outcome":
                outcomes.append(record)

    return signals, outcomes


def join_signal_outcome(signals: dict, outcomes: list) -> list:
    """将 outcome 关联到对应的 signal, 返回合并记录列表"""
    joined = []
    for out in outcomes:
        ref = out.get("signal_ref", {})
        run_id = ref.get("run_id")
        key = f"{ref.get('symbol')}_{ref.get('analysis_date')}"
        sig = signals.get(run_id, {}).get(key)

        joined.append({
            "symbol": out["symbol"],
            "level": out.get("signal_level_at_entry") or (sig["level"] if sig else "unknown"),
            "score": out.get("signal_score_at_entry"),
            "regime": sig["regime"] if sig else "unknown",
            "pnl": out["pnl"],
            "pnl_pct": out["pnl_pct"],
            "holding_days": out["holding_days"],
            "exit_reason": out["exit_reason"],
            "executable": sig["executable"] if sig else None,
        })
    return joined


def stats_by_group(records: list, group_key: str, label: str) -> None:
    """按指定字段分组统计并打印"""
    groups = defaultdict(list)
    for r in records:
        groups[r[group_key]].append(r)

    print(f"\n{'=' * 72}")
    print(f"  按 {label} 分组")
    print(f"{'=' * 72}")
    print(f"  {'等级':<12s} {'笔数':>4s} {'胜率':>6s} {'平均收益':>10s} {'总盈亏':>12s} {'均持仓':>6s}")
    print(f"  {'-' * 56}")

    for name in sorted(groups.keys()):
        trades = groups[name]
        n = len(trades)
        wins = sum(1 for t in trades if t["pnl"] > 0)
        win_rate = wins / n * 100 if n > 0 else 0
        avg_pnl_pct = sum(t["pnl_pct"] for t in trades) / n * 100 if n > 0 else 0
        total_pnl = sum(t["pnl"] for t in trades)
        avg_hold = sum(t["holding_days"] for t in trades) / n if n > 0 else 0
        print(f"  {name:<12s} {n:>4d} {win_rate:>5.0f}% {avg_pnl_pct:>+9.2f}% {total_pnl:>+12.0f} {avg_hold:>5.0f}d")


def main():
    # 确定分析文件
    if len(sys.argv) > 1:
        file_path = sys.argv[1]
    else:
        file_path = find_latest_memory_file()

    if not file_path or not os.path.exists(file_path):
        print("未找到记忆文件. 请先运行回测生成数据.")
        print(f"查找路径: {_BACKTEST_MEM_DIR}")
        sys.exit(1)

    print(f"分析文件: {file_path}")

    # 加载并关联
    signals, outcomes = load_records(file_path)
    print(f"信号记录: {sum(len(v) for v in signals.values())} 条")
    print(f"结果记录: {len(outcomes)} 条")

    if not outcomes:
        print("无交易结果, 无法分析收益率.")
        sys.exit(0)

    joined = join_signal_outcome(signals, outcomes)
    print(f"成功关联: {len(joined)} 笔交易")

    # 统计
    stats_by_group(joined, "level", "信号等级")
    stats_by_group(joined, "exit_reason", "退出原因")
    stats_by_group(joined, "regime", "市场 Regime")

    # 整体汇总
    print(f"\n{'=' * 72}")
    print(f"  整体汇总")
    print(f"{'=' * 72}")
    n = len(joined)
    wins = sum(1 for t in joined if t["pnl"] > 0)
    total_pnl = sum(t["pnl"] for t in joined)
    avg_pnl_pct = sum(t["pnl_pct"] for t in joined) / n * 100
    print(f"  总交易数:   {n}")
    print(f"  胜率:       {wins / n * 100:.1f}%")
    print(f"  总盈亏:     {total_pnl:+,.2f}")
    print(f"  平均收益率: {avg_pnl_pct:+.2f}%")


if __name__ == "__main__":
    main()
