"""P8 进化 A/B 双跑 CLI (dry-run)

用法:
  python scripts/evolution_ab.py --start 2026-01-01 --end 2026-06-30
  python scripts/evolution_ab.py --start 2026-01-01 --end 2026-06-30 --mode trending --json

说明:
  - 基线 = 当前 strategy_config.json 参数, 正常写记忆层
  - 候选 = 基线 + 全部 validated 教训的 rule_action, 不写记忆层, 不写回配置
  - 双跑结果写入版本谱系 (data/failure_cases/version_lineage.jsonl)
"""
import argparse
import json
import os
import sys

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from src.evolution.auto_applier import EvolutionAB  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description="教训驱动 A/B 双跑 (P8 dry-run)")
    parser.add_argument("--start", default=None, help="回测开始日期 YYYY-MM-DD")
    parser.add_argument("--end", default=None, help="回测结束日期 YYYY-MM-DD")
    parser.add_argument("--mode", default="base", choices=["base", "trending", "ranging"])
    parser.add_argument("--capital", type=float, default=100000)
    parser.add_argument("--json", action="store_true", help="输出完整 JSON 报告")
    parser.add_argument("--promote", action="store_true",
                        help="固化已验证教训到 strategy_config.json (需先有 validated 教训)")
    args = parser.parse_args()

    ab = EvolutionAB()

    if args.promote:
        result = ab.promote()
        print(json.dumps(result, ensure_ascii=False, indent=2, default=str))
        return 0 if result.get("written_config") else 1

    report = ab.run_ab(
        start_date=args.start, end_date=args.end,
        mode=args.mode, initial_capital=args.capital)

    if args.json:
        print(json.dumps(report, ensure_ascii=False, indent=2, default=str))
    else:
        _print_report(report)
    return 0 if report.get("verdict") != "error" else 1


def _print_report(r: dict) -> None:
    print("=" * 64)
    print("P8 进化 A/B (dry-run) 报告")
    print("=" * 64)

    if not r.get("has_candidate"):
        print(f"结论: {r.get('verdict')}")
        print(f"说明: {r.get('reason')}")
        baseline = r.get("baseline") or {}
        if baseline.get("metrics"):
            _print_metrics("基线", baseline)
        return

    for label in ("baseline", "candidate"):
        _print_metrics("基线" if label == "baseline" else "候选", r[label])

    print("-" * 64)
    print(f"alpha 变化: {r.get('alpha_delta', 0):+.4f}")
    print(f"结论: {r.get('verdict')} — {r.get('reason')}")

    if r.get("changes"):
        print(f"\n应用的参数变更 ({len(r['changes'])}):")
        for ch in r["changes"]:
            print(f"  [{ch['lesson_id']}] {ch['group']}.{ch['target']} "
                  f"{ch['before']} -> {ch['after']} (via {ch['via']})")
    if r.get("skipped"):
        print(f"\n跳过的动作 ({len(r['skipped'])}):")
        for sk in r["skipped"]:
            reason = sk.get("reason", "")
            action = sk.get("action") or {}
            print(f"  [{sk.get('lesson_id', '?')}] "
                  f"{action.get('type', '?')}:{action.get('target', '?')} — {reason}")
    if r.get("lineage_recorded"):
        print("\n谱系: 已记录候选版本 (parent=基线版本), 可用 "
              "python scripts/version_lineage.py list 查看")
    print(f"\n{r.get('note', '')}")


def _print_metrics(label: str, summary: dict) -> None:
    m = summary.get("metrics") or {}
    print(f"\n[{label}] run_id={summary.get('run_id')}  "
          f"version={summary.get('strategy_version')}")
    print(f"  alpha={m.get('alpha')}  总收益={m.get('total_return')}  "
          f"年化={m.get('annual_return')}")
    print(f"  回撤={m.get('max_drawdown')}  胜率={m.get('win_rate')}  "
          f"夏普={m.get('sharpe_ratio')}  交易数={m.get('trade_count')}")


if __name__ == "__main__":
    raise SystemExit(main())
