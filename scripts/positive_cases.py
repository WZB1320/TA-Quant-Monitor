"""正样本轨迹 CLI (P5)

用法:
  python scripts/positive_cases.py list                          # 全部正样本
  python scripts/positive_cases.py list --group 科技成长型         # 按组过滤
  python scripts/positive_cases.py best --limit 5                 # 最佳参数组合
"""
import argparse
import json
import os
import sys

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from src.memory.positive_case import PositiveCaseRecorder  # noqa: E402


def _filters(args) -> dict:
    return {
        "group": args.group,
        "strategy_mode": args.mode,
        "regime": args.regime,
    }


def cmd_list(args) -> int:
    recorder = PositiveCaseRecorder()
    cases = recorder.list_positive_cases(min_alpha=args.min_alpha, **_filters(args))
    if args.json:
        print(json.dumps(cases, ensure_ascii=False, indent=2, default=str))
        return 0
    if not cases:
        print("暂无正样本记录 (先跑一次组合回测, 且需要 alpha >= 1% 的运行)。")
        return 0
    print(f"共 {len(cases)} 条正样本 (按 alpha 降序):")
    for c in cases:
        metrics = c.get("metrics") or {}
        print(f"- {c['case_id']}  {c['group_name']}/{c['strategy_mode']}  "
              f"version={c['strategy_version']}  alpha={metrics.get('alpha')}  "
              f"dd={metrics.get('max_drawdown')}  "
              f"regime={c.get('dominant_regime') or '-'}  flags={c['positive_flags']}")
    return 0


def cmd_best(args) -> int:
    recorder = PositiveCaseRecorder()
    configs = recorder.best_configs(limit=args.limit, **_filters(args))
    if args.json:
        print(json.dumps(configs, ensure_ascii=False, indent=2, default=str))
        return 0
    if not configs:
        print("暂无可用参数组合。")
        return 0
    print(f"表现最好的 {len(configs)} 个参数版本:")
    for c in configs:
        metrics = c.get("metrics") or {}
        print(f"- version={c['strategy_version']}  {c['group_name']}/{c['strategy_mode']}  "
              f"alpha={metrics.get('alpha')}  dd={metrics.get('max_drawdown')}  "
              f"flags={c['positive_flags']}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="正样本轨迹 (P5)")
    sub = parser.add_subparsers(dest="cmd", required=True)

    def add_filters(p):
        p.add_argument("--group", default=None, help="分组名")
        p.add_argument("--mode", default=None, help="策略模式")
        p.add_argument("--regime", default=None, help="主导体制")
        p.add_argument("--json", action="store_true", help="输出 JSON")

    p_list = sub.add_parser("list", help="列出正样本")
    add_filters(p_list)
    p_list.add_argument("--min-alpha", type=float, default=None, help="alpha 下限")
    p_list.set_defaults(func=cmd_list)

    p_best = sub.add_parser("best", help="按版本去重后的最佳参数组合")
    add_filters(p_best)
    p_best.add_argument("--limit", type=int, default=5, help="最多返回几条")
    p_best.set_defaults(func=cmd_best)

    args = parser.parse_args()
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
