"""建议自动验证 CLI (P4)

用法:
  python scripts/validate_suggestions.py            # 预览 (dry-run, 不写状态)
  python scripts/validate_suggestions.py --apply    # 实际写回建议/教训状态
  python scripts/validate_suggestions.py --json     # 输出 JSON (便于脚本消费)
"""
import argparse
import json
import os
import sys

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from src.ai.suggestion_validator import SuggestionValidator  # noqa: E402
from src.memory.version_lineage import VersionLineage  # noqa: E402


def cmd_validate(args) -> int:
    result = SuggestionValidator(
        version_lineage=VersionLineage(),
    ).validate(dry_run=not args.apply)

    if args.json:
        print(json.dumps(result, ensure_ascii=False, indent=2, default=str))
        return 0

    mode = "写入" if args.apply else "预览 (dry-run, 加 --apply 写入)"
    print(f"建议自动验证 — {mode}")
    print(f"待验证: {result['checked']} 条 | 有效: {result['validated']} | "
          f"证伪: {result['invalidated']} | 无法判定: {result['inconclusive']}")

    for item in result["items"]:
        verdict_label = {
            "validated": "✅ 有效",
            "invalidated": "❌ 证伪",
            "inconclusive": "⏸ 无法判定",
        }.get(item["verdict"], item["verdict"])
        print(f"\n- [{item['suggestion_id']}] target={item['target']} → {verdict_label}")
        print(f"  原因: {item['reason']}")
        if item.get("delta"):
            delta = item["delta"]
            parts = [f"{k} {v:+.2%}" for k, v in delta.items()
                     if k in ("alpha", "total_return") and v is not None]
            if parts:
                print(f"  变化: {', '.join(parts)}")
        if item.get("before_run_id") or item.get("after_run_id"):
            print(f"  before={item.get('before_run_id')} → after={item.get('after_run_id')}")
        if item.get("lessons_updated"):
            print(f"  回写教训: {', '.join(item['lessons_updated'])}")

    return 0


def main():
    parser = argparse.ArgumentParser(description="建议自动验证器 (P4)")
    parser.add_argument("--apply", action="store_true",
                        help="实际写回建议/教训状态 (默认仅预览)")
    parser.add_argument("--json", action="store_true", help="输出 JSON")
    parser.set_defaults(func=cmd_validate)
    args = parser.parse_args()
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
