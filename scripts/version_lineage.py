"""策略版本谱系 CLI (P7)

用法:
  python scripts/version_lineage.py list                  # 全部版本 (按首次出现时间)
  python scripts/version_lineage.py show <version>        # 单版本详情
  python scripts/version_lineage.py chain <version>       # 父版本链
  python scripts/version_lineage.py diff <a> <b>          # 参数差异
"""
import argparse
import json
import os
import sys

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from src.memory.version_lineage import VersionLineage  # noqa: E402


def _print_json(data) -> None:
    print(json.dumps(data, ensure_ascii=False, indent=2, default=str))


def cmd_list(args) -> int:
    lineage = VersionLineage()
    versions = lineage.list_versions()
    if args.json:
        _print_json(versions)
        return 0
    if not versions:
        print("暂无版本记录 (先跑一次组合回测)。")
        return 0
    print(f"共 {len(versions)} 个版本:")
    for node in versions:
        parent = node.get("parent_version") or "-"
        print(f"- {node['version']}  first={node.get('first_seen')}  "
              f"runs={len(node['runs'])}  lessons={len(node['lessons'])}  "
              f"parent={parent}{' (推断)' if node.get('parent_inferred') else ''}")
    return 0


def cmd_show(args) -> int:
    node = VersionLineage().get_version(args.version)
    if node is None:
        print(f"版本不存在: {args.version}")
        return 1
    _print_json(node)
    return 0


def cmd_chain(args) -> int:
    lineage = VersionLineage()
    chain = lineage.ancestors(args.version)
    if not chain:
        print(f"{args.version} 没有可追溯的父版本")
        return 0
    print(f"{args.version} 的父版本链 (由近及远):")
    for node in chain:
        print(f"- {node['version']}  first={node.get('first_seen')}")
    return 0


def cmd_diff(args) -> int:
    try:
        diff = VersionLineage().diff_versions(args.before, args.after)
    except ValueError as e:
        print(f"无法比较: {e}")
        return 1
    _print_json(diff)
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="策略版本谱系 (P7)")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_list = sub.add_parser("list", help="列出全部版本")
    p_list.add_argument("--json", action="store_true", help="输出 JSON")
    p_list.set_defaults(func=cmd_list)

    p_show = sub.add_parser("show", help="查看单版本详情")
    p_show.add_argument("version")
    p_show.set_defaults(func=cmd_show)

    p_chain = sub.add_parser("chain", help="查看父版本链")
    p_chain.add_argument("version")
    p_chain.set_defaults(func=cmd_chain)

    p_diff = sub.add_parser("diff", help="比较两个版本的参数")
    p_diff.add_argument("before")
    p_diff.add_argument("after")
    p_diff.set_defaults(func=cmd_diff)

    args = parser.parse_args()
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
