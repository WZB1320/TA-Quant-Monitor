"""失效案例复盘 CLI — 从 L1 记录提炼 L2 教训

子命令:
  list    列出 L1 记录, 按 (group, strategy_mode, failure_level) 分组,
          显示每组案例数 (>=2 才可提炼教训)
  draft   对指定 case_ids AI 起草 root_cause (不写入)
  create  创建一条 L2 教训 (linked_cases >= 2), root_cause 走 AI 起草,
          默认只打印; 加 --write 才落盘

用法:
  python scripts/review_failures.py list
  python scripts/review_failures.py draft --cases fc_xxx,fc_yyy
  python scripts/review_failures.py create \
      --cases fc_xxx,fc_yyy,fc_zzz \
      --level param \
      --rule "score_threshold 30 -> 38" \
      --do-not-repeat "score_threshold <= 30" \
      --write
"""
import argparse
import json
import os
import sys
from collections import defaultdict
from typing import Dict, List

# 允许脚本直接运行 (项目根目录在 sys.path)
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.memory.failure_case import FAILURE_LEVELS  # noqa: E402
from src.memory.lesson_library import (  # noqa: E402
    LessonLibrary, infer_scope_from_records, load_l1_records,
)


def _group_key(rec: dict) -> tuple:
    return (
        rec.get("group_name") or "?",
        rec.get("strategy_mode") or "?",
        rec.get("failure_level") or "unknown",
    )


def cmd_list(args) -> int:
    records = load_l1_records()
    if not records:
        print("L1 暂无记录 (data/failure_cases/l1_failures.jsonl 不存在或为空)")
        return 0

    groups: Dict[tuple, List[dict]] = defaultdict(list)
    for rec in records:
        groups[_group_key(rec)].append(rec)

    # 按案例数降序
    sorted_groups = sorted(groups.items(), key=lambda kv: -len(kv[1]))

    print(f"L1 共 {len(records)} 条, 分为 {len(sorted_groups)} 组 (group × mode × level)\n")
    for (group, mode, level), recs in sorted_groups:
        flag = "✓ 可提炼" if len(recs) >= 2 else "  单条"
        print(f"[{flag}] {group} / {mode} / {level}  ({len(recs)} 条)")
        for rec in recs[:5]:
            flags = ",".join(rec.get("failure_flags", [])[:3])
            alpha = (rec.get("metrics") or {}).get("alpha")
            print(f"    {rec['case_id']}  alpha={alpha}  flags=[{flags}]")
        if len(recs) > 5:
            print(f"    ... 还有 {len(recs) - 5} 条")
    return 0


def cmd_draft(args) -> int:
    case_ids = [c.strip() for c in args.cases.split(",") if c.strip()]
    if len(case_ids) < 2:
        print("错误: draft 至少需要 2 个 case_id", file=sys.stderr)
        return 2

    records = load_l1_records(case_ids)
    if len(records) < 2:
        print(f"错误: 只找到 {len(records)} 条 L1 记录 (指定了 {len(case_ids)} 个)", file=sys.stderr)
        return 2

    lib = LessonLibrary()
    draft = lib.draft_root_cause(records)
    print("=" * 60)
    print(f"AI 起草 root_cause (基于 {len(records)} 条 L1):")
    print("=" * 60)
    print(draft)
    print("=" * 60)
    return 0


def cmd_create(args) -> int:
    case_ids = [c.strip() for c in args.cases.split(",") if c.strip()]
    if len(case_ids) < 2:
        print("错误: create 至少需要 2 个 case_id", file=sys.stderr)
        return 2

    records = load_l1_records(case_ids)
    if len(records) < 2:
        print(f"错误: 只找到 {len(records)} 条 L1 记录", file=sys.stderr)
        return 2

    if args.level not in FAILURE_LEVELS:
        print(f"错误: --level 必须是 {FAILURE_LEVELS} 之一", file=sys.stderr)
        return 2

    # scope: 从 L1 记录推断 (group/mode 唯一才确定; regime 仅体制失配时确定)
    scope = infer_scope_from_records(records)
    if args.regime:  # 人工覆盖推断结果
        scope["regime"] = args.regime
    if args.group:
        scope["group"] = args.group

    lib = LessonLibrary()
    root_cause = lib.draft_root_cause(records)

    do_not_repeat = [s.strip() for s in (args.do_not_repeat or "").split(";") if s.strip()]
    flags_summary = list({f for r in records for f in r.get("failure_flags", [])})

    rule_action = None
    if args.action:
        from src.memory.rule_action import parse_action_spec
        rule_action = parse_action_spec(args.action)

    payload = {
        "linked_cases": case_ids,
        "scope": scope,
        "failure_level": args.level,
        "root_cause": root_cause,
        "rule": args.rule,
        "rule_action": rule_action,
        "do_not_repeat": do_not_repeat,
        "failure_flags_summary": flags_summary,
        "ai_drafted": True,
    }

    if not args.write:
        print("【预览模式】未写入。加 --write 落盘。Lesson 内容:")
        print(json.dumps(payload, ensure_ascii=False, indent=2))
        return 0

    lesson = lib.create_lesson(**payload)
    print(f"已创建 L2 教训: {lesson['lesson_id']}")
    print(f"  level={lesson['failure_level']}  support={lesson['support_count']}  "
          f"confidence={lesson['confidence']}")
    print(f"  scope={lesson['scope']}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="失效案例复盘 CLI (L1 → L2)")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_list = sub.add_parser("list", help="列出 L1 记录并按相似性分组")
    p_list.set_defaults(func=cmd_list)

    p_draft = sub.add_parser("draft", help="AI 起草 root_cause (不写入)")
    p_draft.add_argument("--cases", required=True, help="逗号分隔的 case_id 列表 (>=2)")
    p_draft.set_defaults(func=cmd_draft)

    p_create = sub.add_parser("create", help="创建 L2 教训")
    p_create.add_argument("--cases", required=True, help="逗号分隔的 case_id 列表 (>=2)")
    p_create.add_argument("--level", required=True, choices=FAILURE_LEVELS,
                          help="确认的 failure_level")
    p_create.add_argument("--rule", required=True, help="可执行的改进规则")
    p_create.add_argument("--do-not-repeat", default="",
                          help="分号分隔的禁忌方向")
    p_create.add_argument("--action", default=None,
                          help="结构化动作, 格式 set_param:target=value 或 "
                               "adjust_param:target=+delta (可被程序执行)")
    p_create.add_argument("--regime", default=None,
                          help="覆盖自动推断的 regime (如 ranging/trending)")
    p_create.add_argument("--group", default=None,
                          help="覆盖自动推断的 group")
    p_create.add_argument("--write", action="store_true",
                          help="实际写入 (默认只预览)")
    p_create.set_defaults(func=cmd_create)

    args = parser.parse_args()
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
