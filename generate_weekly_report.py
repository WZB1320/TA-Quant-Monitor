"""策略周报 CLI — 从记忆层数据生成 AI 分析周报

用法:
  python generate_weekly_report.py                       # 分析最新回测记忆
  python generate_weekly_report.py data/strategy_memory.jsonl  # 分析指定文件
  python generate_weekly_report.py --no-save             # 不保存到文件, 仅打印

环境配置:
  复制 .env.example 为 .env, 填入 DEEPSEEK_API_KEY
"""
import os
import sys

# 确保项目根目录在 sys.path
_PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from src.ai import WeeklyReportGenerator


def main():
    # 解析参数
    args = sys.argv[1:]
    memory_file = None
    save = True

    for arg in args:
        if arg == "--no-save":
            save = False
        elif not arg.startswith("--"):
            memory_file = arg

    # 生成周报
    gen = WeeklyReportGenerator()

    if not gen.llm.is_available:
        print("=" * 60)
        print("  ⚠️  LLM 未配置, 将仅输出统计摘要")
        print("  配置方法: 复制 .env.example 为 .env, 填入 DEEPSEEK_API_KEY")
        print("=" * 60)

    try:
        report = gen.generate(memory_file=memory_file, save=save)
        print()
        print(report)
    except FileNotFoundError as e:
        print(f"错误: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"生成周报失败: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
