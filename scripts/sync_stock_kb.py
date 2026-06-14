"""独立脚本：手动同步 A 股股票知识库

用法:
    python scripts/sync_stock_kb.py           # 使用 stock_info_a_code_name (轻量)
    python scripts/sync_stock_kb.py --full    # 使用 stock_zh_a_spot_em (含行情)
    python scripts/sync_stock_kb.py --status  # 仅查看状态
"""
import os
import sys

# 确保项目根目录在 sys.path 中
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)


def main():
    import argparse

    parser = argparse.ArgumentParser(description="A 股股票知识库同步工具")
    parser.add_argument(
        "--full", action="store_true",
        help="使用 stock_zh_a_spot_em 接口（含实时行情，数据更全但稍慢）"
    )
    parser.add_argument(
        "--status", action="store_true",
        help="仅查看知识库状态，不执行同步"
    )
    args = parser.parse_args()

    from services.knowledge_base import KnowledgeBase

    kb = KnowledgeBase()

    if args.status:
        print(f"数据库路径: {kb.db_path}")
        print(f"股票数量:   {kb.count}")
        print(f"是否为空:   {'是' if kb.is_empty() else '否'}")
        return

    print("正在从 akshare 同步 A 股数据...")
    print(f"模式: {'完整行情' if args.full else '轻量基础'}")
    try:
        count = kb.sync_from_akshare(use_slim=not args.full)
        print(f"同步完成！共 {count} 只股票。")
    except ImportError:
        print("错误: 缺少依赖，请先执行 pip install akshare pypinyin")
        sys.exit(1)
    except Exception as e:
        print(f"同步失败: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()