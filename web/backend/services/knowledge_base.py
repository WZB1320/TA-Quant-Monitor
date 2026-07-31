"""A 股股票知识库

基于 SQLite + FTS5 存储全量 A 股股票列表，支持：
  - 从 akshare 全量同步
  - 混合搜索策略：代码前缀 / 拼音 FTS5 / 中文 LIKE
  - 单只股票校验
"""
import logging
import os
import sqlite3
import threading
from datetime import datetime, timezone
from typing import Optional, Union

logger = logging.getLogger(__name__)


def _get_default_db_path() -> str:
    """自动推断 stock_knowledge.db 路径"""
    # 向上找到项目根目录（stock_knowledge_base_plan.md 所在目录）
    current = os.path.dirname(os.path.abspath(__file__))
    # services/ -> backend/ -> web/ -> 项目根
    root = os.path.dirname(os.path.dirname(os.path.dirname(current)))
    return os.path.join(root, "data", "stock_knowledge.db")


class KnowledgeBase:
    """A 股股票知识库（单例模式，线程安全）"""

    _instance: Optional["KnowledgeBase"] = None
    _lock = threading.Lock()

    def __new__(cls, db_path: Optional[str] = None) -> "KnowledgeBase":
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._initialized = False
        return cls._instance

    def __init__(self, db_path: Optional[str] = None):
        if self._initialized:
            return
        self._initialized = True
        self.db_path = db_path or _get_default_db_path()
        self._init_tables()

    # ── 数据库初始化 ──

    def _init_tables(self):
        """创建 stocks 主表和 FTS5 全文搜索索引"""
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS stocks (
                    code        TEXT PRIMARY KEY,
                    name        TEXT NOT NULL,
                    market      TEXT NOT NULL,
                    py_initials TEXT,
                    full_py     TEXT,
                    industry    TEXT,
                    updated_at  TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            # FTS5 外部内容表：搜索拼音字段
            conn.execute("""
                CREATE VIRTUAL TABLE IF NOT EXISTS stock_search USING fts5(
                    py_initials,
                    full_py,
                    content='stocks',
                    content_rowid='rowid'
                )
            """)
        logger.info("知识库表结构初始化完成: %s", self.db_path)

    # ── 数据同步 ──

    def sync_from_akshare(self, use_slim: bool = True) -> int:
        """从 akshare 拉取全量 A 股股票列表并写入知识库

        Args:
            use_slim: True=使用 stock_info_a_code_name（轻量），False=stock_zh_a_spot_em（含行情）

        Returns:
            int: 同步的股票数量
        """
        try:
            import akshare as ak
            from pypinyin import lazy_pinyin, Style

            if use_slim:
                df = ak.stock_info_a_code_name()
                df = df.rename(columns={"code": "代码", "name": "名称"})
            else:
                df = ak.stock_zh_a_spot_em()

            logger.info("从 akshare 获取到 %d 条股票记录", len(df))

            count = 0
            with sqlite3.connect(self.db_path) as conn:
                for _, row in df.iterrows():
                    code = str(row["代码"]).zfill(6)
                    name = str(row["名称"])
                    market = self._guess_market(code)

                    # 拼音首字母: 平安银行 → payh
                    initials = "".join(
                        p[0].upper() for p in lazy_pinyin(name, style=Style.NORMAL)
                    )
                    # 全拼: 平安银行 → pinganyinhang
                    full_py = "".join(lazy_pinyin(name, style=Style.NORMAL))

                    conn.execute(
                        """INSERT OR REPLACE INTO stocks
                           (code, name, market, py_initials, full_py, updated_at)
                           VALUES (?, ?, ?, ?, ?, ?)""",
                        (code, name, market, initials, full_py,
                         datetime.now(timezone.utc).isoformat()),
                    )
                    count += 1

                # 重建 FTS5 索引（外部内容表需要手动同步）
                conn.execute(
                    "INSERT INTO stock_search(stock_search) VALUES('rebuild')"
                )

            logger.info("知识库同步完成，共 %d 只股票", count)
            return count

        except ImportError:
            logger.error("缺少依赖: pip install akshare pypinyin")
            raise
        except Exception:
            logger.exception("知识库同步失败")
            raise

    # ── 混合搜索 ──

    def search(self, query: str, limit: int = 10) -> list[dict]:
        """多字段混合搜索

        策略：
        - 纯数字     → code LIKE 前缀匹配
        - 纯 ASCII   → FTS5 MATCH 搜拼音字段 + LIKE 回退
        - 含中文     → name LIKE 模糊匹配

        Returns:
            list[dict]: [{"code": "000001", "name": "平安银行", "market": "sz"}, ...]
        """
        query = query.strip()
        if not query:
            return []

        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row

            if query.isdigit():
                rows = self._search_by_code(conn, query, limit)

            elif query.isascii():
                rows = self._search_by_pinyin(conn, query, limit)

            else:
                rows = self._search_by_name(conn, query, limit)

        return [
            {"code": r["code"], "name": r["name"], "market": r["market"]}
            for r in rows
        ]

    def _search_by_code(self, conn: sqlite3.Connection, query: str, limit: int):
        """代码前缀匹配"""
        return conn.execute(
            "SELECT code, name, market FROM stocks WHERE code LIKE ? LIMIT ?",
            (f"{query}%", limit),
        ).fetchall()

    def _search_by_pinyin(self, conn: sqlite3.Connection, query: str, limit: int):
        """FTS5 拼音搜索 + LIKE 回退"""
        query_upper = query.upper()

        # 尝试 FTS5 前缀搜索
        try:
            fts_query = f'"{query_upper}"*'
            rows = conn.execute(
                """SELECT s.code, s.name, s.market
                   FROM stock_search ss
                   JOIN stocks s ON ss.rowid = s.rowid
                   WHERE stock_search MATCH ?
                   ORDER BY rank
                   LIMIT ?""",
                (fts_query, limit),
            ).fetchall()
        except sqlite3.OperationalError:
            rows = []

        # FTS5 无结果时回退到 LIKE
        if not rows:
            rows = conn.execute(
                """SELECT code, name, market FROM stocks
                   WHERE py_initials LIKE ? ESCAPE '\\'
                      OR full_py LIKE ? ESCAPE '\\'
                   LIMIT ?""",
                (f"{query_upper}%", f"{query}%", limit),
            ).fetchall()

        return rows

    def _search_by_name(self, conn: sqlite3.Connection, query: str, limit: int):
        """中文名称模糊匹配"""
        return conn.execute(
            "SELECT code, name, market FROM stocks WHERE name LIKE ? LIMIT ?",
            (f"%{query}%", limit),
        ).fetchall()

    # ── 校验 ──

    def get_by_code(self, code: str) -> Optional[dict]:
        """根据代码查询单只股票"""
        code = code.strip().zfill(6)
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            row = conn.execute(
                "SELECT code, name, market FROM stocks WHERE code = ?",
                (code,),
            ).fetchone()
        if row is None:
            return None
        return {"code": row["code"], "name": row["name"], "market": row["market"]}

    def get_by_name(self, name: str) -> Optional[dict]:
        """根据名称精确查询单只股票"""
        name = name.strip()
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            row = conn.execute(
                "SELECT code, name, market FROM stocks WHERE name = ?",
                (name,),
            ).fetchone()
        if row is None:
            return None
        return {"code": row["code"], "name": row["name"], "market": row["market"]}

    # ── 工具 ──

    @staticmethod
    def _guess_market(code: str) -> str:
        """根据代码判断市场"""
        code = code.zfill(6)
        if code.startswith(("6", "9")):
            return "sh"
        elif code.startswith(("4", "8")):
            return "bj"
        return "sz"

    def is_empty(self) -> bool:
        """知识库是否为空"""
        with sqlite3.connect(self.db_path) as conn:
            row = conn.execute("SELECT COUNT(*) FROM stocks").fetchone()
            return row[0] == 0

    @property
    def count(self) -> int:
        """知识库股票数量"""
        with sqlite3.connect(self.db_path) as conn:
            row = conn.execute("SELECT COUNT(*) FROM stocks").fetchone()
            return row[0]