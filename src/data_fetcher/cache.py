"""SQLite 本地缓存层

缓存已获取的K线数据，避免重复请求。同时作为远程数据源全部不可用时的兜底方案。
"""
import sqlite3
import json
import os
from typing import Optional
import pandas as pd

from src.config.settings import CACHE_DB


class KLineCache:
    """K线数据 SQLite 缓存"""

    def __init__(self, db_path: str = CACHE_DB):
        self.db_path = db_path
        os.makedirs(os.path.dirname(db_path), exist_ok=True)
        self._init_db()

    # 表结构定义 (单一事实来源, 建表与迁移共用, 避免两处定义漂移)
    # 注意: 必须保留 IF NOT EXISTS —— 存量库已存在该表, 缺此子句会报
    # "table kline_cache already exists" (迁移场景下必然命中).
    _SCHEMA = """
        CREATE TABLE IF NOT EXISTS kline_cache (
            symbol TEXT NOT NULL,
            date TEXT NOT NULL,
            adjust TEXT NOT NULL DEFAULT 'qfq',
            open REAL,
            high REAL,
            low REAL,
            close REAL,
            volume REAL,
            amount REAL,
            updated_at TEXT DEFAULT (datetime('now')),
            PRIMARY KEY (symbol, date, adjust)
        )
    """

    def _init_db(self):
        with sqlite3.connect(self.db_path) as conn:
            # 首次运行: 建表 (含 adjust 复权维度)
            conn.execute(self._SCHEMA)

            # 迁移 1: 旧表无 adjust 列时, 用 ALTER 加列保留存量数据
            cols = [r[1] for r in conn.execute("PRAGMA table_info(kline_cache)")]
            if "adjust" not in cols:
                conn.execute(
                    "ALTER TABLE kline_cache ADD COLUMN adjust TEXT NOT NULL DEFAULT 'qfq'"
                )

            # 迁移 2: 旧表主键缺 adjust 维度时, 重建表保留数据
            # 背景: 存量库由旧版 CREATE TABLE 建立, 主键为 (symbol, date);
            # 后续 ALTER 加列无法补主键, 导致 CREATE TABLE IF NOT EXISTS 静默不生效.
            # 危害: 并存多种复权方式时, INSERT OR REPLACE 按旧主键冲突,
            #       后写入的 adjust 会直接覆写先写入的, load(adjust=...) 读到错误数据.
            if not self._pk_has_adjust(conn):
                self._migrate_primary_key(conn)

            conn.execute("CREATE INDEX IF NOT EXISTS idx_symbol_date ON kline_cache(symbol, date)")

            # 公司行动事件表 (除权除息): 与行情同库, 便于按 (symbol, date) join.
            # DDL 唯一定义在 event_book.CORPORATE_ACTION_SCHEMA, 此处仅复用,
            # 保证"事件表存在"这件事由 KLineCache 初始化一并兜住.
            from .event_book import EventBook

            EventBook.ensure_table(conn)

    @staticmethod
    def _pk_has_adjust(conn) -> bool:
        """检测 kline_cache 主键是否含 adjust 列

        PRAGMA table_info 返回 (cid, name, type, notnull, dflt_value, pk),
        其中 pk 为 0 表示非主键, >0 表示主键中的第几列.
        """
        for row in conn.execute("PRAGMA table_info(kline_cache)"):
            if row[5] > 0 and row[1] == "adjust":
                return True
        return False

    def _migrate_primary_key(self, conn) -> None:
        """将主键由 (symbol, date) 重建为 (symbol, date, adjust), 保留全部存量数据

        SQLite 的 ALTER TABLE 无法修改主键, 只能"建新表 → 搬迁 → 改名".
        安全措施:
          - 全程在同一事务内, 任一步失败则整体回滚, 不产生半截状态
          - 搬迁前统计去重后的行数, 搬迁后校验一致, 不一致则抛错回滚
          - 旧表不 DROP, 改名留存为 kline_cache__old_<ts>, 可随时回滚
        """
        from datetime import datetime

        tmp = "kline_cache__new"
        backup = f"kline_cache__old_{datetime.now().strftime('%Y%m%d%H%M%S')}"

        # 脏数据兜底: 上次迁移中断可能残留临时表
        conn.execute(f"DROP TABLE IF EXISTS {tmp}")

        # 临时表用不带 IF NOT EXISTS 的形式: 刚 DROP 过, 若仍存在说明有异常, 应暴露而非静默复用
        conn.execute(self._SCHEMA.replace(
            "CREATE TABLE IF NOT EXISTS kline_cache", f"CREATE TABLE {tmp}"))

        # 搬迁 (显式列名, 不依赖列顺序; OR IGNORE 防重复键中断)
        conn.execute(f"""
            INSERT OR IGNORE INTO {tmp}
                (symbol, date, adjust, open, high, low, close, volume, amount, updated_at)
            SELECT symbol, date, adjust, open, high, low, close, volume, amount, updated_at
            FROM kline_cache
        """)

        # 校验: 新表行数必须等于去重后的 (symbol, date, adjust) 组合数
        expected = conn.execute(
            "SELECT COUNT(*) FROM (SELECT DISTINCT symbol, date, adjust FROM kline_cache)"
        ).fetchone()[0]
        actual = conn.execute(f"SELECT COUNT(*) FROM {tmp}").fetchone()[0]
        if actual != expected:
            raise RuntimeError(
                f"kline_cache 主键迁移数据校验失败: 期望 {expected} 行, 实际 {actual} 行, 已回滚"
            )

        conn.execute(f"ALTER TABLE kline_cache RENAME TO {backup}")
        conn.execute(f"ALTER TABLE {tmp} RENAME TO kline_cache")

    def save(self, symbol: str, df: pd.DataFrame, adjust: str = "qfq") -> int:
        """保存K线数据到缓存，返回写入行数"""
        if df is None or df.empty:
            return 0

        rows = df.where(df.notna(), None).to_dict("records")
        count = 0
        with sqlite3.connect(self.db_path) as conn:
            for row in rows:
                conn.execute(
                    """INSERT OR REPLACE INTO kline_cache
                       (symbol, date, adjust, open, high, low, close, volume, amount, updated_at)
                       VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, datetime('now'))""",
                    (
                        symbol,
                        str(row.get("date", "")),
                        adjust,
                        row.get("open"),
                        row.get("high"),
                        row.get("low"),
                        row.get("close"),
                        row.get("volume"),
                        row.get("amount"),
                    ),
                )
                count += 1
        return count

    def load(
        self, symbol: str, start_date: str = None, end_date: str = None,
        adjust: str = "qfq",
    ) -> Optional[pd.DataFrame]:
        """从缓存加载K线数据 (按复权方式隔离, 避免不同 adjust 命中错误缓存)"""
        query = "SELECT date, open, high, low, close, volume, amount FROM kline_cache WHERE symbol = ? AND adjust = ?"
        params = [symbol, adjust]

        if start_date:
            query += " AND date >= ?"
            params.append(start_date)  # 缓存中日期格式为 YYYY-MM-DD (带横线)
        if end_date:
            query += " AND date <= ?"
            params.append(end_date)

        query += " ORDER BY date ASC"

        with sqlite3.connect(self.db_path) as conn:
            df = pd.read_sql_query(query, conn, params=params)

        if df.empty:
            return None

        df["date"] = pd.to_datetime(df["date"]).dt.strftime("%Y-%m-%d")
        for col in ["open", "high", "low", "close", "volume", "amount"]:
            df[col] = pd.to_numeric(df[col], errors="coerce")
        return df

    def get_cached_date_range(self, symbol: str, adjust: str = "qfq") -> tuple:
        """获取某只股票在缓存中的日期范围 (最早, 最晚)

        Args:
            adjust: 复权方式, 与 load/save 保持同一维度. 不传时默认 qfq (向后兼容).
        """
        with sqlite3.connect(self.db_path) as conn:
            row = conn.execute(
                "SELECT MIN(date), MAX(date) FROM kline_cache WHERE symbol = ? AND adjust = ?",
                (symbol, adjust),
            ).fetchone()
        if row and row[0]:
            return (row[0], row[1])
        return (None, None)