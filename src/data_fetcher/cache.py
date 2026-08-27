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

    def _init_db(self):
        with sqlite3.connect(self.db_path) as conn:
            # 首次运行: 建表 (含 adjust 复权维度)
            conn.execute("""
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
            """)
            # 迁移: 旧表无 adjust 维度时, 用 ALTER 加列保留存量数据
            # (严禁 DROP 重建, 会清空缓存; 缓存重建需远程拉取, 离线环境会丢数据)
            cols = [r[1] for r in conn.execute("PRAGMA table_info(kline_cache)")]
            if "adjust" not in cols:
                conn.execute(
                    "ALTER TABLE kline_cache ADD COLUMN adjust TEXT NOT NULL DEFAULT 'qfq'"
                )
            conn.execute("CREATE INDEX IF NOT EXISTS idx_symbol_date ON kline_cache(symbol, date)")

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

    def get_cached_date_range(self, symbol: str) -> tuple:
        """获取某只股票在缓存中的日期范围 (最早, 最晚)"""
        with sqlite3.connect(self.db_path) as conn:
            row = conn.execute(
                "SELECT MIN(date), MAX(date) FROM kline_cache WHERE symbol = ?",
                (symbol,),
            ).fetchone()
        if row and row[0]:
            return (row[0], row[1])
        return (None, None)