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
            conn.execute("""
                CREATE TABLE IF NOT EXISTS kline_cache (
                    symbol TEXT NOT NULL,
                    date TEXT NOT NULL,
                    open REAL,
                    high REAL,
                    low REAL,
                    close REAL,
                    volume REAL,
                    amount REAL,
                    updated_at TEXT DEFAULT (datetime('now')),
                    PRIMARY KEY (symbol, date)
                )
            """)
            conn.execute("CREATE INDEX IF NOT EXISTS idx_symbol_date ON kline_cache(symbol, date)")

    def save(self, symbol: str, df: pd.DataFrame) -> int:
        """保存K线数据到缓存，返回写入行数"""
        if df is None or df.empty:
            return 0

        rows = df.where(df.notna(), None).to_dict("records")
        count = 0
        with sqlite3.connect(self.db_path) as conn:
            for row in rows:
                conn.execute(
                    """INSERT OR REPLACE INTO kline_cache
                       (symbol, date, open, high, low, close, volume, amount, updated_at)
                       VALUES (?, ?, ?, ?, ?, ?, ?, ?, datetime('now'))""",
                    (
                        symbol,
                        str(row.get("date", "")),
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
        self, symbol: str, start_date: str = None, end_date: str = None
    ) -> Optional[pd.DataFrame]:
        """从缓存加载K线数据"""
        query = "SELECT date, open, high, low, close, volume, amount FROM kline_cache WHERE symbol = ?"
        params = [symbol]

        if start_date:
            query += " AND date >= ?"
            params.append(start_date.replace("-", ""))
        if end_date:
            query += " AND date <= ?"
            params.append(end_date.replace("-", ""))

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