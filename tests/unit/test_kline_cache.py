"""KLineCache 单元测试

覆盖 (2026-10-14 复权口径审计发现的两个 bug):
  - 主键必须含 adjust 维度, 否则多种复权方式并存时会静默互覆
  - 旧库 (主键为 (symbol,date)) 能正确迁移且数据零丢失
  - load/save 按 adjust 严格隔离

设计原则: 全部用例使用 tmp_path 构造临时库, 不触碰真实的
data/kline_cache.db (该库重建需远程拉取, 离线环境会丢数据).
"""
import sqlite3

import pandas as pd
import pytest

from src.data_fetcher.cache import KLineCache


def _df(close_values, dates=None):
    """构造标准列 K 线 DataFrame"""
    n = len(close_values)
    dates = dates or [f"2026-01-{i + 1:02d}" for i in range(n)]
    return pd.DataFrame({
        "date": dates,
        "open": close_values,
        "high": close_values,
        "low": close_values,
        "close": close_values,
        "volume": [1000.0] * n,
        "amount": [10000.0] * n,
    })


@pytest.fixture
def cache(tmp_path):
    return KLineCache(db_path=str(tmp_path / "test_cache.db"))


def _seq_dates(n, start="2026-01-01"):
    """生成 n 个连续有效日期 (跨月安全, 避免 2026-01-32 这类非法日期)"""
    return [(pd.Timestamp(start) + pd.Timedelta(days=i)).strftime("%Y-%m-%d")
            for i in range(n)]


# ── 主键维度 ──

class TestPrimaryKeyHasAdjust:
    def test_new_db_pk_includes_adjust(self, cache):
        """新建库的主键必须含 adjust"""
        with sqlite3.connect(cache.db_path) as conn:
            assert KLineCache._pk_has_adjust(conn) is True

    def test_different_adjust_coexist_without_overwrite(self, cache):
        """同一 (symbol,date) 下 qfq 与 hfq 必须并存, 互不覆写

        这是本次修复的核心: 修复前主键为 (symbol,date),
        第二个 save 会把第一个整行替换掉.
        """
        cache.save("600887", _df([1.0, 2.0]), adjust="qfq")
        cache.save("600887", _df([9.0, 9.5]), adjust="hfq")

        q = cache.load("600887", adjust="qfq")
        h = cache.load("600887", adjust="hfq")

        assert q is not None and list(q["close"]) == [1.0, 2.0]
        assert h is not None and list(h["close"]) == [9.0, 9.5]

    def test_load_does_not_leak_across_adjust(self, cache):
        """load(adjust='qfq') 不得读到 hfq 的行"""
        cache.save("600887", _df([9.0]), adjust="hfq")
        assert cache.load("600887", adjust="qfq") is None

    def test_same_adjust_overwrite_still_works(self, cache):
        """同一 adjust 重复保存仍应覆盖 (而非报错)"""
        cache.save("600887", _df([1.0]), adjust="qfq")
        cache.save("600887", _df([3.0]), adjust="qfq")
        assert list(cache.load("600887", adjust="qfq")["close"]) == [3.0]


# ── 旧库迁移 ──

class TestLegacyMigration:
    def _make_legacy_db(self, path):
        """构造修复前的旧库: 主键 (symbol,date) + 通过 ALTER 补的 adjust 列"""
        with sqlite3.connect(path) as conn:
            conn.execute("""
                CREATE TABLE kline_cache (
                    symbol TEXT NOT NULL,
                    date TEXT NOT NULL,
                    open REAL, high REAL, low REAL, close REAL,
                    volume REAL, amount REAL,
                    updated_at TEXT DEFAULT (datetime('now')),
                    adjust TEXT NOT NULL DEFAULT 'qfq',
                    PRIMARY KEY (symbol, date)
                )
            """)
            conn.executemany(
                "INSERT INTO kline_cache (symbol,date,open,high,low,close,volume,amount,adjust)"
                " VALUES (?,?,?,?,?,?,?,?,?)",
                [("600887", d, 1.0, 1.0, 1.0, float(i + 1), 1.0, 1.0, "qfq")
                 for i, d in enumerate(_seq_dates(50))],
            )

    def test_legacy_db_migrated_with_zero_loss(self, tmp_path):
        """旧库迁移后: 主键含 adjust 且行数不丢"""
        path = tmp_path / "legacy.db"
        self._make_legacy_db(str(path))

        c = KLineCache(db_path=str(path))

        with sqlite3.connect(str(path)) as conn:
            assert KLineCache._pk_has_adjust(conn) is True
            assert conn.execute("SELECT COUNT(*) FROM kline_cache").fetchone()[0] == 50

        loaded = c.load("600887", adjust="qfq")
        assert loaded is not None and len(loaded) == 50
        assert list(loaded["close"]) == [float(i + 1) for i in range(50)]
        assert list(loaded["date"]) == _seq_dates(50)

    def test_legacy_db_keeps_backup_table(self, tmp_path):
        """旧表改名留存, 不 DROP (便于回滚)"""
        path = tmp_path / "legacy.db"
        self._make_legacy_db(str(path))

        KLineCache(db_path=str(path))

        with sqlite3.connect(str(path)) as conn:
            names = [r[0] for r in conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name LIKE 'kline_cache__old%'"
            )]
        assert len(names) == 1

    def test_migration_is_idempotent(self, tmp_path):
        """重复打开已迁移的库不产生二次迁移/重复备份"""
        path = tmp_path / "legacy.db"
        self._make_legacy_db(str(path))

        KLineCache(db_path=str(path))
        KLineCache(db_path=str(path))
        KLineCache(db_path=str(path))

        with sqlite3.connect(str(path)) as conn:
            backups = [r[0] for r in conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name LIKE 'kline_cache__old%'"
            )]
            assert len(backups) == 1
            assert conn.execute("SELECT COUNT(*) FROM kline_cache").fetchone()[0] == 50

    def test_migrated_db_supports_multi_adjust(self, tmp_path):
        """迁移后的库能正确并存多种复权 (修复的最终目的)"""
        path = tmp_path / "legacy.db"
        self._make_legacy_db(str(path))

        c = KLineCache(db_path=str(path))
        c.save("600887", _df([9.0]), adjust="hfq")

        assert list(c.load("600887", adjust="qfq")["close"]) == [float(i + 1) for i in range(50)]
        assert list(c.load("600887", adjust="hfq")["close"]) == [9.0]


# ── 其余行为 ──

class TestBasics:
    def test_save_empty_df_returns_zero(self, cache):
        assert cache.save("600887", pd.DataFrame()) == 0

    def test_load_missing_symbol_returns_none(self, cache):
        assert cache.load("000000", adjust="qfq") is None

    def test_date_range_respects_adjust(self, cache):
        """日期范围按 adjust 隔离"""
        cache.save("600887", _df([1.0], dates=["2026-03-01"]), adjust="qfq")
        cache.save("600887", _df([9.0], dates=["2026-06-01"]), adjust="hfq")

        assert cache.get_cached_date_range("600887", "qfq") == ("2026-03-01", "2026-03-01")
        assert cache.get_cached_date_range("600887", "hfq") == ("2026-06-01", "2026-06-01")
