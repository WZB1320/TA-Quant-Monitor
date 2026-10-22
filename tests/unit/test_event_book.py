"""EventBook（公司行动事件表）单元测试

覆盖:
  - 建表幂等、与 KLineCache 同库共存
  - 事件读写与区间查询
  - 除权调整: 纯派息 / 送转 / 派息+送转 三种形态
  - 除权参考价（涨跌停基准）
  - 除权日不误止损（成本基准同步下移的意义）
  - 红利税三档
  - 过期哨兵

全部用例使用 tmp_path 构造临时库, 不触碰真实 data/kline_cache.db.
"""
import sqlite3
from datetime import datetime, timedelta

import pytest

from src.data_fetcher.cache import KLineCache
from src.data_fetcher.event_book import (
    CorporateAction,
    EventBook,
    check_staleness,
    dividend_tax_rate,
)


@pytest.fixture
def book(tmp_path):
    return EventBook(db_path=str(tmp_path / "test_cache.db"))


def _a(**kw) -> CorporateAction:
    """构造事件: 默认纯派息"""
    base = dict(symbol="600887", ex_date="2025-06-06", dps_pre=1.22, dps_after=1.098)
    base.update(kw)
    return CorporateAction(**base)


# ── 建表 ──

class TestSchema:
    def test_table_created_on_init(self, book):
        with sqlite3.connect(book.db_path) as conn:
            names = [r[0] for r in conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table'")]
        assert "corporate_action" in names

    def test_init_is_idempotent(self, tmp_path):
        path = str(tmp_path / "a.db")
        EventBook(path).upsert(_a())
        EventBook(path)
        EventBook(path)
        assert EventBook(path).count() == 1

    def test_coexists_with_kline_cache(self, tmp_path):
        """事件表与行情同库: KLineCache 初始化应一并建出事件表"""
        path = str(tmp_path / "b.db")
        KLineCache(db_path=path)
        with sqlite3.connect(path) as conn:
            names = [r[0] for r in conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table'")]
        assert "kline_cache" in names and "corporate_action" in names


# ── 读写 ──

class TestReadWrite:
    def test_upsert_and_get(self, book):
        book.upsert(_a())
        got = book.get("600887", "2025-06-06")
        assert got is not None
        assert got.dps_pre == pytest.approx(1.22)
        assert got.dps_after == pytest.approx(1.098)

    def test_upsert_same_key_overwrites(self, book):
        book.upsert(_a(dps_pre=1.0))
        book.upsert(_a(dps_pre=2.0))
        assert book.count() == 1
        assert book.get("600887", "2025-06-06").dps_pre == pytest.approx(2.0)

    def test_events_between_inclusive_and_sorted(self, book):
        book.upsert_many([
            _a(ex_date="2025-06-06"),
            _a(ex_date="2024-06-05"),
            _a(ex_date="2026-06-05"),
            _a(ex_date="2023-06-13"),
        ])
        got = book.events_between("600887", "2024-01-01", "2025-12-31")
        assert [a.ex_date for a in got] == ["2024-06-05", "2025-06-06"]

    def test_checked_at_defaults_to_now(self, book):
        book.upsert(_a())
        assert book.get("600887", "2025-06-06").checked_at is not None


# ── 除权调整: 核心计算 ──

class TestApplyToPosition:
    def test_cash_dividend_only(self, book):
        """纯派息: 股数不变, 成本基准与最高价同步下移 D

        前收 28.62, 每股派 1.22 → 除权参考价 27.40, 成本基准同幅下移,
        故除权日的价格跳空不会被误判为暴跌.
        """
        book.upsert(_a(share_ratio=0, transfer_ratio=0))
        shares, ref, high, cash, s = book.apply_to_position(
            "600887", "2025-06-06", shares=1000, ref_price=28.62, highest_price=30.0
        )
        assert shares == 1000
        assert ref == pytest.approx(28.62 - 1.22)
        assert high == pytest.approx(30.0 - 1.22)
        assert cash == pytest.approx(1.22 * 1000)   # 含税基数
        assert s == 0.0

    def test_share_split_uses_after_tax_dps(self, book):
        """含送转: 派息基数改用税后 dps_after（送股按面值代扣的税已内扣）

        公告口径: 送股/转增 0.3, 每股派 0.45(含税) / 0.405(税后)
        """
        book.upsert(_a(share_ratio=0.3, transfer_ratio=0.0, dps_pre=0.45, dps_after=0.405))
        shares, ref, high, cash, s = book.apply_to_position(
            "600887", "2025-06-06", shares=1000, ref_price=166.96, highest_price=170.0
        )
        assert shares == 1300
        assert s == pytest.approx(0.3)
        assert cash == pytest.approx(0.405 * 1000)     # 税后基数
        assert ref == pytest.approx((166.96 - 0.405) / 1.3)
        assert high == pytest.approx((170.0 - 0.405) / 1.3)

    def test_dividend_uses_pre_split_shares(self, book):
        """送转新增的股份不参与本次派息: 现金按除权日持有股数计算"""
        book.upsert(_a(share_ratio=1.0, dps_pre=0.45, dps_after=0.405))
        _, _, _, cash, _ = book.apply_to_position(
            "600887", "2025-06-06", shares=1000, ref_price=10.0
        )
        assert cash == pytest.approx(1000 * 0.405)     # 不是 2000 股

    def test_no_event_returns_unchanged(self, book):
        shares, ref, high, cash, s = book.apply_to_position(
            "600887", "2025-07-01", shares=1000, ref_price=10.0, highest_price=11.0
        )
        assert (shares, ref, high, cash, s) == (1000, 10.0, 11.0, 0.0, 0.0)

    def test_ex_div_day_does_not_false_trigger_stop_loss(self, book):
        """除权日不误止损: 成本基准下移后, 派息造成的跳空不再触及硬止损

        场景: 成本 28.62, 硬止损 -12% → 25.19. 除权日派 1.22, 价格跌到 24.30.
        - 若 ref_price 不调整: 24.30 <= 25.19 → 误触发止损
        - 调整后: ref = 27.40, 止损线 24.11, 24.30 > 24.11 → 不触发
        """
        book.upsert(_a())
        _, ref, _, _, _ = book.apply_to_position(
            "600887", "2025-06-06", shares=1000, ref_price=28.62
        )
        price_after_ex = 24.30
        stop_unadjusted = 28.62 * (1 - 0.12)
        stop_adjusted = ref * (1 - 0.12)
        assert price_after_ex <= stop_unadjusted      # 未调整会误触发
        assert price_after_ex > stop_adjusted         # 调整后不触发


    def test_multiplier_takes_priority_for_cost_basis(self, book):
        """有市场乘数时, 成本基准按 ref/mult 下移 (保证 ref/price 比值连续)

        纯派息且 multiplier 与公告口径基本一致时, ref/mult 与 ref−D 应几乎相等。
        """
        # 前收 28.62, 派 1.22 → mult = 28.62 / 27.42
        mult = 28.62 / 27.42
        book.upsert(_a(multiplier=mult))
        shares, ref, high, cash, _ = book.apply_to_position(
            "600887", "2025-06-06", shares=1000, ref_price=28.62, highest_price=30.0
        )
        assert ref == pytest.approx(27.42, abs=1e-6)
        assert high == pytest.approx(30.0 / mult)

    def test_cash_uses_announced_dps_not_price_adjustment(self, book):
        """回购股不参与分配: 现金按公告值, 价格调整按市场乘数 —— 两者不等

        实测 600519 2025-06-26: 公告每股派 27.6730, 行情价格调整 27.6007
        （差额 0.0723 元是回购股放弃的那份, 由其余股东无偿获得, 真实存在）。
        现金账必须用公告值, 否则会少记股东实际到账的钱。
        """
        prev, d_announced, d_price = 1435.86, 27.6730, 27.6007
        mult = prev / (prev - d_price)
        book.upsert(_a(symbol="600519", ex_date="2025-06-26", dps_pre=d_announced,
                       dps_after=d_announced * 0.9, multiplier=mult))
        _, ref, _, cash, _ = book.apply_to_position(
            "600519", "2025-06-26", shares=100, ref_price=1400.0, highest_price=1500.0
        )
        assert cash == pytest.approx(100 * d_announced)      # 公告口径
        assert ref == pytest.approx(1400.0 / mult)           # 市场口径
        assert ref != pytest.approx(1400.0 - d_announced)    # 两者确实不同


# ── 除权参考价（涨跌停基准） ──

class TestAdjustedPrevClose:
    def test_ex_day_uses_reference_price(self, book):
        book.upsert(_a(dps_pre=1.20, share_ratio=0.0))
        # 前收 28.62, 派 1.20 → 除权参考价 27.42
        assert book.adjusted_prev_close("600887", "2025-06-06", 28.62) == pytest.approx(27.42)

    def test_ex_day_with_split(self, book):
        book.upsert(_a(dps_pre=0.45, share_ratio=0.4))
        # 交易所公布的除权参考价按"分"四舍五入
        assert book.adjusted_prev_close("600887", "2025-06-06", 166.96) == pytest.approx(
            round((166.96 - 0.45) / 1.4, 2)
        )

    def test_normal_day_unchanged(self, book):
        book.upsert(_a())
        assert book.adjusted_prev_close("600887", "2025-07-01", 28.62) == pytest.approx(28.62)


# ── 事件属性 ──

class TestActionProperties:
    def test_s_is_share_plus_transfer(self):
        a = _a(share_ratio=0.2, transfer_ratio=0.3)
        assert a.s == pytest.approx(0.5)

    def test_ex_reference_price(self):
        a = _a(dps_pre=1.0, share_ratio=0.0)
        assert a.ex_reference_price(25.0) == pytest.approx(24.0)


# ── 红利税 ──

class TestDividendTaxRate:
    @pytest.mark.parametrize("days,expected", [
        (1, 0.20), (30, 0.20),        # ≤1个月(含) 20%
        (31, 0.10), (365, 0.10),      # 1个月~1年(含) 10%
        (366, 0.0), (1000, 0.0),      # >1年 0%
    ])
    def test_three_tiers(self, days, expected):
        assert dividend_tax_rate(days) == pytest.approx(expected)

    def test_zero_percent_never_triggers_in_this_strategy(self):
        """基线 229 笔最长持仓 103 天 → 0% 档本策略永不触发"""
        assert dividend_tax_rate(103) == pytest.approx(0.10)


# ── 过期哨兵 ──

class TestStaleness:
    def test_missing_symbols(self, book):
        book.upsert(_a())
        st = book.staleness(["600887", "300308"])
        assert st["missing"] == ["300308"]
        assert st["count"] == 1

    def test_fresh_record_not_stale(self, book):
        book.upsert(_a())
        assert check_staleness(["600887"], db_path=book.db_path) is None

    def test_aged_record_warns(self, book):
        old = (datetime.now() - timedelta(days=200)).strftime("%Y-%m-%d %H:%M:%S")
        book.upsert(_a(checked_at=old))
        msg = check_staleness(["600887"], db_path=book.db_path)
        assert msg is not None and "200" in msg

    def test_warn_does_not_raise(self, book):
        """哨兵只告警不阻断"""
        assert check_staleness(["000001"], db_path=book.db_path) is not None
