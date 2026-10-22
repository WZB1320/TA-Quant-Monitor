"""M2 成交层真实账本 — 除权除息 / 红利税 / 比率接口 单元测试

覆盖 plan §6 风险 R1~R4:
  R1 除权处理顺序错误 → 除权日误止损
  R2 highest_price 未除权调整 → 移动止盈误触发
  R3 用"扣税后"值当派息基数 → 税率被锁死在 10%
  R4 减仓场景分红未按比例拆分

以及"比率接口"的尺度不变性 (ATR/close 而非 ATR 元).
"""
from datetime import date

import pytest

from src.backtest.position import PositionManager, Trade, Side
from src.data_fetcher.event_book import CorporateAction, dividend_tax_rate


# ── 工具 ──

def ev(ex_date, dps_pre=0.0, dps_after=None, share_ratio=0.0,
       transfer_ratio=0.0, multiplier=None):
    """构造一条事件 (dps_after 默认 = dps_pre, 测试中显式给税率时另传)"""
    return CorporateAction(
        symbol="TEST", ex_date=ex_date,
        dps_pre=dps_pre,
        dps_after=dps_pre if dps_after is None else dps_after,
        share_ratio=share_ratio, transfer_ratio=transfer_ratio,
        multiplier=multiplier,
    )


@pytest.fixture
def pm():
    """risk_per_trade=0.05 → 在 price=100 时开 300 股 (3万), 刚好触及单票上限"""
    return PositionManager(
        initial_capital=100_000,
        position_ratio=0.30,
        commission_rate=0.00025,
        risk_per_trade=0.05,
        atr_stop_mult=2.5,
    )


def open_at(pm, price=100.0, ratio=0.05, day=date(2026, 1, 5)):
    return pm.open_long("TEST", day, price, "BUY", atr_ratio=ratio)


# ── 红利税三档 ──

class TestDividendTaxRate:
    def test_three_tiers(self):
        """财税[2015]101号: <=1月 20%, ~1年 10%, >1年 0%"""
        assert dividend_tax_rate(1) == 0.20
        assert dividend_tax_rate(30) == 0.20      # 含 1 个月
        assert dividend_tax_rate(31) == 0.10
        assert dividend_tax_rate(365) == 0.10     # 含 1 年
        assert dividend_tax_rate(366) == 0.0

    def test_tax_deducted_on_close_not_on_ex_date(self, pm):
        """派息时不扣缴, 平仓时才扣 —— 除权日现金流是**含税**全额"""
        t = open_at(pm)
        shares = t.shares
        pm._event_map = {"TEST": {"2026-02-02": ev("2026-02-02", dps_pre=2.0,
                                                   multiplier=100 / 98)}}
        cash_before = pm.cash
        credited = pm.apply_corporate_actions(date(2026, 2, 2))
        # 除权日: 全额含税入账
        assert credited == {"TEST": pytest.approx(2.0 * shares)}
        assert pm.cash - cash_before == pytest.approx(2.0 * shares)
        assert t.dividend_gross == pytest.approx(2.0 * shares)
        assert t.dividend_tax == 0.0        # 此刻尚未卖出, 税未定

        # 持有 100 天卖出 → 10% 档
        pm.close_position("TEST", date(2026, 5, 15), 98.0, "SELL")
        assert t.holding_days > 30
        assert t.dividend_tax == pytest.approx(2.0 * shares * 0.10)

    def test_short_holding_pays_20pct(self, pm):
        t = open_at(pm, day=date(2026, 1, 5))
        shares = t.shares
        pm._event_map = {"TEST": {"2026-01-20": ev("2026-01-20", dps_pre=1.0,
                                                   multiplier=100 / 99)}}
        pm.apply_corporate_actions(date(2026, 1, 20))
        pm.close_position("TEST", date(2026, 1, 25), 99.0, "SELL")
        assert t.holding_days == 20
        assert t.dividend_tax == pytest.approx(1.0 * shares * 0.20)


# ── 除权调整 ──

class TestCorporateActionAdjustment:
    def test_cash_dividend_moves_ref_price_and_credits_cash(self, pm):
        t = open_at(pm)
        shares = t.shares
        assert t.ref_price == 100.0            # 无事件时 == entry_price
        pm._event_map = {"TEST": {"2026-02-02": ev("2026-02-02", dps_pre=2.0,
                                                   multiplier=100 / 98)}}
        pm.apply_corporate_actions(date(2026, 2, 2))
        assert t.shares == shares               # 纯派息不改股数
        assert t.ref_price == pytest.approx(98.0)
        assert t.dividend_gross == pytest.approx(2.0 * shares)

    def test_split_increases_shares_and_halves_ref(self, pm):
        """10送10: 股数翻倍, 成本基准减半, 本金不变"""
        t = open_at(pm)
        shares = t.shares
        entry_amount = t.entry_amount
        pm._event_map = {"TEST": {"2026-02-02": ev("2026-02-02", share_ratio=1.0,
                                                   multiplier=2.0)}}
        pm.apply_corporate_actions(date(2026, 2, 2))
        assert t.shares == shares * 2
        assert t.ref_price == pytest.approx(50.0)
        assert t.entry_amount == entry_amount     # 本金不因送股而变

        # 每股 60 元卖出: 市值 shares×2×60, 成本基准 50 → 每股赚 10
        gross = (60.0 - 50.0) * t.shares
        pm.close_position("TEST", date(2026, 3, 2), 60.0, "SELL")
        assert t.pnl == pytest.approx(gross - t.commission)
        assert t.pnl_pct == pytest.approx(t.pnl / entry_amount, abs=1e-9)

    def test_highest_price_is_adjusted(self, pm):
        """R2: highest_price 必须同步下移, 否则除权后移动止盈立即误触发"""
        t = open_at(pm)
        t.highest_price = 120.0
        pm._event_map = {"TEST": {"2026-02-02": ev("2026-02-02", share_ratio=1.0,
                                                   multiplier=2.0)}}
        pm.apply_corporate_actions(date(2026, 2, 2))
        assert t.highest_price == pytest.approx(60.0)


# ── R1: 除权日不得误止损 ──

class TestNoFalseStopOnExDate:
    def test_big_dividend_does_not_trigger_stop(self, pm):
        """派息 15% 后价格自然下移到 85 —— 不得被当成暴跌止损"""
        t = open_at(pm, price=100.0, ratio=0.05)
        pm._event_map = {"TEST": {"2026-02-02": ev("2026-02-02", dps_pre=15.0,
                                                   multiplier=100 / 85)}}
        pm.apply_corporate_actions(date(2026, 2, 2))
        assert t.ref_price == pytest.approx(85.0)
        # 除权日收盘 85 (= 除权参考价), 不应触发任何止损
        assert pm.check_stop_loss("TEST", 85.0, date(2026, 2, 2)) is None

    def test_without_adjustment_it_would_false_trigger(self, pm):
        """反向验证机制有牙: 若 ref_price 不下移, 同一行情必然误止损"""
        t = open_at(pm, price=100.0, ratio=0.05)
        pm._event_map = {"TEST": {"2026-02-02": ev("2026-02-02", dps_pre=15.0,
                                                   multiplier=100 / 85)}}
        pm.apply_corporate_actions(date(2026, 2, 2))
        t.ref_price = 100.0          # 人为还原成"忘记除权调整"的 bug 状态
        closed = pm.check_stop_loss("TEST", 85.0, date(2026, 2, 2))
        assert closed is not None    # 12% 硬止损被误触发

    def test_stop_distance_follows_ref_price(self, pm):
        """ATR 止损距离按 ref_price 现算: 除权后门槛随之下移"""
        t = open_at(pm, price=100.0, ratio=0.05)     # ATR=5, stop=12.5
        pm._event_map = {"TEST": {"2026-02-02": ev("2026-02-02", dps_pre=10.0,
                                                   multiplier=100 / 90)}}
        pm.apply_corporate_actions(date(2026, 2, 2))
        assert t.ref_price == pytest.approx(90.0)
        # 新止损距离 = 0.05 × 2.5 × 90 = 11.25 → 门槛 78.75
        assert pm.check_stop_loss("TEST", 80.0, date(2026, 2, 3)) is None
        assert pm.check_stop_loss("TEST", 78.0, date(2026, 2, 4)) is not None


# ── 记账口径: 分红不得重复计入 ──

class TestNoDoubleCounting:
    def test_total_return_equals_price_return(self):
        """总收益口径与价格口径必须恒等 —— 分红不得因基准下移而被算两次

        总收益: pnl = (exit − ref_price) × shares − 费 − 税   (ref_price 已内含分红)
        价  格: pnl = (exit − entry)    × shares + 分红 − 税 − 费
        二者差额应仅为浮点噪声. 实施中曾因两处都加分红而重复计算 58 元.
        """
        pm = PositionManager(initial_capital=100_000, position_ratio=0.30,
                             commission_rate=0.00025, risk_per_trade=0.05,
                             atr_stop_mult=2.5)
        t = pm.open_long("TEST", date(2026, 1, 5), 100.0, "BUY", atr_ratio=0.05)
        shares = t.shares
        dps = 2.0
        pm._event_map = {"TEST": {"2026-02-02": ev("2026-02-02", dps_pre=dps,
                                                   multiplier=100 / (100 - dps))}}
        pm.apply_corporate_actions(date(2026, 2, 2))
        pm.close_position("TEST", date(2026, 5, 16), 99.0, "SELL")   # >30 天 → 10%

        price_return = ((99.0 - 100.0) * shares + dps * shares
                        - dps * shares * 0.10 - t.commission)
        assert t.pnl == pytest.approx(price_return, abs=0.01)
        assert t.dividend_gross == pytest.approx(dps * shares)
        assert t.dividend_tax == pytest.approx(dps * shares * 0.10)

    def test_dividend_not_added_to_pnl(self, pm):
        """无除权事件时 pnl 里不应出现分红项 (向后兼容)"""
        t = open_at(pm)
        pm.close_position("TEST", date(2026, 3, 5), 110.0, "SELL")
        assert t.pnl == pytest.approx(10.0 * t.shares - t.commission)


# ── R3: 派息基数必须用含税值 ──

class TestDividendBase:
    def test_pure_cash_uses_dps_pre(self, pm):
        """s == 0 时现金以 dps_pre (含税) 入账, 税率由 holding_days 决定"""
        t = open_at(pm)
        shares = t.shares
        pm._event_map = {"TEST": {"2026-02-02": ev(
            "2026-02-02", dps_pre=1.0, dps_after=0.9, multiplier=100 / 99)}}
        pm.apply_corporate_actions(date(2026, 2, 2))
        assert t.dividend_gross == pytest.approx(1.0 * shares)   # 不是 0.9×shares

    def test_with_split_uses_dps_after(self, pm):
        """s > 0 时送股已按面值代扣个税并内扣于派息 → 不再叠加自算税率"""
        t = open_at(pm)
        shares = t.shares
        pm._event_map = {"TEST": {"2026-02-02": ev(
            "2026-02-02", dps_pre=0.45, dps_after=0.405,
            share_ratio=0.4, multiplier=1.4)}}
        pm.apply_corporate_actions(date(2026, 2, 2))
        assert t.dividend_gross == pytest.approx(0.405 * shares)
        # 源头代扣的税 (含税 0.45 − 到账 0.405) 立即记为成本, 不再二次计税
        assert t.dividend_tax == pytest.approx((0.45 - 0.405) * shares)
        assert t.dividend_taxable == 0.0


# ── R4: 减仓时分红按比例拆分 ──

class TestReducePositionSplit:
    def test_dividend_split_proportionally(self, pm):
        t = open_at(pm)
        shares = t.shares
        pm._event_map = {"TEST": {"2026-02-02": ev("2026-02-02", dps_pre=2.0,
                                                   multiplier=100 / 98)}}
        pm.apply_corporate_actions(date(2026, 2, 2))
        assert t.dividend_gross == pytest.approx(2.0 * shares)

        partial = pm.reduce_position("TEST", date(2026, 3, 2), 98.0, reduce_ratio=0.5)
        sold = partial.shares
        assert partial.dividend_gross == pytest.approx(2.0 * shares * (sold / shares))
        assert t.dividend_gross == pytest.approx(2.0 * shares * (1 - sold / shares))
        # 两笔的分红之和 == 原累计分红
        assert partial.dividend_gross + t.dividend_gross == pytest.approx(2.0 * shares)
        # 待税基数同样按比例拆分
        assert partial.dividend_taxable + t.dividend_taxable == pytest.approx(
            partial.dividend_gross + t.dividend_gross)


# ── 比率接口的尺度不变性 ──

class TestAtrRatioInterface:
    def test_ratio_equals_absolute_atr_when_same_scale(self, pm):
        """atr_ratio × price 必须与直接传 atr_value 完全等价"""
        a = pm.open_long("A", date(2026, 1, 5), 100.0, "BUY", atr_ratio=0.05)
        b = PositionManager(initial_capital=100_000, position_ratio=0.30,
                            commission_rate=0.00025, risk_per_trade=0.05,
                            atr_stop_mult=2.5)
        c = b.open_long("A", date(2026, 1, 5), 100.0, "BUY", atr_value=5.0)
        assert a.shares == c.shares

    @staticmethod
    def _mk():
        return PositionManager(initial_capital=100_000, position_ratio=0.30,
                               commission_rate=0.00025, risk_per_trade=0.015,
                               atr_stop_mult=2.5)

    def test_ratio_path_matches_absolute_path_at_any_price(self):
        """atr_ratio × price 与直接传 atr_value 在任意价位都一致"""
        for price in (20.0, 100.0):
            a = self._mk().open_long("X", date(2026, 1, 5), price, "BUY",
                                     atr_ratio=0.05)
            b = self._mk().open_long("X", date(2026, 1, 5), price, "BUY",
                                     atr_value=0.05 * price)
            assert a.shares == b.shares and a.shares > 0

    def test_absolute_atr_from_other_scale_is_wrong(self):
        """反向验证: 沿用另一复权尺度的 ATR(元) 会让仓位系统性错配

        这正是复权改造前"信号层 qfq / 成交层 raw"会踩的坑: ATR 元来自 qfq
        尺度 (~100 元), 却拿去除 raw 尺度 (~20 元) 的风险预算, 仓位差 6 倍.
        """
        right = self._mk().open_long("X", date(2026, 1, 5), 20.0, "BUY",
                                     atr_ratio=0.05)         # ATR=1.0
        wrong = self._mk().open_long("X", date(2026, 1, 5), 20.0, "BUY",
                                     atr_value=5.0)           # 旧尺度的 ATR 元
        assert right.shares > wrong.shares > 0

    def test_atr_ratio_stored_on_trade(self, pm):
        t = open_at(pm, ratio=0.05)
        assert t._atr_ratio == pytest.approx(0.05)


# ── 涨跌停基准: 除权参考价 ──

class TestAdjustedPrevClose:
    def test_uses_multiplier_when_available(self):
        a = ev("2026-06-05", dps_pre=1.2, multiplier=1.043764)
        assert a.adjust_prev_close(28.62) == pytest.approx(round(28.62 / 1.043764, 2))

    def test_falls_back_to_announced_when_no_multiplier(self):
        a = ev("2026-06-05", dps_pre=1.2, share_ratio=0.4)
        assert a.adjust_prev_close(28.62) == pytest.approx(round((28.62 - 1.2) / 1.4, 2))

    def test_rounds_to_fen(self):
        """交易所按分公布除权参考价"""
        a = ev("2026-06-05", dps_pre=1.2, multiplier=1.043764)
        assert a.adjust_prev_close(28.62) == round(a.adjust_prev_close(28.62), 2)


# ── 向后兼容 ──

class TestBackwardCompatibility:
    def test_no_event_map_equals_legacy_behaviour(self, pm):
        """不传 event_map 时 ref_price == entry_price, 所有判定逐位同改造前"""
        t = pm.open_long("TEST", date(2026, 1, 5), 100.0, "BUY", atr_value=5.0)
        assert t.ref_price == t.entry_price
        assert t.dividend_gross == 0.0
        pm.close_position("TEST", date(2026, 3, 5), 110.0, "SELL")
        assert t.pnl == pytest.approx(10.0 * t.shares - t.commission)
        assert t.pnl_pct == pytest.approx(t.pnl / (100.0 * t.shares))
        assert pm.apply_corporate_actions(date(2026, 2, 2)) == {}
