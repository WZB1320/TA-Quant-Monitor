# -*- coding: utf-8 -*-
"""verify_trade_ledger 独立对账工具单测 (M4)

测试重点不是"能对上一笔" —— 那是自证循环. 每个正向用例都配一个**篡改版本**,
断言对账必须报警. 若工具只会在正确数据上说 OK, 它对回归毫无价值.
"""
import importlib.util
import os
import sqlite3
import sys

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from src.data_fetcher.event_book import CorporateAction, EventBook  # noqa: E402


def _load_vtl():
    path = os.path.join(ROOT, "scripts", "verify_trade_ledger.py")
    spec = importlib.util.spec_from_file_location("verify_trade_ledger", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


vtl = _load_vtl()


def bar(dt, o, c):
    return vtl.Bar(dt, o, c)


def make_trade(**kw):
    """构造一笔回测记录 (字段与 run_raw_ledger_backtest.trade_rows 对齐)"""
    t = {
        "symbol": "600000", "entry_date": "2024-01-02", "exit_date": "2024-01-10",
        "entry_price": 10.001, "exit_price": 9.5, "shares": 1000,
        "entry_amount": 10001.0, "pnl": -520.5, "commission": 19.5,
        "dividend_gross": 0.0, "dividend_taxable": 0.0, "dividend_tax": 0.0,
        "exit_reason": "ATR硬止损 (-5.0%, ATR=0.30)",
    }
    t.update(kw)
    return t


BARS = {
    "2024-01-02": bar("2024-01-02", 10.00, 10.20),
    "2024-01-10": bar("2024-01-10", 9.60, 9.50),
}


# ─────────────────────────── 正向: 两条撮合路径 ───────────────────────────

def test_stop_exit_uses_close_price():
    """止损出场按当日收盘价成交, 不经滑点"""
    calc = vtl.replay(make_trade(), BARS, [])
    assert calc["is_stop"] is True
    assert calc["exp_exit"] == pytest.approx(9.50)
    assert calc["exp_entry"] == pytest.approx(10.00 * 1.0001)
    assert calc["pnl"] == pytest.approx(-520.5)


def test_signal_exit_uses_open_minus_slippage():
    """信号卖出按开盘价×(1−滑点)"""
    t = make_trade(exit_reason="卖出 score=-12.0", entry_date="2024-01-02")
    # 用同一组 bars: open 9.60 → 9.60×0.9999 = 9.59904
    calc = vtl.replay(t, BARS, [])
    assert calc["is_stop"] is False
    assert calc["exp_exit"] == pytest.approx(9.60 * 0.9999)
    assert calc["exp_exit"] != pytest.approx(9.50)


def test_commission_uses_minimum_and_stamp_on_sell():
    """买入佣金取下限 5 元; 卖出佣金取下限 + 印花税千1"""
    calc = vtl.replay(make_trade(), BARS, [])
    assert calc["comm_buy"] == pytest.approx(5.0)
    assert calc["comm_sell"] == pytest.approx(5.0 + 9500 * 0.001)
    assert calc["commission"] == pytest.approx(19.5)


# ─────────────────────────── 正向: 红利税三档 ───────────────────────────

def _div_bars(exit_date, exit_close=11.0):
    bars = dict(BARS)
    bars["2024-02-01"] = bar("2024-02-01", 10.5, 10.5)
    bars[exit_date] = bar(exit_date, exit_close, exit_close)
    return bars


def test_dividend_60days_10pct():
    """持股 60 天 → 10%"""
    ev = CorporateAction(symbol="600000", ex_date="2024-02-01",
                         dps_pre=0.5, dps_after=0.45)
    t = make_trade(exit_date="2024-03-02", exit_price=11.0,
                   exit_reason="卖出 score=-10", pnl=0.0, commission=0.0)
    calc = vtl.replay(t, _div_bars("2024-03-02"), [ev])
    assert calc["holding_days"] == 60
    assert calc["tax_rate"] == pytest.approx(0.10)
    assert calc["div_gross"] == pytest.approx(500.0)
    assert calc["div_tax"] == pytest.approx(50.0)


def test_dividend_30days_20pct_boundary():
    """持股 30 天(含) → 20%"""
    ev = CorporateAction(symbol="600000", ex_date="2024-01-15",
                         dps_pre=0.5, dps_after=0.45)
    t = make_trade(exit_date="2024-02-01", exit_price=11.0, exit_reason="卖出 x")
    calc = vtl.replay(t, _div_bars("2024-02-01"), [ev])
    assert calc["holding_days"] == 30
    assert calc["div_tax"] == pytest.approx(500.0 * 0.20)


def test_dividend_31days_10pct_boundary():
    ev = CorporateAction(symbol="600000", ex_date="2024-01-15",
                         dps_pre=0.5, dps_after=0.45)
    t = make_trade(exit_date="2024-02-02", exit_price=11.0, exit_reason="卖出 x")
    calc = vtl.replay(t, _div_bars("2024-02-02"), [ev])
    assert calc["holding_days"] == 31
    assert calc["div_tax"] == pytest.approx(500.0 * 0.10)


def test_dividend_over_1year_zero_pct():
    """> 1 年 → 0%"""
    ev = CorporateAction(symbol="600000", ex_date="2024-06-01",
                         dps_pre=0.5, dps_after=0.45)
    bars = dict(BARS)
    bars["2024-06-01"] = bar("2024-06-01", 10.5, 10.5)
    bars["2025-03-01"] = bar("2025-03-01", 12.0, 12.0)
    t = make_trade(exit_date="2025-03-01", exit_price=12.0, exit_reason="卖出 x")
    calc = vtl.replay(t, bars, [ev])
    assert calc["holding_days"] > 365
    assert calc["div_tax"] == pytest.approx(0.0)


# ─────────────────────────── 正向: 送转 ───────────────────────────

def test_share_transfer_inflates_shares_and_uses_after_tax_dps():
    """10送4: 股数 ×1.4, 现金按税后 dps 到账, 面值税显式记为已代扣"""
    ev = CorporateAction(symbol="600000", ex_date="2024-02-01",
                         dps_pre=1.0, dps_after=0.9,
                         share_ratio=0.4, transfer_ratio=0.0)
    t = make_trade(exit_date="2024-03-02", exit_price=8.0,
                   shares=1400, exit_reason="卖出 x")
    calc = vtl.replay(t, _div_bars("2024-03-02", 8.0), [ev])
    assert calc["shares_final"] == 1400
    assert calc["div_gross"] == pytest.approx(0.9 * 1000)
    assert calc["div_tax_at_source"] == pytest.approx(0.1 * 1000)
    # 送转场景不再叠加自算税率
    assert calc["div_tax"] == pytest.approx(100.0)


# ─────────────────────── 领息权边界 (T+1 买入无领权) ───────────────────────

def test_ex_date_equals_entry_date_not_entitled():
    """买入日 = 除权日 → 无领息权 (登记日收盘时尚未持有)"""
    ev = CorporateAction(symbol="600000", ex_date="2024-01-02",
                         dps_pre=0.5, dps_after=0.45)
    calc = vtl.replay(make_trade(), BARS, [ev])
    assert calc["div_gross"] == 0.0
    assert calc["events"] == []


def test_ex_date_equals_exit_date_entitled():
    """平仓日 = 除权日 → 仍有权 (除权日当天卖出照领)"""
    ev = CorporateAction(symbol="600000", ex_date="2024-01-10",
                         dps_pre=0.5, dps_after=0.45)
    calc = vtl.replay(make_trade(), BARS, [ev])
    assert calc["div_gross"] == pytest.approx(500.0)


# ─────────────────────────── 反向: 必须抓到错 ───────────────────────────

def _assert_fails(calc, item):
    diffs = vtl.diff_items(calc)
    bad = [d for d in diffs if d["item"] == item]
    assert bad and not bad[0]["ok"], f"篡改 {item} 未被抓到: {bad}"


def test_detects_tampered_dividend():
    ev = CorporateAction(symbol="600000", ex_date="2024-02-01",
                         dps_pre=0.5, dps_after=0.45)
    t = make_trade(exit_date="2024-03-02", exit_price=11.0,
                   exit_reason="卖出 x", dividend_gross=480.0)
    calc = vtl.replay(t, _div_bars("2024-03-02"), [ev])
    _assert_fails(calc, "分红毛额")


def test_detects_tampered_tax():
    ev = CorporateAction(symbol="600000", ex_date="2024-02-01",
                         dps_pre=0.5, dps_after=0.45)
    t = make_trade(exit_date="2024-03-02", exit_price=11.0,
                   exit_reason="卖出 x", dividend_tax=25.0)
    calc = vtl.replay(t, _div_bars("2024-03-02"), [ev])
    _assert_fails(calc, "红利税")


def test_detects_tampered_commission():
    t = make_trade(commission=12.34)
    calc = vtl.replay(t, BARS, [])
    _assert_fails(calc, "佣金")


def test_detects_tampered_pnl():
    t = make_trade(pnl=-400.0)
    calc = vtl.replay(t, BARS, [])
    _assert_fails(calc, "净盈亏")


def test_detects_wrong_shares():
    """股数错 (漏了送股) 必须被抓到"""
    ev = CorporateAction(symbol="600000", ex_date="2024-02-01",
                         dps_pre=1.0, dps_after=0.9, share_ratio=0.4)
    t = make_trade(exit_date="2024-03-02", exit_price=8.0,
                   shares=1000, exit_reason="卖出 x")  # 应为 1400
    calc = vtl.replay(t, _div_bars("2024-03-02", 8.0), [ev])
    _assert_fails(calc, "股数")


def test_detects_wrong_exit_price_path():
    """把止损笔误记成信号卖出价 → 卖出价必须报警"""
    t = make_trade(exit_price=9.60 * 0.9999)  # 用了 open 路径
    calc = vtl.replay(t, BARS, [])            # 但 exit_reason 说止损
    _assert_fails(calc, "卖出价")


# ─────────────────────────── 金额到分语义 ───────────────────────────

def test_cent_rounding_is_not_a_failure():
    """导出侧 round(,2) 造成的 0.004 级差异不应误报"""
    calc = vtl.replay(make_trade(commission=19.5049, pnl=-520.4951), BARS, [])
    diffs = {d["item"]: d for d in vtl.diff_items(calc)}
    assert diffs["佣金"]["ok"]
    assert diffs["净盈亏"]["ok"]


def test_real_one_cent_error_is_caught():
    calc = vtl.replay(make_trade(commission=19.51), BARS, [])
    diffs = {d["item"]: d for d in vtl.diff_items(calc)}
    assert not diffs["佣金"]["ok"]


# ─────────────────────────── 数据缺失 ───────────────────────────

def test_missing_bar_raises():
    t = make_trade(exit_date="2024-01-11")  # BARS 里没有
    with pytest.raises(KeyError):
        vtl.replay(t, BARS, [])


# ─────────────────────────── DB 读取与选笔 ───────────────────────────

def test_load_bars_reads_raw_adjust_only(tmp_path):
    db = tmp_path / "t.db"
    with sqlite3.connect(db) as conn:
        conn.execute("CREATE TABLE kline_cache (symbol TEXT, date TEXT, "
                     "adjust TEXT, open REAL, close REAL)")
        conn.executemany("INSERT INTO kline_cache VALUES (?,?,?,?,?)", [
            ("600000", "2024-01-02", "", 10.0, 10.2),
            ("600000", "2024-01-02", "qfq", 8.0, 8.1),   # 不应被读到
        ])
    bars = vtl.load_bars(str(db), "600000", "2024-01-01", "2024-12-31")
    assert list(bars) == ["2024-01-02"]
    assert bars["2024-01-02"].open == 10.0


def test_pick_trades_covers_both_exit_types_and_dividends():
    trades = (
        [{"symbol": "A", "entry_date": "2024-01-01", "exit_date": "2024-01-05",
          "dividend_gross": 10.0, "exit_reason": "ATR硬止损"}]
        + [{"symbol": "B", "entry_date": "2024-02-01", "exit_date": "2024-02-05",
            "dividend_gross": 20.0, "exit_reason": "卖出 score"}]
        + [{"symbol": "C", "entry_date": "2024-03-01", "exit_date": "2024-03-05",
            "dividend_gross": 0.0, "exit_reason": "ATR移动止盈"}]
        + [{"symbol": "D", "entry_date": "2024-04-01", "exit_date": "2024-04-05",
            "dividend_gross": 0.0, "exit_reason": "强卖出"}]
    )
    picked = vtl.pick_trades(trades, 2, 2)
    assert {t["symbol"] for t in picked} == {"A", "B", "C", "D"}
    div = [t for t in picked if t["dividend_gross"] > 0]
    plain = [t for t in picked if t["dividend_gross"] == 0]
    assert len(div) == 2 and len(plain) == 2
    # 两类出场都要覆盖
    assert any(vtl.is_stop_exit(t["exit_reason"]) for t in div)
    assert any(not vtl.is_stop_exit(t["exit_reason"]) for t in div)
