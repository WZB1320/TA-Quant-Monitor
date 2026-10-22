# -*- coding: utf-8 -*-
"""M5 四来源归因单测

核心不变量: 四个 bucket 之和必须等于 actual delta, 且逐笔残差为 0.
若哪天改了账本口径却没更新归因公式, 这里必须先红.
"""
import importlib.util
import os
import sys

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)


def _load():
    path = os.path.join(ROOT, "scripts", "run_raw_baseline.py")
    spec = importlib.util.spec_from_file_location("run_raw_baseline", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


rb = _load()


def trade(symbol="600000", entry="2024-01-02", exit_="2024-01-10",
          entry_price=10.0, exit_price=11.0, shares=1000,
          pnl=None, commission=20.0, dividend_gross=0.0, dividend_tax=0.0):
    """构造一条 trade_rows 格式的记录"""
    if pnl is None:
        pnl = round((exit_price - entry_price) * shares - commission
                    + dividend_gross - dividend_tax, 4)
    return {
        "symbol": symbol, "entry_date": entry, "exit_date": exit_,
        "entry_price": entry_price, "exit_price": exit_price,
        "shares": shares, "pnl": pnl,
        "entry_amount": entry_price * shares,
        "dividend_gross": dividend_gross, "dividend_tax": dividend_tax,
        "commission": commission,
    }


def test_no_change_means_zero_buckets():
    """两臂完全相同 → 所有 bucket 为 0"""
    t = trade()
    a = rb.attribute([t], [dict(t)])
    assert a["balanced"]
    assert all(v == 0 for v in a["buckets"].values())
    assert a["actual_delta"] == 0


def test_dividend_and_tax_go_to_dividend_bucket():
    """红利与红利税归入同一类, 只影响该 bucket"""
    base = trade()
    raw = trade(dividend_gross=500.0, dividend_tax=50.0)
    a = rb.attribute([base], [raw])
    assert a["buckets"]["红利(含税)"] == pytest.approx(450.0)
    assert a["buckets"]["一手取整"] == pytest.approx(0.0)
    assert a["balanced"]


def test_lot_rounding_from_share_change():
    """股数从 1000 降到 900 (价格尺度变), 价差收益按新股数缩小"""
    base = trade(entry_price=10.0, exit_price=11.0, shares=1000)
    raw = trade(entry_price=11.0, exit_price=12.1, shares=900)
    # 价格同比例上涨 10%, 但股数少 100 → 收益少
    a = rb.attribute([base], [raw])
    assert a["balanced"]
    assert a["buckets"]["一手取整"] < 0
    assert a["stats"]["shares_changed"] == 1


def test_commission_delta_bucket():
    base = trade(commission=20.0)
    raw = trade(commission=35.0)
    a = rb.attribute([base], [raw])
    assert a["buckets"]["佣金"] == pytest.approx(-15.0)
    assert a["balanced"]


def test_exit_date_divergence_goes_to_path_bucket():
    """平仓日不同 = 交易路径分叉, 整笔 delta 归入路径类"""
    base = trade(exit_="2024-01-10", pnl=1000.0)
    raw = trade(exit_="2024-01-15", pnl=700.0)
    a = rb.attribute([base], [raw])
    assert a["buckets"]["除权日路径变化"] == pytest.approx(-300.0)
    assert a["stats"]["exit_date_changed"] == 1
    assert a["balanced"]


def test_one_sided_trades_go_to_path_bucket():
    """只在旧臂存在的交易 = 损失其盈亏; 只在新臂存在 = 白得其盈亏"""
    keep = trade(symbol="A")
    only_old = trade(symbol="B", pnl=1234.0)
    only_new = trade(symbol="C", pnl=-500.0)
    a = rb.attribute([keep, only_old], [keep, only_new])
    # -1234 + (-500)
    assert a["buckets"]["除权日路径变化"] == pytest.approx(-1734.0)
    assert a["stats"]["legacy_only"] == 1
    assert a["stats"]["raw_only"] == 1
    assert a["balanced"]


def test_unexplained_residual_is_reported_not_hidden():
    """人为破坏 pnl, 残差必须被抓出并标为不守恒"""
    base = trade()
    raw = trade(pnl=base["pnl"] + 99.0)   # 凭空多 99 元
    a = rb.attribute([base], [raw])
    assert not a["balanced"]
    assert len(a["unexplained_per_trade"]) == 1
    assert abs(a["unexplained_per_trade"][0]["residual"] - 99.0) < 0.01


def test_multi_trade_aggregation_closes():
    """多笔混合场景, 四类加和仍严格等于总 delta"""
    base = [
        trade(symbol="A", shares=1000, entry_price=10.0, exit_price=11.0),
        trade(symbol="B", dividend_gross=300.0, dividend_tax=30.0),
        trade(symbol="C", commission=5.0),
    ]
    raw = [
        trade(symbol="A", shares=900, entry_price=11.0, exit_price=12.1),
        trade(symbol="B", dividend_gross=600.0, dividend_tax=60.0),
        trade(symbol="C", commission=5.0),
        trade(symbol="D", pnl=88.0),          # 新臂独有
    ]
    a = rb.attribute(base, raw)
    assert a["balanced"]
    assert abs(sum(a["buckets"].values()) - a["actual_delta"]) < 0.02
