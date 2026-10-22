# -*- coding: utf-8 -*-
"""券商交割单对账工具的单测

重点是两类:
  1. 正向: 真实费率能被正确反推, 且模型偏差被报出来
  2. 反向: 工具不许误报 —— 地板检测拿不到证据时必须说"无样本",
     剔除地板单后佣金率必须回到真实值
"""

from __future__ import annotations

import csv
import importlib.util
import os

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
_PATH = os.path.join(ROOT, "scripts", "reconcile_broker_statement.py")
_spec = importlib.util.spec_from_file_location("reconcile_broker_statement", _PATH)
rbs = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(rbs)


def _write(tmp_path, rows, name="s.csv", encoding="utf-8"):
    p = tmp_path / name
    with open(p, "w", encoding=encoding, newline="") as f:
        w = csv.writer(f)
        w.writerow(rbs.CSV_COLUMNS)
        for r in rows:
            w.writerow(r)
    return str(p)


def _buy(price, shares, amount, comm, transfer, settlement,
         symbol="600887", date="2026-10-20"):
    return [date, symbol, "buy", price, shares, amount, comm, 0, transfer, 0,
            settlement]


def _sell(price, shares, amount, comm, stamp, transfer, settlement,
          symbol="600887", date="2026-11-03"):
    return [date, symbol, "sell", price, shares, amount, comm, stamp, transfer,
            0, settlement]


# 按 A 股真实费率生成: 佣金万2.5(地板5元) / 印花税万5 / 过户费万0.1 双向
TRUE_ROWS = [
    _buy(26.15, 3000, 78450.00, 19.61, 0.78, -78470.39),
    _sell(27.80, 3000, 83400.00, 20.85, 41.70, 0.83, 83336.62),
    _buy(4.35, 20000, 87000.00, 21.75, 0.87, -87022.62, symbol="000725",
         date="2026-09-15"),
    _sell(4.50, 20000, 90000.00, 22.50, 45.00, 0.90, 89931.60, symbol="000725",
          date="2026-10-10"),
]

# 含两笔踩 5 元地板的小额单
FLOOR_ROWS = [
    _buy(10.00, 1000, 10000.00, 5.00, 0.10, -10005.10),
    _buy(10.00, 2000, 20000.00, 5.00, 0.20, -20005.20, date="2026-10-21"),
    _sell(10.50, 3000, 31500.00, 7.88, 15.75, 0.32, 31476.05),
]


# ─────────────────────────── 自洽校验 ───────────────────────────

def test_self_consistent_statement_passes(tmp_path):
    rows = rbs.load_statement(_write(tmp_path, TRUE_ROWS))
    findings = rbs.check_self_consistency(rows)
    assert all(f["ok"] for f in findings), [f for f in findings if not f["ok"]]


def test_wrong_amount_is_caught(tmp_path):
    """price×shares 与 amount 不符必须抓到"""
    bad = [list(r) for r in TRUE_ROWS]
    bad[0][5] = 78000.00       # amount 写错
    rows = rbs.load_statement(_write(tmp_path, bad))
    f = rbs.check_self_consistency(rows)[0]
    assert not f["ok"]
    assert f["problems"][0]["item"] == "成交金额"


def test_wrong_settlement_is_caught(tmp_path):
    """各项费用加总与资金变动不符必须抓到"""
    bad = [list(r) for r in TRUE_ROWS]
    bad[1][10] = 83000.00      # settlement 写错
    rows = rbs.load_statement(_write(tmp_path, bad))
    f = rbs.check_self_consistency(rows)[1]
    assert not f["ok"]
    assert f["problems"][0]["item"] == "资金变动"


def test_stamp_tax_on_buy_is_caught(tmp_path):
    """印花税只对卖方征收"""
    bad = [list(r) for r in TRUE_ROWS]
    bad[0][7] = 39.23          # 买入却收了印花税
    rows = rbs.load_statement(_write(tmp_path, bad))
    f = rbs.check_self_consistency(rows)[0]
    assert not f["ok"]
    # 注意: 多收一笔费用会让资金变动也不平, 两条问题同时在.
    # 只断言"买入收印花税"被报出来, 不断言顺序.
    assert any(p["item"] == "买入收印花税" for p in f["problems"])


# ─────────────────────────── 费率反推 ───────────────────────────

def test_stamp_tax_rate_is_inferred_correctly(tmp_path):
    """真实印花税 万5 应被反推出来, 且与模型千1 判定为不符"""
    rows = rbs.load_statement(_write(tmp_path, TRUE_ROWS))
    inf = rbs.infer_rates(rows)
    assert inf["stamp_tax_rate"] == pytest.approx(0.0005, rel=0.01)

    cmp_ = {c["item"]: c for c in rbs.compare_with_model(inf)}
    assert not cmp_["印花税率"]["match_model"]      # 模型是 0.001
    assert cmp_["印花税率"]["match_regulation"]     # 监管是 0.0005


def test_transfer_fee_rate_is_inferred_correctly(tmp_path):
    """过户费 万0.1 应被反推出来, 而模型是 0(完全未建模)"""
    rows = rbs.load_statement(_write(tmp_path, TRUE_ROWS))
    inf = rbs.infer_rates(rows)
    assert inf["transfer_fee_rate"] == pytest.approx(0.00001, rel=0.01)

    cmp_ = {c["item"]: c for c in rbs.compare_with_model(inf)}
    assert not cmp_["过户费率"]["match_model"]


def test_commission_rate_matches_model(tmp_path):
    rows = rbs.load_statement(_write(tmp_path, TRUE_ROWS))
    inf = rbs.infer_rates(rows)
    assert inf["commission_rate"] == pytest.approx(0.00025, rel=0.01)
    cmp_ = {c["item"]: c for c in rbs.compare_with_model(inf)}
    assert cmp_["佣金率"]["match_model"]


# ─────────────────── 地板检测: 不许拿样本最小值冒充 ───────────────────

def test_commission_floor_detected_when_evidence_exists(tmp_path):
    """金额变、费用不变 -> 认作地板"""
    rows = rbs.load_statement(_write(tmp_path, FLOOR_ROWS))
    assert rbs.detect_commission_floor(rows) == 5.00


def test_no_floor_evidence_means_no_sample_not_a_false_alarm(tmp_path):
    """纯大额样本: 最小实收佣金 19.61 只是样本最小值, 不是地板

    曾误报成"地板 19.61 vs 模型 5 元不符"。地板必须靠证据, 没有证据就
    必须回答"无样本", 不能拿最小值充数。
    """
    rows = rbs.load_statement(_write(tmp_path, TRUE_ROWS))
    inf = rbs.infer_rates(rows)
    assert inf["commission_min_charged"] is None
    assert inf["commission_min_observed"] == pytest.approx(19.61)

    cmp_ = {c["item"]: c for c in rbs.compare_with_model(inf)}
    assert cmp_["最低佣金"]["actual"] is None
    assert cmp_["最低佣金"]["match_model"] is False   # 无样本不算"一致"


def test_floor_rows_excluded_from_commission_rate(tmp_path):
    """地板单的费用不含费率信息, 混进来会把万2.5 抬成 0.00029 而误报

    这是本工具最隐蔽的坑: 含地板样本的全样本总额法 = 0.00029073,
    与模型 0.00025 相差 16%, 会被误判"佣金率不符"。剔除后必须回到真实值。
    """
    rows = rbs.load_statement(_write(tmp_path, FLOOR_ROWS))
    inf = rbs.infer_rates(rows)
    assert inf["commission_rate_all"] == pytest.approx(0.00029, rel=0.02)
    assert inf["commission_rate"] == pytest.approx(0.00025, rel=0.01)
    assert inf["commission_rate"] < inf["commission_rate_all"]


# ─────────────────────────── 解析健壮性 ───────────────────────────

def test_gbk_encoded_statement_is_read(tmp_path):
    """券商导出多为 GBK"""
    rows = rbs.load_statement(_write(tmp_path, TRUE_ROWS, "g.csv", encoding="gbk"))
    assert len(rows) == 4


def test_blank_and_placeholder_become_zero(tmp_path):
    p = tmp_path / "b.csv"
    with open(p, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(rbs.CSV_COLUMNS)
        w.writerow(["2026-10-20", "600887", "buy", 26.15, 1000, 26150.00,
                    "", "--", "", "", ""])
    rows = rbs.load_statement(str(p))
    assert rows[0]["commission"] == 0.0
    assert rows[0]["settlement"] == 0.0


def test_illegal_side_raises(tmp_path):
    bad = [list(r) for r in TRUE_ROWS]
    bad[0][2] = "买入"
    with pytest.raises(ValueError):
        rbs.load_statement(_write(tmp_path, bad))


def test_missing_column_raises(tmp_path):
    p = tmp_path / "m.csv"
    with open(p, "w", encoding="utf-8", newline="") as f:
        f.write("date,symbol,price\n2026-10-20,600887,26.15\n")
    with pytest.raises(ValueError):
        rbs.load_statement(str(p))


def test_emit_template_has_standard_columns(tmp_path):
    out = str(tmp_path / "tpl.csv")
    rbs.emit_template(out)
    with open(out, encoding="utf-8-sig") as f:
        header = next(csv.reader(f))
    assert header == rbs.CSV_COLUMNS


def test_model_constants_match_code():
    """白名单常量必须与 src/backtest/broker.py 实际生效值一致

    这是双写的代价: 改了被验对象而没改这里, 工具会失去意义。
    """
    assert rbs.MODEL["commission_rate"] == 0.00025
    assert rbs.MODEL["stamp_tax"] == 0.001
    assert rbs.MODEL["min_commission"] == 5.0
    # 过户费在回测里完全未建模, 这是已知缺陷而非本测试的目标值
    assert rbs.MODEL["transfer_fee"] == 0.0
