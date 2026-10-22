#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""券商交割单对账 + 费率反推（复权改造 选项3：用真实交割单反向检验回测模型）

回测里的费率全是"拍的常量"（佣金万2.5 / 最低5元 / 印花税千1 / 过户费 0），
只有真实交割单能证明它们对不对。本工具做两件事:

  1. 自洽校验: 交割单内部是否算得平 (price×shares == amount? 各项费用加总 ==
     settlement?) —— 先证明输入可信, 否则反推出的费率也是错的
  2. 费率反推: 从真实交割单反算佣金率 / 最低佣金 / 印花税率 / 过户费率,
     与回测模型的常量逐一对照, 输出可直接填回代码的建议值

独立性: 与 verify_trade_ledger.py 一样, **刻意不 import broker.py / position.py** ——
它们是本工具要检验的对象, 复用其常量等于自证循环。

标准中间格式 (CSV, UTF-8 或 GBK 自动识别):

    date,symbol,side,price,shares,amount,commission,stamp_tax,transfer_fee,other_fee,settlement

    date          成交日 YYYY-MM-DD
    symbol        6 位代码
    side          buy / sell
    price         成交均价
    shares        成交股数 (买入为正, 卖出为正)
    amount        成交金额 = price × shares
    commission    佣金 (券商收)
    stamp_tax     印花税 (卖出才有)
    transfer_fee  过户费 (中国结算收)
    other_fee     其它费用 (经手费/证管费等, 若券商单列)
    settlement    资金变动: 买入为负, 卖出为正

不同券商导出列名不同, 请先用 --emit-template 生成模板, 按语义映射后填入。
金额列允许为空 (视为 0); settlement 为空则跳过自洽校验但保留费率反推。

用法:
    # 1) 生成模板
    python scripts/reconcile_broker_statement.py --emit-template my.csv

    # 2) 填好交割单后对账
    python scripts/reconcile_broker_statement.py --statement my.csv \\
           --json outputs/m6_broker_reconcile.json
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from typing import Dict, List, Optional

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

# ── 回测模型的费率常量（白名单双写，检验对象在 src/backtest/） ──
MODEL = {
    "commission_rate": 0.00025,   # src/backtest/broker.py:23 实际生效值
    "min_commission": 5.0,
    "stamp_tax": 0.001,           # ⚠️ 实际应为 0.0005（2023-08-28 起减半）
    "transfer_fee": 0.0,          # ⚠️ 回测完全未建模，实际 0.00001 双向
}

# ── A 股现行规则（政策依据，用于提示而非判定） ──
REGULATION = {
    "stamp_tax": (0.0005, "财政部税务总局公告2023年第39号: 2023-08-28 起减半, 卖方单边 0.5‰"),
    "transfer_fee": (0.00001, "中国结算 2022-04-29 起: 沪深京统一按成交金额 0.01‰ 双向"),
}

CSV_COLUMNS = ["date", "symbol", "side", "price", "shares", "amount",
               "commission", "stamp_tax", "transfer_fee", "other_fee", "settlement"]

TOL = 0.01            # 金额自洽判定：1 分钱
# 费率比较必须用**相对**容差。交割单各项费用都四舍五入到分，反推出的费率必然
# 带 1e-8 级误差（例：过户费 0.7845→0.78 使 0.00001 变成 0.00000997），
# 用绝对容差会永远判"不符"。1% 远大于舍入噪声，远小于真实的费率差异
# （如印花税 千1 vs 万5 差 100%）。
REL_TOL = 0.01


def _f(x) -> float:
    """空值/占位符统一转 0"""
    if x is None:
        return 0.0
    s = str(x).strip().replace(",", "").replace("--", "")
    if s in ("", "-", "None"):
        return 0.0
    try:
        return float(s)
    except ValueError:
        raise ValueError(f"无法解析为数字: {x!r}")


def load_statement(path: str) -> List[dict]:
    """读交割单 CSV。编码自动尝试 utf-8-sig / gbk（券商导出多为 GBK）"""
    rows = None
    for enc in ("utf-8-sig", "gbk", "utf-8"):
        try:
            with open(path, encoding=enc, newline="") as f:
                rows = list(csv.DictReader(f))
            break
        except (UnicodeDecodeError, UnicodeError):
            continue
    if rows is None:
        raise IOError(f"无法解码 {path}（已尝试 utf-8-sig / gbk / utf-8）")

    missing = [c for c in CSV_COLUMNS if c not in (rows[0].keys() if rows else [])]
    if missing:
        raise ValueError(f"交割单缺列 {missing}；标准列为 {CSV_COLUMNS}，"
                         f"可用 --emit-template 生成模板")

    out = []
    for i, r in enumerate(rows, 1):
        side = (r.get("side") or "").strip().lower()
        if side not in ("buy", "sell"):
            raise ValueError(f"第 {i} 行 side={side!r} 非法，应为 buy / sell")
        out.append({
            "row": i,
            "date": (r.get("date") or "").strip(),
            "symbol": (r.get("symbol") or "").strip(),
            "side": side,
            "price": _f(r.get("price")),
            "shares": _f(r.get("shares")),
            "amount": _f(r.get("amount")),
            "commission": _f(r.get("commission")),
            "stamp_tax": _f(r.get("stamp_tax")),
            "transfer_fee": _f(r.get("transfer_fee")),
            "other_fee": _f(r.get("other_fee")),
            "settlement": _f(r.get("settlement")),
        })
    return out


def check_self_consistency(rows: List[dict]) -> List[dict]:
    """交割单内部自洽：price×shares==amount？费用加总==settlement？

    这一步必须先过。输入不可信时，反推出的费率也是错的。
    """
    findings = []
    for r in rows:
        problems = []

        # 1) 成交金额 = 价 × 量
        if r["amount"] and r["price"] and r["shares"]:
            implied = r["price"] * r["shares"]
            if abs(implied - r["amount"]) > TOL:
                problems.append(("成交金额", implied, r["amount"], implied - r["amount"]))

        # 2) 资金变动 = ±成交金额 ∓ 各项费用
        if r["settlement"]:
            fees = r["commission"] + r["stamp_tax"] + r["transfer_fee"] + r["other_fee"]
            expected = (r["amount"] - fees) if r["side"] == "sell" else -(r["amount"] + fees)
            if abs(expected - r["settlement"]) > TOL:
                problems.append(("资金变动", expected, r["settlement"],
                                 expected - r["settlement"]))

        # 3) 印花税只对卖方征收
        if r["side"] == "buy" and r["stamp_tax"]:
            problems.append(("买入收印花税", 0.0, r["stamp_tax"], -r["stamp_tax"]))

        findings.append({"row": r["row"], "symbol": r["symbol"], "side": r["side"],
                         "ok": not problems,
                         "problems": [{"item": n, "expected": e, "actual": a, "diff": d}
                                      for n, e, a, d in problems]})
    return findings


LARGE_TRADE_AMOUNT = 50000.0   # 大于此成交额的单笔不会踩最低佣金地板


def _total_rate(rows: List[dict], fee_key: str,
                side: Optional[str] = None, min_amount: float = 0.0,
                exclude_floor: Optional[float] = None) -> Optional[float]:
    """总额法反推费率: Σ费用 / Σ成交额

    逐笔除法会被"佣金四舍五入到分"污染 (6.5375→6.54 使费率从 0.00025000
    变成 0.00025009), 众数聚类直接失效。总额法让舍入误差在聚合中摊掉。

    exclude_floor: 剔除佣金恰好等于最低佣金的单。踩地板的单其费用不含费率
    信息 (金额变了费用不变), 混进来会把费率系统性抬高 —— 实测三笔含两笔
    地板单时, 真实万 2.5 被反推成 0.00029 而误报"模型不符"。
    """
    num = den = 0.0
    for r in rows:
        if not r["amount"] or r["amount"] < min_amount:
            continue
        if side and r["side"] != side:
            continue
        if (exclude_floor is not None and fee_key == "commission"
                and round(r["commission"], 2) == exclude_floor):
            continue
        num += r[fee_key]
        den += r["amount"]
    return num / den if den else None


def detect_commission_floor(rows: List[dict]) -> Optional[float]:
    """检测最低佣金地板

    不能拿"样本里最小的实收佣金"当地板 —— 若样本全是大额单，那个值只是
    样本最小值，不代表券商的下限（实测四笔 8 万级单得出 19.61，误报成
    "地板 19.61 vs 模型 5 元不符"）。

    地板的**证据**是: 存在多笔**成交额显著不同**但**佣金完全相同**的单。
    只有踩到地板才会出现这种"金额变、费用不变"的现象。
    """
    groups: Dict[float, List[float]] = {}
    for r in rows:
        if r["commission"] > 0 and r["amount"] > 0:
            groups.setdefault(round(r["commission"], 2), []).append(r["amount"])
    for comm, amounts in groups.items():
        if len(amounts) >= 2 and max(amounts) / min(amounts) > 1.5:
            return comm
    return None


def infer_rates(rows: List[dict]) -> dict:
    """从真实交割单反推费率

    两个坑，分别处理:
    1. 逐笔除法被"佣金四舍五入到分"污染 -> 改用总额法 (Σ费用/Σ成交额)
    2. 最低 5 元地板会抬高小额单的费率 -> 分层: 只看成交额 >= 5 万的单,
       那类单按万2.5 算佣金 >= 12.5 元, 不可能踩地板
    """
    comm_charged = [r["commission"] for r in rows if r["commission"] > 0]
    floor = detect_commission_floor(rows)

    return {
        "n_rows": len(rows),
        "n_with_amount": sum(1 for r in rows if r["amount"]),
        "n_large": sum(1 for r in rows if r["amount"] >= LARGE_TRADE_AMOUNT),
        # 主估计: 剔除地板单后的总额法. 地板单的费用不含费率信息, 必须排除
        "commission_rate": _total_rate(rows, "commission", exclude_floor=floor),
        # 参考: 含地板单(会偏高) 与 仅大额单(可能无样本)
        "commission_rate_all": _total_rate(rows, "commission"),
        "commission_rate_large": _total_rate(rows, "commission",
                                             min_amount=LARGE_TRADE_AMOUNT),
        # 只有检测到"金额变、费用不变"的证据才认作地板; 否则 None(无样本)
        "commission_min_charged": floor,
        "commission_min_observed": min(comm_charged) if comm_charged else None,
        "stamp_tax_rate": _total_rate(rows, "stamp_tax", side="sell"),
        "transfer_fee_rate": _total_rate(rows, "transfer_fee"),
        "other_fee_rate": _total_rate(rows, "other_fee"),
        "has_other_fee": any(r["other_fee"] > 0 for r in rows),
    }


def compare_with_model(inferred: dict) -> List[dict]:
    """反推值 vs 回测模型常量 vs 监管规则，三方对照"""
    # 佣金率取"剔除地板单"的总额法; 若全部单都踩地板则退化到全样本(会偏高,
    # 但此时样本确实无法给出费率, 报告里会一并显示全样本值供人工判断)
    comm = inferred["commission_rate"]
    if comm is None:
        comm = inferred["commission_rate_all"]

    pairs = [
        ("佣金率", comm, MODEL["commission_rate"],
         None, "券商议定, 无统一标准"),
        ("最低佣金", inferred["commission_min_charged"], MODEL["min_commission"],
         None, "券商议定"),
        ("印花税率", inferred["stamp_tax_rate"], MODEL["stamp_tax"],
         *REGULATION["stamp_tax"]),
        ("过户费率", inferred["transfer_fee_rate"], MODEL["transfer_fee"],
         *REGULATION["transfer_fee"]),
    ]
    def _rel_match(a: Optional[float], b: Optional[float]) -> bool:
        """相对容差比较。基准为 0 时只判断 a 是否也为 0"""
        if a is None or b is None:
            return False
        if b == 0:
            return abs(a) <= 0.0
        return abs(a - b) / abs(b) <= REL_TOL

    out = []
    for name, actual, model, reg, note in pairs:
        out.append({
            "item": name,
            "actual": actual,
            "model": model,
            "regulation": reg,
            "note": note,
            "match_model": _rel_match(actual, model),
            "match_regulation": _rel_match(actual, reg),
        })
    return out


def emit_template(path: str) -> None:
    with open(path, "w", encoding="utf-8-sig", newline="") as f:
        w = csv.writer(f)
        w.writerow(CSV_COLUMNS)
        w.writerow(["2026-10-20", "600887", "buy", 26.15, 1000,
                    26150.00, 6.54, 0, 0.26, 0, -26156.80])
        w.writerow(["2026-11-03", "600887", "sell", 27.80, 1000,
                    27800.00, 6.95, 13.90, 0.28, 0, 27778.87])
    print(f"模板已写入 {path}")
    print("说明: 买入 settlement = -(amount + 各项费用); 卖出 = amount - 各项费用")
    print("      列允许留空(视为 0); 不同券商列名不同, 请按语义映射")


def format_report(rows: List[dict], findings: List[dict],
                  inferred: dict, comparison: List[dict]) -> str:
    L = []
    L.append("=" * 68)
    L.append("券商交割单对账 —— 用真实数据反向检验回测费率模型")
    L.append("=" * 68)
    L.append(f"\n读入 {len(rows)} 笔（其中 {inferred['n_with_amount']} 笔有成交金额）")

    bad = [f for f in findings if not f["ok"]]
    L.append(f"\n【一】交割单自洽校验: 通过 {len(findings) - len(bad)} / {len(findings)}")
    if bad:
        L.append("  以下行内部算不平，请核对导出格式:")
        for f in bad[:10]:
            for p in f["problems"]:
                L.append(f"    第{f['row']}行 {f['symbol']} {f['side']} "
                         f"{p['item']}: 应为 {p['expected']:.2f} 实为 {p['actual']:.2f} "
                         f"(差 {p['diff']:+.2f})")
    else:
        L.append("  成交金额与资金变动均自洽，输入可信")

    L.append("\n【二】费率反推（真实交割单 → 实际费率，总额法 Σ费用/Σ成交额）")
    for k, label in (("commission_rate", "佣金率(剔除地板单)"),
                     ("commission_rate_all", "佣金率(全样本, 含地板会偏高)"),
                     ("commission_rate_large", "佣金率(>=5万单)"),
                     ("commission_min_charged", "最低佣金地板(需证据)"),
                     ("commission_min_observed", "样本最小实收佣金(元)"),
                     ("stamp_tax_rate", "印花税率"),
                     ("transfer_fee_rate", "过户费率"),
                     ("other_fee_rate", "其它费率")):
        v = inferred.get(k)
        L.append(f"  {label:<22}{'-' if v is None else f'{v:.8f}'}")
    L.append(f"  样本: {inferred['n_with_amount']} 笔有金额, "
             f"其中 {inferred['n_large']} 笔 >= {LARGE_TRADE_AMOUNT:,.0f} 元")
    if inferred["has_other_fee"]:
        L.append("  注意: 交割单单列了其它费用(经手费/证管费)。若券商报的是净佣金，")
        L.append("        回测的 commission_rate 需加上这部分，否则系统性低估成本。")

    L.append("\n【三】三方对照: 实际 vs 回测模型 vs 监管规则")
    L.append(f"  {'项目':<8}{'实际':>14}{'回测模型':>14}{'监管规则':>14}  判定")
    for c in comparison:
        a = "-" if c["actual"] is None else f"{c['actual']:.8f}"
        m = f"{c['model']:.8f}"
        g = "-" if c["regulation"] is None else f"{c['regulation']:.8f}"
        if c["actual"] is None:
            verdict = "无样本"
        elif c["match_model"]:
            verdict = "模型一致"
        else:
            verdict = "模型不符"
        L.append(f"  {c['item']:<8}{a:>14}{m:>14}{g:>14}  {verdict}")

    L.append("\n【四】结论")
    mism = [c for c in comparison
            if c["actual"] is not None and not c["match_model"]]
    if not mism:
        L.append("  回测费率与真实交割单一致，模型可继续使用。")
    else:
        L.append("  以下费率与真实交割单不符，回测的成本被系统性错估:")
        for c in mism:
            L.append(f"    {c['item']}: 实际 {c['actual']:.8f} vs 模型 {c['model']:.8f}"
                     f"  ({c['note']})")
        L.append("  请按实际值修改 src/backtest/broker.py 与 position.py 后重跑基线。")
    return "\n".join(L)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="券商交割单对账 + 费率反推")
    ap.add_argument("--statement", help="交割单 CSV")
    ap.add_argument("--emit-template", metavar="PATH", help="生成空白模板并退出")
    ap.add_argument("--json", dest="json_out", default=None, help="结果落盘 JSON")
    ap.add_argument("--strict", action="store_true",
                    help="自洽校验有失败或费率不符则 exit 1")
    args = ap.parse_args(argv)

    if args.emit_template:
        emit_template(args.emit_template)
        return 0
    if not args.statement:
        ap.error("需要 --statement 或 --emit-template")

    rows = load_statement(args.statement)
    findings = check_self_consistency(rows)
    inferred = infer_rates(rows)
    comparison = compare_with_model(inferred)

    print(format_report(rows, findings, inferred, comparison))

    payload = {
        "n_rows": len(rows),
        "self_consistency": findings,
        "inferred_rates": inferred,
        "comparison": comparison,
        "model": MODEL,
        "regulation": {k: {"value": v[0], "basis": v[1]} for k, v in REGULATION.items()},
    }
    if args.json_out:
        with open(args.json_out, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)
        print(f"\n明细已写入 {args.json_out}")

    if args.strict:
        bad = [f for f in findings if not f["ok"]]
        mism = [c for c in comparison
                if c["actual"] is not None and not c["match_model"]]
        if bad or mism:
            return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
