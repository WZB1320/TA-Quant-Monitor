#!/usr/bin/env python
"""三方交叉校验除权除息事件，并写入 corporate_action 表。

三条独立来源，任一单独都不可信，必须交叉：

| 来源                        | 取数                                   | 角色                       |
|-----------------------------|----------------------------------------|----------------------------|
| 行情反推  hfq/raw 比值跳变  | akshare(新浪) stock_zh_a_daily         | **裁判**：市场实际发生的乘数 |
| baostock backAdjustFactor  | query_adjust_factor 逐行比值            | 厂商因子（补召回）         |
| 东财 stock_fhps_detail_em   | 公告 D / s 重构 prev/((prev−D)/(1+s))   | D 与 s 的拆分（记账依据）  |

判据：两个乘数对应的**除权参考价之差 < 0.01 元**（一个最小报价单位）。
    err = |prev/mult_a − prev/mult_b|
注意不要用 |Δmult| × prev —— 送转事件下会被 mult² 放大（实测 001309：
    |Δmult|×prev = 0.0129 元 误判，除权参考价实际只差 0.0076 元）。

为什么取三源并集而不是"行情单边"：行情检测阈值必须保守（防噪声误报），
代价是小额派息会被漏掉 —— 600392 每股派 0.02 元（1.74e-3）低于行情阈值，
但东财明细与 baostock 都确认存在。所以候选日期 = 三源并集，±7 天聚类。

**禁止**使用 baostock 的 foreAdjustFactor：它不是逐事件乘数
（实测 2021-06-11 给 0.8218，实际乘数 1.0212），只能取 backAdjustFactor 逐行比值。

用法::

    python scripts/validate_corporate_actions.py                    # 全量个股
    python scripts/validate_corporate_actions.py --symbols 600887,300308
    python scripts/validate_corporate_actions.py --adjudicate adj.json
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sqlite3
import sys
import time
import warnings
from datetime import datetime
from typing import List, Optional

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.config.settings import CACHE_DB, DATA_DIR  # noqa: E402
from src.data_fetcher.event_book import CorporateAction, EventBook  # noqa: E402
from src.data_fetcher.manager import DataManager  # noqa: E402

# 判据：两个乘数对应的除权参考价之差 < 一个最小报价单位
TOL_YUAN = 0.01
# 源间事件日期匹配窗口（公告登记日/除权日与行情日期可能差几天）
MATCH_DAYS = 7
# 行情检测阈值：一分钱在 raw 前收上的相对分辨率 × 3 倍裕度（保守，宁漏勿错）
# 漏掉的小额派息由 baostock / 东财明细补召回
DETECT_SIGMA = 3.0
DETECT_FLOOR = 5e-5
# baostock 占位行判定：因子完全没变且无他源支持 → 不是事件
NOOP_EPS = 1e-9

_DPS_AFTER_PAT = re.compile(r"([\d.]+)元\(含税[,，]扣税后([\d.]+)元\)")


# --------------------------------------------------------------------- 取数

def _is_stock_code(symbol: str) -> bool:
    """是否纯 6 位个股代码（过滤指数 sh.000300 与异常格式 sz000977）"""
    return bool(re.fullmatch(r"\d{6}", symbol.strip()))


def fetch_price_frame(dm: DataManager, code: str, start: str, end: str):
    """取 raw + hfq 两份行情，算出逐日复权比值跳变

    返回 (frame, err)。frame 列: date / close_raw / mult / prev_close
    """
    raw = dm.get_daily_kline(code, start, end, adjust=dm.ADJUST_RAW)
    hfq = dm.get_daily_kline(code, start, end, adjust=dm.ADJUST_HFQ)
    if raw is None or raw.empty:
        return None, "raw 行情获取失败"
    if hfq is None or hfq.empty:
        return None, "hfq 行情获取失败"

    m = raw[["date", "close"]].merge(
        hfq[["date", "close"]], on="date", suffixes=("_raw", "_hfq")
    ).sort_values("date").reset_index(drop=True)
    if len(m) < 2:
        return None, "行情行数不足"

    m["ratio"] = m["close_hfq"] / m["close_raw"]
    m["mult"] = m["ratio"] / m["ratio"].shift(1)
    m["prev_close"] = m["close_raw"].shift(1)
    return m, None


def detect_market_events(frame: pd.DataFrame) -> pd.DataFrame:
    """从比值跳变中检出除权事件（保守阈值）"""
    m = frame.dropna(subset=["mult", "prev_close"])
    m = m[m["prev_close"] > 0]
    if m.empty:
        return m.assign(tol=[]).iloc[0:0]
    tol = np.maximum(DETECT_FLOOR, DETECT_SIGMA * 0.01 / m["prev_close"])
    return m[(m["mult"] - 1).abs() > tol][["date", "mult", "prev_close"]].reset_index(drop=True)


_bs_logged_in = False


def baostock_factor_events(code: str, start: str, end: str):
    """baostock 累积后复权因子的逐行比值

    只取 backAdjustFactor。foreAdjustFactor 不是逐事件乘数，禁用。
    """
    global _bs_logged_in
    try:
        import baostock as bs
    except ImportError:
        return None

    if not _bs_logged_in:
        lg = bs.login()
        if lg.error_code != "0":
            return None
        _bs_logged_in = True

    pre = "sh." if code[0] in "69" else ("bj." if code[0] in "48" else "sz.")
    try:
        rs = bs.query_adjust_factor(code=pre + code, start_date="2015-01-01", end_date="2026-12-31")
        rows = []
        while rs.next():
            rows.append(rs.get_row_data())
    except Exception:
        return None
    if not rows:
        return None

    f = pd.DataFrame(rows, columns=rs.fields)
    f["d"] = pd.to_datetime(f["dividOperateDate"]).dt.strftime("%Y-%m-%d")
    f["back"] = pd.to_numeric(f["backAdjustFactor"], errors="coerce")
    f = f.dropna(subset=["back"]).sort_values("d").reset_index(drop=True)
    if len(f) < 2:
        return None
    f["mult"] = f["back"] / f["back"].shift(1)
    f = f.iloc[1:][["d", "mult"]].rename(columns={"d": "date"})
    f = f[(f["date"] >= start) & (f["date"] <= end)]
    f = f[(f["mult"] - 1).abs() > NOOP_EPS]          # 丢弃因子未变的占位行
    return f.reset_index(drop=True) if not f.empty else None


def em_announced_events(code: str, start: str, end: str):
    """东财分红送配明细：拆出每股派息 D（含税/税后）与送转比例 s

    数据源给的是"每 10 股"，这里统一换算成"每股"再返回。
    """
    try:
        import akshare as ak
        df = ak.stock_fhps_detail_em(symbol=code)
    except Exception:
        return None
    if df is None or df.empty or "除权除息日" not in df.columns:
        return None

    out = []
    for _, r in df.iterrows():
        exd = str(r.get("除权除息日") or "")[:10]
        if not exd or exd in ("nan", "None"):
            continue
        if not (start <= exd <= end):
            continue
        # 只认已实施的方案，预案不计入账本
        prog = str(r.get("方案进度") or "")
        if prog and "实施" not in prog:
            continue

        def per_share(col):
            v = pd.to_numeric(r.get(col), errors="coerce")
            return 0.0 if pd.isna(v) else float(v) / 10.0

        dps_pre = per_share("现金分红-现金分红比例")
        share_r = per_share("送转股份-送股比例")
        trans_r = per_share("送转股份-转股比例")

        dps_after = None
        m = _DPS_AFTER_PAT.search(str(r.get("现金分红-现金分红比例描述") or ""))
        if m and abs(float(m.group(1)) / 10.0 - dps_pre) < 1e-6:
            dps_after = float(m.group(2)) / 10.0
        if dps_after is None and dps_pre > 0:
            dps_after = dps_pre * 0.9     # 兜底：公告隐含税率恒为 10%

        recd = str(r.get("股权登记日") or "")[:10]
        out.append(
            {
                "date": exd,
                "record_date": recd if recd and recd != "nan" else None,
                "dps_pre": dps_pre,
                "dps_after": dps_after if dps_after is not None else dps_pre,
                "share_ratio": share_r,
                "transfer_ratio": trans_r,
            }
        )
    return pd.DataFrame(out) if out else None


# --------------------------------------------------------------------- 工具

def mult_diff_yuan(m1: Optional[float], m2: Optional[float], prev: float) -> Optional[float]:
    """两个乘数对应的除权参考价之差（元）—— 绝对金额判据

    不要用 |Δmult| × prev：送转时会被 mult² 放大，导致误判。
    """
    if m1 is None or m2 is None or m1 <= 0 or m2 <= 0 or not prev:
        return None
    return abs(prev / m1 - prev / m2)


def _nearest_row(df: Optional[pd.DataFrame], date: str, days: int = MATCH_DAYS):
    if df is None or df.empty:
        return None
    d = pd.to_datetime(date)
    delta = (pd.to_datetime(df["date"]) - d).abs()
    idx = delta.idxmin()
    return df.loc[idx] if delta.loc[idx] <= pd.Timedelta(f"{days}D") else None


def _prev_close_at(frame: pd.DataFrame, date: str) -> Optional[float]:
    """取 ex_date 当日（或之前最近一个交易日）的 raw 前收"""
    if frame is None or frame.empty:
        return None
    sub = frame[frame["date"] <= date]
    if sub.empty:
        return None
    row = sub.iloc[-1]
    return float(row["prev_close"]) if pd.notna(row.get("prev_close")) else float(row["close_raw"])


def _implied_dps(mult: Optional[float], prev: Optional[float]) -> Optional[float]:
    """由权威乘数反解等效每股派息（假设 s=0）:  D = prev × (1 − 1/mult)"""
    if mult is None or not prev or mult <= 0:
        return None
    return prev * (1 - 1 / mult)


def _implied_split(mult: Optional[float], prev: Optional[float], dps: float):
    """已知 D，由权威乘数反解实际送转比例:  s = (prev − D) × mult / prev − 1"""
    if mult is None or not prev:
        return None
    return (prev - dps) * mult / prev - 1


# --------------------------------------------------------------------- 校验

def build_actions(code: str, frame, mk_ev, bs_ev, em_ev) -> List[CorporateAction]:
    """三源聚类 → 逐事件定 D/s 与 verified"""
    # 1) 聚类：候选日期取三源并集，中心日期优先用行情日期（真实交易日）
    clusters = []          # {ex_date, mult_s, prev, mult_b, em}
    used_b, used_e = set(), set()

    def add(ex_date, mult_s, prev, mult_b, em):
        clusters.append({"ex_date": ex_date, "mult_s": mult_s, "prev": prev,
                         "mult_b": mult_b, "em": em})

    for _, e in (mk_ev if mk_ev is not None else pd.DataFrame()).iterrows():
        d = e["date"]
        b = _nearest_row(bs_ev, d)
        if b is not None:
            used_b.add(b["date"])
        em = _nearest_row(em_ev, d)
        if em is not None:
            used_e.add(em["date"])
        add(d, float(e["mult"]), float(e["prev_close"]),
            float(b["mult"]) if b is not None else None, em)

    # baostock 补召回：行情阈值保守会漏掉小额派息
    if bs_ev is not None:
        for _, b in bs_ev.iterrows():
            if b["date"] in used_b:
                continue
            em = _nearest_row(em_ev, b["date"])
            if em is not None:
                used_e.add(em["date"])
            add(b["date"], None, _prev_close_at(frame, b["date"]), float(b["mult"]), em)

    # 东财补召回：公告已实施但行情/因子都没反应（影响低于检测阈值）
    if em_ev is not None:
        for _, em in em_ev.iterrows():
            if em["date"] in used_e:
                continue
            add(em["date"], None, _prev_close_at(frame, em["date"]), None, em)

    # 2) 逐簇定值与判定
    actions = []
    for c in sorted(clusters, key=lambda x: x["ex_date"]):
        date, prev, mult_s, mult_b, em = c["ex_date"], c["prev"], c["mult_s"], c["mult_b"], c["em"]
        authority = mult_s if mult_s is not None else mult_b
        mult_em = None
        if em is not None and prev:
            s_em = float(em["share_ratio"]) + float(em["transfer_ratio"])
            ex_ref = (prev - float(em["dps_pre"])) / (1 + s_em)
            mult_em = prev / ex_ref if ex_ref > 0 else None

        reasons = []

        if em is not None:
            dps_pre = float(em["dps_pre"])
            dps_after = float(em["dps_after"])
            share_r = float(em["share_ratio"])
            trans_r = float(em["transfer_ratio"])
            record_date = em["record_date"]
            verified = 1

            # 明细与行情是否冲突，用"除权参考价差多少元"判定（不用 s 的绝对差，
            # 浮点噪声会让 s 在无实质差异时也触发）
            d_em = mult_diff_yuan(mult_em, authority, prev)
            if d_em is not None and d_em >= TOL_YUAN:
                s_ann = share_r + trans_r
                if s_ann > 0:
                    # 含送转：派息是每股固定金额，确定无歧义 → 保留公告 D，反解 s。
                    # 典型 300308：公告"10转4"（s=0.4），实际 s=0.3918
                    #（回购股份不参与分配 / 零股尾数），差 0.6% 会直接进股数账本。
                    s_true = _implied_split(authority, prev, dps_pre)
                    if s_true is not None and s_true >= 0:
                        k = s_true / s_ann
                        share_r, trans_r = share_r * k, trans_r * k
                        reasons.append(
                            f"已修正：明细名义 s={s_ann:.4f} 与行情差 {d_em:.4f} 元，"
                            f"按行情反解实际 s={s_true:.4f}"
                        )
                else:
                    # 纯派息：s 恒为 0，差异只能来自派息金额本身。分方向判定：
                    #
                    # (a) d_price < d_announced —— 有标准解释：回购股份不参与利润分配
                    #     公告 D      = 每股实际到账现金（含权股东拿到的）→ 现金账用这个
                    #     d_price    = 分红总额 / 总股本(含回购)          → 除权价用这个
                    #     差额是回购股放弃的那份，由其余股东无偿获得，真实存在。
                    #     实测 600519 2025-06-26: 公告 27.6730 vs 行情 27.6007 (回购 0.26%)
                    #     → 可直接采信，但仍**不能**用行情值覆盖 dps_pre。
                    #
                    # (b) d_price > d_announced —— 无合理解释（回购只会让调整变小）
                    #     实测 603259 2025-05-21: 公告 0.9817 vs 行情 1.3314，
                    #     开盘价 61.28 也更贴近行情口径的除权参考价 61.33。
                    #     baostock 分红接口与东财明细都写 0.9817，行情因子却按 1.3314。
                    #     → 无法判断股东实际到账现金，必须人工裁决。
                    d_price = _implied_dps(authority, prev)
                    if d_price is None:
                        reasons.append(f"未闭合：明细派息与行情差 {d_em:.4f} 元，且无法反解价格调整")
                        verified = 0
                    elif d_price < dps_pre:
                        reasons.append(
                            f"已标注：公告派息 {dps_pre:.4f} 与行情价格调整 {d_price:.4f} "
                            f"差 {d_em:.4f} 元 → 回购股不参与分配；"
                            f"现金按公告值记账，价格调整按行情乘数"
                        )
                    else:
                        verified = 0
                        reasons.append(
                            f"未闭合：行情价格调整 {d_price:.4f} 反而大于公告派息 {dps_pre:.4f} "
                            f"（差 {d_em:.4f} 元），回购股不参与无法解释该方向，需人工裁决"
                        )
        else:
            # 明细缺失 → 只能由乘数反解（假设 s=0）。无法区分"派息"与"配股"：
            # 配股是持有人要掏钱，反解出的"派息"会变成一笔根本拿不到的现金。
            d_true = _implied_dps(authority, prev)
            dps_pre = d_true if (d_true is not None and d_true >= 0) else 0.0
            dps_after = dps_pre
            share_r, trans_r, record_date = 0.0, 0.0, None
            verified = 0
            reasons.append(
                "未闭合：分红明细无对应事件 → 无法区分派息/配股，D 由乘数反解（假设 s=0）"
            )

        # 交叉验证：两个独立厂商（新浪行情 vs baostock 因子）必须一致
        if mult_s is not None and mult_b is not None:
            d = mult_diff_yuan(mult_s, mult_b, prev)
            if d is not None and d >= TOL_YUAN:
                verified = 0
                reasons.append(f"未闭合：行情 vs baostock 除权参考价差 {d:.4f} 元")
        elif mult_s is None and mult_b is None:
            verified = 0
            reasons.append("未闭合：行情与因子均无对应事件，仅公告有记录")

        actions.append(
            CorporateAction(
                symbol=code,
                ex_date=date,
                record_date=record_date,
                dps_pre=dps_pre,
                dps_after=dps_after,
                share_ratio=share_r,
                transfer_ratio=trans_r,
                multiplier=mult_s,
                mult_baostock=mult_b,
                mult_em=mult_em,
                verified=verified,
                reason="; ".join(reasons) if reasons else None,
            )
        )
    return actions


def validate_symbol(dm: DataManager, code: str, start: str, end: str) -> dict:
    frame, err = fetch_price_frame(dm, code, start, end)
    if err:
        return {"symbol": code, "status": "fetch_failed", "error": err, "events": []}
    mk_ev = detect_market_events(frame)
    bs_ev = baostock_factor_events(code, start, end)
    em_ev = em_announced_events(code, start, end)
    return {
        "symbol": code,
        "status": "ok",
        "error": None,
        "events": build_actions(code, frame, mk_ev, bs_ev, em_ev),
    }


# --------------------------------------------------------------------- 裁决回写

def apply_adjudications(actions: List[CorporateAction], path: str) -> int:
    """人工裁决回写：JSON 里显式指定的事件覆盖自动结果，并在 reason 留痕

    格式: [{"symbol":"600519","ex_date":"2023-12-20","dps_pre":19.106,
            "share_ratio":0,"verified":1,"reason":"人工裁决: 茅台2023特别分红"}]
    """
    with open(path, encoding="utf-8") as f:
        items = json.load(f)
    idx = {(a.symbol, a.ex_date): a for a in actions}
    n = 0
    for it in items:
        a = idx.get((it["symbol"], it["ex_date"]))
        if a is None:
            continue
        for k in ("dps_pre", "dps_after", "share_ratio", "transfer_ratio", "verified"):
            if k in it:
                setattr(a, k, it[k])
        a.reason = "人工裁决: " + it.get("reason", "")
        n += 1
    return n


# --------------------------------------------------------------------- 主流程

def load_symbols(args) -> List[str]:
    if args.symbols:
        return [s.strip() for s in args.symbols.split(",") if s.strip()]
    with sqlite3.connect(CACHE_DB) as conn:
        rows = conn.execute("SELECT DISTINCT symbol FROM kline_cache ORDER BY symbol").fetchall()
    return [r[0] for r in rows]


def main():
    ap = argparse.ArgumentParser(description="三方交叉校验除权除息事件")
    ap.add_argument("--symbols", help="逗号分隔的代码，默认取缓存内全部个股")
    ap.add_argument("--start", default="2023-01-01")
    ap.add_argument("--end", default="2026-09-28")
    ap.add_argument("--adjudicate", help="人工裁决 JSON 路径，覆盖自动结果")
    ap.add_argument(
        "--output-dir",
        default=os.path.abspath(os.path.join(DATA_DIR, "..", "outputs")),
    )
    ap.add_argument("--dry-run", action="store_true", help="只出报告，不写库")
    args = ap.parse_args()

    symbols = load_symbols(args)
    stocks = [s for s in symbols if _is_stock_code(s)]
    skipped = [s for s in symbols if not _is_stock_code(s)]

    dm = DataManager()
    book = EventBook()

    t0 = time.time()
    all_actions: List[CorporateAction] = []
    per_symbol, failed = [], []

    for i, code in enumerate(stocks, 1):
        res = validate_symbol(dm, code, args.start, args.end)
        if res["status"] != "ok":
            failed.append({"symbol": code, "error": res["error"]})
            print(f"[{i}/{len(stocks)}] {code}  FAIL  {res['error']}")
            continue
        evs = res["events"]
        all_actions.extend(evs)
        n_ok = sum(1 for a in evs if a.verified)
        per_symbol.append({"symbol": code, "events": len(evs), "verified": n_ok})
        print(f"[{i}/{len(stocks)}] {code}  事件 {len(evs)}  闭合 {n_ok}")

    if args.adjudicate:
        n = apply_adjudications(all_actions, args.adjudicate)
        print(f"\n人工裁决覆盖 {n} 条")

    if not args.dry_run:
        # 先清后写: 事件表是校验脚本唯一写入方, 不清会留下上次的过期事件行
        for code in per_symbol:
            book.delete_symbol(code["symbol"])
        book.upsert_many(all_actions)

    n = len(all_actions)
    n_ok = sum(1 for a in all_actions if a.verified)
    report = {
        "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "window": [args.start, args.end],
        "tolerance_yuan": TOL_YUAN,
        "symbols_total": len(symbols),
        "symbols_validated": len(stocks),
        "symbols_failed": failed,
        "symbols_skipped_non_stock": skipped,
        "events_total": n,
        "events_verified": n_ok,
        "closure_rate_pct": round(n_ok / n * 100, 2) if n else None,
        "elapsed_sec": round(time.time() - t0, 1),
        "unverified_events": [
            {
                "symbol": a.symbol, "ex_date": a.ex_date,
                "multiplier": a.multiplier, "mult_baostock": a.mult_baostock,
                "mult_em": a.mult_em, "dps_pre": a.dps_pre, "reason": a.reason,
            }
            for a in all_actions if not a.verified
        ],
        "corrected_events": [
            {
                "symbol": a.symbol, "ex_date": a.ex_date,
                "share_ratio": a.share_ratio, "transfer_ratio": a.transfer_ratio,
                "reason": a.reason,
            }
            for a in all_actions if a.verified and a.reason
        ],
        "per_symbol": per_symbol,
    }

    os.makedirs(args.output_dir, exist_ok=True)
    out = os.path.join(args.output_dir, f"event_validation_{datetime.now():%Y%m%d}.json")
    with open(out, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    print("\n" + "=" * 60)
    print(f"标的 {len(stocks)} 只（跳过非个股 {len(skipped)}），失败 {len(failed)}")
    print(f"事件 {n} 个，闭合 {n_ok} 个，闭合率 {report['closure_rate_pct']}%")
    print(f"耗时 {report['elapsed_sec']}s   报告: {out}")
    if not args.dry_run:
        print(f"已写入 corporate_action（当前表内共 {book.count()} 行）")

    fixed = report["corrected_events"]
    if fixed:
        print(f"\n已修正事件 {len(fixed)} 个（明细比例与行情不符，s 按行情反解）:")
        for b in fixed[:20]:
            print(f"  {b['symbol']} {b['ex_date']}  {b['reason']}")
    bad = report["unverified_events"]
    if bad:
        print(f"\n未闭合事件 {len(bad)} 个（需人工裁决）:")
        for b in bad[:20]:
            print(f"  {b['symbol']} {b['ex_date']}  D={b['dps_pre']:.4f}  {b['reason']}")


if __name__ == "__main__":
    main()
