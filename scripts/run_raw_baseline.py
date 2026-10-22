# -*- coding: utf-8 -*-
"""复权改造 M5 — 生成并固化「真实账本」新基线

与 run_baseline_backtest.py (pre-adjust 基线) 的关系
------------------------------------------------
  pre-adjust : 信号层 qfq + 成交层 qfq   → 历史对照, 已冻结
  raw-ledger : 信号层 qfq + 成交层 raw   → **本次固化的现行口径**

两者信号层喂的是同一份 qfq 快照 (data/baseline_snapshot_20261016.db),
故策略信号逐日相同; 差异只来自成交/账本层.

本脚本比 A/B 脚本多做的三件事 (M5 交付)
------------------------------------
  1. **数据完整性闸门 (T5.1)**: raw 必须单批次, 且与 qfq 的交易日集合逐标的一致.
     任一不满足直接 abort —— 这类问题会静默产生错误结果, 不能只 warn.
     (背景: qfq 历史是 9 个抓取批次拼接, 38/46 标的跨批次; raw 是 2026-10-22 单批)
  2. **过期哨兵 (T5.2)**: 事件表 checked_at 距今 >90 天或标的缺行 → warn 不阻断.
  3. **四来源归因 (T5.4)**: 与 pre-adjust 基线逐笔比对, 归因到
     ①一手取整 ②红利(含税) ③佣金最低阈值 ④除权日路径变化, 并要求零不明原因.

用法:
    python scripts/run_raw_baseline.py                 # 跑双跑 + 出基线 + 归因
    python scripts/run_raw_baseline.py --tag foo       # 自定义 tag (可复现性校验用)
    python scripts/run_raw_baseline.py --skip-legacy   # 只跑 raw 臂 (省一半时间)
"""
import argparse
import importlib.util
import json
import os
import shutil
import sqlite3
import sys
from datetime import datetime

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from src.config.runtime_mode import set_mode, RuntimeMode
set_mode(RuntimeMode.BACKTEST)   # 必须在导入回测层之前

from src.data_fetcher.event_book import EventBook, check_staleness  # noqa: E402

MAIN_DB = os.path.join(ROOT, "data", "kline_cache.db")
SNAPSHOT_QFQ = os.path.join(ROOT, "data", "baseline_snapshot_20261016.db")
PRE_ADJUST_JSON = os.path.join(ROOT, "outputs", "backtest_baseline_20261016.json")

DATA_START, DATA_END, BT_START = "2024-01-01", "2026-09-28", "2024-07-01"
STALENESS_MAX_DAYS = 90
RESIDUAL_TOL = 0.02   # 浮点级; 超过即视为"不明原因", 不允许静默放过


def _load_ab_module():
    """复用 A/B 脚本的装载/运行/序列化函数, 避免两套口径漂移"""
    path = os.path.join(ROOT, "scripts", "run_raw_ledger_backtest.py")
    spec = importlib.util.spec_from_file_location("run_raw_ledger_backtest", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# ───────────────────────── T5.1 数据完整性闸门 ─────────────────────────

def check_batch_integrity(symbols) -> dict:
    """raw 数据必须单批次; 违反即 abort (会静默污染结果的问题不能只 warn)"""
    ph = ",".join("?" * len(symbols))
    with sqlite3.connect(MAIN_DB) as conn:
        rows = conn.execute(
            f"SELECT substr(updated_at,1,10), COUNT(*) FROM kline_cache "
            f"WHERE adjust='' AND symbol IN ({ph}) GROUP BY 1 ORDER BY 1",
            list(symbols)).fetchall()
    batches = [r[0] for r in rows]
    return {"batches": batches, "batch_count": len(batches),
            "rows": sum(r[1] for r in rows)}


def check_calendar_alignment(qfq_map, raw_map) -> list:
    """逐标比对 qfq 与 raw 的交易日集合

    engine 用 qfq 决定信号日、raw 决定成交价; 两者日历不一致会让
    "信号日 T → 执行日 T+1" 错位, 且不会报错, 只会静默换价.
    """
    bad = []
    for code, qdf in qfq_map.items():
        rdf = raw_map.get(code)
        if rdf is None:
            bad.append({"symbol": code, "reason": "missing_raw"})
            continue
        qd = set(qdf["date"].astype(str).str[:10])
        rd = set(rdf["date"].astype(str).str[:10])
        if qd != rd:
            bad.append({"symbol": code, "reason": "calendar_mismatch",
                        "qfq_only": sorted(qd - rd)[:5],
                        "raw_only": sorted(rd - qd)[:5],
                        "n_qfq": len(qd), "n_raw": len(rd)})
    return bad


# ───────────────────────── T5.4 四来源归因 ─────────────────────────

def attribute(legacy_trades, raw_trades) -> dict:
    """把总 pnl 差异归因到 4 类, 残差必须为 0

    ①一手取整  : 价格尺度 qfq→raw 改变可买手数 (含未被手数吸收的价格残余)
    ②红利      : 实收股息 − 红利税
    ③佣金      : 成交额变化 + 最低 5 元阈值
    ④路径变化  : 单边交易 (只在一臂出现) + 平仓日分叉
    """
    kl = {(t["symbol"], t["entry_date"]) for t in legacy_trades}
    kr = {(t["symbol"], t["entry_date"]) for t in raw_trades}
    dl = {(t["symbol"], t["entry_date"]): t for t in legacy_trades}
    dr = {(t["symbol"], t["entry_date"]): t for t in raw_trades}

    b_lot = b_div = b_comm = b_path = 0.0
    unexplained = []
    n_div = n_shares = n_exit = 0

    for k in sorted(kl & kr):
        a, b = dl[k], dr[k]
        if a["exit_date"] != b["exit_date"]:
            b_path += b["pnl"] - a["pnl"]
            n_exit += 1
            continue
        spread_l = (a["exit_price"] or 0) - a["entry_price"]
        spread_r = (b["exit_price"] or 0) - b["entry_price"]
        e_scale = (spread_r - spread_l) * a["shares"]
        e_lot = spread_r * (b["shares"] - a["shares"])
        # 用差值而非 raw 单边值: 真实场景 legacy 分红恒为 0(旧路径不计分红),
        # 但公式不应依赖这个前提 —— 否则将来给旧路径补分红时归因会静默失真.
        e_div = ((b["dividend_gross"] - b["dividend_tax"])
                 - (a.get("dividend_gross", 0.0) - a.get("dividend_tax", 0.0)))
        e_comm = -(b["commission"] - a["commission"])
        b_lot += e_scale + e_lot
        b_div += e_div
        b_comm += e_comm
        if b["shares"] != a["shares"]:
            n_shares += 1
        if b["dividend_gross"] > 0:
            n_div += 1
        resid = b["pnl"] - a["pnl"] - (e_scale + e_lot + e_div + e_comm)
        if abs(resid) > 0.01:
            unexplained.append({"symbol": k[0], "entry_date": k[1],
                                "residual": round(resid, 4)})

    # 单边交易: 只在 legacy 存在的等于"丢了这笔盈亏", 只在 raw 的等于"白得"
    only_l = sorted(kl - kr)
    only_r = sorted(kr - kl)
    b_path += -sum(dl[k]["pnl"] for k in only_l)
    b_path += sum(dr[k]["pnl"] for k in only_r)

    total_l = sum(t["pnl"] for t in legacy_trades)
    total_r = sum(t["pnl"] for t in raw_trades)
    actual = total_r - total_l
    buckets = {"一手取整": b_lot, "红利(含税)": b_div,
               "佣金": b_comm, "除权日路径变化": b_path}
    resid_total = actual - sum(buckets.values())
    return {
        "buckets": {k: round(v, 2) for k, v in buckets.items()},
        "bucket_sum": round(sum(buckets.values()), 2),
        "actual_delta": round(actual, 2),
        "residual": round(resid_total, 4),
        "balanced": abs(resid_total) <= RESIDUAL_TOL,
        "unexplained_per_trade": unexplained,
        "stats": {
            "common": len(kl & kr), "legacy_only": len(only_l),
            "raw_only": len(only_r), "shares_changed": n_shares,
            "exit_date_changed": n_exit, "trades_with_dividend": n_div,
        },
        "legacy_only_trades": [{"symbol": s, "entry_date": d} for s, d in only_l],
        "raw_only_trades": [{"symbol": s, "entry_date": d} for s, d in only_r],
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="raw-ledger-20261022")
    ap.add_argument("--skip-legacy", action="store_true",
                    help="跳过 legacy 臂(省时)。注意: pre-adjust 基线缺 commission "
                         "等字段, 该模式下**不出归因**, 只产出新基线指标")
    args = ap.parse_args()

    ab = _load_ab_module()
    print("[1/6] 装载数据 (qfq 来自冻结快照, raw 来自主库)")
    qfq_map, raw_map, skipped = ab.load_maps()
    print(f"      qfq {len(qfq_map)} 只 / raw {len(raw_map)} 只 / skipped {skipped}")

    print("[2/6] T5.1 数据完整性闸门")
    integ = check_batch_integrity(list(qfq_map))
    print(f"      raw 批次 {integ['batch_count']} 个: {integ['batches']}"
          f"  ({integ['rows']} 行)")
    if integ["batch_count"] != 1:
        print("[ABORT] raw 跨批次拼接, 会静默污染成交价 → 拒绝出基线", file=sys.stderr)
        return 2
    misalign = check_calendar_alignment(qfq_map, raw_map)
    if misalign:
        print(f"[ABORT] {len(misalign)} 只标的日历不一致 → 拒绝出基线", file=sys.stderr)
        for m in misalign[:10]:
            print(f"        {m}", file=sys.stderr)
        return 2
    print(f"      日历逐标一致: {len(qfq_map)}/{len(qfq_map)}")

    print("[3/6] T5.2 事件表过期哨兵 (warn 不阻断)")
    stale = check_staleness(list(qfq_map), db_path=MAIN_DB,
                            max_age_days=STALENESS_MAX_DAYS)
    print(f"      {stale if stale else 'OK: 事件表在 90 天内'}")

    print("[4/6] 冻结 raw 快照 (保证日后可复现)")
    raw_snap = os.path.join(ROOT, "data", f"raw_snapshot_{args.tag.split('-')[-1]}.db")
    if os.path.exists(raw_snap):
        print(f"      已存在, 不覆盖: {os.path.basename(raw_snap)}")
    else:
        shutil.copy2(MAIN_DB, raw_snap)
        print(f"      已冻结 -> {os.path.basename(raw_snap)}")

    event_book = EventBook(db_path=MAIN_DB)
    benchmark_df = None   # 与 pre-adjust 基线一致, 不传基准

    if args.skip_legacy:
        print("[5/6] legacy 臂: 跳过, 改读 pre-adjust 基线")
        with open(PRE_ADJUST_JSON, encoding="utf-8") as f:
            pre = json.load(f)
        tr_l = pre["trades"]
        head_l = pre["headline"]
        # pre-adjust 基线的 trade_rows 只含基础字段(无 commission/entry_amount/
        # dividend_*). 缺字段归因会算出假项(实测佣金被算成 -5,349 元), 宁可
        # 明确不出归因, 也不能给一份看起来精确的错报告.
        missing = [k for k in ("commission", "entry_amount", "dividend_gross")
                   if tr_l and k not in tr_l[0]]
        legacy_complete = not missing
        if not legacy_complete:
            print(f"[warn] pre-adjust trades 缺字段 {missing} → 本次跳过归因, "
                  f"只出新基线指标")
    else:
        print("[5/6] 跑 legacy 臂 (对照, 应复现 pre-adjust 基线)")
        eng_l, m_l = ab.run_arm(qfq_map, raw_map, benchmark_df, False, event_book)
        tr_l = ab.trade_rows(eng_l.position_mgr.closed_trades)
        head_l = {"trade_count": m_l.trade_count,
                  "total_return_pct": m_l.total_return * 100,
                  "total_pnl": sum(t["pnl"] for t in tr_l)}
        print(f"      {head_l['trade_count']} 笔 / {head_l['total_return_pct']:+.2f}%"
              f" / {head_l['total_pnl']:,.2f}")
        legacy_complete = True

    print("[6/6] 跑 raw 臂 (正式基线)")
    t0 = datetime.now()
    eng_r, m_r = ab.run_arm(qfq_map, raw_map, benchmark_df, True, event_book)
    tr_r = ab.trade_rows(eng_r.position_mgr.closed_trades)
    elapsed = (datetime.now() - t0).total_seconds()
    head_r = {"trade_count": m_r.trade_count,
              "total_return_pct": m_r.total_return * 100,
              "total_pnl": sum(t["pnl"] for t in tr_r)}
    print(f"      {head_r['trade_count']} 笔 / {head_r['total_return_pct']:+.2f}%"
          f" / {head_r['total_pnl']:,.2f}   ({elapsed:.0f}s)")

    attr = attribute(tr_l, tr_r) if legacy_complete else None

    out = {
        "baseline_tag": args.tag,
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "purpose": "复权改造后的现行口径基线 (信号层 qfq / 成交层 raw + 除权事件表)",
        "config": {
            "qfq_snapshot": os.path.basename(SNAPSHOT_QFQ),
            "raw_snapshot": os.path.basename(raw_snap),
            "data_window": [DATA_START, DATA_END],
            "backtest_window": [BT_START, DATA_END],
            "engine": ab.ENGINE_KWARGS,
            "symbols_qfq": len(qfq_map), "symbols_raw": len(raw_map),
            "symbols_skipped": skipped,
        },
        "integrity": {
            "raw_batch_count": integ["batch_count"],
            "raw_batches": integ["batches"],
            "raw_rows": integ["rows"],
            "calendar_aligned": len(qfq_map) - len(misalign),
            "calendar_mismatch": misalign,
            "staleness": stale,
        },
        "headline": {
            "total_return_pct": m_r.total_return * 100,
            "annual_return_pct": m_r.annual_return * 100,
            "max_drawdown_pct": m_r.max_drawdown * 100,
            "volatility_pct": m_r.volatility * 100,
            "sharpe_ratio": m_r.sharpe_ratio,
            "trade_count": m_r.trade_count,
            "win_count": m_r.win_count,
            "win_rate_pct": m_r.win_rate * 100,
            "profit_factor": m_r.profit_factor,
            "avg_win_pct": m_r.avg_win_pct * 100,
            "avg_loss_pct": m_r.avg_loss_pct * 100,
            "avg_holding_days": m_r.avg_holding_days,
            "total_pnl": head_r["total_pnl"],
        },
        "vs_pre_adjust": {"legacy": head_l, "raw": head_r, "attribution": attr},
        "trades": tr_r,
    }

    out_json = os.path.join(ROOT, "outputs", f"backtest_baseline_{args.tag}.json")
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)

    # ── 控制台摘要 (纯 ASCII/中文, 规避 Windows GBK) ──
    H = out["headline"]
    print("\n" + "=" * 68)
    print(f"  NEW BASELINE  {args.tag}")
    print("=" * 68)
    print(f"  total return   : {H['total_return_pct']:+.2f}%")
    print(f"  annual return  : {H['annual_return_pct']:+.2f}%")
    print(f"  max drawdown   : {H['max_drawdown_pct']:+.2f}%")
    print(f"  sharpe         : {H['sharpe_ratio']:.2f}")
    print(f"  trades         : {H['trade_count']}  (win {H['win_count']}, "
          f"{H['win_rate_pct']:.1f}%)")
    print(f"  profit factor  : {H['profit_factor']:.2f}")
    print(f"  total pnl      : {H['total_pnl']:+,.0f}")
    if attr:
        print(f"\n  四来源归因:")
        for k, v in attr["buckets"].items():
            print(f"      {k:<18}{v:>+12,.2f}")
        print(f"      {'合计':<18}{attr['bucket_sum']:>+12,.2f}"
              f"   实际 {attr['actual_delta']:>+12,.2f}")
        print(f"  残差 {attr['residual']:+.4f}  平衡={attr['balanced']}  "
              f"不明原因 {len(attr['unexplained_per_trade'])} 笔")
    else:
        print("\n  归因: 跳过 (legacy 记录字段不全, 见上方 warn)")
    print(f"\n  saved -> {out_json}")
    return 0 if (attr is None or attr["balanced"]) else 3


if __name__ == "__main__":
    sys.exit(main())
