#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""交易账本独立对账工具 (M4)

用途
----
把回测产出的任意一笔交易, 从**原始行情 + 除权事件表**出发独立重算现金流,
逐项与回测记录比对, 要求一致到分. 兼作将来对接券商流水的对账雏形.

设计原则: 刻意**不 import** position.py / broker.py
-----------------------------------------------
这两个模块正是被验对象. 复用它们的函数等于自己验自己 —— 只要公式错了,
对账工具会跟着一起错, 永远对得上. 所以这里:
  - 行情与事件只从 SQLite 原始表读
  - 撮合规则(滑点/佣金/印花税)以**白名单常量**重述一遍
  - 唯一复用是 event_book.dividend_tax_rate: 它是财税规则的单一事实来源,
    属于"业务口径"而非"被验算法"

代价是费率/滑点改动时这里也要跟着改. 这是有意的双写:
换来的是"对账真的能抓到错", 否则它只是一个自证循环.

两条撮合路径 (易错点)
------------------
  - 信号卖出: exit_price = 当日 raw open × (1 − slippage), 且计入卖出费用
  - 止损/止盈: exit_price = 当日 raw close, **不经滑点**
两者混用会让对账全表报错, 故按 exit_reason 分流.

用法
----
    # 自动挑 5 笔跨除权 + 5 笔普通
    python scripts/verify_trade_ledger.py --ledger outputs/m2_ab_compare.json --arm raw

    # 指定单笔
    python scripts/verify_trade_ledger.py --ledger outputs/m2_ab_compare.json \\
        --arm raw --symbol 600887 --entry-date 2025-06-03

    # 严格模式: 任何一分钱差异都 exit 1 (供回归 / CI 用)
    python scripts/verify_trade_ledger.py --ledger ... --strict
"""
from __future__ import annotations

import argparse
import json
import os
import sqlite3
import sys
from datetime import date
from typing import Dict, List, Optional

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from src.data_fetcher.event_book import EventBook, dividend_tax_rate  # noqa: E402

# ── 撮合规则白名单常量 (必须与 src/backtest/broker.py 一致, 有意双写) ──
SLIPPAGE = 0.0001          # 滑点 万一
COMMISSION_RATE = 0.00025  # 佣金 万 2.5
MIN_COMMISSION = 5.0       # 最低佣金 5 元
STAMP_TAX = 0.001          # 印花税 千 1, 仅卖出
TOLERANCE = 0.005          # 到分: 半分钱
# 源值一致阈值. 取 1e-4 而非 1e-6, 因为要覆盖**导出侧自身的精度损失**:
#   pnl / dividend_* 是 round(,4) -> 单笔最大损失 5e-5; entry_amount 是 round(,6).
# 实测最坏一笔源值差 4.6e-5 (600887 2026-07-17), 纯属 round(,4) 的尾数丢失.
# 1e-4 仍比 1 分钱小 100 倍, 吞不掉真实错误; commission 的 0.0049 级噪声
# 超过此阈值, 会正确退化到分位比对.
EXACT_TOL = 1e-4

DEFAULT_DB = os.path.join(ROOT, "data", "kline_cache.db")
STOP_KEYWORDS = ("止损", "止盈")


class Bar:
    __slots__ = ("dt", "open", "close")

    def __init__(self, dt: str, o: float, c: float):
        self.dt = dt
        self.open = o
        self.close = c


def load_bars(db_path: str, symbol: str, start: str, end: str) -> Dict[str, Bar]:
    """读不复权原始行情 (adjust='')"""
    sql = ("SELECT date, open, close FROM kline_cache "
           "WHERE symbol=? AND adjust='' AND date BETWEEN ? AND ? ORDER BY date")
    with sqlite3.connect(db_path) as conn:
        rows = conn.execute(sql, (symbol, start, end)).fetchall()
    return {r[0]: Bar(r[0], float(r[1]), float(r[2])) for r in rows}


def is_stop_exit(reason: str) -> bool:
    return any(k in (reason or "") for k in STOP_KEYWORDS)


def replay(trade: dict, bars: Dict[str, Bar], events: List) -> dict:
    """从原始数据独立重算一笔交易的现金流

    与 PositionManager 走不同代码路径: 这里一次性算完整个生命周期,
    PositionManager 是逐日增量累加. 两条独立路径同结果才说明账本自洽.
    """
    entry_date = trade["entry_date"]
    exit_date = trade["exit_date"]
    rec_entry = float(trade["entry_price"])
    rec_exit = float(trade["exit_price"])
    entry_amount = float(trade["entry_amount"])

    for d in (entry_date, exit_date):
        if d not in bars:
            raise KeyError(f"raw 行情缺 {trade['symbol']} {d}")

    # ── 1. 买入价 = 开盘 × (1 + 滑点); 开仓股数由本金反推 ──
    exp_entry = bars[entry_date].open * (1 + SLIPPAGE)
    shares0 = int(round(entry_amount / rec_entry)) if rec_entry else 0

    # ── 2. 持有期内除权 (登记日收盘持有即有权: entry_date < ex_date <= exit_date) ──
    shares = shares0
    div_gross = 0.0
    div_taxable = 0.0        # 纯派息: 卖出时按持股期限补税
    div_tax_at_source = 0.0  # 含送转: 送股按面值代扣的税, 已在派息时扣掉
    applied: List[dict] = []
    for ev in sorted(events, key=lambda e: e.ex_date):
        if not (entry_date < ev.ex_date <= exit_date):
            continue
        s = ev.s
        dps = ev.dps_pre if s == 0 else ev.dps_after
        cash = dps * shares
        div_gross += cash
        if s == 0:
            div_taxable += cash
        else:
            div_tax_at_source += max(ev.dps_pre - ev.dps_after, 0.0) * shares
        applied.append({"ex_date": ev.ex_date, "dps": dps, "dps_pre": ev.dps_pre,
                        "s": s, "shares_before": shares, "cash": cash})
        shares = int(round(shares * (1 + s)))

    # ── 3. 红利税: 持股期限三档, 在卖出时确定 ──
    holding_days = (date.fromisoformat(exit_date)
                    - date.fromisoformat(entry_date)).days
    rate = dividend_tax_rate(holding_days)
    div_tax = div_tax_at_source + div_taxable * rate

    # ── 4. 卖出价: 止损走收盘价(无滑点), 信号卖出走开盘价−滑点 ──
    stop = is_stop_exit(trade.get("exit_reason", ""))
    exp_exit = bars[exit_date].close if stop else bars[exit_date].open * (1 - SLIPPAGE)

    # ── 5. 费用: 买卖各一次佣金(取 max), 卖出加印花税 ──
    buy_amt = rec_entry * shares0
    sell_amt = rec_exit * shares
    comm_buy = max(buy_amt * COMMISSION_RATE, MIN_COMMISSION)
    comm_sell = max(sell_amt * COMMISSION_RATE, MIN_COMMISSION) + sell_amt * STAMP_TAX
    commission = comm_buy + comm_sell

    # ── 6. 净盈亏: 现金流法 (与 cost_basis/ref_price 无关) ──
    pnl = sell_amt - buy_amt - commission + div_gross - div_tax

    return {
        "shares0": shares0,
        "shares_final": shares,
        "entry_open": bars[entry_date].open,
        "exp_entry": exp_entry,
        "exit_open": bars[exit_date].open,
        "exit_close": bars[exit_date].close,
        "is_stop": stop,
        "exp_exit": exp_exit,
        "holding_days": holding_days,
        "tax_rate": rate,
        "events": applied,
        "div_gross": div_gross,
        "div_tax_at_source": div_tax_at_source,
        "div_taxable": div_taxable,
        "div_tax": div_tax,
        "buy_amt": buy_amt,
        "sell_amt": sell_amt,
        "comm_buy": comm_buy,
        "comm_sell": comm_sell,
        "commission": commission,
        "pnl": pnl,
        # 回测记录侧
        "rec_entry": rec_entry,
        "rec_exit": rec_exit,
        "rec_shares": int(trade["shares"]),
        "rec_commission": float(trade.get("commission") or 0.0),
        "rec_div_gross": float(trade.get("dividend_gross") or 0.0),
        "rec_div_tax": float(trade.get("dividend_tax") or 0.0),
        "rec_pnl": float(trade["pnl"]),
    }


def diff_items(calc: dict) -> List[dict]:
    """逐项差异 (计算值 − 记录值)

    金额类的判定**必须先比源值, 再比到分**, 两步缺一不可:

    - 只比源值(容差 1e-6): 会把导出侧 round(commission,2) 造成的 0.0049
      判成差异 —— 那是精度损失不是账本错误, 报出来反而掩盖真问题.
    - 只比到分: 会在**半分边界**假报警. 实测 233 笔中有 3 笔 pnl 精确值落在
      x.xx5 上(-32.935 / -1284.425 / -210.765), 浮点表示在边界两侧差 1e-14,
      round 到分的方向相反, 于是报出 0.01 的"差异"—— 实际两值同源.

    所以: 源值一致(<=1e-6) 即通过, 无论分位是否相同; 否则退化为到分比对.
    这样 1 分钱的真实错误仍会被抓到(它源值就差 0.01 > 1e-6), 而边界
    舍入噪声不会被误报. 注意不能用 0.01 容差一刀切, 那会连真错一起吞掉.

    价格类不金额化: 导出侧是 round(,4), 量化到分会让 0.0001 的滑点消失.
    """
    def money(x: float) -> float:
        return round(float(x) + 0.0, 2)

    pairs = [
        ("买入价", calc["exp_entry"], calc["rec_entry"], False),
        ("卖出价", calc["exp_exit"], calc["rec_exit"], False),
        ("股数", float(calc["shares_final"]), float(calc["rec_shares"]), False),
        ("分红毛额", calc["div_gross"], calc["rec_div_gross"], True),
        ("红利税", calc["div_tax"], calc["rec_div_tax"], True),
        ("佣金", calc["commission"], calc["rec_commission"], True),
        ("净盈亏", calc["pnl"], calc["rec_pnl"], True),
    ]
    out = []
    for name, c, r, is_money in pairs:
        raw_diff = c - r
        if is_money:
            c, r = money(c), money(r)
        d = c - r
        ok = abs(raw_diff) <= EXACT_TOL or abs(d) <= TOLERANCE
        out.append({"item": name, "calc": c, "rec": r, "diff": d,
                    "raw_diff": raw_diff, "ok": ok})
    return out


def format_report(trade: dict, calc: dict, diffs: List[dict]) -> str:
    sym = trade["symbol"]
    lines = [
        f"标的/期间: {sym}  {trade['entry_date']} → {trade['exit_date']}"
        f"  持股 {calc['holding_days']} 天",
        f"  买入  {calc['shares0']:>6d} 股 @ {calc['rec_entry']:>10.4f} "
        f"(raw open {calc['entry_open']:.2f} × (1+{SLIPPAGE})) "
        f"{-calc['buy_amt']:>12,.2f}",
    ]
    for ev in calc["events"]:
        lines.append(
            f"  除权日 {ev['ex_date']}  每股派 {ev['dps']:.4f}"
            + (f" (含税 {ev['dps_pre']:.4f})" if ev["s"] else "")
            + (f"  送转 s={ev['s']:.4f}" if ev["s"] else ""))
        lines.append(
            f"        持有 {ev['shares_before']:>6d} 股 → 分红现金 "
            f"{ev['cash']:>10,.2f}")
    if calc["events"]:
        lines.append(
            f"        红利税  持股{calc['holding_days']}天 → "
            f"{calc['tax_rate']*100:.0f}%  {-calc['div_tax']:>10,.2f}")
    lines.append(
        f"  卖出  {calc['shares_final']:>6d} 股 @ {calc['rec_exit']:>10.4f} "
        f"({'raw close 止损' if calc['is_stop'] else 'raw open 信号'}) "
        f"{calc['sell_amt']:>+13,.2f}")
    lines.append(
        f"  佣金  买 {calc['comm_buy']:.2f} + 卖 {calc['comm_sell']:.2f} "
        f"(含印花税) {-calc['commission']:>+10,.2f}")
    lines.append(f"  净盈亏 {calc['pnl']:>+38,.2f}")
    worst = max(abs(d["diff"]) for d in diffs)
    flag = "OK" if all(d["ok"] for d in diffs) else "FAIL"
    lines.append(f"  === 与回测记录最大差异: {worst:.4f} 元 [{flag}] ===")
    bad = [d for d in diffs if not d["ok"]]
    if bad:
        for d in bad:
            lines.append(
                f"     ! {d['item']}: 计算 {d['calc']:.4f} vs 记录 {d['rec']:.4f} "
                f"(差 {d['diff']:+.4f})")
    return "\n".join(lines)


def pick_trades(trades: List[dict], n_div: int, n_plain: int) -> List[dict]:
    """自动挑选: 跨除权(有分红) n_div 笔 + 普通 n_plain 笔

    优先覆盖不同标的, 并尽量分散在不同年份, 避免 10 笔都来自同一只股票
    的同一种情形 —— 那样的"全绿"说明不了泛化能力.
    """
    div = [t for t in trades if (t.get("dividend_gross") or 0) > 0]
    plain = [t for t in trades if not (t.get("dividend_gross") or 0) > 0]
    # 出场类型也要覆盖: 止损/止盈 vs 信号卖出
    def spread(pool, n):
        stop = [t for t in pool if is_stop_exit(t.get("exit_reason", ""))]
        sig = [t for t in pool if not is_stop_exit(t.get("exit_reason", ""))]
        out, i = [], 0
        while len(out) < n and (i < len(stop) or i < len(sig)):
            if i < len(stop):
                out.append(stop[i])
            if len(out) < n and i < len(sig):
                out.append(sig[i])
            i += 1
        return out[:n]
    return spread(div, n_div) + spread(plain, n_plain)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="交易账本独立对账 (M4)")
    ap.add_argument("--ledger", required=True,
                    help="含 trades 的 JSON (如 outputs/m2_ab_compare.json)")
    ap.add_argument("--arm", default=None,
                    help="m2_ab_compare 结构的臂名: raw / legacy; 省略则按顶层 trades")
    ap.add_argument("--db", default=DEFAULT_DB, help="含 raw 行情与事件表的 SQLite")
    ap.add_argument("--symbol", default=None)
    ap.add_argument("--entry-date", default=None)
    ap.add_argument("--n-div", type=int, default=5, help="跨除权笔数")
    ap.add_argument("--n-plain", type=int, default=5, help="普通笔数")
    ap.add_argument("--all", dest="all_trades", action="store_true",
                    help="全量对账(覆盖 --n-div/--n-plain). 233 笔约 10 秒, "
                         "改口径后应跑这个而不是抽样")
    ap.add_argument("--strict", action="store_true", help="有差异则 exit 1")
    ap.add_argument("--json", dest="json_out", default=None)
    args = ap.parse_args(argv)

    with open(args.ledger, encoding="utf-8") as f:
        payload = json.load(f)
    if args.arm:
        if args.arm not in payload or "trades" not in payload[args.arm]:
            print(f"错误: {args.ledger} 中无 '{args.arm}.trades'", file=sys.stderr)
            return 2
        trades = payload[args.arm]["trades"]
    else:
        trades = payload["trades"] if isinstance(payload, dict) and "trades" in payload else payload

    if args.symbol and args.entry_date:
        picked = [t for t in trades
                  if t["symbol"] == args.symbol and t["entry_date"] == args.entry_date]
        if not picked:
            print(f"错误: 未找到 {args.symbol} {args.entry_date}", file=sys.stderr)
            return 2
    elif args.all_trades:
        picked = list(trades)
    else:
        picked = pick_trades(trades, args.n_div, args.n_plain)

    book = EventBook(args.db)
    results, n_fail = [], 0
    for t in picked:
        evs = list(book.bulk([t["symbol"]])[t["symbol"]].values())
        bars = load_bars(args.db, t["symbol"], t["entry_date"], t["exit_date"])
        calc = replay(t, bars, evs)
        diffs = diff_items(calc)
        ok = all(d["ok"] for d in diffs)
        n_fail += 0 if ok else 1
        print(format_report(t, calc, diffs))
        print()
        results.append({"symbol": t["symbol"], "entry_date": t["entry_date"],
                        "exit_date": t["exit_date"], "ok": ok,
                        "max_diff": max(abs(d["diff"]) for d in diffs),
                        "calc": calc, "diffs": diffs})

    n_ok = len(results) - n_fail
    print(f"合计 {len(results)} 笔: 通过 {n_ok}, 失败 {n_fail}")
    if args.json_out:
        with open(args.json_out, "w", encoding="utf-8") as f:
            json.dump({"checked": len(results), "passed": n_ok,
                       "failed": n_fail, "results": results},
                      f, ensure_ascii=False, indent=1)
        print(f"明细已写入 {args.json_out}")
    return 1 if (args.strict and n_fail) else 0


if __name__ == "__main__":
    sys.exit(main())
