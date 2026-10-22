# -*- coding: utf-8 -*-
"""M2 A/B 对照回测 — 证明「信号层零影响」+「成交层已真实化」。

同进程跑两条臂, 信号层喂**完全同一份** qfq DataFrame 对象, 因此
"信号是否变化" 这个问题不依赖任何推理, 只依赖两臂交易明细的差异:

  arm=legacy  成交层也用 qfq              → 应逐位复现 pre-adjust-20261016 基线
  arm=raw     成交层用 raw + 除权事件表     → M2 改造后的新结果

数据来源 (刻意分离, 避免互相污染):
  qfq  来自冻结快照 data/baseline_snapshot_20261016.db —— 与基线脚本同源,
        保证信号层输入与 2026-10-16 那次运行 bit-for-bit 一致.
  raw  来自主库 data/kline_cache.db (M1 落库). raw 是位级恒定的真实成交价,
        不存在 qfq 那种随新除权回溯重算的问题, 故不需要冻结.
  事件表 来自主库 corporate_action (M1 三方校验产物).

用法:
  python scripts/run_raw_ledger_backtest.py            # 跑双臂 + 出归因报告
  python scripts/run_raw_ledger_backtest.py --legacy-only
"""
import sys
import os
import json
import sqlite3
import argparse
from datetime import datetime

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from src.config.runtime_mode import set_mode, RuntimeMode
set_mode(RuntimeMode.BACKTEST)   # 必须在导入回测层之前

from src.data_fetcher import DataManager, Watchlist
from src.data_fetcher.cache import KLineCache
from src.data_fetcher.event_book import EventBook, check_staleness
from src.backtest import BacktestEngine

SNAPSHOT_QFQ = os.path.join(ROOT, "data", "baseline_snapshot_20261016.db")
MAIN_DB = os.path.join(ROOT, "data", "kline_cache.db")

DATA_START = "2024-01-01"
DATA_END = "2026-09-28"
BT_START = "2024-07-01"

# 与 run_baseline_backtest.py 完全一致, 否则两臂不可比
ENGINE_KWARGS = dict(
    initial_capital=100_000,
    lookback_days=120,
    position_ratio=0.30,
    signal_dedup_days=5,
    risk_per_trade=0.05,
    atr_stop_mult=2.5,
)


def build_dm(db_path):
    dm = DataManager()
    dm.cache = KLineCache(db_path=db_path)
    dm._source_order = []      # 禁远程; 缓存未命中即失败, 不静默联网
    return dm


def load(dm, code, adjust):
    try:
        return dm.get_daily_kline(code, start_date=DATA_START,
                                  end_date=DATA_END, adjust=adjust)
    except Exception:
        return None


def load_maps():
    dm_qfq = build_dm(SNAPSHOT_QFQ)
    dm_raw = build_dm(MAIN_DB)
    wl = Watchlist()
    qfq_map, raw_map, skipped = {}, {}, []
    for s in wl.get_all():
        code = s["code"]
        df = load(dm_qfq, code, "qfq")
        if df is None or len(df) < 120:
            skipped.append((code, 0 if df is None else len(df)))
            continue
        qfq_map[code] = df
        rdf = load(dm_raw, code, "")
        if rdf is not None and len(rdf) > 0:
            raw_map[code] = rdf
    return qfq_map, raw_map, skipped


def run_arm(qfq_map, raw_map, benchmark_df, use_raw, event_book):
    """跑一条臂. use_raw=False → 完全等价于改造前.

    两条臂传入的是**同一个** qfq_map 里的 DataFrame 对象, 且各自新建
    BacktestEngine (SignalEngine/PositionManager 全新), 因此信号层输入
    与状态都一致 —— 差异只可能来自成交层口径.
    """
    engine = BacktestEngine(**ENGINE_KWARGS,
                            event_book=event_book if use_raw else None)
    metrics = engine.run(
        qfq_map,
        benchmark_df=benchmark_df,
        start_date=BT_START,
        end_date=DATA_END,
        raw_data_map=raw_map if use_raw else None,
    )
    return engine, metrics


def trade_rows(trades):
    return [{
        "symbol": t.symbol, "entry_date": str(t.entry_date),
        "exit_date": str(t.exit_date),
        # 价格保留高精度: 归因分解要用到 (exit−entry)×shares, 4 位小数 × 上千股
        # 会引入 0.1~0.3 元的假残差, 掩盖真正的"不明原因"
        "entry_price": round(float(t.entry_price), 8),
        "exit_price": round(float(t.exit_price), 8) if t.exit_price else None,
        "shares": int(t.shares), "pnl": round(float(t.pnl), 4),
        "entry_amount": round(float(t.entry_amount), 6),
        "pnl_pct": round(float(t.pnl_pct) * 100, 6),
        "holding_days": int(t.holding_days),
        "entry_signal": t.entry_signal, "exit_reason": t.exit_signal,
        "dividend_gross": round(float(getattr(t, "dividend_gross", 0.0)), 4),
        "dividend_tax": round(float(getattr(t, "dividend_tax", 0.0)), 4),
        "commission": round(float(t.commission), 2),
        # 成本基准 (除权后已下移, 内含分红) —— 归因分解必需
        "cost_basis": round(float(t.cost_basis), 4),
    } for t in sorted(trades, key=lambda x: (str(x.entry_date), x.symbol))]


def key(t):
    return (t["symbol"], t["entry_date"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--legacy-only", action="store_true")
    args = ap.parse_args()

    qfq_map, raw_map, skipped = load_maps()
    print(f"[data] qfq {len(qfq_map)} symbols (from frozen snapshot 20261016)")
    print(f"[data] raw {len(raw_map)} symbols (from main db)")
    print(f"[data] skipped {skipped}")

    missing_raw = [s for s in qfq_map if s not in raw_map]
    if missing_raw:
        print(f"[warn] {len(missing_raw)} symbols without raw -> fall back to qfq: "
              f"{missing_raw[:8]}")

    event_book = EventBook(db_path=MAIN_DB)
    warn = check_staleness(list(qfq_map.keys()), db_path=MAIN_DB)
    if warn:
        print(f"[warn][staleness] {warn}")

    # 基线脚本 run_baseline_backtest.py 不传 benchmark_df —— 为逐位复现基线,
    # 这里同样传 None (否则 sharpe/alpha 会多一条基准曲线, 两臂不可比).
    benchmark_df = None

    print("\n[arm] legacy (qfq only, 等价于改造前)")
    eng_l, m_l = run_arm(qfq_map, raw_map, benchmark_df, False, event_book)
    tr_l = trade_rows(eng_l.position_mgr.closed_trades)
    print(f"      trades={len(tr_l)}  total_return={m_l.total_return*100:+.2f}%")

    out = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "window": [BT_START, DATA_END],
        "data": {
            "qfq_source": os.path.basename(SNAPSHOT_QFQ),
            "raw_source": os.path.basename(MAIN_DB),
            "symbols_qfq": len(qfq_map), "symbols_raw": len(raw_map),
            "missing_raw": missing_raw,
        },
        "legacy": {
            "total_return_pct": m_l.total_return * 100,
            "annual_return_pct": m_l.annual_return * 100,
            "max_drawdown_pct": m_l.max_drawdown * 100,
            "sharpe_ratio": m_l.sharpe_ratio,
            "trade_count": m_l.trade_count,
            "win_rate_pct": m_l.win_rate * 100,
            "profit_factor": m_l.profit_factor,
            "avg_holding_days": m_l.avg_holding_days,
            "total_pnl": sum(t["pnl"] for t in tr_l),
            "trades": tr_l,
        },
    }

    if args.legacy_only:
        path = os.path.join(ROOT, "outputs", "m2_legacy_check.json")
        with open(path, "w", encoding="utf-8") as f:
            json.dump(out, f, ensure_ascii=False, indent=2)
        print(f"\n  saved -> {path}")
        return out

    print("\n[arm] raw ledger (成交层 raw + 除权事件表)")
    eng_r, m_r = run_arm(qfq_map, raw_map, benchmark_df, True, event_book)
    tr_r = trade_rows(eng_r.position_mgr.closed_trades)
    print(f"      trades={len(tr_r)}  total_return={m_r.total_return*100:+.2f}%")

    # 先落盘再归因: 双臂各跑 ~6 分钟, 归因段出错也不该丢回测结果
    path = os.path.join(ROOT, "outputs", "m2_ab_compare.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)

    # ── 归因 ──
    kl = {key(t) for t in tr_l}
    kr = {key(t) for t in tr_r}
    only_l, only_r = sorted(kl - kr), sorted(kr - kl)
    dl = {key(t): t for t in tr_l}
    dr = {key(t): t for t in tr_r}
    common = []
    for k in sorted(kl & kr):
        a, b = dl[k], dr[k]
        common.append({
            "symbol": k[0], "entry_date": k[1],
            "shares_legacy": a["shares"], "shares_raw": b["shares"],
            "shares_delta": b["shares"] - a["shares"],
            "price_legacy": a["entry_price"], "price_raw": b["entry_price"],
            "price_delta_pct": ((b["entry_price"] / a["entry_price"] - 1) * 100
                               if a["entry_price"] else 0.0),
            "pnl_legacy": a["pnl"], "pnl_raw": b["pnl"],
            "pnl_delta": round(b["pnl"] - a["pnl"], 2),
            "dividend_gross": b["dividend_gross"], "dividend_tax": b["dividend_tax"],
            "holding_days": b["holding_days"],
            "exit_legacy": a["exit_date"], "exit_raw": b["exit_date"],
        })
    # 成交路径分叉 = 股数变了 (整手取整随价格尺度变化)
    lot_fork = [c for c in common if c["shares_delta"] != 0]
    # 退出日不同 = 交易路径彻底分叉 (持仓盈亏变了导致止损/止盈时点不同)
    exit_fork = [c for c in common if c["exit_legacy"] != c["exit_raw"]]

    out["raw"] = {
        "total_return_pct": m_r.total_return * 100,
        "annual_return_pct": m_r.annual_return * 100,
        "max_drawdown_pct": m_r.max_drawdown * 100,
        "sharpe_ratio": m_r.sharpe_ratio,
        "trade_count": m_r.trade_count,
        "win_rate_pct": m_r.win_rate * 100,
        "profit_factor": m_r.profit_factor,
        "avg_holding_days": m_r.avg_holding_days,
        "total_pnl": sum(t["pnl"] for t in tr_r),
        "trades": tr_r,
    }
    out["attribution"] = {
        "legacy_only_trades": [{"symbol": s, "entry_date": d} for s, d in only_l],
        "raw_only_trades": [{"symbol": s, "entry_date": d} for s, d in only_r],
        "common_trades": len(common),
        "shares_changed": len(lot_fork),
        "exit_date_changed": len(exit_fork),
        "total_dividend_gross": round(sum(t["dividend_gross"] for t in tr_r), 2),
        "total_dividend_tax": round(sum(t["dividend_tax"] for t in tr_r), 2),
        "trades_with_dividend": sum(1 for t in tr_r if t["dividend_gross"] > 0),
        "unexplained_pnl_delta": None,   # 下面计算
    }
    # ── pnl 差异精确分解 ──
    # 记账口径 (position.py): pnl = 卖出所得 − 开仓本金 + 实收股息 − 红利税 − 费用
    #   legacy: 本金 = entry×shares(qfq), 无股息项
    #   raw   : 本金 = entry×shares(raw), 有股息/红利税
    # 对退出日相同的共有交易, delta 必须能由以下来源**加和还原**:
    #   ① price_scale    价差 (exit−entry) 从 qfq 尺度变 raw 尺度
    #   ② lot_rounding    股数随价格尺度跳档 (整手取整)
    #   ③ dividend        实收股息现金
    #   ④ dividend_tax    红利税 (真实现金流出)
    #   ⑤ commission      成交额变化导致的费用变化
    # 另有两类整体项: 单边交易 (只在一条臂出现) 与 退出日分叉 (路径分叉).
    # 五项 + 两类 + 残差 加和必须等于 total_pnl 的实际 delta, 否则分解有漏.
    buckets = {"price_scale": 0.0, "lot_rounding": 0.0, "dividend": 0.0,
               "dividend_tax": 0.0, "commission": 0.0}
    unexplained, divergence, divergence_pnl = [], 0, 0.0
    for k in sorted(kl & kr):
        a, b = dl[k], dr[k]
        if a["exit_date"] != b["exit_date"]:
            divergence += 1                      # 交易路径分叉, 单独归类
            divergence_pnl += b["pnl"] - a["pnl"]
            continue
        spread_l = (a["exit_price"] or 0) - a["entry_price"]
        spread_r = (b["exit_price"] or 0) - b["entry_price"]
        e_scale = (spread_r - spread_l) * a["shares"]
        e_lot = spread_r * (b["shares"] - a["shares"])
        e_div = b["dividend_gross"]
        e_tax = -b["dividend_tax"]
        e_comm = -(b["commission"] - a["commission"])
        for name, val in (("price_scale", e_scale), ("lot_rounding", e_lot),
                          ("dividend", e_div), ("dividend_tax", e_tax),
                          ("commission", e_comm)):
            buckets[name] += val
        resid = b["pnl"] - a["pnl"] - (e_scale + e_lot + e_div + e_tax + e_comm)
        if abs(resid) > 0.01:
            unexplained.append({"symbol": k[0], "entry_date": k[1],
                                "residual": round(resid, 4)})
    only_l_pnl = -sum(dl[k]["pnl"] for k in only_l)   # legacy 有而 raw 无 → 全额损失
    only_r_pnl = sum(dr[k]["pnl"] for k in only_r)    # raw 有而 legacy 无 → 全额增量
    buckets["legacy_only_trades"] = only_l_pnl
    buckets["raw_only_trades"] = only_r_pnl
    buckets["exit_date_divergence"] = divergence_pnl
    out["attribution"]["pnl_delta_buckets"] = {k: round(v, 2) for k, v in buckets.items()}
    out["attribution"]["bucket_sum"] = round(sum(buckets.values()), 2)
    actual_delta = out["raw"]["total_pnl"] - out["legacy"]["total_pnl"]
    out["attribution"]["actual_total_pnl_delta"] = round(actual_delta, 2)
    out["attribution"]["ledger_balanced"] = abs(
        sum(buckets.values()) - actual_delta) < 0.05
    out["attribution"]["exit_date_divergence"] = divergence
    out["attribution"]["unexplained_pnl_delta"] = unexplained

    with open(path, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)

    L, R, A = out["legacy"], out["raw"], out["attribution"]
    print("\n" + "=" * 72)
    print("  M2 A/B  (legacy=qfq成交, raw=raw成交+除权事件表)")
    print("=" * 72)
    print(f"  {'':<16}{'legacy':>14}{'raw':>14}{'delta':>14}")
    for k in ["total_return_pct", "annual_return_pct", "max_drawdown_pct",
              "sharpe_ratio", "trade_count", "win_rate_pct",
              "avg_holding_days", "total_pnl"]:
        print(f"  {k:<16}{L[k]:>14.2f}{R[k]:>14.2f}{R[k]-L[k]:>14.2f}")
    print(f"\n  共有交易 {A['common_trades']}  股数变化 {A['shares_changed']}  "
          f"退出日变化 {A['exit_date_changed']}")
    print(f"  legacy独有 {len(only_l)}  raw独有 {len(only_r)}")
    print(f"  含分红交易 {A['trades_with_dividend']}  "
          f"分红总额 {A['total_dividend_gross']:,.2f}  红利税 {A['total_dividend_tax']:,.2f}")
    print(f"  退出日分叉(路径分叉) {A['exit_date_divergence']} 笔")
    print("  pnl 差异分解:")
    for k, v in A["pnl_delta_buckets"].items():
        print(f"      {k:<22}{v:>+14,.2f}")
    print(f"      {'合计':<22}{A['bucket_sum']:>+14,.2f}"
          f"   实际 delta {A['actual_total_pnl_delta']:>+12,.2f}"
          f"   平衡={A['ledger_balanced']}")
    print(f"  不明原因残差: {len(unexplained)} 笔")
    print(f"\n  saved -> {path}")
    return out


if __name__ == "__main__":
    main()
