# -*- coding: utf-8 -*-
"""复权改造基线回测 — 固化「改造前」的数字。

设计要点 (与直接跑 run_backtest.py 的关键区别):
  1. 数据来自**冻结快照** data/baseline_snapshot_<date>.db, 而非主库。
     后续复权改造会改写主库的 adjust/批次, 快照保证基线永远可复现。
  2. **显式 end_date = 缓存末日**, 且清空 _source_order 禁用远程。
     直接跑 run_backtest.py 时 end_date 默认是今天 -> 缓存判为过期 ->
     触发 35 只标的全量重抓 -> qfq 基准被改写。那样跑出来的不是"现状基线",
     而是被这次运行污染后的新基准, 前后无法对比。
  3. 输出数据指纹 (每只标的的行数/批次/收盘价 sha256), 用于日后证明
     "改造前后喂的是同一份数据"。
"""
import sys
import os
import json
import shutil
import sqlite3
import hashlib
from datetime import datetime

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from src.config.runtime_mode import set_mode, RuntimeMode
# 必须在导入回测层之前: SignalFilter 不写盘, 用户偏好不持久化, 保证可复现
set_mode(RuntimeMode.BACKTEST)

from src.data_fetcher import DataManager, Watchlist
from src.data_fetcher.cache import KLineCache
from src.backtest import BacktestEngine
from src.config.group_config import GroupConfig

DATE_TAG = datetime.now().strftime("%Y%m%d")
TAG = sys.argv[1] if len(sys.argv) > 1 else DATE_TAG   # 支持重复跑做可复现性校验
SNAPSHOT = os.path.join(ROOT, "data", f"baseline_snapshot_{DATE_TAG}.db")
MAIN_DB = os.path.join(ROOT, "data", "kline_cache.db")

DATA_START = "2024-01-01"   # 数据加载起点 (含指标预热)
DATA_END = "2026-09-28"     # 数据/回测末日 = 缓存末日 (显式指定, 禁远程)
BT_START = "2024-07-01"     # 回测起点: 该日后才产生交易, 之前仅用于 120 日预热

# 与 run_backtest.py 完全一致的参数, 保证口径对齐
ENGINE_KWARGS = dict(
    initial_capital=100_000,
    lookback_days=120,
    position_ratio=0.30,
    signal_dedup_days=5,
    risk_per_trade=0.05,
    atr_stop_mult=2.5,
)


def freeze_snapshot():
    """把当前主库冻结为快照; 已存在则不覆盖(保证同一天重复运行结果一致)。"""
    if os.path.exists(SNAPSHOT):
        print(f"[snapshot] already exists: {SNAPSHOT}")
        return
    shutil.copy2(MAIN_DB, SNAPSHOT)
    print(f"[snapshot] frozen -> {SNAPSHOT}")


def build_data_manager():
    """DataManager, 但缓存指向快照且禁止任何远程请求。"""
    dm = DataManager()
    dm.cache = KLineCache(db_path=SNAPSHOT)
    dm._source_order = []          # 禁用远程; 缓存未命中即失败, 不静默联网
    return dm


def compute_fingerprint(symbol, df, con):
    """标的级数据指纹: 用于日后证明 '改造前后是同一份数据'。"""
    h = hashlib.sha256()
    for d, c in zip(df["date"].astype(str), df["close"].astype(float)):
        h.update(f"{d}|{c!r}".encode())
    batches = con.execute(
        "SELECT substr(updated_at,1,10), COUNT(*) FROM kline_cache "
        "WHERE symbol=? GROUP BY 1 ORDER BY 1", (symbol,)
    ).fetchall()
    return {
        "rows": len(df),
        "first_date": str(df["date"].iloc[0])[:10],
        "last_date": str(df["date"].iloc[-1])[:10],
        "first_close": float(df["close"].iloc[0]),
        "last_close": float(df["close"].iloc[-1]),
        "close_sha256": h.hexdigest()[:16],
        "qfq_batches": [b[0] for b in batches],   # >1 即该标的历史为跨批次拼接
    }


def main():
    freeze_snapshot()
    con = sqlite3.connect(SNAPSHOT)
    dm = build_data_manager()
    wl = Watchlist()

    print(f"[data] window {DATA_START} ~ {DATA_END} (adjust=qfq, snapshot only)")
    data_map, fingerprints, skipped = {}, {}, []
    for s in wl.get_all():
        code = s["code"]
        df = dm.get_daily_kline(code, start_date=DATA_START, end_date=DATA_END, adjust="qfq")
        if df is None or len(df) < 120:
            skipped.append((code, 0 if df is None else len(df)))
            continue
        data_map[code] = df
        fingerprints[code] = compute_fingerprint(code, df, con)

    print(f"[data] loaded {len(data_map)} symbols, skipped {len(skipped)} {skipped}")
    multi = {k: v for k, v in fingerprints.items() if len(v["qfq_batches"]) > 1}
    print(f"[data] symbols with >1 qfq batch (cross-batch splice): {len(multi)}/{len(fingerprints)}")

    print(f"[run] backtest {BT_START} ~ {DATA_END}")
    t0 = datetime.now()
    engine = BacktestEngine(**ENGINE_KWARGS)
    metrics = engine.run(data_map, start_date=BT_START, end_date=DATA_END)
    trades = engine.position_mgr.closed_trades
    elapsed = (datetime.now() - t0).total_seconds()
    print(f"[run] done in {elapsed:.1f}s, {len(trades)} trades")

    # ── 分组统计 ──
    gc = GroupConfig()
    groups = {}
    for t in trades:
        g = gc.get_group(t.symbol)
        groups.setdefault(g, []).append(t)

    def pnl_sum(ts): return sum(x.pnl for x in ts)
    def wins(ts): return [x for x in ts if x.pnl > 0]
    def losses(ts): return [x for x in ts if x.pnl <= 0]

    group_stats = {}
    for g, ts in sorted(groups.items()):
        w, l = wins(ts), losses(ts)
        group_stats[g] = {
            "trades": len(ts),
            "wins": len(w),
            "losses": len(l),
            "win_rate": len(w) / len(ts) * 100 if ts else 0,
            "total_pnl": pnl_sum(ts),
            "avg_win_pct": sum(x.pnl_pct for x in w) / len(w) * 100 if w else 0,
            "avg_loss_pct": sum(x.pnl_pct for x in l) / len(l) * 100 if l else 0,
            "avg_holding_days": sum(x.holding_days for x in ts) / len(ts) if ts else 0,
        }

    trade_rows = [{
        "symbol": t.symbol, "entry_date": str(t.entry_date), "exit_date": str(t.exit_date),
        "entry_price": round(float(t.entry_price), 4), "exit_price": round(float(t.exit_price), 4),
        "shares": int(t.shares), "pnl": round(float(t.pnl), 2),
        "pnl_pct": round(float(t.pnl_pct) * 100, 4), "holding_days": int(t.holding_days),
        "entry_signal": t.entry_signal, "exit_reason": getattr(t, "exit_reason", None),
    } for t in sorted(trades, key=lambda x: x.entry_date)]

    result = {
        "baseline_tag": f"pre-adjust-{TAG}",
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "purpose": "复权改造前的对照基线 (信号层 qfq / 成交层 qfq, 未做滚动复权)",
        "config": {
            "snapshot_db": os.path.basename(SNAPSHOT),
            "adjust": "qfq",
            "data_window": [DATA_START, DATA_END],
            "backtest_window": [BT_START, DATA_END],
            "engine": ENGINE_KWARGS,
            "watchlist_size": len(wl.get_all()),
            "symbols_loaded": len(data_map),
            "symbols_skipped": skipped,
        },
        "data_fingerprint": {
            "cross_batch_symbols": sorted(multi.keys()),
            "cross_batch_count": len(multi),
            "symbols": fingerprints,
        },
        "headline": {
            "total_return_pct": metrics.total_return * 100,
            "annual_return_pct": metrics.annual_return * 100,
            "max_drawdown_pct": metrics.max_drawdown * 100,
            "volatility_pct": metrics.volatility * 100,
            "sharpe_ratio": metrics.sharpe_ratio,
            "trade_count": metrics.trade_count,
            "win_count": metrics.win_count,
            "win_rate_pct": metrics.win_rate * 100,
            "profit_factor": metrics.profit_factor,
            "avg_win_pct": metrics.avg_win_pct * 100,
            "avg_loss_pct": metrics.avg_loss_pct * 100,
            "avg_holding_days": metrics.avg_holding_days,
            "total_pnl": sum(t.pnl for t in trades),
        },
        "group_stats": group_stats,
        "trades": trade_rows,
    }

    out_json = os.path.join(ROOT, "outputs", f"backtest_baseline_{TAG}.json")
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    # ── 控制台摘要 (纯 ASCII, 规避 Windows GBK) ──
    H = result["headline"]
    print("\n" + "=" * 66)
    print(f"  BASELINE (pre-adjust)  {result['baseline_tag']}")
    print("=" * 66)
    print(f"  total return   : {H['total_return_pct']:+.2f}%")
    print(f"  annual return  : {H['annual_return_pct']:+.2f}%")
    print(f"  max drawdown   : {H['max_drawdown_pct']:+.2f}%")
    print(f"  sharpe         : {H['sharpe_ratio']:.2f}")
    print(f"  trades         : {H['trade_count']}  (win {H['win_count']}, {H['win_rate_pct']:.1f}%)")
    print(f"  profit factor  : {H['profit_factor']:.2f}")
    print(f"  avg win/loss   : {H['avg_win_pct']:+.2f}% / {H['avg_loss_pct']:+.2f}%")
    print(f"  avg holding    : {H['avg_holding_days']:.0f} d")
    print(f"  total pnl      : {H['total_pnl']:+,.0f}")
    print(f"\n  saved -> {out_json}")
    return result


if __name__ == "__main__":
    main()
