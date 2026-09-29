"""收盘复盘大屏 · demo 数据生成（独立脚本，不改任何核心代码）

⚠️ DEPRECATED（2026-09-29）: 已被正式实现取代，本脚本冻结不再迭代。
   - 数据生成 → web/backend/services/dashboard_builder.py（python services/dashboard_builder.py）
   - 页面渲染 → web/frontend/src/pages/Dashboard/（应用内 /dashboard 路由）
   本脚本仅保留用于离线 fallback（无后端环境下产出独立 HTML 的数据源）。

产出: data/dashboard.json
用法: python scripts/dashboard_demo.py [--force]
  若 data/dashboard.json 已存在且未指定 --force，直接退出（供渲染端复用）
"""
import sys, os, time, json, warnings
warnings.filterwarnings("ignore")

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(ROOT)
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "web", "backend"))

from src.config.runtime_mode import set_mode, RuntimeMode
set_mode(RuntimeMode.BACKTEST)

import numpy as np
import pandas as pd
from src.data_fetcher import DataManager
from src.config.group_config import GroupConfig
from src.backtest.engine import BacktestEngine
from services.signal_analyzer import SignalAnalyzer

CAP = 100000.0          # 模拟本金（需求冻结：回测口径）
BT_START = "2025-06-01" # 数据拉取起点（含 lookback 120 余量）
NAV_START = "2026-01-01" # 净值曲线起点（需求冻结）

OUT = os.path.join(ROOT, "data", "dashboard.json")
INDICES = [("sh.000300", "沪深300"), ("sz.399673", "创业板50")]


def log(msg):
    print(f"[demo] {msg}", flush=True)


# ── 1. 探远程最新交易日（绕开节假日全量重下：让 end_date == 缓存末条）──
def probe_data_date():
    from src.data_fetcher.baostock_source import BaoStockSource
    src = BaoStockSource()
    df = src.get_daily_kline("sh.000300", "2026-09-01", "2026-12-31", "qfq")
    if df is None or df.empty:
        raise RuntimeError("探数据日期失败（baostock 无返回）")
    d = str(df.iloc[-1]["date"])[:10]
    log(f"远程最新交易日 = {d}")
    return d


# ── 2. 指数带 ──
def fetch_indices(dm, data_date):
    out = []
    for code, name in INDICES:
        df = dm.get_daily_kline(code, start_date="2025-01-01", end_date=data_date)
        if df is None or df.empty:
            log(f"指数 {code} 无数据")
            out.append({"code": code, "name": name, "close": 0.0, "chg": 0.0, "chg_pct": 0.0})
            continue
        c = df["close"].astype(float)
        close = float(c.iloc[-1])
        prev = float(c.iloc[-2]) if len(c) > 1 else close
        spark = [round(float(v), 2) for v in c.tail(20).values]
        out.append({
            "code": code, "name": name,
            "close": round(close, 2),
            "chg": round(close - prev, 2),
            "chg_pct": round((close / prev - 1) * 100, 2) if prev else 0.0,
            "spark": spark,
        })
        log(f"指数 {name} 收 {close:.2f} ({(close/prev-1)*100:+.2f}%)")
    return out


# ── 3. 信号分析（口径与看板一致：全部分组 + auto）──
def run_signals():
    res = SignalAnalyzer().run_analysis(group="", user_regime="auto")
    if res.get("status") != "done":
        raise RuntimeError(f"信号分析失败: {res.get('message')}")
    log(f"信号分析完成: {res['summary']}")
    return res


# ── 4. 五组组合回测（编排逻辑与探针一致，纯只读复用）──
def run_backtest(dm, data_date):
    from src.data_fetcher import Watchlist
    stocks = Watchlist().get_all()
    gc = GroupConfig()
    data_map, stock_info = {}, {}
    for s in stocks:
        code = s["code"]
        try:
            df = dm.get_daily_kline(code, start_date=BT_START, end_date=data_date)
            if df is not None and len(df) >= 120:
                data_map[code] = df
                stock_info[code] = {"name": s.get("name", code), "group": gc.get_group(code)}
        except Exception as e:
            log(f"  {code} 数据失败 {type(e).__name__}: {e}")
    log(f"回测数据就绪 {len(data_map)}/{len(stocks)}")

    bench_df = dm.get_daily_kline("sh.000300", start_date=BT_START, end_date=data_date)
    if bench_df is None or bench_df.empty:
        raise RuntimeError("基准数据获取失败")

    pc = json.load(open("config/strategy_config.json", encoding="utf-8"))["strategy_config"]["portfolio_config_p5"]
    weights = dict(pc["weights"])
    regimes_cfg = {k: set(v) for k, v in pc.get("regimes_filter", {}).items()}
    atr_ov = dict(pc.get("atr_override", {}))
    dd_cfg = {
        "threshold": pc.get("dd_protection_threshold", -0.12),
        "recovery": pc.get("dd_protection_recovery", -0.06),
        "reduced_ratio": pc.get("dd_protection_reduced_ratio", 0.5),
    }

    nav, open_pos = None, {}
    for g, w in weights.items():
        if w <= 0:
            continue
        codes = [c for c, i in stock_info.items() if i["group"] == g and c in data_map]
        if not codes:
            log(f"  {g}: 无可用股票, 跳过")
            continue
        gcfg = gc._groups.get(g, {})
        smap = {c: data_map[c] for c in codes}
        if gcfg.get("strategy_mode") == "rotation":
            from src.backtest.rotation_strategy import RotationStrategy
            r = RotationStrategy(initial_capital=CAP * w)
            res = r.run(smap, bench_df=bench_df, start_date=NAV_START, end_date=data_date)
            dv, op = res["daily_values"], res["open_positions"]
        else:
            eng = BacktestEngine(
                initial_capital=CAP * w, lookback_days=120, position_ratio=0.3,
                commission_rate=0.00025, stamp_tax=0.001, slippage=0.0001,
                signal_dedup_days=5, risk_per_trade=0.05,
                atr_stop_mult=atr_ov.get(g, 2.0), group_config=gc,
                forced_regime="auto", benchmark_df_for_memory=bench_df,
                trade_regimes=regimes_cfg.get(g), dd_protection_config=dd_cfg,
                mean_reversion_config=gcfg.get("mean_reversion_exit") or None,
            )
            eng.run(smap, benchmark_df=bench_df, start_date=NAV_START, end_date=data_date)
            dv, op = eng.daily_values, eng.position_mgr.open_positions
        if dv is not None:
            nav = dv.copy() if nav is None else nav.add(dv, fill_value=0)
        if isinstance(op, dict):
            open_pos.update(op)
        else:
            for t in op:
                open_pos[getattr(t, "symbol", id(t))] = t
        log(f"  {g} w={w} 股票{len(codes)}只 持仓{len(op)}")

    if nav is None or len(nav) == 0:
        raise RuntimeError("回测未产生净值数据")
    nav = nav + (CAP - sum(w for w in weights.values() if w > 0) * CAP)
    nav.index = pd.to_datetime(nav.index)
    nav = nav.sort_index()
    nav = nav[nav.index >= pd.Timestamp(NAV_START)]
    log(f"组合净值 {len(nav)} 天 {nav.index[0]} ~ {nav.index[-1]} 末值 {nav.iloc[-1]:.0f}")
    return nav, open_pos, data_map, stock_info, bench_df


# ── 5. 组装快照 ──
def build_snapshot(data_date, t_start, indices, sig_res, nav, open_pos, data_map, stock_info, bench_df):
    held_codes = set(open_pos.keys())

    # 信号行: 按信号强度(等级value)降序, 同级按得分降序
    rows = []
    for r in sig_res["results"]:
        code = r["code"]
        df = data_map.get(code)
        chg_pct = 0.0
        if df is not None and len(df) >= 2:
            c = df["close"].astype(float)
            chg_pct = round((float(c.iloc[-1]) / float(c.iloc[-2]) - 1) * 100, 2)
        rows.append({
            "code": code, "name": r["name"], "group": r["group"],
            "close": r["close"], "chg_pct": chg_pct,
            "level": r["level"], "label": r["label"], "score": r["score"],
            "execution": r["execution"], "held": code in held_codes,
            "indicators": r["indicators"],
        })
    rows.sort(key=lambda x: (x["score"],))
    # 等级权重排序 (实际枚举: NEUTRAL=0, 非 HOLD)
    level_rank = {"STRONG_BUY": 7, "BUY": 6, "WEAK_BUY": 5, "NEUTRAL": 4, "WEAK_SELL": 3, "SELL": 2, "STRONG_SELL": 1}
    rows.sort(key=lambda x: (level_rank.get(x["level"], 0), x["score"]), reverse=True)

    # 持仓行
    total_cost = 0.0
    pos_rows = []
    for sym, t in open_pos.items():
        entry_price = float(getattr(t, "entry_price", 0) or 0)
        shares = int(getattr(t, "shares", 0) or 0)
        entry_date = str(getattr(t, "entry_date", ""))
        df = data_map.get(sym)
        cur = float(df["close"].astype(float).iloc[-1]) if df is not None and len(df) else entry_price
        cost = shares * entry_price
        total_cost += cost
        pos_rows.append({
            "code": sym,
            "name": (stock_info.get(sym) or {}).get("name", sym),
            "group": (stock_info.get(sym) or {}).get("group", ""),
            "entry_date": entry_date[:10],
            "entry_price": round(entry_price, 4),
            "shares": shares,
            "cost": round(cost, 2),
            "current_price": round(cur, 2),
            "pnl": round((cur - entry_price) * shares, 2),
            "pnl_pct": round((cur / entry_price - 1) * 100, 2) if entry_price else 0.0,
        })
    pos_rows.sort(key=lambda x: -x["cost"])
    for p in pos_rows:
        p["weight_pct"] = round(p["cost"] / total_cost * 100, 1) if total_cost else 0.0
    position_pct = round(total_cost / CAP * 100, 1)

    # 净值与基准
    nav_norm = (nav / CAP).round(6)
    bench_series = BacktestEngine._align_benchmark(bench_df, nav.index)
    dates = [str(d)[:10] for d in nav_norm.index]
    total_return = float(nav_norm.iloc[-1] - 1)
    max_dd = float((nav_norm / nav_norm.cummax() - 1).min())
    bench_return = float(bench_series.iloc[-1] - 1) if bench_series is not None and len(bench_series) else 0.0

    snap = {
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "data_date": data_date,
        "data_lag_note": "工作日近似",
        "elapsed_sec": round(time.time() - t_start, 1),
        "regime_mode": "auto",
        "nav_start": NAV_START,
        "param_fixed_date": "2026-08-07",
        "initial_capital": CAP,
        "indices": indices,
        "signals": {"summary": sig_res["summary"], "rows": rows},
        "positions": {
            "count": len(pos_rows),
            "position_pct": position_pct,
            "cash_pct": round(100 - position_pct, 1),
            "rows": pos_rows,
        },
        "nav": {
            "latest": round(float(nav_norm.iloc[-1]), 4),
            "latest_value": round(float(nav.iloc[-1]), 2),
            "total_return": round(total_return, 4),
            "max_drawdown": round(max_dd, 4),
            "benchmark_return": round(bench_return, 4),
            "alpha": round(total_return - bench_return, 4),
            "strategy_curve": {"dates": dates, "values": [float(v) for v in nav_norm.values]},
            "benchmark_curve": {
                "dates": [str(d)[:10] for d in bench_series.index] if bench_series is not None else [],
                "values": [round(float(v), 6) for v in bench_series.values] if bench_series is not None else [],
            },
        },
    }
    return snap


def main():
    force = "--force" in sys.argv
    if os.path.exists(OUT) and not force:
        log(f"{OUT} 已存在，跳过数据阶段（--force 重跑）")
        return
    t0 = time.time()
    data_date = probe_data_date()
    dm = DataManager()
    indices = fetch_indices(dm, data_date)
    sig_res = run_signals()
    nav, open_pos, data_map, stock_info, bench_df = run_backtest(dm, data_date)
    snap = build_snapshot(data_date, t0, indices, sig_res, nav, open_pos, data_map, stock_info, bench_df)
    with open(OUT, "w", encoding="utf-8") as f:
        json.dump(snap, f, ensure_ascii=False, indent=1)
    log(f"快照已写入 {OUT}  总耗时 {snap['elapsed_sec']}s")


if __name__ == "__main__":
    main()
