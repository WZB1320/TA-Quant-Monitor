"""收盘复盘大屏 — 快照生成服务

从 scripts/dashboard_demo.py 迁入（2026-09-29），供 /api/dashboard/* 与 CLI 共用。

流程:
  1. 探远程最新交易日（绕开 DataManager 工作日近似导致的节假日全量重下）
  2. 指数带（沪深300 / 创业板50，含 20 日 sparkline）
  3. 组合回测（复用 services/portfolio_backtest.run_portfolio_backtest，纯只读复用）
  4. 信号分析（口径与信号看板一致: 全部分组 + auto）
  5. 组装快照 → 原子写 data/dashboard.json（失败不覆盖旧快照）

口径（需求冻结）:
  - 持仓 = 策略模拟持仓（回测引擎推导，非实盘）
  - 净值 = 回测口径，模拟本金 ¥100,000，区间 2026-01-01 起
  - 体制 = 固定 auto
  - 参数固化日 2026-08-07（净值 2026-01~08 段存在参数前视，需在页面标注）
"""
import json
import logging
import os
import sys
import threading
import time
from datetime import datetime, timedelta
from typing import Callable, Optional

# 确保项目根目录与 web/backend 在 sys.path（支持 uvicorn 与直接执行两种方式）
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
_BACKEND_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for _p in (_PROJECT_ROOT, _BACKEND_ROOT):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from src.config.runtime_mode import set_mode, RuntimeMode
from src.data_fetcher import DataManager

logger = logging.getLogger(__name__)

# ── 常量（需求冻结） ──
CAP = 100000.0            # 模拟本金
NAV_START = "2026-01-01"  # 净值曲线起点
INDICES = [("sh.000300", "沪深300"), ("sz.399673", "创业板50")]
PARAM_FIXED_DATE = "2026-08-07"   # P5 参数固化日
SNAPSHOT_PATH = os.path.join(_PROJECT_ROOT, "data", "dashboard.json")

# ── 生成状态（单例，同一时间只允许一个生成任务） ──
_state = {
    "state": "idle",       # idle | running | success | failed
    "stage": "",
    "progress": 0,
    "error": None,
    "started_at": None,
    "finished_at": None,
    "lock": threading.Lock(),
}


def get_status() -> dict:
    """生成状态快照（供 /status 轮询）"""
    return {
        "state": _state["state"],
        "stage": _state["stage"],
        "progress": _state["progress"],
        "error": _state["error"],
        "started_at": _state["started_at"],
        "finished_at": _state["finished_at"],
    }


def is_running() -> bool:
    return _state["state"] == "running"


def _set(progress: int, stage: str) -> None:
    _state["progress"] = progress
    if stage:
        _state["stage"] = stage


# ── 1. 探远程最新交易日 ──
def probe_latest_trade_date() -> str:
    """探明数据源实际最新交易日。

    必要性: DataManager._expected_last_trade_date 为工作日近似（源码自述），
    法定节假日会被判为「预期最新交易日」，导致缓存被永久判定过期 →
    每次生成都全量重下且下完数据不变。用探明值当 end_date 可命中缓存。
    """
    from src.data_fetcher.baostock_source import BaoStockSource
    src = BaoStockSource()
    start = (datetime.now() - timedelta(days=90)).strftime("%Y-%m-%d")
    end = (datetime.now() + timedelta(days=5)).strftime("%Y-%m-%d")
    df = src.get_daily_kline("sh.000300", start, end, "qfq")
    if df is None or df.empty:
        raise RuntimeError("探最新交易日失败（数据源无返回）")
    d = str(df.iloc[-1]["date"])[:10]
    logger.info("探明远程最新交易日 = %s", d)
    return d


# ── 2. 指数带 ──
def fetch_indices(dm: DataManager, data_date: str) -> list:
    out = []
    for code, name in INDICES:
        df = dm.get_daily_kline(code, start_date="2025-01-01", end_date=data_date)
        if df is None or df.empty:
            logger.warning("指数 %s 无数据", code)
            out.append({"code": code, "name": name, "close": 0.0, "chg": 0.0,
                        "chg_pct": 0.0, "spark": []})
            continue
        c = df["close"].astype(float)
        close = float(c.iloc[-1])
        prev = float(c.iloc[-2]) if len(c) > 1 else close
        out.append({
            "code": code, "name": name,
            "close": round(close, 2),
            "chg": round(close - prev, 2),
            "chg_pct": round((close / prev - 1) * 100, 2) if prev else 0.0,
            "spark": [round(float(v), 2) for v in c.tail(20).values],
        })
        logger.info("指数 %s 收 %.2f (%+.2f%%)", name, close, (close / prev - 1) * 100 if prev else 0.0)
    return out


# ── 3. 信号分析（口径与看板一致） ──
def run_signals(on_progress: Optional[Callable[[int, str], None]] = None) -> dict:
    from services.signal_analyzer import SignalAnalyzer
    res = SignalAnalyzer().run_analysis(group="", user_regime="auto")
    if res.get("status") != "done":
        raise RuntimeError(f"信号分析失败: {res.get('message')}")
    logger.info("信号分析完成: %s", res.get("summary"))
    return res


# ── 4. 组装快照 ──
def _attach_chg_pct(rows: list, dm: DataManager, data_date: str) -> None:
    """为信号行补当日涨跌幅（取自缓存，窄区间查询命中已缓存数据）。"""
    start = (datetime.strptime(data_date, "%Y-%m-%d") - timedelta(days=30)).strftime("%Y-%m-%d")
    for r in rows:
        df = dm.get_daily_kline(r["code"], start_date=start, end_date=data_date)
        if df is not None and len(df) >= 2:
            c = df["close"].astype(float)
            prev = float(c.iloc[-2])
            r["chg_pct"] = round((float(c.iloc[-1]) / prev - 1) * 100, 2) if prev else 0.0
        else:
            r["chg_pct"] = 0.0


def build_snapshot(data_date: str, dm: DataManager, indices: list, sig_res: dict,
                   bt_res: dict, t_start: float,
                   on_progress: Optional[Callable[[int, str], None]] = None) -> dict:
    """由回测/信号结果 + 指数带组装大屏快照 JSON。"""
    def _p(p: int, s: str = "") -> None:
        if on_progress:
            on_progress(p, s)

    _p(96, "组装快照...")

    bt_trades = bt_res["trades"]
    open_trades = [t for t in bt_trades if t["status"] == "open"]
    held_codes = {t["symbol"] for t in open_trades}

    # 信号行: 按信号强度(等级)降序, 同级按得分降序
    rows = []
    for r in sig_res["results"]:
        rows.append({
            "code": r["code"], "name": r["name"], "group": r["group"],
            "close": r["close"], "chg_pct": 0.0,
            "level": r["level"], "label": r["label"], "score": r["score"],
            "execution": r["execution"], "held": r["code"] in held_codes,
            "indicators": r.get("indicators", {}),
        })
    _attach_chg_pct(rows, dm, data_date)
    # 实际枚举: NEUTRAL=0（非 HOLD）
    level_rank = {"STRONG_BUY": 7, "BUY": 6, "WEAK_BUY": 5, "NEUTRAL": 4,
                  "WEAK_SELL": 3, "SELL": 2, "STRONG_SELL": 1}
    rows.sort(key=lambda x: (level_rank.get(x["level"], 0), x["score"]), reverse=True)

    # 持仓行（策略模拟持仓; 现价取回测末日常规收盘价）
    total_cost = 0.0
    pos_rows = []
    for t in open_trades:
        entry_price = float(t.get("entry_price") or 0)
        shares = int(t.get("shares") or 0)
        cur = float(t.get("exit_price") or entry_price)  # 未平仓: exit_price 存的是末日现价
        cost = shares * entry_price
        total_cost += cost
        pos_rows.append({
            "code": t["symbol"],
            "name": t["name"],
            "group": t.get("group", ""),
            "entry_date": str(t.get("entry_date", ""))[:10],
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

    # 净值与基准（daily_values 为金额量纲，含现金；benchmark 已归一化到 1.0）
    dv = bt_res["daily_values"]
    nav_dates = list(dv["dates"])
    nav_values = list(dv["values"])
    nav_norm = [round(v / CAP, 6) for v in nav_values]
    bench = bt_res.get("benchmark_values") or {"dates": [], "values": []}
    total_return = (nav_norm[-1] - 1) if nav_norm else 0.0
    peak = nav_norm[0] if nav_norm else 1.0
    max_dd = 0.0
    for v in nav_norm:
        peak = max(peak, v)
        if peak:
            max_dd = min(max_dd, v / peak - 1)
    bench_return = (bench["values"][-1] - 1) if bench.get("values") else 0.0

    snap = {
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "data_date": data_date,
        "probed_latest_trade_date": data_date,   # 生成时探明的远程最新交易日
        "elapsed_sec": round(time.time() - t_start, 1),
        "regime_mode": "auto",
        "nav_start": NAV_START,
        "param_fixed_date": PARAM_FIXED_DATE,
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
            "latest": round(nav_norm[-1], 4) if nav_norm else 0.0,
            "latest_value": round(nav_values[-1], 2) if nav_values else 0.0,
            "total_return": round(total_return, 4),
            "max_drawdown": round(max_dd, 4),
            "benchmark_return": round(bench_return, 4),
            "alpha": round(total_return - bench_return, 4),
            "strategy_curve": {"dates": nav_dates, "values": nav_norm},
            "benchmark_curve": {"dates": list(bench.get("dates", [])),
                                "values": [round(float(v), 6) for v in bench.get("values", [])]},
        },
    }
    return snap


# ── 5. 原子写 / 读 ──
def _atomic_write(snap: dict, path: str = SNAPSHOT_PATH) -> None:
    """临时文件 + os.replace 原子替换：任何失败都不会破坏旧快照。"""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(snap, f, ensure_ascii=False, indent=1)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def load_snapshot(path: str = SNAPSHOT_PATH) -> Optional[dict]:
    """读取已有快照；不存在或损坏返回 None。"""
    if not os.path.exists(path):
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        logger.warning("读取快照失败 %s: %s", path, e)
        return None


# ── 6. 数据新鲜度 ──
def compute_freshness(snap: Optional[dict], now: Optional[datetime] = None) -> dict:
    """判定快照数据是否已是最近交易日。

    规则:
      1. data_date >= 工作日近似的预期最新交易日 → 新鲜
      2. 否则: 若快照生成于当天、且生成时探明的远程最新交易日 <= data_date
         （即当日已确认源端最新就是这条），仍视为新鲜 —— 覆盖法定节假日，
         避免每次打开都白等一次全量重下。
      3. 其余 → 过期，给出滞后交易日数（工作日近似）
    """
    now = now or datetime.now()
    if not snap:
        return {"data_date": None, "expected_last_trade_date": None,
                "stale_trade_days": 0, "is_fresh": False, "has_snapshot": False}

    data_date = str(snap.get("data_date") or "")
    expected = DataManager._expected_last_trade_date(now, now=now)
    expected_s = expected.strftime("%Y-%m-%d")
    generated_at = str(snap.get("generated_at") or "")
    probed = snap.get("probed_latest_trade_date")

    is_fresh = bool(data_date) and data_date >= expected_s
    if not is_fresh and generated_at[:10] == now.strftime("%Y-%m-%d") and probed and probed <= data_date:
        is_fresh = True   # 当日已探明源端最新 == 本地数据（节假日/源端延迟）

    stale_days = 0
    if data_date and data_date < expected_s:
        import numpy as np
        d0 = datetime.strptime(data_date, "%Y-%m-%d").date()
        stale_days = int(np.busday_count(d0 + timedelta(days=1), expected.date() + timedelta(days=1)))
        stale_days = max(stale_days, 0)
        if is_fresh:
            stale_days = 0

    return {
        "data_date": data_date or None,
        "expected_last_trade_date": expected_s,
        "probed_latest_trade_date": probed,
        "stale_trade_days": stale_days,
        "is_fresh": is_fresh,
        "has_snapshot": True,
        "generated_at": generated_at,
    }


# ── 7. 主流程 ──
def generate(force: bool = True, on_progress: Optional[Callable[[int, str], None]] = None) -> dict:
    """执行完整生成管线并落盘。互斥：已在运行直接返回 running。

    失败不覆盖：任何异常都不触碰旧快照，错误写入 _state["error"]。
    """
    def _p(p: int, s: str = "") -> None:
        _set(p, s)
        if on_progress:
            on_progress(p, s)

    if not _state["lock"].acquire(blocking=False):
        return {"status": "running", "message": "已有生成任务正在运行", **get_status()}

    t0 = time.time()
    _state.update({"state": "running", "stage": "初始化", "progress": 0,
                   "error": None, "started_at": time.strftime("%Y-%m-%d %H:%M:%S"),
                   "finished_at": None})
    # 生成期使用 BACKTEST 模式: 信号过滤器走纯内存历史, 不写 signal_history.json,
    # 避免污染实盘信号去重记录（与 demo 一致）
    prev_mode = None
    try:
        from services.portfolio_backtest import BacktestRequest, run_portfolio_backtest
        try:
            from src.config.runtime_mode import get_mode
            prev_mode = get_mode()
        except Exception:
            prev_mode = None
        set_mode(RuntimeMode.BACKTEST)

        _p(3, "探测最新交易日...")
        data_date = probe_latest_trade_date()

        dm = DataManager()

        _p(10, "拉取指数...")
        indices = fetch_indices(dm, data_date)

        _p(20, "组合回测...")
        req = BacktestRequest(groups=[], mode="base", start_date=NAV_START,
                              end_date=data_date, initial_capital=CAP,
                              benchmark="sh.000300")
        # 回测内部进度(0-100)映射到 20-78 区间
        bt_res = run_portfolio_backtest(
            req, on_progress=lambda p, txt: _p(20 + int(p * 0.58), txt or "组合回测..."))
        if bt_res.get("status") != "done":
            raise RuntimeError(f"回测失败: {bt_res.get('message')}")

        _p(80, "信号分析...")
        sig_res = run_signals()

        snap = build_snapshot(data_date, dm, indices, sig_res, bt_res, t0, on_progress=_p)
        _atomic_write(snap)

        _p(100, "完成")
        _state["state"] = "success"
        _state["finished_at"] = time.strftime("%Y-%m-%d %H:%M:%S")
        logger.info("大屏快照生成完成: %s 天 %s 末值 %.2f 耗时 %.1fs",
                    len(snap["nav"]["strategy_curve"]["dates"]), snap["data_date"],
                    snap["nav"]["latest_value"], snap["elapsed_sec"])
        return {"status": "done", "snapshot": snap}
    except Exception as e:
        logger.exception("大屏快照生成失败")
        _state["state"] = "failed"
        _state["error"] = f"{type(e).__name__}: {e}"
        _state["finished_at"] = time.strftime("%Y-%m-%d %H:%M:%S")
        return {"status": "error", "message": _state["error"], **get_status()}
    finally:
        if prev_mode is not None:
            set_mode(prev_mode)
        _state["lock"].release()


def main() -> int:
    """CLI 入口: python -m services.dashboard_builder (在 web/backend 下) 或直接执行本文件"""
    import argparse
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    ap = argparse.ArgumentParser(description="收盘复盘大屏 — 快照生成")
    ap.add_argument("--skip-if-fresh", action="store_true",
                    help="快照已是最新交易日则跳过生成（秒开路径）")
    args = ap.parse_args()

    if args.skip_if_fresh:
        snap = load_snapshot()
        fr = compute_freshness(snap)
        if snap and fr["is_fresh"]:
            print(f"[skip] 快照已是最新交易日 {fr['data_date']}，跳过生成")
            return 0

    res = generate(on_progress=lambda p, s: print(f"[{p:3d}%] {s}", flush=True))
    if res.get("status") != "done":
        print(f"[FAIL] {res.get('message')}")
        return 1
    snap = res["snapshot"]
    print(f"[OK] 数据日期 {snap['data_date']} | 净值 {snap['nav']['latest']:.4f} "
          f"({snap['nav']['total_return']:+.2%}) | 持仓 {snap['positions']['count']} 只 "
          f"| 耗时 {snap['elapsed_sec']}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
