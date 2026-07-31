"""回测路由 — 运行回测 / 获取结果"""
import logging
import os
import sys
import threading
from datetime import datetime, date
from typing import Optional

from fastapi import APIRouter
from pydantic import BaseModel

# 确保项目根目录在 sys.path
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from src.data_fetcher import DataManager, Watchlist
from src.config.group_config import GroupConfig
from src.config.runtime_mode import set_mode, RuntimeMode, get_mode
from src.backtest import BacktestEngine
from src.backtest.metrics import compute_metrics

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/backtest", tags=["回测分析"])


class BacktestRequest(BaseModel):
    groups: list[str] = []          # 参与分组, 空则全部
    mode: str = "base"              # base / trending / ranging
    start_date: Optional[str] = None  # YYYY-MM-DD
    end_date: Optional[str] = None    # YYYY-MM-DD
    initial_capital: float = 100000
    benchmark: str = "000001"       # 基准指数代码


# ── 全局状态 (单例, 同一时间只允许一个回测任务) ──
_state = {
    "running": False,
    "progress": 0,
    "total": 0,
    "progress_text": "",
    "result": None,
    "lock": threading.Lock(),
}


@router.post("/run", summary="运行回测")
def run_backtest(req: BacktestRequest):
    """同步执行回测（阻塞直到完成）"""
    if not _state["lock"].acquire(blocking=False):
        return {"status": "error", "message": "已有回测任务正在运行"}

    # 切换到 BACKTEST 模式: SignalFilter 使用纯内存历史, 不读写 signal_history.json
    # 避免污染实时信号数据, 且避免上次回测/分析的去重记录影响本次回测
    prev_mode = get_mode()
    set_mode(RuntimeMode.BACKTEST)

    try:
        _state["running"] = True
        _state["progress"] = 0
        _state["progress_text"] = "加载自选股..."

        # 1. 读取自选股
        wl = Watchlist()
        all_stocks = wl.get_all()
        if not all_stocks:
            return {"status": "error", "message": "自选股列表为空"}

        # 清除信号历史文件, 避免上次回测/分析的去重记录影响本次回测
        _history_file = os.path.join(_PROJECT_ROOT, "data", "signal_history.json")
        if os.path.exists(_history_file):
            os.remove(_history_file)

        gc = GroupConfig()

        # 按分组筛选
        if req.groups:
            filtered = [s for s in all_stocks if gc.get_group(s["code"]) in req.groups]
            if not filtered:
                return {"status": "error", "message": f"所选分组中无股票: {req.groups}"}
            all_stocks = filtered

        # regime 作为请求级参数传入 BacktestEngine, 不再写盘 user_preferences.json
        # 避免污染其他页面 (信号看板/个股回测) 的 regime 配置
        # base → auto (ADX 自动判断); trending/ranging → 该模式
        forced_regime = req.mode if req.mode in ("trending", "ranging") else "auto"

        _state["total"] = len(all_stocks)
        _state["progress_text"] = f"加载日线数据 (共{_state['total']}只)..."
        _state["progress"] = 10

        # 2. 拉取日线数据
        # 数据获取起点要比回测起点早至少 120 个交易日 (lookback), 否则引擎无法计算指标
        dm = DataManager()
        if req.start_date:
            # 回测起点往前推 8 个月 (约 120 个交易日), 确保有足够 lookback
            from datetime import timedelta
            start_dt = datetime.strptime(req.start_date, "%Y-%m-%d")
            fetch_start = (start_dt - timedelta(days=240)).strftime("%Y-%m-%d")
        else:
            fetch_start = "2024-01-01"
        end = req.end_date or datetime.now().strftime("%Y-%m-%d")

        data_map = {}
        stock_info = {}  # code → {name, group}
        for i, stock in enumerate(all_stocks):
            code = stock["code"]
            try:
                df = dm.get_daily_kline(code, start_date=fetch_start, end_date=end)
                if df is not None and len(df) >= 120:
                    data_map[code] = df
                    stock_info[code] = {
                        "name": stock.get("name", code),
                        "group": gc.get_group(code),
                    }
            except Exception as e:
                logger.warning("获取 %s 数据失败: %s", code, e)
            _state["progress"] = 10 + int((i + 1) / len(all_stocks) * 50)

        if not data_map:
            return {"status": "error", "message": "所有股票数据获取失败"}

        _state["progress_text"] = "执行回测引擎..."
        _state["progress"] = 65

        # 3. 获取基准数据
        bench_df = None
        try:
            bench_df = dm.get_daily_kline(req.benchmark, start_date=fetch_start, end_date=end)
        except Exception as e:
            logger.warning("获取基准数据失败: %s", e)

        # 4. 运行回测引擎
        # risk_per_trade=0.05 与命令行回测脚本一致, 控制单笔风险敞口 (影响仓位大小)
        # benchmark_df 不传入 engine.run: 避免 MarketFilter 在大盘空头时减仓,
        # 与命令行回测脚本保持一致 (命令行不传 benchmark, 不做大盘减仓)
        # benchmark_df_for_memory: 传入已拉取的 bench_df, 仅用于 OutcomeRecord 记录超额收益
        # forced_regime: 请求级 regime 覆盖, 不写盘 user_preferences.json
        engine = BacktestEngine(
            initial_capital=req.initial_capital,
            lookback_days=120,
            position_ratio=0.3,
            signal_dedup_days=5,
            risk_per_trade=0.05,
            atr_stop_mult=2.5,
            group_config=gc,
            forced_regime=forced_regime,
            benchmark_df_for_memory=bench_df,
        )
        metrics = engine.run(
            data_map,
            benchmark_df=None,
            start_date=req.start_date,
            end_date=end,
        )

        # 单独计算基准指标 (仅用于展示 benchmark_return / alpha, 不影响仓位)
        if bench_df is not None and engine.daily_values is not None:
            bench_series = BacktestEngine._align_benchmark(bench_df, engine.daily_values.index)
            if bench_series is not None and len(bench_series) > 0:
                metrics = compute_metrics(
                    daily_values=engine.daily_values,
                    trades=engine.position_mgr.closed_trades,
                    initial_capital=engine.initial_capital,
                    benchmark_values=bench_series,
                )

        _state["progress_text"] = "格式化结果..."
        _state["progress"] = 90

        # 5. 格式化交易记录 (含已平仓 + 未平仓)
        trades_out = []
        for t in engine.position_mgr.closed_trades:
            info = stock_info.get(t.symbol, {})
            trades_out.append({
                "symbol": t.symbol,
                "name": info.get("name", t.symbol),
                "group": info.get("group", ""),
                "mode": req.mode,
                "entry_date": str(t.entry_date),
                "entry_price": round(t.entry_price, 3),
                "exit_date": str(t.exit_date) if t.exit_date else "",
                "exit_price": round(t.exit_price, 3) if t.exit_price else None,
                "shares": t.shares,
                "cost": round(t.entry_price * t.shares, 2),
                "pnl": round(t.pnl, 2),
                "pnl_pct": round(t.pnl_pct * 100, 2),
                "holding_days": t.holding_days,
                "signal": t.entry_signal,
                "exit_signal": t.exit_signal,
                "commission": round(t.commission, 2),
                "status": "closed",
            })

        # 加入未平仓持仓 (回测结束时仍持有的仓位)
        # 使用 daily_values 的最后一个日期作为最后日期
        last_date = None
        if metrics.daily_values is not None and len(metrics.daily_values) > 0:
            last_date = metrics.daily_values.index[-1]
            # 获取最后一天的收盘价
            from src.backtest.calendar import TradingCalendar
            calendar = TradingCalendar(data_map)
            prices_last = calendar.get_closing_prices(data_map, last_date)
            for symbol, t in engine.position_mgr.open_positions.items():
                info = stock_info.get(symbol, {})
                current_price = prices_last.get(symbol)
                # 计算浮动盈亏 (未扣除卖出手续费)
                unrealized_pnl = (current_price - t.entry_price) * t.shares - t.commission if current_price else 0
                unrealized_pnl_pct = (unrealized_pnl / (t.entry_price * t.shares) * 100) if t.shares > 0 else 0
                trades_out.append({
                    "symbol": symbol,
                    "name": info.get("name", symbol),
                    "group": info.get("group", ""),
                    "mode": req.mode,
                    "entry_date": str(t.entry_date),
                    "entry_price": round(t.entry_price, 3),
                    "exit_date": "",  # 未平仓
                    "exit_price": round(current_price, 3) if current_price else None,
                    "shares": t.shares,
                    "cost": round(t.entry_price * t.shares, 2),
                    "pnl": round(unrealized_pnl, 2),
                    "pnl_pct": round(unrealized_pnl_pct, 2),
                    "holding_days": (last_date - t.entry_date).days if hasattr(last_date, '__sub__') else 0,
                    "signal": t.entry_signal,
                    "exit_signal": "未平仓",
                    "commission": round(t.commission, 2),
                    "status": "open",
                })

        # 按买入日期排序
        trades_out.sort(key=lambda x: x["entry_date"])

        # 6. 格式化日净值序列 (供前端绘图)
        daily_values = metrics.daily_values
        dates = []
        values = []
        if daily_values is not None and len(daily_values) > 0:
            for d, v in daily_values.items():
                dates.append(str(d)[:10])
                values.append(round(float(v), 2))

        # 基准净值
        bench_dates = []
        bench_values = []
        if metrics.benchmark_return != 0 and hasattr(engine, '_bench_series'):
            # 从 metrics 重新对齐
            pass

        _state["progress"] = 100
        _state["progress_text"] = "完成"

        result = {
            "status": "done",
            "metrics": {
                "total_return": round(metrics.total_return, 6),
                "annual_return": round(metrics.annual_return, 6),
                "max_drawdown": round(metrics.max_drawdown, 6),
                "sharpe_ratio": round(metrics.sharpe_ratio, 4),
                "trade_count": metrics.trade_count,
                "win_rate": round(metrics.win_rate, 4),
                "profit_factor": round(metrics.profit_factor, 4) if metrics.profit_factor != float('inf') else 999.0,
                "total_pnl": round(metrics.total_pnl, 2),
                "initial_capital": metrics.initial_capital,
                "final_value": round(metrics.final_value, 2),
                "win_count": metrics.win_count,
                "avg_holding_days": round(metrics.avg_holding_days, 1),
                "volatility": round(metrics.volatility, 4),
                "benchmark_return": round(metrics.benchmark_return, 6),
                "alpha": round(metrics.alpha, 6),
            },
            "trades": trades_out,
            "daily_values": {"dates": dates, "values": values},
            "mode": req.mode,
            "groups": req.groups,
        }

        _state["result"] = result
        return result

    except Exception as e:
        logger.exception("回测执行异常")
        return {"status": "error", "message": str(e)}
    finally:
        _state["running"] = False
        set_mode(prev_mode)  # 恢复原运行时模式
        _state["lock"].release()


@router.get("/status", summary="查询回测状态")
def get_status():
    return {
        "running": _state["running"],
        "progress": _state["progress"],
        "progress_text": _state["progress_text"],
        "has_result": _state["result"] is not None,
    }
