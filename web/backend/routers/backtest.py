"""回测路由 — 运行回测 / 获取结果"""
import logging
import os
import sys
import threading
from datetime import datetime, date
from typing import Optional

import pandas as pd
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
from src.backtest.rotation_strategy import RotationStrategy

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/backtest", tags=["回测分析"])


# ── P5 组合配置 (2026-08-07 固化) ──
# 分组独立回测: 每组独立engine, 按权重分配资金, 合并daily_values计算组合级指标.
# 与命令行脚本(ytd_2026_group_comparison.py)配置完全一致, 确保前端回测结果可复现.
PORTFOLIO_WEIGHTS = {
    "科技成长型": 0.40,    # 核心引擎 (2026YTD +41%)
    "消费稳健型": 0.05,    # 全组启用 (P2均值回归, Alpha为负但参与组合分散)
    "周期资源型": 0.375,   # 核心引擎 (2026YTD +38%, 降5%让位消费/医药)
    "医药创新型": 0.05,    # 全组启用 (P3均值回归, Alpha为负但参与组合分散)
    "机械制造型": 0.10,    # P4c启用 (动量轮动策略, 测试窗Alpha-9.97%)
}
# 周期组仅趋势市交易 (ADX>25), 避免震荡市频繁止损
REGIMES_CFG = {"周期资源型": {"trending"}}
# 科技组ATR收紧至1.8 (默认2.0), 控制回撤
ATR_OVERRIDE = {"科技成长型": 1.8}
# 组级12%回撤保护 (引擎真实降仓, 非事后净值调整)
DD_CONFIG = BacktestEngine.DEFAULT_DD_PROTECTION_CONFIG


class BacktestRequest(BaseModel):
    groups: list[str] = []          # 展示分组(仅过滤前端交易明细展示, 不影响回测参与)
    mode: str = "base"              # base / trending / ranging
    start_date: Optional[str] = None  # YYYY-MM-DD
    end_date: Optional[str] = None    # YYYY-MM-DD
    initial_capital: float = 100000
    benchmark: str = "sh.000300"       # 基准指数代码 (沪深300, 与回测脚本一致)


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

        # 筛选展示语义: req.groups 仅用于前端交易明细展示过滤, 不影响回测参与.
        # 回测始终跑全部活跃组(权重>0), 确保收益与P5基线一致, 避免选子集导致现金稀释.

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
            from datetime import timedelta
            start_dt = datetime.strptime(req.start_date, "%Y-%m-%d")
            if start_dt >= datetime(2025, 12, 1):
                # P5 验证场景 (2026-01-01 起): 硬编码与脚本一致, 确保结果可复现.
                # 对应 verify_frontend_p5.py / ytd_2026_group_comparison.py 的 DATA_START.
                fetch_start = "2025-06-01"
            else:
                # 其他场景 (如训练窗 2024-07-01): 动态往前推 240 天, 确保有足够 lookback
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

        # 4. 运行回测引擎 — P5 分组独立回测 + 权重合并
        # 每组独立创建 BacktestEngine, 按PORTFOLIO_WEIGHTS分配资金, 应用差异化配置:
        #   - 周期组: trade_regimes={trending} 仅趋势市交易
        #   - 科技组: atr_stop_mult=1.8 收紧止损
        #   - 所有组: dd_protection_config 12%引擎真实降仓
        # 权重0%的组(消费/医药/机械)跳过不交易, 资金留作现金缓冲(17.5%).
        # benchmark_df 传入 engine.run: RegimeDetector 需要它判断 trending/ranging,
        # 周期组的 trade_regimes={trending} 过滤依赖此判断.

        # 确定参与的组: 始终用全部活跃组(权重>0), 不受 req.groups 影响.
        # req.groups 仅用于前端交易明细展示过滤 (筛选展示语义).
        active_groups = [g for g, w in PORTFOLIO_WEIGHTS.items() if w > 0]

        if not active_groups:
            return {"status": "error", "message": "全部分组权重均为0(已暂停), 无可回测的组"}

        # 权重: 始终用原始 PORTFOLIO_WEIGHTS, 不归一化.
        # P5 策略设计了 17.5% 现金缓冲作为风控组成部分, 归一化会抹掉现金缓冲,
        # 导致满仓运行 + MarketFilter 减仓 → 收益大幅下降 (33% → 7%).
        # 与命令行脚本(ytd_2026_group_comparison.py)和验证脚本(verify_frontend_p5.py)保持一致.
        weights_normalized = {g: PORTFOLIO_WEIGHTS[g] for g in active_groups}

        portfolio_daily_values = None
        all_closed_trades = []
        all_open_positions = {}
        total_invested = 0

        for idx, group_name in enumerate(active_groups):
            weight = weights_normalized[group_name]
            group_capital = req.initial_capital * weight
            total_invested += group_capital

            # 该组的股票
            group_codes = [c for c, info in stock_info.items() if info["group"] == group_name]
            group_codes = [c for c in group_codes if c in data_map]
            if not group_codes:
                continue

            _state["progress_text"] = f"执行回测引擎: {group_name} ({idx+1}/{len(active_groups)})..."

            # 重置 GroupConfig 单例, 避免多次 run() 之间状态残留
            GroupConfig._instance = None
            GroupConfig._config = None

            # 读取该组的策略配置
            group_cfg = gc._groups.get(group_name, {})
            strategy_mode = group_cfg.get("strategy_mode", "trend_following")

            if strategy_mode == "rotation":
                # ── 动量轮动策略 (机械组专用) ──
                # 使用独立RotationStrategy, 不走BacktestEngine
                rotation = RotationStrategy(initial_capital=group_capital)
                sub_map = {c: data_map[c] for c in group_codes}
                rot_result = rotation.run(
                    sub_map,
                    bench_df=bench_df,
                    start_date=req.start_date or "2024-07-01",
                    end_date=end,
                )

                # 合并 daily_values
                if rot_result["daily_values"] is not None:
                    if portfolio_daily_values is None:
                        portfolio_daily_values = rot_result["daily_values"].copy()
                    else:
                        portfolio_daily_values = portfolio_daily_values.add(
                            rot_result["daily_values"], fill_value=0)

                # 收集交易记录 (RotationTrade对象与Trade对象格式兼容)
                all_closed_trades.extend(rot_result["closed_trades"])
                all_open_positions.update(rot_result["open_positions"])

            else:
                # ── 标准趋势跟踪/均值回归策略 ──
                # 桥接: strategy_config.mean_reversion_exit → BacktestEngine.mean_reversion_config
                mean_reversion_config = group_cfg.get("mean_reversion_exit") or None

                group_engine = BacktestEngine(
                    initial_capital=group_capital,
                    lookback_days=120,
                    position_ratio=0.3,
                    commission_rate=0.00025,
                    stamp_tax=0.001,
                    slippage=0.0001,
                    signal_dedup_days=5,
                    risk_per_trade=0.05,
                    atr_stop_mult=ATR_OVERRIDE.get(group_name, 2.0),
                    group_config=gc,
                    forced_regime=forced_regime,
                    benchmark_df_for_memory=bench_df,
                    trade_regimes=REGIMES_CFG.get(group_name),
                    dd_protection_config=DD_CONFIG,
                    mean_reversion_config=mean_reversion_config,
                )

                sub_map = {c: data_map[c] for c in group_codes}
                group_engine.run(
                    sub_map,
                    benchmark_df=bench_df,  # 传基准数据: RegimeDetector 需要它判断 trending/ranging
                                            # (周期组 trade_regimes={trending} 过滤依赖 regime 判断)
                    start_date=req.start_date,
                    end_date=end,
                )

                # 合并 daily_values (各组净值按日期对齐相加)
                if group_engine.daily_values is not None:
                    if portfolio_daily_values is None:
                        portfolio_daily_values = group_engine.daily_values.copy()
                    else:
                        portfolio_daily_values = portfolio_daily_values.add(
                            group_engine.daily_values, fill_value=0)

                # 收集交易记录 (含部分平仓的回撤保护降仓记录)
                all_closed_trades.extend(group_engine.position_mgr.closed_trades)
                all_open_positions.update(group_engine.position_mgr.open_positions)

            _state["progress"] = 65 + int((idx + 1) / len(active_groups) * 20)

        # 加上现金部分 (权重0%的组的资金 + 归一化后的剩余现金)
        cash = req.initial_capital - total_invested
        if portfolio_daily_values is not None:
            portfolio_daily_values = portfolio_daily_values + cash

        # 用合并后的 daily_values 计算组合级 metrics
        if portfolio_daily_values is None or len(portfolio_daily_values) == 0:
            return {"status": "error", "message": "回测未产生净值数据"}

        if bench_df is not None:
            bench_series = BacktestEngine._align_benchmark(
                bench_df, portfolio_daily_values.index)
            if bench_series is not None and len(bench_series) > 0:
                metrics = compute_metrics(
                    daily_values=portfolio_daily_values,
                    trades=all_closed_trades,
                    initial_capital=req.initial_capital,
                    benchmark_values=bench_series,
                )
            else:
                metrics = compute_metrics(
                    daily_values=portfolio_daily_values,
                    trades=all_closed_trades,
                    initial_capital=req.initial_capital,
                )
        else:
            metrics = compute_metrics(
                daily_values=portfolio_daily_values,
                trades=all_closed_trades,
                initial_capital=req.initial_capital,
            )

        _state["progress_text"] = "格式化结果..."
        _state["progress"] = 90

        # 5. 格式化交易记录 (含已平仓 + 未平仓)
        # 合并各组的 closed_trades, 按 entry_date 排序
        trades_out = []
        for t in all_closed_trades:
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
            for symbol, t in all_open_positions.items():
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
        if metrics.benchmark_return != 0 and bench_df is not None:
            # 从 bench_df 重新对齐到组合净值日期
            try:
                bench_aligned = BacktestEngine._align_benchmark(
                    bench_df, portfolio_daily_values.index)
                if bench_aligned is not None and len(bench_aligned) > 0:
                    for d, v in bench_aligned.items():
                        bench_dates.append(str(d)[:10])
                        bench_values.append(round(float(v), 2))
            except Exception:
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
            "benchmark_values": {"dates": bench_dates, "values": bench_values},
            "mode": req.mode,
            "groups": active_groups,
            "display_groups": req.groups,
            "portfolio_config": {
                "weights": {g: w for g, w in PORTFOLIO_WEIGHTS.items() if w > 0},
                "cash_ratio": round(cash / req.initial_capital, 4),
            },
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
