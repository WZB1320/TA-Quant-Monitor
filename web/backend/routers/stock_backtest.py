"""个股信号回测路由 — 对单只自选股逐日运行信号分析

本质: 在用户指定时间段内, 逐日截取历史数据切片调用 SignalEngine.analyze,
输出每日的信号结果表格 (日期/收盘价/信号/得分/置信度/判断缘由)。

关键设计:
  1. set_mode(BACKTEST) — SignalFilter 纯内存历史, 不写 signal_history.json, 不污染实时数据
  2. filter 历史按日累积 — cooldown/dedup 跨日生效, 模拟真实逐日分析
  3. analysis_date 传入 — 让 filter 的去重和冷却检查使用正确历史日期
  4. 零侵入 — 不修改 SignalEngine / filter / 现有路由; 模式切换仅在本请求内
"""
import logging
import os
import sys
import threading
from datetime import datetime, timedelta
from typing import Optional

from fastapi import APIRouter
from pydantic import BaseModel
import pandas as pd

# 确保项目根目录在 sys.path
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from src.data_fetcher import DataManager, Watchlist
from src.config.group_config import GroupConfig
from src.config.runtime_mode import set_mode, RuntimeMode, get_mode
from src.signal_engine import SignalEngine
from src.indicators import IndicatorPipeline
from src.memory import StrategyMemory
from src.backtest.position_tracker import PositionStateTracker

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/stock-backtest", tags=["个股信号回测"])


class StockBacktestRequest(BaseModel):
    code: str                        # 自选股中的股票代码
    mode: str = "base"               # base / trending / ranging
    start_date: Optional[str] = None  # YYYY-MM-DD
    end_date: Optional[str] = None    # YYYY-MM-DD


# ── 全局状态 (单例, 同一时间只允许一个个股信号回测任务) ──
_state = {
    "running": False,
    "progress": 0,
    "progress_text": "",
    "result": None,
    "lock": threading.Lock(),
}


@router.post("/run", summary="个股信号回测 — 逐日信号分析")
def run_stock_backtest(req: StockBacktestRequest):
    """对单只自选股在指定时间段内逐日运行信号分析"""
    if not _state["lock"].acquire(blocking=False):
        return {"status": "error", "message": "已有信号回测任务正在运行"}

    # 切换到 BACKTEST 模式 (filter 纯内存, 不污染实时数据)
    prev_mode = get_mode()
    set_mode(RuntimeMode.BACKTEST)

    try:
        _state["running"] = True
        _state["progress"] = 5
        _state["progress_text"] = "验证股票..."

        # 1. 验证股票在自选股中
        wl = Watchlist()
        all_stocks = wl.get_all()
        stock_info = next((s for s in all_stocks if s["code"] == req.code), None)
        if stock_info is None:
            return {"status": "error", "message": f"股票 {req.code} 不在自选股列表中"}

        gc = GroupConfig()
        group = gc.get_group(req.code)
        stock_name = stock_info.get("name", req.code)

        # 2. 策略模式作为请求级参数, 不再写盘 user_preferences.json
        # base → auto (ADX 自动判断); trending/ranging → 该模式
        forced_regime = req.mode if req.mode in ("trending", "ranging") else "auto"

        _state["progress"] = 15
        _state["progress_text"] = "加载日线数据..."

        # 3. 拉取股票日线 (起点往前推 240 自然日作为 lookback 预热, 保证 MA60 等指标可用)
        dm = DataManager()
        if req.start_date:
            start_dt = datetime.strptime(req.start_date, "%Y-%m-%d")
            fetch_start = (start_dt - timedelta(days=240)).strftime("%Y-%m-%d")
        else:
            # 默认最近半年
            fetch_start = (datetime.now() - timedelta(days=400)).strftime("%Y-%m-%d")
            req_start_date_obj = datetime.now() - timedelta(days=180)
        end = req.end_date or datetime.now().strftime("%Y-%m-%d")

        df = dm.get_daily_kline(req.code, start_date=fetch_start, end_date=end)
        if df is None or len(df) < 120:
            return {"status": "error", "message": f"股票 {req.code} 数据不足 (需至少120根K线)"}

        # 确保 date 列为 datetime
        if "date" in df.columns:
            df["date"] = pd.to_datetime(df["date"])

        # 4. 确定回测时间段的起止索引
        #    find_start_idx: 第一个 date >= req.start_date 的行
        if req.start_date:
            start_threshold = pd.Timestamp(req.start_date)
            candidate = df[df["date"] >= start_threshold]
            if candidate.empty:
                return {"status": "error", "message": f"起始日期 {req.start_date} 之后无数据"}
            start_idx = candidate.index[0]
        else:
            # 默认最近 180 自然日
            threshold = pd.Timestamp.now() - pd.Timedelta(days=180)
            candidate = df[df["date"] >= threshold]
            start_idx = candidate.index[0] if not candidate.empty else max(120, len(df) // 2)

        if req.end_date:
            end_threshold = pd.Timestamp(req.end_date)
            end_candidate = df[df["date"] <= end_threshold]
            if end_candidate.empty:
                return {"status": "error", "message": f"结束日期 {req.end_date} 之前无数据"}
            end_idx = end_candidate.index[-1]
        else:
            end_idx = len(df) - 1

        # 需要 start_idx >= 120 (lookback)
        if start_idx < 120:
            start_idx = 120

        total_days = end_idx - start_idx + 1
        if total_days <= 0:
            return {"status": "error", "message": "时间区间无效"}

        logger.info("个股信号回测: %s %s, 回测区间 %s ~ %s, 共 %d 个交易日",
                    req.code, stock_name,
                    df.iloc[start_idx]["date"].strftime("%Y-%m-%d") if start_idx < len(df) else "?",
                    df.iloc[end_idx]["date"].strftime("%Y-%m-%d") if end_idx < len(df) else "?",
                    total_days)

        _state["progress"] = 25
        _state["progress_text"] = f"逐日分析 (共{total_days}天)..."

        # 5. 创建 SignalEngine, 清除 filter 历史 (BACKTEST 模式下仅清内存)
        # forced_regime: 请求级 regime 覆盖, 不写盘 user_preferences.json
        # 每次个股回测独立一个 memory 文件, 便于按 run_id 检索
        stock_memory = StrategyMemory(source="backtest")
        logger.info("个股回测记忆 run_id=%s, file=%s",
                    stock_memory.run_id, stock_memory.file_path)
        sig_engine = SignalEngine(
            dedup_days=5, group_config=gc,
            forced_regime=forced_regime,
            memory=stock_memory,
        )
        sig_engine.filter.clear_history()
        pipeline = IndicatorPipeline()  # 用于提取展示指标

        # 创建持仓状态追踪器 (让止损/止盈/冷却期在个股回测中生效)
        tracker = PositionStateTracker(
            symbol=req.code,
            group_config=gc,
            signal_filter=sig_engine.filter,
        )

        # 6. 逐日分析
        results = []
        for i, idx in enumerate(range(start_idx, end_idx + 1)):
            row = df.iloc[idx]
            analysis_date = row["date"].date() if hasattr(row["date"], 'date') else row["date"]
            close = float(row["close"])

            # 截取截止到当日的数据切片
            df_slice = df.iloc[:idx + 1]

            try:
                r = sig_engine.analyze(req.code, df_slice, analysis_date=analysis_date)

                # 提取关键展示指标 (与信号看板一致)
                ind = {}
                atr_value = None  # 用于持仓状态追踪器的止损计算
                try:
                    ir = pipeline.run(df_slice)
                    ma60 = ir.get("MA60")
                    if ma60:
                        ma60_v = ma60.values.get("ma60")
                        if ma60_v is not None:
                            ind["ma60"] = round(float(ma60_v), 2)
                            ind["ma60_dir"] = "多头" if close > ma60_v else "空头"
                    rsi = ir.get("RSI")
                    if rsi:
                        rsi_v = rsi.values.get("rsi")
                        if rsi_v is not None:
                            ind["rsi"] = round(float(rsi_v), 1)
                    macd = ir.get("MACD")
                    if macd:
                        dif = macd.values.get("dif")
                        dea = macd.values.get("dea")
                        if dif is not None:
                            ind["dif"] = round(float(dif), 3)
                        if dea is not None:
                            ind["dea"] = round(float(dea), 3)
                    adx = ir.get("ADX")
                    if adx:
                        adx_v = adx.values.get("adx")
                        if adx_v is not None:
                            ind["adx"] = round(float(adx_v), 1)
                    vol_ratio = ir.get("VOL_RATIO")
                    if vol_ratio:
                        vr = vol_ratio.values.get("volume_ratio")
                        if vr is not None:
                            ind["volume_ratio"] = round(float(vr), 2)
                    atr = ir.get("ATR")
                    if atr:
                        atr_v = atr.values.get("atr")
                        if atr_v is not None:
                            atr_value = float(atr_v)
                            if close > 0:
                                ind["atr_pct"] = round(atr_value / close * 100, 2)
                except Exception as e:
                    logger.warning("提取指标失败 %s @ %s: %s", req.code, analysis_date, e)

                # 执行状态 (简洁分类标签, 具体原因保留在 execution.reason 供 tooltip 显示)
                execution = r.execution
                if execution is None:
                    from src.signal_engine.classifier import ExecutionConstraint
                    execution = ExecutionConstraint()

                if r.hard_filter_blocked:
                    exec_status = "硬过滤"
                elif not r.level.is_actionable:
                    exec_status = "无需操作"
                elif execution.is_executable:
                    exec_status = "可执行"
                elif not execution.score_passes:
                    exec_status = "得分不达标"
                elif execution.in_cooldown:
                    exec_status = "冷却期内"
                elif execution.suspended:
                    exec_status = "连亏暂停"
                elif execution.is_duplicate:
                    exec_status = "信号去重"
                else:
                    exec_status = "不可执行"

                # 持仓状态追踪 (让止损/止盈/冷却期在个股回测中生效)
                pos_info = tracker.process_day(r, close, atr_value, analysis_date)

                results.append({
                    "date": str(analysis_date)[:10],
                    "code": req.code,
                    "name": stock_name,
                    "group": group,
                    "close": round(close, 2),
                    "level": r.level.name,
                    "label": r.level.label,
                    "score": round(float(r.score), 1),
                    "confidence": round(float(r.confidence), 2),
                    "reason": r.reason,
                    "details": r.details,
                    "block_detail": r.block_detail or "",
                    "hard_filter_blocked": bool(r.hard_filter_blocked),
                    "block_reason": r.block_reason or "",
                    "initial_level": r.initial_level.name if r.initial_level else r.level.name,
                    "demotion_chain": list(r.demotion_chain) if r.demotion_chain else [],
                    "execution": {
                        "executable": bool(execution.is_executable),
                        "status": exec_status,
                        "reason": execution.blocking_reason,
                    },
                    "indicators": ind,
                    # 持仓状态机字段
                    "position_state": pos_info['state'],
                    "action": pos_info['action'],
                    "entry_price": pos_info['entry_price'],
                    "stop_loss_price": pos_info['stop_loss_price'],
                    "trailing_stop_price": pos_info['trailing_stop_price'],
                    "highest_price": pos_info['highest_price'],
                    "holding_pnl_pct": pos_info['holding_pnl_pct'],
                    "holding_days": pos_info['holding_days'],
                    "cooldown_remaining": pos_info['cooldown_remaining'],
                    "exit_reason": pos_info['exit_reason'],
                    "exit_pnl_pct": pos_info['exit_pnl_pct'],
                })
            except Exception as e:
                logger.warning("分析 %s @ %s 失败: %s", req.code, analysis_date, e)
                results.append({
                    "date": str(analysis_date)[:10],
                    "code": req.code,
                    "name": stock_name,
                    "group": group,
                    "close": round(close, 2),
                    "level": "NEUTRAL",
                    "label": "观望",
                    "score": 0.0,
                    "confidence": 0.0,
                    "reason": f"分析异常: {e}",
                    "details": str(e),
                    "block_detail": "分析异常",
                    "hard_filter_blocked": False,
                    "block_reason": "",
                    "initial_level": "NEUTRAL",
                    "demotion_chain": [],
                    "execution": {"executable": False, "status": "不可执行", "reason": "分析异常"},
                    "indicators": {},
                    # 持仓状态机字段 (异常时保持原状态, 不推进)
                    "position_state": tracker.state,
                    "action": "NONE",
                    "entry_price": None,
                    "stop_loss_price": None,
                    "trailing_stop_price": None,
                    "highest_price": None,
                    "holding_pnl_pct": None,
                    "holding_days": None,
                    "cooldown_remaining": None,
                    "exit_reason": None,
                    "exit_pnl_pct": None,
                })

            # 更新进度
            progress = 25 + int((i + 1) / total_days * 70)
            _state["progress"] = progress
            if (i + 1) % 10 == 0:
                _state["progress_text"] = f"已分析 {i + 1}/{total_days} 天..."

        _state["progress"] = 100
        _state["progress_text"] = "完成"

        # 7. 统计汇总
        bullish = sum(1 for r in results if r["level"] in ("STRONG_BUY", "BUY", "WEAK_BUY"))
        bearish = sum(1 for r in results if r["level"] in ("WEAK_SELL", "SELL", "STRONG_SELL"))
        neutral = sum(1 for r in results if r["level"] == "NEUTRAL")
        actionable = sum(1 for r in results if r["execution"]["executable"])

        result = {
            "status": "done",
            "stock": {"code": req.code, "name": stock_name, "group": group},
            "mode": req.mode,
            "summary": {
                "total": len(results),
                "bullish": bullish,
                "neutral": neutral,
                "bearish": bearish,
                "actionable": actionable,
            },
            "trade_summary": tracker.get_trade_summary(),
            "results": results,
            "memory_run_id": stock_memory.run_id,
            "memory_file": stock_memory.file_path,
        }

        _state["result"] = result
        return result

    except Exception as e:
        logger.exception("个股信号回测异常")
        return {"status": "error", "message": str(e)}
    finally:
        _state["running"] = False
        # 恢复原运行时模式
        set_mode(prev_mode)
        _state["lock"].release()


@router.get("/status", summary="查询个股信号回测状态")
def get_status():
    return {
        "running": _state["running"],
        "progress": _state["progress"],
        "progress_text": _state["progress_text"],
        "has_result": _state["result"] is not None,
    }


@router.get("/result", summary="获取最近一次个股信号回测结果")
def get_result():
    if _state["result"] is None:
        return {
            "status": "no_result",
            "message": "尚未运行过信号回测",
            "results": [],
        }
    return _state["result"]


@router.get("/stocks", summary="获取可选股票列表")
def get_stocks():
    """返回自选股列表, 供前端股票选择器使用"""
    wl = Watchlist()
    all_stocks = wl.get_all()
    gc = GroupConfig()
    stocks = []
    for s in all_stocks:
        stocks.append({
            "code": s["code"],
            "name": s.get("name", s["code"]),
            "market": s.get("market", ""),
            "group": gc.get_group(s["code"]),
        })
    stocks.sort(key=lambda x: (x["group"], x["code"]))
    return {"stocks": stocks}
