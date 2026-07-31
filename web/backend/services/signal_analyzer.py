"""信号分析服务 — 封装自选股信号分析逻辑

流程:
  读取自选股 → 拉取日线数据 → SignalEngine 批量分析 → 格式化输出
"""
import json
import logging
import os
import sys
import threading
from datetime import datetime
from typing import Optional

# 确保项目根目录在 sys.path
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from src.data_fetcher import DataManager, Watchlist
from src.signal_engine import SignalEngine
from src.signal_engine.signals import SignalLevel
from src.config.group_config import GroupConfig
from src.indicators import IndicatorPipeline
from src.memory import StrategyMemory

logger = logging.getLogger(__name__)


class SignalAnalyzer:
    """信号分析器 — 单例模式"""

    _instance = None
    _lock = threading.Lock()

    def __new__(cls):
        with cls._lock:
            if cls._instance is None:
                cls._instance = super().__new__(cls)
                cls._instance._initialized = False
            return cls._instance

    def __init__(self):
        if self._initialized:
            return
        self._initialized = True
        self._running = False
        self._progress = 0
        self._total = 0
        self._result: Optional[dict] = None
        self._analyzed_at: Optional[str] = None
        self._run_lock = threading.Lock()
        # 进程级单例 memory: live 路径所有信号分析共享同一个 run_id
        # 重启后端 = 新 run_id, 自然形成会话边界
        self._memory = StrategyMemory(source="live")
        logger.info("Live 记忆层已初始化, run_id=%s, file=%s",
                    self._memory.run_id, self._memory.file_path)

    @property
    def is_running(self) -> bool:
        return self._running

    @property
    def progress(self) -> tuple:
        return (self._progress, self._total)

    def get_result(self) -> Optional[dict]:
        return self._result

    def get_status(self) -> dict:
        return {
            "running": self._running,
            "progress": self._progress,
            "total": self._total,
            "analyzed_at": self._analyzed_at,
            "has_result": self._result is not None,
            "memory_run_id": self._memory.run_id,
            "memory_file": self._memory.file_path,
        }

    def run_analysis(self, group: str = "", user_regime: str = "auto") -> dict:
        """同步运行信号分析（在后台线程中调用）

        Args:
            group: 指定分组名称，为空则分析全部自选股
            user_regime: 手动模式 — "trending"=趋势上涨 / "ranging"=震荡 / "auto"=自动判断
        """
        logger.info("run_analysis 开始: group=%s, regime=%s", group, user_regime)
        if not self._run_lock.acquire(blocking=False):
            return {"status": "already_running", "message": "分析任务正在运行中"}

        try:
            self._running = True
            self._progress = 0

            # 1. 读取自选股
            wl = Watchlist()
            all_stocks = wl.get_all()
            logger.info("自选股数量: %d", len(all_stocks))
            if not all_stocks:
                self._running = False
                return {"status": "error", "message": "自选股列表为空，请先添加股票"}

            # 按分组筛选
            gc = GroupConfig()
            if group:
                filtered = [s for s in all_stocks if gc.get_group(s["code"]) == group]
                logger.info("分组筛选: group=%s, 筛选前=%d, 筛选后=%d", group, len(all_stocks), len(filtered))
                if not filtered:
                    self._running = False
                    return {"status": "error", "message": f"分组「{group}」中没有股票"}
                all_stocks = filtered

            self._total = len(all_stocks)

            # 2. 拉取日线数据
            dm = DataManager()
            # regime 作为请求级参数传入 SignalEngine, 不再写盘 user_preferences.json
            # 避免污染其他页面 (回测页/个股回测页) 的 regime 配置
            sig_engine = SignalEngine(
                dedup_days=5, group_config=gc,
                forced_regime=user_regime,
                memory=self._memory,
            )
            sig_engine.filter.clear_history()  # 新分析前清除旧去重记录
            pipeline = IndicatorPipeline()

            data_map = {}
            for i, stock in enumerate(all_stocks):
                code = stock["code"]
                try:
                    df = dm.get_daily_kline(code, start_date="2024-01-01")
                    if df is not None and len(df) >= 120:
                        data_map[code] = df
                except Exception as e:
                    logger.warning("获取 %s 数据失败: %s", code, e)
                self._progress = i + 1

            if not data_map:
                self._running = False
                return {"status": "error", "message": "所有股票数据获取失败"}

            # 3. 批量信号分析
            results = sig_engine.analyze_batch(data_map)

            # 4. 格式化输出
            formatted = []
            bullish = 0
            bearish = 0
            neutral = 0
            actionable = 0

            for r in results:
                df = data_map.get(r.symbol)
                latest = df.iloc[-1] if df is not None else None
                close = float(latest["close"]) if latest is not None else 0.0
                latest_date = str(latest["date"])[:10] if latest is not None else ""

                # 获取股票名称和分组
                stock_info = next((s for s in all_stocks if s["code"] == r.symbol), {})
                name = stock_info.get("name", r.symbol)
                market = stock_info.get("market", "")
                group = gc.get_group(r.symbol) if gc else ""

                # 统计
                if r.level.is_bullish:
                    bullish += 1
                elif r.level.is_bearish:
                    bearish += 1
                else:
                    neutral += 1
                if r.level.is_actionable:
                    actionable += 1

                # 提取关键指标
                indicators = self._extract_indicators(r, df, pipeline)

                # 执行约束 (来自 classifier 接入后的 SignalResult.execution)
                execution = r.execution
                if execution is None:
                    from src.signal_engine.classifier import ExecutionConstraint
                    execution = ExecutionConstraint()

                # 执行状态 (独立于 7 级信号, 仅表示"能否操作")
                # 返回简洁分类标签, 具体原因(含数值)保留在 execution.reason 供 tooltip 显示
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

                formatted.append({
                    "code": r.symbol,
                    "name": name,
                    "market": market,
                    "group": group,
                    "close": round(close, 2),
                    "date": latest_date,
                    "level": r.level.name,
                    "label": r.level.label,
                    "score": round(float(r.score), 1),
                    "confidence": round(float(r.confidence), 2),
                    "reason": r.reason,
                    "details": r.details,
                    "actionable": bool(r.level.is_actionable),
                    # ── 执行约束 (独立于信号级别, 替代旧 action 标签) ──
                    "execution": {
                        "executable": bool(execution.is_executable),
                        "status": exec_status,
                        "reason": execution.blocking_reason,
                    },
                    "initial_level": r.initial_level.name if r.initial_level else r.level.name,
                    "demotion_chain": list(r.demotion_chain) if r.demotion_chain else [],
                    "hard_filter_blocked": bool(r.hard_filter_blocked),
                    "block_reason": r.block_reason,
                    "block_detail": r.block_detail,
                    "indicators": indicators,
                })

            # 按得分降序排列
            formatted.sort(key=lambda x: x["score"], reverse=True)

            self._analyzed_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            self._result = {
                "status": "done",
                "analyzed_at": self._analyzed_at,
                "group": group or "",
                "summary": {
                    "bullish": bullish,
                    "neutral": neutral,
                    "bearish": bearish,
                    "actionable": actionable,
                    "total": len(formatted),
                },
                "results": formatted,
            }

            return self._result

        except Exception as e:
            logger.exception("信号分析异常")
            return {"status": "error", "message": str(e)}
        finally:
            self._running = False
            self._run_lock.release()

    @staticmethod
    def _extract_indicators(result, df, pipeline) -> dict:
        """从信号结果中提取关键指标"""
        indicators = {}
        try:
            if df is not None and not df.empty:
                ir = pipeline.run(df)

                # MA60
                ma60 = ir.get("MA60")
                if ma60:
                    ma60_v = ma60.values.get("ma60")
                    close = float(df.iloc[-1]["close"])
                    if ma60_v is not None:
                        indicators["ma60"] = round(float(ma60_v), 2)
                        indicators["ma60_dir"] = "多头" if close > ma60_v else "空头"

                # RSI
                rsi = ir.get("RSI")
                if rsi:
                    rsi_v = rsi.values.get("rsi")
                    if rsi_v is not None:
                        indicators["rsi"] = round(float(rsi_v), 1)

                # MACD
                macd = ir.get("MACD")
                if macd:
                    dif = macd.values.get("dif")
                    dea = macd.values.get("dea")
                    if dif is not None:
                        indicators["dif"] = round(float(dif), 3)
                    if dea is not None:
                        indicators["dea"] = round(float(dea), 3)

                # ADX
                adx = ir.get("ADX")
                if adx:
                    adx_v = adx.values.get("adx")
                    if adx_v is not None:
                        indicators["adx"] = round(float(adx_v), 1)

                # 量比
                vol_ratio = ir.get("VOL_RATIO")
                if vol_ratio:
                    vr = vol_ratio.values.get("volume_ratio")
                    if vr is not None:
                        indicators["volume_ratio"] = round(float(vr), 2)

                # ATR
                atr = ir.get("ATR")
                if atr:
                    atr_v = atr.values.get("atr")
                    if atr_v is not None and close > 0:
                        indicators["atr_pct"] = round(float(atr_v) / close * 100, 2)
        except Exception as e:
            logger.warning("提取指标失败 %s: %s", result.symbol, e)

        return indicators
