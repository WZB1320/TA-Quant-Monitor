"""
回测引擎 — 逐日滚动, 信号触发, 模拟交易

流程:
  1. 对每只股票, 从 lookback 天后开始逐日计算信号
  2. 信号触发当日生成, T+1 以次日开盘价执行
  3. 每日记录组合净值
  4. 回测结束计算绩效指标

架构:
  本类仅负责回测主循环编排, 具体职责委托给独立组件:
    - TradingCalendar: 交易日历构建 + O(1) 日期定位
    - RegimeDetector: 市场体制检测 (统一入口, 消除双套逻辑)
    - MarketFilter: 大盘 MA60 过滤
    - SignalExecutor: T+1 信号执行
  各组件可通过构造函数注入, 便于测试替换 (依赖倒置)。
"""
from datetime import date
from typing import Dict, Optional

import pandas as pd

from src.signal_engine import SignalEngine, SignalResult
from src.signal_engine.filter import SignalFilter
from src.config.group_config import GroupConfig
from src.memory import StrategyMemory
from .position import PositionManager
from .broker import Broker
from .metrics import compute_metrics, BacktestMetrics
from .calendar import TradingCalendar
from .regime_detector import RegimeDetector
from .market_filter import MarketFilter
from .signal_executor import SignalExecutor


class BacktestEngine:
    """回测引擎 — 仅负责主循环编排"""

    def __init__(self,
                 initial_capital: float = 100000,
                 lookback_days: int = 120,
                 position_ratio: float = 0.3,
                 commission_rate: float = 0.00025,
                 stamp_tax: float = 0.001,
                 slippage: float = 0.0001,
                 signal_dedup_days: int = 5,
                 risk_per_trade: float = 0.015,
                 atr_stop_mult: float = 2.5,
                 # 依赖注入参数 (可选, 默认内部创建)
                 signal_engine: Optional[SignalEngine] = None,
                 broker: Optional[Broker] = None,
                 group_config: Optional[GroupConfig] = None,
                 regime_detector: Optional[RegimeDetector] = None,
                 forced_regime: Optional[str] = None,
                 memory: Optional[StrategyMemory] = None,
                 benchmark_df_for_memory: Optional[pd.DataFrame] = None):
        """
        Args:
            initial_capital: 初始资金
            lookback_days: 指标计算需要的最少天数
            position_ratio: 单只股票最大仓位占比
            commission_rate: 佣金率
            stamp_tax: 印花税率
            slippage: 滑点率
            signal_dedup_days: 信号去重天数
            risk_per_trade: 每笔交易风险敞口 (默认1.5%)
            atr_stop_mult: ATR 止损倍率 (默认2.5x)
            signal_engine: 可选, 注入自定义信号引擎 (测试用)
            broker: 可选, 注入自定义撮合器 (测试用)
            group_config: 可选, 注入自定义分组配置 (测试用)
            regime_detector: 可选, 注入自定义体制检测器 (测试用)
            forced_regime: 请求级 regime 覆盖, 不写盘 (None/"auto"/"trending"/"ranging")
            memory: 可选, 策略记忆层实例. None 则自动创建 (source="backtest")
            benchmark_df_for_memory: 可选, 仅用于 OutcomeRecord 记录基准收益,
                不影响交易逻辑 (MarketFilter 用的是 run() 的 benchmark_df 参数).
                路由层拉取后传入, 避免 engine 内部重复拉取.
        """
        self.initial_capital = initial_capital
        self.lookback_days = lookback_days
        self.position_ratio = position_ratio
        self.commission_rate = commission_rate
        self.risk_per_trade = risk_per_trade
        self.atr_stop_mult = atr_stop_mult

        # 依赖注入: 优先用外部传入, 否则内部创建
        self.group_config = group_config or GroupConfig()
        self.memory = memory or StrategyMemory(source="backtest")
        # benchmark 数据仅用于记忆层记录超额收益, 不参与交易决策
        self._benchmark_df_for_memory = benchmark_df_for_memory
        self.signal_engine = signal_engine or SignalEngine(
            dedup_days=signal_dedup_days,
            group_config=self.group_config,
            forced_regime=forced_regime,
            memory=self.memory,
        )
        # 确保注入的 signal_engine 也持有记忆层 (测试场景)
        if self.signal_engine.memory is None:
            self.signal_engine.memory = self.memory
        self.broker = broker or Broker(
            commission_rate=commission_rate,
            stamp_tax=stamp_tax,
            slippage=slippage,
        )
        self.regime_detector = regime_detector or RegimeDetector()

        self.position_mgr: Optional[PositionManager] = None
        self.daily_values: Optional[pd.Series] = None
        self.metrics: Optional[BacktestMetrics] = None

    def run(self, data_map: Dict[str, pd.DataFrame],
            benchmark_df: Optional[pd.DataFrame] = None,
            start_date: Optional[str] = None,
            end_date: Optional[str] = None,
            ) -> BacktestMetrics:
        """
        执行回测

        Args:
            data_map: {symbol: DataFrame} 每只股票的全量日线数据 (含 lookback 预热数据)
            benchmark_df: 基准指数的日线数据 (可选)
            start_date: 回测起始日期 (YYYY-MM-DD), 仅该日期后产生交易, 之前仅用于指标预热
            end_date: 回测结束日期 (YYYY-MM-DD)

        Returns:
            BacktestMetrics 绩效指标
        """
        # ── 初始化 ──
        self.position_mgr = PositionManager(
            initial_capital=self.initial_capital,
            position_ratio=self.position_ratio,
            commission_rate=self.commission_rate,
            risk_per_trade=self.risk_per_trade,
            atr_stop_mult=self.atr_stop_mult,
        )

        # 重置信号引擎的过滤器状态，避免多次 run() 之间状态残留
        self.signal_engine.filter = SignalFilter(
            dedup_days=self.signal_engine.filter.dedup_days,
            market_ma60_filter=self.signal_engine.filter.market_ma60_filter,
            cooldown_days=self.signal_engine.filter.cooldown_days,
        )

        # 构建交易日历 (O(1) 日期定位)
        calendar = TradingCalendar(data_map)
        # 大盘 MA60 过滤器 (pandas rolling, O(N))
        market_filter = MarketFilter(benchmark_df)
        # 信号执行器
        executor = SignalExecutor(
            broker=self.broker,
            position_mgr=self.position_mgr,
            signal_engine=self.signal_engine,
            group_config=self.group_config,
            market_filter=market_filter,
            calendar=calendar,
        )

        all_dates = calendar.all_dates

        # 解析回测区间 (仅该区间内产生交易, 之前仅用于指标预热)
        from datetime import datetime as _dt
        bt_start = _dt.strptime(start_date, "%Y-%m-%d").date() if start_date else None
        bt_end = _dt.strptime(end_date, "%Y-%m-%d").date() if end_date else None

        # ── 逐日回测 ──
        values = {}
        pending_signals: Dict = {}

        # 策略记忆层: 入场信号元数据 (用于 outcome 关联) + 已记录交易计数
        self._entry_signal_meta: Dict[str, dict] = {}
        self._last_recorded_trade_count = 0

        for i, today in enumerate(all_dates):
            # 回测结束日之后的日子无需处理, 直接跳出
            if bt_end and today > bt_end:
                break

            # 0. 每日更新市场体制 → 仓位管理器自适应
            regime = self.regime_detector.detect(benchmark_df, today, calendar=None)
            self.position_mgr.set_regime(regime)

            # 1. 检查持仓止损/移动止盈 (自适应倍率)
            prices_today = calendar.get_closing_prices(data_map, today)
            for symbol in list(self.position_mgr.open_positions.keys()):
                if symbol in prices_today:
                    score = self._get_position_score(symbol, data_map, today, calendar)
                    closed = self.position_mgr.check_stop_loss(
                        symbol, prices_today[symbol], today,
                        signal_score=score,
                    )
                    if closed is not None:
                        self.signal_engine.filter.record_exit(symbol, today)
                        if closed.pnl > 0:
                            self.signal_engine.filter.record_win(symbol)
                        else:
                            self.signal_engine.filter.record_loss(symbol, today)

            # 记录止损/止盈平仓的 outcome
            self._record_new_outcomes(today, regime)

            # 2. T+1 执行: 取出昨天暂存的信号, 以今天开盘价执行
            if i > 0:
                prev_date = all_dates[i - 1]
                prev_signals = pending_signals.pop(prev_date, {})
                if prev_signals:
                    positions_before = set(self.position_mgr.open_positions.keys())
                    executor.execute(prev_signals, data_map, prev_date)
                    # 记录新开仓信号元数据 (用于 outcome 关联)
                    for symbol, result in prev_signals.items():
                        if (symbol in self.position_mgr.open_positions
                                and symbol not in positions_before):
                            self._entry_signal_meta[symbol] = {
                                "analysis_date": prev_date,
                                "level": result.level.label,
                                "score": result.score,
                            }
                    # 记录信号平仓的 outcome
                    self._record_new_outcomes(today, regime)

            # 3. 对每只股票, 判断今日是否可计算信号
            #    仅在回测区间内产生新信号, 之前的日子只用于指标预热
            signals_today: Dict[str, SignalResult] = {}
            in_backtest_range = (bt_start is None or today >= bt_start)

            if in_backtest_range:
                for symbol, df in data_map.items():
                    idx = calendar.locate(symbol, today)
                    if idx is None or idx < self.lookback_days:
                        continue

                    df_slice = df.iloc[:idx + 1]

                    try:
                        result = self.signal_engine.analyze(symbol, df_slice,
                                                            analysis_date=today)
                    except Exception:
                        continue

                    if result.level.is_actionable:
                        signals_today[symbol] = result

            # 4. 暂存今天的信号, 明天执行
            pending_signals[today] = signals_today

            # 5. 记录当日净值 (仅回测区间内记录, 避免预热期污染曲线)
            if in_backtest_range:
                values[today] = self.position_mgr.total_value(prices_today)

        # ── 回测结束: 执行最后一天未处理的信号 ──
        executor.flush(pending_signals, data_map)

        # 最后再记录一次净值 (使用回测区间内最后一个实际交易日, 确保持仓市值被正确计入)
        if all_dates:
            # 找到 <= bt_end 的最后一个交易日 (避免 bt_end 非交易日时拿不到收盘价)
            if bt_end:
                valid_dates = [d for d in all_dates if d <= bt_end]
                last_bt_date = valid_dates[-1] if valid_dates else all_dates[-1]
            else:
                last_bt_date = all_dates[-1]
            # 记录 flush 阶段平仓的 outcome
            last_regime = self.regime_detector.detect(benchmark_df, last_bt_date, calendar=None)
            self._record_new_outcomes(last_bt_date, last_regime)
            prices_last = calendar.get_closing_prices(data_map, last_bt_date)
            values[last_bt_date] = self.position_mgr.total_value(prices_last)

        # ── 构建日净值序列 ──
        self.daily_values = pd.Series(values).sort_index()

        # ── 基准处理 ──
        bench_series = None
        if benchmark_df is not None:
            bench_series = self._align_benchmark(benchmark_df, self.daily_values.index)

        # ── 计算指标 ──
        self.metrics = compute_metrics(
            daily_values=self.daily_values,
            trades=self.position_mgr.closed_trades,
            initial_capital=self.initial_capital,
            benchmark_values=bench_series,
        )
        return self.metrics

    def _get_position_score(self, symbol: str, data_map: Dict[str, pd.DataFrame],
                            today, calendar: TradingCalendar) -> Optional[float]:
        """计算持仓股当日信号得分, 用于自适应移动止盈"""
        df = data_map.get(symbol)
        if df is None:
            return None
        idx = calendar.locate(symbol, today)
        if idx is None or idx < self.lookback_days:
            return None
        df_slice = df.iloc[:idx + 1]
        try:
            indicator_results = self.signal_engine.pipeline.run(df_slice)
            from src.signal_engine.scorer import Scorer
            group_weights = self.group_config.get_regime_weights(symbol)
            return Scorer().score(indicator_results, regime_weights=group_weights)
        except Exception:
            return None

    @staticmethod
    def _align_benchmark(bench_df: pd.DataFrame, dates) -> pd.Series:
        """将基准日线对齐到回测日期"""
        if "close" not in bench_df.columns:
            return None
        bench_df = bench_df.copy()
        if "date" in bench_df.columns:
            bench_df["date"] = pd.to_datetime(bench_df["date"]).dt.date
        bench_series = bench_df.set_index("date")["close"]
        target_dates = []
        for d in dates:
            if isinstance(d, pd.Timestamp):
                target_dates.append(d.date())
            elif isinstance(d, date):
                target_dates.append(d)
            else:
                target_dates.append(pd.Timestamp(d).date())
        bench_series = bench_series.reindex(target_dates, method="ffill").dropna()
        if len(bench_series) > 0:
            bench_series = bench_series / bench_series.iloc[0]
        return bench_series

    # ── 策略记忆层: 结果记录 ──

    _EXIT_REASON_MAP = [
        ("ATR硬止损", "atr_hard_stop"),
        ("ATR移动止盈", "atr_trailing"),
        ("安全网", "safety_net"),
    ]

    def _record_new_outcomes(self, today, regime: str) -> None:
        """记录自上次调用以来新平仓的交易结果到记忆层

        在止损检查后、信号执行后、flush 后各调用一次,
        通过对比 closed_trades 增量捕获所有平仓路径.

        benchmark_5d_return 使用 self._benchmark_df_for_memory 计算,
        与 run() 的 benchmark_df 参数完全解耦 (不影响交易逻辑).
        """
        new_trades = self.position_mgr.closed_trades[self._last_recorded_trade_count:]
        self._last_recorded_trade_count = len(self.position_mgr.closed_trades)

        for trade in new_trades:
            meta = self._entry_signal_meta.pop(trade.symbol, {})
            analysis_date = meta.get("analysis_date")
            signal_level = meta.get("level")
            signal_score = meta.get("score")

            self.memory.record_outcome({
                "signal_ref": {
                    "symbol": trade.symbol,
                    "analysis_date": str(analysis_date) if analysis_date else None,
                    "run_id": self.memory.run_id,
                },
                "signal_level_at_entry": signal_level,
                "signal_score_at_entry": round(signal_score, 2) if signal_score is not None else None,
                "symbol": trade.symbol,
                "entry_date": str(trade.entry_date) if trade.entry_date else None,
                "entry_price": round(trade.entry_price, 4) if trade.entry_price else None,
                "exit_date": str(trade.exit_date) if trade.exit_date else None,
                "exit_price": round(trade.exit_price, 4) if trade.exit_price else None,
                "shares": trade.shares,
                "pnl": round(trade.pnl, 2),
                "pnl_pct": round(trade.pnl_pct, 4),
                "holding_days": trade.holding_days,
                "commission_total": round(trade.commission, 2),
                "exit_reason": self._categorize_exit_reason(trade.exit_signal),
                "exit_reason_detail": trade.exit_signal,
                "market_context_exit": {
                    "regime_at_exit": regime,
                    "benchmark_5d_return": self._benchmark_5d_return(
                        self._benchmark_df_for_memory, today),
                },
            })

    def _categorize_exit_reason(self, exit_signal: str) -> str:
        """将退出信号文本映射为类别标签"""
        for keyword, category in self._EXIT_REASON_MAP:
            if keyword in exit_signal:
                return category
        if "score=" in exit_signal:
            return "signal_exit"
        return "other"

    def _benchmark_5d_return(self, benchmark_df, today) -> Optional[float]:
        """计算基准指数近 5 日涨幅 (无基准数据时返回 None)"""
        if benchmark_df is None or benchmark_df.empty:
            return None
        try:
            if "date" not in benchmark_df.columns or "close" not in benchmark_df.columns:
                return None
            bench = benchmark_df.copy()
            bench["date"] = pd.to_datetime(bench["date"]).dt.date
            recent = bench.loc[bench["date"] <= today].tail(6)
            if len(recent) < 2:
                return None
            return round((recent["close"].iloc[-1] / recent["close"].iloc[0]) - 1.0, 4)
        except Exception:
            return None
