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
                 forced_regime: Optional[str] = None):
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
        """
        self.initial_capital = initial_capital
        self.lookback_days = lookback_days
        self.position_ratio = position_ratio
        self.commission_rate = commission_rate
        self.risk_per_trade = risk_per_trade
        self.atr_stop_mult = atr_stop_mult

        # 依赖注入: 优先用外部传入, 否则内部创建
        self.group_config = group_config or GroupConfig()
        self.signal_engine = signal_engine or SignalEngine(
            dedup_days=signal_dedup_days,
            group_config=self.group_config,
            forced_regime=forced_regime,
        )
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

        for i, today in enumerate(all_dates):
            # 跳过回测结束日之后的日子
            if bt_end and today > bt_end:
                continue

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

            # 2. T+1 执行: 取出昨天暂存的信号, 以今天开盘价执行
            if i > 0:
                prev_date = all_dates[i - 1]
                prev_signals = pending_signals.pop(prev_date, {})
                if prev_signals:
                    executor.execute(prev_signals, data_map, prev_date)

            # 3. 对每只股票, 判断今日是否可计算信号
            #    仅在回测区间内产生新信号, 之前的日子只用于指标预热
            signals_today: Dict[str, SignalResult] = {}
            in_backtest_range = (bt_start is None or today >= bt_start)

            if in_backtest_range:
                for symbol, df in data_map.items():
                    idx = calendar.locate(symbol, today)
                    if idx is None or idx < self.lookback_days:
                        continue

                    df_slice = df.iloc[:idx + 1].copy()

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
        df_slice = df.iloc[:idx + 1].copy()
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
