"""
回测引擎 — 逐日滚动, 信号触发, 模拟交易

流程:
  1. 对每只股票, 从 lookback 天后开始逐日计算信号
  2. 信号触发当日生成, T+1 以次日开盘价执行
  3. 每日记录组合净值
  4. 回测结束计算绩效指标
"""
from datetime import date, timedelta
from typing import Dict, List, Optional, Tuple
import pandas as pd
import numpy as np

from src.signal_engine import SignalEngine, SignalResult
from src.signal_engine.signals import SignalLevel
from src.signal_engine.filter import SignalFilter
from src.config.group_config import GroupConfig
from .position import PositionManager, Trade, Side
from .broker import Broker
from .metrics import compute_metrics, BacktestMetrics


class BacktestEngine:
    """回测引擎"""

    def __init__(self, initial_capital: float = 100000,
                 lookback_days: int = 120,
                 position_ratio: float = 0.3,
                 commission_rate: float = 0.00025,
                 stamp_tax: float = 0.001,
                 slippage: float = 0.0001,
                 signal_dedup_days: int = 5,
                 risk_per_trade: float = 0.015,
                 atr_stop_mult: float = 2.5):
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
        """
        self.initial_capital = initial_capital
        self.lookback_days = lookback_days
        self.position_ratio = position_ratio
        self.commission_rate = commission_rate
        self.risk_per_trade = risk_per_trade
        self.atr_stop_mult = atr_stop_mult

        self.group_config = GroupConfig()
        self.signal_engine = SignalEngine(
            dedup_days=signal_dedup_days,
            group_config=self.group_config,
        )
        self.broker = Broker(
            commission_rate=commission_rate,
            stamp_tax=stamp_tax,
            slippage=slippage,
        )

        self.position_mgr: Optional[PositionManager] = None
        self.daily_values: Optional[pd.Series] = None
        self.metrics: Optional[BacktestMetrics] = None

    def run(self, data_map: Dict[str, pd.DataFrame],
            benchmark_df: Optional[pd.DataFrame] = None
            ) -> BacktestMetrics:
        """
        执行回测

        Args:
            data_map: {symbol: DataFrame} 每只股票的全量日线数据
            benchmark_df: 基准指数的日线数据 (可选)

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

        # 构建交易日历 + 大盘MA60过滤
        all_dates = self._build_calendar(data_map)
        self._market_ma60_trend = self._compute_market_ma60(benchmark_df, all_dates) if benchmark_df is not None else {}

        # ── 逐日回测 ──
        values = {}
        self._pending_signals: Dict = {}  # 显式初始化, 避免多次 run() 残留

        for i, today in enumerate(all_dates):
            # 0. 每日更新市场体制 → 仓位管理器自适应
            regime = self._get_market_regime(benchmark_df, today)
            self.position_mgr.set_regime(regime)

            # 1. 检查持仓止损/移动止盈 (自适应倍率)
            prices_today = self._get_closing_prices(data_map, today)
            for symbol in list(self.position_mgr.open_positions.keys()):
                if symbol in prices_today:
                    # 计算持仓股当日信号得分, 用于自适应移动止盈
                    score = self._get_position_score(symbol, data_map, today)
                    closed = self.position_mgr.check_stop_loss(
                        symbol, prices_today[symbol], today,
                        signal_score=score,
                    )
                    # 止损/止盈触发后, 记录冷却期 + 连亏保护
                    if closed is not None:
                        self.signal_engine.filter.record_exit(symbol, today)
                        if closed.pnl > 0:
                            self.signal_engine.filter.record_win(symbol)
                        else:
                            self.signal_engine.filter.record_loss(symbol, today)

            # 2. T+1 执行: 取出昨天暂存的信号, 以今天开盘价执行
            if i > 0:
                prev_date = all_dates[i - 1]
                prev_signals = self._pending_signals.pop(prev_date, {})
                if prev_signals:
                    self._apply_pending(prev_signals, data_map, prev_date)

            # 3. 对每只股票, 判断今日是否可计算信号
            signals_today: Dict[str, SignalResult] = {}

            for symbol, df in data_map.items():
                # 找到 today 在 df 中的位置
                idx = self._locate_date(df, today)
                if idx is None or idx < self.lookback_days:
                    continue  # 数据不够, 跳过

                # 截取到今天为止的数据
                df_slice = df.iloc[:idx + 1].copy()

                # 计算信号 (传入实际分析日期, 用于去重)
                try:
                    result = self.signal_engine.analyze(symbol, df_slice,
                                                        analysis_date=today)
                except Exception:
                    continue

                # 只保留有操作价值的信号
                if result.level.is_actionable:
                    signals_today[symbol] = result

            # 4. 暂存今天的信号, 明天执行
            self._pending_signals[today] = signals_today

            # 4. 记录当日净值
            prices_today = self._get_closing_prices(data_map, today)
            values[today] = self.position_mgr.total_value(prices_today)

        # ── 回测结束: 执行最后一天未处理的信号 ──
        self._flush_pending(data_map)

        # 最后再记录一次净值 (包含最后一天信号执行后的结果)
        if all_dates:
            prices_last = self._get_closing_prices(data_map, all_dates[-1])
            values[all_dates[-1]] = self.position_mgr.total_value(prices_last)

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

    # ── 内部: 执行最后一天未处理的信号 ──

    def _flush_pending(self, data_map: Dict[str, pd.DataFrame]):
        """回测结束时, 把最后一天未执行的信号执行掉"""
        if not self._pending_signals:
            return
        # 取出最后一天暂存的信号 (回测循环中已 pop 掉之前的)
        pending_keys = sorted(self._pending_signals.keys())
        if len(pending_keys) >= 1:
            last_date = pending_keys[-1]
            self._apply_pending(self._pending_signals[last_date], data_map, last_date)
            self._pending_signals.clear()

    def _apply_pending(self, signals: Dict[str, SignalResult],
                       data_map: Dict[str, pd.DataFrame], signal_date):
        """执行 pending 信号 (T+1 成交)"""
        if not signals:
            return

        # 按得分绝对值排序, 优先执行强信号
        sorted_signals = sorted(
            signals.items(),
            key=lambda x: abs(x[1].score), reverse=True
        )

        for symbol, result in sorted_signals:
            df = data_map.get(symbol)
            if df is None:
                continue

            # 找到 signal_date 在 df 中的位置
            idx = self._locate_date(df, signal_date)
            if idx is None:
                continue

            # 获取次日开盘信息
            next_info = self.broker.get_next_open(df, idx)
            if next_info is None:
                continue

            try:
                next_date = next_info["date"]
                if isinstance(next_date, pd.Timestamp):
                    next_date = next_date.date()
                elif isinstance(next_date, str):
                    next_date = pd.Timestamp(next_date).date()
            except Exception:
                continue

            open_price = next_info["open"]
            prev_close = next_info["prev_close"]

            # 涨跌停检查
            if not self.broker.can_trade(open_price, prev_close):
                continue

            if result.level.is_bullish:
                # 买入信号

                # 大盘MA60过滤: 指数在MA60下方时, 仓位减半
                market_dir = self._market_ma60_trend.get(next_date, 1)
                market_bearish = (market_dir == -1)

                # 冷却期检查: 最近卖出/止损后 4天内不反向买入
                if self.signal_engine.filter.is_in_cooldown(symbol, next_date, True):
                    continue

                # MA60 执行日重验证: 信号日MA60多头, 但执行日可能已转空
                exec_idx = self._locate_date(df, next_date)
                exec_atr = None
                if exec_idx is not None and exec_idx >= 60:
                    exec_df = df.iloc[:exec_idx + 1].copy()
                    try:
                        exec_ind = self.signal_engine.pipeline.run(exec_df)
                        exec_ma60 = exec_ind.get("MA60")
                        if exec_ma60 and exec_ma60.direction != 1:
                            continue  # 执行日MA60已转空, 放弃买入
                        # 获取执行日 ATR 用于动态仓位
                        exec_atr_ind = exec_ind.get("ATR")
                        if exec_atr_ind:
                            exec_atr = exec_atr_ind.values.get("atr")
                    except Exception:
                        pass

                if self.position_mgr.has_position(symbol):
                    continue  # 已有持仓, 不重复买

                buy_price = self.broker.buy_price(open_price)
                # 信号强度加成: 强买入 ×1.3, 普通买入 ×1.0
                signal_strength = 1.3 if "强买入" in result.level.label else 1.0
                # 分组专属仓位加成
                group_boost = self.group_config.get_max_per_stock_boost(symbol)
                signal_strength *= group_boost
                # 分组专属止损倍率
                group_stop_mult = self.group_config.get_atr_stop_mult(symbol)
                trade = self.position_mgr.open_long(
                    symbol=symbol,
                    entry_date=next_date,
                    entry_price=buy_price,
                    signal=f"{result.level.label} score={result.score:+.1f}",
                    atr_value=exec_atr,
                    bearish_market=market_bearish,
                    signal_strength=signal_strength,
                    atr_stop_mult=group_stop_mult,
                )

            elif result.level.is_bearish:
                # 卖出信号 → 平多仓
                if not self.position_mgr.has_position(symbol):
                    continue

                sell_price = self.broker.sell_price(open_price)
                closed = self.position_mgr.close_position(
                    symbol=symbol,
                    exit_date=next_date,
                    exit_price=sell_price,
                    signal=f"{result.level.label} score={result.score:+.1f}",
                )
                # 卖出后记录冷却期 + 连亏保护
                if closed is not None:
                    self.signal_engine.filter.record_exit(symbol, next_date)
                    if closed.pnl > 0:
                        self.signal_engine.filter.record_win(symbol)
                    else:
                        self.signal_engine.filter.record_loss(symbol, next_date)

    # ── 辅助方法 ──

    def _get_position_score(self, symbol: str, data_map: Dict[str, pd.DataFrame],
                            today) -> Optional[float]:
        """计算持仓股当日信号得分, 用于自适应移动止盈"""
        df = data_map.get(symbol)
        if df is None:
            return None
        idx = self._locate_date(df, today)
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
    def _get_market_regime(benchmark_df: pd.DataFrame, today) -> str:
        """根据基准指数 ADX 判断当前市场体制 (trending/transition/ranging)"""
        if benchmark_df is None or len(benchmark_df) < 30:
            return "transition"

        # 找到 today 在 benchmark_df 中的位置
        idx = BacktestEngine._locate_date(benchmark_df, today)
        if idx is None or idx < 28:
            return "transition"

        close = benchmark_df["close"].values[:idx + 1].astype(np.float64)

        # 计算 ADX(14): 先算 TR, +DM, -DM, 再平滑
        if len(close) < 28:
            return "transition"

        highs = benchmark_df["high"].values[:idx + 1].astype(np.float64) if "high" in benchmark_df.columns else close
        lows = benchmark_df["low"].values[:idx + 1].astype(np.float64) if "low" in benchmark_df.columns else close

        # 简化: 用价格波动率近似判断体制
        # 最近20日收益率标准差 vs 长期均值
        if len(close) < 40:
            return "transition"

        returns = np.diff(close[-21:]) / close[-21:-1]
        recent_vol = np.std(returns) * np.sqrt(252)

        # 最近60日波动率
        if len(close) >= 61:
            long_returns = np.diff(close[-61:]) / close[-61:-1]
            long_vol = np.std(long_returns) * np.sqrt(252)
        else:
            long_vol = recent_vol

        # 趋势强度: 20日均线 vs 60日均线
        ma20 = np.mean(close[-20:])
        ma60 = np.mean(close[-60:]) if len(close) >= 60 else ma20
        trend_strength = (ma20 - ma60) / ma60

        # 综合判断
        if abs(trend_strength) > 0.01 and recent_vol < long_vol * 1.5:
            return "trending"   # 有趋势 + 波动可控
        elif recent_vol > long_vol * 1.3:
            return "ranging"    # 高波动震荡
        elif abs(trend_strength) > 0.005:
            return "transition" # 弱趋势
        else:
            return "ranging"    # 无趋势

    @staticmethod
    def _build_calendar(data_map: Dict[str, pd.DataFrame]) -> List:
        """构建交易日历 (多只股票的日期并集, 排序)"""
        all_dates = set()
        for df in data_map.values():
            # 从 date 列提取日期
            dates = df["date"] if "date" in df.columns else df.index
            for d in dates:
                if isinstance(d, pd.Timestamp):
                    all_dates.add(d.date())
                elif isinstance(d, date):
                    all_dates.add(d)
                elif isinstance(d, str):
                    all_dates.add(pd.Timestamp(d).date())
        return sorted(all_dates)

    @staticmethod
    def _locate_date(df: pd.DataFrame, target) -> Optional[int]:
        """找到 target 日期在 df 中的位置索引"""
        if df.index.name == "date" or isinstance(df.index, pd.DatetimeIndex):
            # 日期索引
            if isinstance(target, date) and not isinstance(target, pd.Timestamp):
                target = pd.Timestamp(target)
            try:
                loc = df.index.get_loc(target)
                if isinstance(loc, slice):
                    return loc.start
                if isinstance(loc, np.ndarray):
                    return int(loc[0]) if len(loc) > 0 else None
                return int(loc)
            except KeyError:
                return None
        elif "date" in df.columns:
            # date 列 (可能是字符串, 统一转字符串比较)
            target_str = target.strftime("%Y-%m-%d") if isinstance(target, date) else str(target)
            mask = df["date"] == target_str
            if mask.any():
                return int(mask.idxmax())
            return None
        return None

    @staticmethod
    def _get_closing_prices(data_map: Dict[str, pd.DataFrame],
                            today) -> Dict[str, float]:
        """获取当天收盘价"""
        prices = {}
        for symbol, df in data_map.items():
            idx = BacktestEngine._locate_date(df, today)
            if idx is not None:
                try:
                    prices[symbol] = float(df.iloc[idx]["close"])
                except (KeyError, IndexError):
                    pass
        return prices

    @staticmethod
    def _compute_market_ma60(benchmark_df: pd.DataFrame,
                              all_dates: list) -> dict:
        """计算大盘(沪深300)每日MA60趋势, 用于过滤熊市"""
        result = {}
        if benchmark_df is None or len(benchmark_df) < 60:
            return result

        close = benchmark_df["close"].values.astype(np.float64)
        ma60_vals = np.zeros_like(close)
        for i in range(60, len(close)):
            ma60_vals[i] = np.mean(close[i - 60:i])

        dates_col = benchmark_df["date"].values
        for target in all_dates:
            if isinstance(target, pd.Timestamp):
                target = target.date()
            for i in range(len(dates_col) - 1, -1, -1):
                d = dates_col[i]
                if hasattr(d, 'date'):
                    d = d.date()
                elif isinstance(d, str):
                    d = pd.Timestamp(d).date()
                if d <= target and i >= 60 and ma60_vals[i] > 0:
                    result[target] = 1 if close[i] > ma60_vals[i] else -1
                    break
        return result

    @staticmethod
    def _align_benchmark(bench_df: pd.DataFrame,
                         dates) -> pd.Series:
        """将基准日线对齐到回测日期"""
        if "close" not in bench_df.columns:
            return None
        # 统一日期类型: 将 bench_df 的 date 列转为 date 对象
        bench_df = bench_df.copy()
        if "date" in bench_df.columns:
            bench_df["date"] = pd.to_datetime(bench_df["date"]).dt.date
        bench_series = bench_df.set_index("date")["close"]
        # 将 target dates 也转为 date 对象
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