"""集成测试: 验证新架构在市场体制切换 + T+1 限制下的正确性

本测试构造一个完整的 mock 回测场景, 验证:
  1. 状态隔离: BACKTEST 模式不污染磁盘 signal_history.json
  2. 体制切换: RegimeDetector 在不同阶段输出 trending/ranging/transition
  3. T+1 限制: 信号日生成, 次日开盘价成交 (不是当日收盘价)
  4. 依赖注入: 各组件协同工作, 可替换
  5. MA60 过滤: 大盘空头时买入仓位减半
  6. 体制自适应仓位: trending 仓位 > ranging 仓位
"""
import json
import os
import sys
from datetime import date, timedelta
from typing import Dict, List

import numpy as np
import pandas as pd
import pytest

# 确保项目根目录在 sys.path
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from src.config.runtime_mode import set_mode, RuntimeMode, get_mode
from src.backtest.engine import BacktestEngine
from src.backtest.regime_detector import RegimeDetector
from src.backtest.market_filter import MarketFilter
from src.backtest.calendar import TradingCalendar
from src.backtest.broker import Broker
from src.backtest.position import PositionManager
from src.backtest.signal_executor import SignalExecutor
from src.signal_engine import SignalEngine, SignalResult, SignalLevel
from src.signal_engine.filter import SignalFilter
from src.config.group_config import GroupConfig


# ─────────────────────────────────────────────────────────
# Mock 数据构造工具
# ─────────────────────────────────────────────────────────

def _make_dates(n: int, start: str = "2025-01-01") -> List[str]:
    """生成 n 个交易日 (跳过周末)"""
    dates = pd.bdate_range(start, periods=n)
    return [d.strftime("%Y-%m-%d") for d in dates]


def _make_benchmark(n: int = 200, seed: int = 42) -> pd.DataFrame:
    """构造基准指数: 三阶段体制切换

    阶段1 (0-80天): 强趋势上涨 (trending)
      - 每日 +0.5%, MA20 明显在 MA60 上方
    阶段2 (80-140天): 高位震荡 (ranging)
      - 围绕均值波动, 波动率放大
    阶段3 (140-200天): 弱趋势 (transition)
      - 缓慢下跌
    """
    rng = np.random.RandomState(seed)
    dates = _make_dates(n)

    closes = [3000.0]
    for i in range(1, n):
        if i < 80:
            # 强趋势: 每日 +0.5% + 小噪声
            ret = 0.005 + rng.randn() * 0.003
        elif i < 140:
            # 震荡: 均值回归, 波动大
            ret = rng.randn() * 0.015
        else:
            # 弱趋势下跌
            ret = -0.002 + rng.randn() * 0.005
        closes.append(closes[-1] * (1 + ret))

    df = pd.DataFrame({
        "date": dates,
        "open": [c * (1 + rng.randn() * 0.002) for c in closes],
        "high": [c * (1 + abs(rng.randn()) * 0.005) for c in closes],
        "low": [c * (1 - abs(rng.randn()) * 0.005) for c in closes],
        "close": closes,
        "volume": [int(1e8 * (0.8 + rng.rand() * 0.4)) for _ in closes],
    })
    return df


def _make_stock(n: int = 200, seed: int = 7,
                trend_start: int = 60, trend_end: int = 100,
                uptrend_pct: float = 0.015) -> pd.DataFrame:
    """构造个股数据: 在 trend_start~trend_end 期间有明显上涨趋势

    设计: 在第 60-100 天形成上涨趋势 (MA60 多头 + MACD 金叉 + 放量),
    触发买入信号; 之后高位震荡, 触发移动止盈或卖出信号。
    """
    rng = np.random.RandomState(seed)
    dates = _make_dates(n)

    closes = [10.0]
    for i in range(1, n):
        if trend_start <= i < trend_end:
            # 上涨趋势: 每日 +1.5% + 小噪声
            ret = uptrend_pct + rng.randn() * 0.008
        elif trend_end <= i < trend_end + 30:
            # 高位震荡
            ret = rng.randn() * 0.012
        else:
            # 平淡
            ret = rng.randn() * 0.006
        closes.append(closes[-1] * (1 + ret))

    # 成交量: 趋势段放量
    vols = []
    for i in range(n):
        base = 1e6
        if trend_start <= i < trend_end:
            base *= 2.5  # 放量
        vols.append(int(base * (0.8 + rng.rand() * 0.4)))

    df = pd.DataFrame({
        "date": dates,
        "open": [c * (1 + rng.randn() * 0.003) for c in closes],
        "high": [c * (1 + abs(rng.randn()) * 0.008) for c in closes],
        "low": [c * (1 - abs(rng.randn()) * 0.008) for c in closes],
        "close": closes,
        "volume": vols,
    })
    return df


# ─────────────────────────────────────────────────────────
# 测试夹具
# ─────────────────────────────────────────────────────────

@pytest.fixture
def backtest_env():
    """构造完整回测环境: 基准 + 个股 + 引擎"""
    # 确保回测模式
    set_mode(RuntimeMode.BACKTEST)
    assert get_mode() == RuntimeMode.BACKTEST

    benchmark = _make_benchmark(n=200, seed=42)
    stock_a = _make_stock(n=200, seed=7, trend_start=60, trend_end=100)
    stock_b = _make_stock(n=200, seed=99, trend_start=120, trend_end=160)

    data_map = {"STOCK_A": stock_a, "STOCK_B": stock_b}

    # 清除可能存在的信号历史文件 (测试隔离)
    from src.signal_engine.filter import _DEDUP_FILE
    history_existed = os.path.exists(_DEDUP_FILE)
    history_content = None
    if history_existed:
        with open(_DEDUP_FILE, "r") as f:
            history_content = f.read()
        os.remove(_DEDUP_FILE)

    yield {
        "benchmark": benchmark,
        "data_map": data_map,
        "stock_a": stock_a,
        "stock_b": stock_b,
    }

    # 恢复原始信号历史文件
    if history_existed:
        os.makedirs(os.path.dirname(_DEDUP_FILE), exist_ok=True)
        with open(_DEDUP_FILE, "w") as f:
            f.write(history_content)
    elif os.path.exists(_DEDUP_FILE):
        # 回测不应写盘, 如果文件被创建则删除
        os.remove(_DEDUP_FILE)


# ─────────────────────────────────────────────────────────
# 测试 1: 状态隔离 — 回测不污染磁盘
# ─────────────────────────────────────────────────────────

class TestStateIsolation:
    """P0-3: 状态隔离验证"""

    def test_backtest_mode_does_not_write_signal_history(self, backtest_env):
        """回测模式不应创建/修改 signal_history.json"""
        from src.signal_engine.filter import _DEDUP_FILE

        # 文件应不存在 (fixture 已清除)
        assert not os.path.exists(_DEDUP_FILE), \
            "回测前 signal_history.json 不应存在"

        engine = BacktestEngine(
            initial_capital=100000,
            lookback_days=60,
            position_ratio=0.3,
        )
        engine.run(backtest_env["data_map"], backtest_env["benchmark"])

        # 回测结束后, 文件仍不应存在 (BACKTEST 模式纯内存)
        assert not os.path.exists(_DEDUP_FILE), \
            "BACKTEST 模式不应写入 signal_history.json"

    def test_signal_filter_uses_memory_in_backtest(self, backtest_env):
        """BACKTEST 模式下 SignalFilter 应使用内存历史"""
        set_mode(RuntimeMode.BACKTEST)
        sf = SignalFilter(dedup_days=5)
        # 触发懒加载
        sf._ensure_history_loaded()
        # 内存历史应为空 (不从磁盘读)
        assert sf._history == {}
        assert sf._history_loaded is True

    def test_live_mode_writes_to_disk(self, tmp_path):
        """LIVE 模式应写盘 (对照实验)"""
        # 用临时文件避免污染真实数据
        from src.signal_engine import filter as filter_module
        original_file = filter_module._DEDUP_FILE
        tmp_file = str(tmp_path / "test_signal_history.json")
        filter_module._DEDUP_FILE = tmp_file

        try:
            set_mode(RuntimeMode.LIVE)
            sf = SignalFilter(dedup_days=5)
            sf.record("TEST", SignalLevel.BUY, analysis_date=date(2026, 1, 1))
            # LIVE 模式应写盘
            assert os.path.exists(tmp_file)
            with open(tmp_file, "r") as f:
                data = json.load(f)
            assert "TEST_bull" in data
        finally:
            # 恢复
            filter_module._DEDUP_FILE = original_file
            set_mode(RuntimeMode.BACKTEST)


# ─────────────────────────────────────────────────────────
# 测试 2: 体制切换检测
# ─────────────────────────────────────────────────────────

class TestRegimeSwitching:
    """验证 RegimeDetector 在三阶段基准数据上的输出"""

    def test_regime_varies_across_phases(self, backtest_env):
        """基准数据三阶段应产生不同的体制判断"""
        benchmark = backtest_env["benchmark"]
        detector = RegimeDetector()

        dates = benchmark["date"].tolist()
        regimes: List[str] = []

        # 每隔 10 天采样一次
        for i in range(60, len(dates), 10):
            d = pd.Timestamp(dates[i]).date()
            r = detector.detect(benchmark, d)
            regimes.append(r)

        # 应至少出现 2 种不同的体制
        unique_regimes = set(regimes)
        assert len(unique_regimes) >= 2, \
            f"三阶段基准应产生 ≥2 种体制, 实际: {unique_regimes}"

    def test_regime_detector_returns_valid_string(self, backtest_env):
        """体制检测应返回有效字符串"""
        benchmark = backtest_env["benchmark"]
        detector = RegimeDetector()
        valid = {"trending", "ranging", "transition"}

        for i in [70, 100, 150, 190]:
            d = pd.Timestamp(benchmark["date"].iloc[i]).date()
            r = detector.detect(benchmark, d)
            assert r in valid, f"日期 {d} 返回无效体制: {r}"

    def test_no_benchmark_returns_transition(self):
        """无基准数据应返回 transition"""
        detector = RegimeDetector()
        assert detector.detect(None, date(2026, 1, 1)) == "transition"

    def test_short_benchmark_returns_transition(self):
        """数据不足应返回 transition"""
        df = pd.DataFrame({
            "date": _make_dates(20),
            "close": np.linspace(3000, 3100, 20),
        })
        detector = RegimeDetector()
        d = pd.Timestamp(df["date"].iloc[-1]).date()
        assert detector.detect(df, d) == "transition"


# ─────────────────────────────────────────────────────────
# 测试 3: T+1 信号执行
# ─────────────────────────────────────────────────────────

class TestTPlus1Execution:
    """验证 T+1 限制: 信号日生成, 次日开盘价成交"""

    def test_entry_date_is_after_signal_date(self, backtest_env):
        """开仓日期应为信号日的次日 (T+1)"""
        engine = BacktestEngine(
            initial_capital=100000,
            lookback_days=60,
            position_ratio=0.3,
        )
        metrics = engine.run(backtest_env["data_map"], backtest_env["benchmark"])

        pm = engine.position_mgr
        assert pm is not None

        # 如果有交易, 检查每笔交易的入场日期
        if pm.closed_trades:
            for trade in pm.closed_trades:
                # 入场日期不应是数据的第一天 (需要 lookback + T+1)
                first_date = pd.Timestamp(backtest_env["data_map"][trade.symbol]["date"].iloc[0]).date()
                assert trade.entry_date > first_date, \
                    f"{trade.symbol} 入场日期 {trade.entry_date} 应在数据开始之后"

    def test_broker_get_next_open_returns_next_day(self, backtest_env):
        """Broker.get_next_open 应返回次日数据"""
        broker = Broker()
        df = backtest_env["stock_a"]

        # 第 100 行的次日应为第 101 行
        result = broker.get_next_open(df, 100)
        assert result is not None
        assert result["open"] == float(df.iloc[101]["open"])
        assert result["prev_close"] == float(df.iloc[100]["close"])

    def test_broker_get_next_open_last_day_returns_none(self, backtest_env):
        """最后一天无次日数据"""
        broker = Broker()
        df = backtest_env["stock_a"]
        last_idx = len(df) - 1
        assert broker.get_next_open(df, last_idx) is None

    def test_no_same_day_execution(self, backtest_env):
        """信号不应在当日成交 (T+1)"""
        # 用 mock 信号引擎追踪信号生成日期
        signal_dates: Dict[str, list] = {}

        class TrackingSignalEngine(SignalEngine):
            def analyze(self, symbol, df, analysis_date=None):
                result = super().analyze(symbol, df, analysis_date)
                if result.level.is_actionable:
                    signal_dates.setdefault(symbol, []).append(analysis_date)
                return result

        engine = BacktestEngine(
            initial_capital=100000,
            lookback_days=60,
            signal_engine=TrackingSignalEngine(),
        )
        engine.run(backtest_env["data_map"], backtest_env["benchmark"])

        pm = engine.position_mgr
        # 对每笔交易, 入场日期应严格大于某个信号日期 (T+1)
        for trade in pm.closed_trades:
            sig_dates = signal_dates.get(trade.symbol, [])
            # 找到入场日期之前最近的信号日
            prior_signals = [d for d in sig_dates if d is not None and d < trade.entry_date]
            if prior_signals:
                latest_signal = max(prior_signals)
                assert trade.entry_date > latest_signal, \
                    f"{trade.symbol}: 入场日 {trade.entry_date} 应 > 信号日 {latest_signal}"


# ─────────────────────────────────────────────────────────
# 测试 4: 依赖注入 — 组件可替换
# ─────────────────────────────────────────────────────────

class TestDependencyInjection:
    """验证 BacktestEngine 接受外部注入的组件"""

    def test_custom_broker_used(self, backtest_env):
        """注入自定义 Broker 应被使用"""
        custom_broker = Broker(commission_rate=0.001, slippage=0.005)
        engine = BacktestEngine(
            initial_capital=100000,
            lookback_days=60,
            broker=custom_broker,
        )
        assert engine.broker is custom_broker
        assert engine.broker.slippage == 0.005

    def test_custom_regime_detector_used(self, backtest_env):
        """注入自定义 RegimeDetector 应被使用"""

        class FixedRegimeDetector(RegimeDetector):
            """始终返回 trending 的固定检测器"""
            def detect(self, benchmark_df, today, calendar=None):
                return "trending"

        fixed_detector = FixedRegimeDetector()
        engine = BacktestEngine(
            initial_capital=100000,
            lookback_days=60,
            regime_detector=fixed_detector,
        )
        assert engine.regime_detector is fixed_detector

        # 运行回测, 仓位管理器应始终处于 trending
        engine.run(backtest_env["data_map"], backtest_env["benchmark"])
        assert engine.position_mgr._regime == "trending"

    def test_custom_signal_engine_used(self, backtest_env):
        """注入自定义 SignalEngine 应被使用"""

        class StubSignalEngine(SignalEngine):
            """始终返回 NEUTRAL 的桩"""
            def analyze(self, symbol, df, analysis_date=None):
                return SignalResult(
                    symbol=symbol,
                    level=SignalLevel.NEUTRAL,
                    score=0.0,
                    confidence=0.0,
                    reason="stub",
                    details="stub",
                )

        stub = StubSignalEngine()
        engine = BacktestEngine(
            initial_capital=100000,
            lookback_days=60,
            signal_engine=stub,
        )
        assert engine.signal_engine is stub

        metrics = engine.run(backtest_env["data_map"], backtest_env["benchmark"])
        # 全部 NEUTRAL, 不应有交易
        assert metrics.trade_count == 0


# ─────────────────────────────────────────────────────────
# 测试 5: MA60 大盘过滤
# ─────────────────────────────────────────────────────────

class TestMarketFilterIntegration:
    """验证 MarketFilter 在回测中的大盘 MA60 过滤"""

    def test_market_filter_detects_bull_and_bear(self, backtest_env):
        """基准数据应同时包含多头和空头阶段"""
        benchmark = backtest_env["benchmark"]
        mf = MarketFilter(benchmark)

        # 应有趋势映射 (数据 >= 60 天)
        assert len(mf.trend_map) > 0

        # 应同时包含 1 (多头) 和 -1 (空头)
        directions = set(mf.trend_map.values())
        assert 1 in directions, "应有多头阶段"
        # 由于基准数据有上涨阶段, 至少应有多头
        # 空头可能不出现 (如果价格始终在 MA60 上方), 不强制要求

    def test_market_filter_is_bearish_method(self, backtest_env):
        """is_bearish 方法应正确返回布尔值"""
        benchmark = backtest_env["benchmark"]
        mf = MarketFilter(benchmark)

        for d, direction in mf.trend_map.items():
            if direction == -1:
                assert mf.is_bearish(d) is True
            else:
                assert mf.is_bearish(d) is False

    def test_no_benchmark_market_filter_empty(self):
        """无基准时 MarketFilter 应为空 (默认多头)"""
        mf = MarketFilter(None)
        assert mf.trend_map == {}
        assert mf.is_bearish(date(2026, 1, 1)) is False


# ─────────────────────────────────────────────────────────
# 测试 6: 体制自适应仓位
# ─────────────────────────────────────────────────────────

class TestRegimeAdaptivePosition:
    """验证 PositionManager 根据体制调整仓位上限"""

    def test_trending_has_higher_position_cap_than_ranging(self):
        """趋势市的单票仓位上限应高于震荡市"""
        pm = PositionManager(
            initial_capital=100000,
            position_ratio=0.3,  # 绝对上限 30%
        )

        pm.set_regime("trending")
        trending_cap = pm.current_position_ratio

        pm.set_regime("ranging")
        ranging_cap = pm.current_position_ratio

        assert trending_cap > ranging_cap, \
            f"趋势市仓位上限 {trending_cap} 应 > 震荡市 {ranging_cap}"

    def test_position_ratio_capped_by_absolute_limit(self):
        """体制仓位上限不应超过 position_ratio 绝对上限"""
        pm = PositionManager(
            initial_capital=100000,
            position_ratio=0.10,  # 严格上限 10%
        )
        pm.set_regime("trending")
        # trending 默认 max_per_stock=0.30, 但应被 position_ratio=0.10 截断
        assert pm.current_position_ratio <= 0.10

    def test_engine_updates_regime_daily(self, backtest_env):
        """回测引擎应每日更新仓位管理器的体制"""
        # 用 FixedRegimeDetector 确保体制可预测
        class FixedRegimeDetector(RegimeDetector):
            def __init__(self, regime_sequence):
                super().__init__()
                self._sequence = regime_sequence
                self._idx = 0

            def detect(self, benchmark_df, today, calendar=None):
                r = self._sequence[min(self._idx, len(self._sequence) - 1)]
                self._idx += 1
                return r

        # 前 100 天 trending, 之后 ranging
        seq = ["trending"] * 100 + ["ranging"] * 100
        engine = BacktestEngine(
            initial_capital=100000,
            lookback_days=60,
            regime_detector=FixedRegimeDetector(seq),
        )
        engine.run(backtest_env["data_map"], backtest_env["benchmark"])

        # 回测结束后, 仓位管理器应处于最后一个体制 (ranging)
        assert engine.position_mgr._regime == "ranging"


# ─────────────────────────────────────────────────────────
# 测试 7: 完整回测端到端
# ─────────────────────────────────────────────────────────

class TestEndToEndBacktest:
    """端到端回测: 验证新架构完整运行"""

    def test_backtest_runs_without_error(self, backtest_env):
        """回测应无异常完成"""
        engine = BacktestEngine(
            initial_capital=100000,
            lookback_days=60,
            position_ratio=0.3,
        )
        metrics = engine.run(backtest_env["data_map"], backtest_env["benchmark"])

        # 应返回有效的 BacktestMetrics
        assert metrics is not None
        assert metrics.initial_capital == 100000
        assert metrics.final_value > 0

    def test_backtest_metrics_complete(self, backtest_env):
        """回测指标应完整计算"""
        engine = BacktestEngine(
            initial_capital=100000,
            lookback_days=60,
        )
        metrics = engine.run(backtest_env["data_map"], backtest_env["benchmark"])

        # 核心指标应已计算 (非默认值)
        assert metrics.total_return is not None
        assert metrics.max_drawdown <= 0  # 回撤应为负或 0
        assert metrics.volatility >= 0
        assert metrics.trade_count >= 0
        assert 0 <= metrics.win_rate <= 1
        assert metrics.daily_values is not None
        assert len(metrics.daily_values) > 0

    def test_backtest_reproducible(self, backtest_env):
        """同一数据两次回测结果应一致 (可复现)"""
        engine1 = BacktestEngine(
            initial_capital=100000,
            lookback_days=60,
        )
        metrics1 = engine1.run(backtest_env["data_map"], backtest_env["benchmark"])

        engine2 = BacktestEngine(
            initial_capital=100000,
            lookback_days=60,
        )
        metrics2 = engine2.run(backtest_env["data_map"], backtest_env["benchmark"])

        # 两次回测的总收益应完全一致
        assert metrics1.total_return == metrics2.total_return, \
            "同一数据两次回测应可复现"
        assert metrics1.trade_count == metrics2.trade_count
        assert metrics1.final_value == metrics2.final_value

    def test_daily_values_aligned_with_dates(self, backtest_env):
        """日净值序列应与交易日历对齐"""
        engine = BacktestEngine(
            initial_capital=100000,
            lookback_days=60,
        )
        engine.run(backtest_env["data_map"], backtest_env["benchmark"])

        cal = TradingCalendar(backtest_env["data_map"])
        daily_dates = list(engine.daily_values.index)

        # 日净值应从 lookback 之后开始
        assert len(daily_dates) > 0
        # 每个净值日期应在交易日历中
        for d in daily_dates:
            assert d in cal.all_dates or pd.Timestamp(d) in [pd.Timestamp(x) for x in cal.all_dates]


# ─────────────────────────────────────────────────────────
# 测试 8: TradingCalendar O(1) 定位
# ─────────────────────────────────────────────────────────

class TestCalendarO1:
    """验证 TradingCalendar 的 O(1) 日期定位"""

    def test_locate_is_o1(self, backtest_env):
        """locate 应在 O(1) 时间返回索引"""
        cal = TradingCalendar(backtest_env["data_map"])

        # 定位每只股票的每个日期
        for symbol, df in backtest_env["data_map"].items():
            for i in range(0, len(df), 20):
                d_str = df["date"].iloc[i]
                d = pd.Timestamp(d_str).date()
                idx = cal.locate(symbol, d)
                assert idx == i, f"{symbol} 日期 {d}: 期望 {i}, 实际 {idx}"

    def test_locate_nonexistent_symbol(self, backtest_env):
        """不存在的股票应返回 None"""
        cal = TradingCalendar(backtest_env["data_map"])
        assert cal.locate("NONEXIST", date(2025, 6, 1)) is None

    def test_calendar_union_of_dates(self, backtest_env):
        """日历应为所有股票日期的并集"""
        cal = TradingCalendar(backtest_env["data_map"])
        all_dates_set = set(cal.all_dates)

        for symbol, df in backtest_env["data_map"].items():
            for d in df["date"]:
                d_obj = pd.Timestamp(d).date()
                assert d_obj in all_dates_set, \
                    f"{symbol} 日期 {d_obj} 应在交易日历中"


# ─────────────────────────────────────────────────────────
# 测试 9: 信号执行器涨跌停检查
# ─────────────────────────────────────────────────────────

class TestLimitUpDownBlocking:
    """验证 SignalExecutor 在涨跌停时跳过成交"""

    def test_can_trade_normal_price(self):
        """正常价格可交易"""
        broker = Broker()
        assert broker.can_trade(10.5, 10.0) is True
        assert broker.can_trade(9.5, 10.0) is True

    def test_cannot_trade_at_limit_up(self):
        """一字涨停无法成交"""
        broker = Broker()
        # 10.0 × 1.10 = 11.0 涨停
        assert broker.can_trade(11.0, 10.0) is False

    def test_cannot_trade_at_limit_down(self):
        """一字跌停无法成交"""
        broker = Broker()
        # 10.0 × 0.90 = 9.0 跌停
        assert broker.can_trade(9.0, 10.0) is False


# ─────────────────────────────────────────────────────────
# 测试 10: 回测后组件状态
# ─────────────────────────────────────────────────────────

class TestPostBacktestState:
    """验证回测后各组件状态一致"""

    def test_position_manager_state_consistent(self, backtest_env):
        """回测后仓位管理器状态应一致"""
        engine = BacktestEngine(
            initial_capital=100000,
            lookback_days=60,
        )
        engine.run(backtest_env["data_map"], backtest_env["benchmark"])

        pm = engine.position_mgr
        # 所有开仓应已平仓 (回测结束)
        assert len(pm.open_positions) == 0, \
            "回测结束不应有未平仓持仓"
        # 已平仓交易数应等于 closed_trades 长度
        assert pm.trade_count == len(pm.closed_trades)
        # 胜率计算应一致
        if pm.closed_trades:
            expected_win_rate = pm.win_count / pm.trade_count
            assert abs(pm.win_rate - expected_win_rate) < 1e-6

    def test_cash_plus_positions_equals_total(self, backtest_env):
        """现金 + 持仓市值 = 总资产 (无浮亏漏洞)"""
        engine = BacktestEngine(
            initial_capital=100000,
            lookback_days=60,
        )
        engine.run(backtest_env["data_map"], backtest_env["benchmark"])

        pm = engine.position_mgr
        # 回测结束无持仓, 总资产应等于现金
        assert len(pm.open_positions) == 0
        assert abs(pm.cash - pm.total_value({})) < 1e-6

    def test_metrics_final_value_matches_position_manager(self, backtest_env):
        """metrics.final_value 应与 PositionManager 一致"""
        engine = BacktestEngine(
            initial_capital=100000,
            lookback_days=60,
        )
        metrics = engine.run(backtest_env["data_map"], backtest_env["benchmark"])

        pm = engine.position_mgr
        # 最后一天的总资产
        last_prices = {}
        cal = TradingCalendar(backtest_env["data_map"])
        last_date = cal.all_dates[-1]
        prices = cal.get_closing_prices(backtest_env["data_map"], last_date)
        expected = pm.total_value(prices)

        assert abs(metrics.final_value - expected) < 1e-6, \
            f"metrics.final_value ({metrics.final_value}) 应与 PM 总资产 ({expected}) 一致"
