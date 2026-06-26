"""开盘价判断 + 收盘价执行 回测对比脚本

对比策略:
  A. 原策略: 收盘价判断信号, T+1 次日开盘价执行
  B. 新策略: 开盘价判断信号, 当日收盘价执行

回测区间: 2026-01-01 ~ 2026-06-22
"""
import os
import sys
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.config.runtime_mode import set_mode, RuntimeMode
set_mode(RuntimeMode.BACKTEST)

from src.data_fetcher import DataManager, Watchlist
from src.backtest import BacktestEngine
from src.backtest.broker import Broker
from src.backtest.position import PositionManager
from src.backtest.calendar import TradingCalendar
from src.backtest.market_filter import MarketFilter
from src.backtest.regime_detector import RegimeDetector
from src.backtest.metrics import compute_metrics
from src.signal_engine import SignalEngine
from src.signal_engine.filter import SignalFilter
from src.config.group_config import GroupConfig

import pandas as pd
from datetime import datetime as _dt


# ── 回测参数 ──
START_DATE = "2026-01-01"
END_DATE = "2026-06-22"
INITIAL_CAPITAL = 100_000
LOOKBACK_DAYS = 120


def load_data():
    """加载股票数据"""
    dm = DataManager()
    wl = Watchlist()
    data_map = {}
    for s in wl.get_all():
        df = dm.get_daily_kline(s['code'], start_date='2024-07-01')
        if df is not None and len(df) >= LOOKBACK_DAYS:
            data_map[s['code']] = df
    print(f"加载 {len(data_map)} 只股票")
    return data_map


# ══════════════════════════════════════════════════════════════
#  策略 A: 原策略 (收盘价判断, T+1 开盘价执行)
# ══════════════════════════════════════════════════════════════

def run_original(data_map):
    """原策略: 收盘价判断, T+1 开盘价执行"""
    engine = BacktestEngine(
        initial_capital=INITIAL_CAPITAL,
        lookback_days=LOOKBACK_DAYS,
        position_ratio=0.30,
        signal_dedup_days=5,
        risk_per_trade=0.05,
        atr_stop_mult=2.5,
    )
    metrics = engine.run(data_map, start_date=START_DATE, end_date=END_DATE)
    return metrics, engine.position_mgr.closed_trades


# ══════════════════════════════════════════════════════════════
#  策略 B: 新策略 (开盘价判断, 当日收盘价执行)
# ══════════════════════════════════════════════════════════════

def run_open_signal(data_map):
    """新策略: 开盘价判断, 当日收盘价执行"""

    # ── 1. 创建信号数据 (close 列替换为 open, 使所有指标基于开盘价计算) ──
    data_map_signal = {}
    for symbol, df in data_map.items():
        df_sig = df.copy()
        df_sig["close"] = df_sig["open"]
        data_map_signal[symbol] = df_sig

    # ── 2. 初始化组件 ──
    gc = GroupConfig()
    signal_engine = SignalEngine(dedup_days=5, group_config=gc)
    broker = Broker(commission_rate=0.00025, stamp_tax=0.001, slippage=0.0001)
    regime_detector = RegimeDetector()

    position_mgr = PositionManager(
        initial_capital=INITIAL_CAPITAL,
        position_ratio=0.30,
        commission_rate=0.00025,
        risk_per_trade=0.05,
        atr_stop_mult=2.5,
    )

    # 重置信号引擎过滤器 (避免状态残留)
    signal_engine.filter = SignalFilter(
        dedup_days=signal_engine.filter.dedup_days,
        market_ma60_filter=signal_engine.filter.market_ma60_filter,
        cooldown_days=signal_engine.filter.cooldown_days,
    )

    calendar = TradingCalendar(data_map)
    market_filter = MarketFilter(None)  # 无基准, 同原策略

    all_dates = calendar.all_dates
    bt_start = _dt.strptime(START_DATE, "%Y-%m-%d").date()
    bt_end = _dt.strptime(END_DATE, "%Y-%m-%d").date()

    values = {}

    # ── 3. 逐日回测 ──
    for today in all_dates:
        if today > bt_end:
            continue

        # 0. 更新市场体制
        regime = regime_detector.detect(None, today)
        position_mgr.set_regime(regime)

        # 1. 检查止损/移动止盈 (使用原始收盘价)
        prices_today = calendar.get_closing_prices(data_map, today)
        for symbol in list(position_mgr.open_positions.keys()):
            if symbol in prices_today:
                closed = position_mgr.check_stop_loss(
                    symbol, prices_today[symbol], today)
                if closed is not None:
                    signal_engine.filter.record_exit(symbol, today)
                    if closed.pnl > 0:
                        signal_engine.filter.record_win(symbol)
                    else:
                        signal_engine.filter.record_loss(symbol, today)

        in_range = today >= bt_start

        # 2. 生成信号 (使用开盘价数据: close=open)
        signals_today = {}
        if in_range:
            for symbol, df_sig in data_map_signal.items():
                idx = calendar.locate(symbol, today)
                if idx is None or idx < LOOKBACK_DAYS:
                    continue
                df_slice = df_sig.iloc[:idx + 1].copy()
                try:
                    result = signal_engine.analyze(
                        symbol, df_slice, analysis_date=today)
                except Exception:
                    continue
                if result.level.is_actionable:
                    signals_today[symbol] = result

        # 3. 当日收盘价执行 (无 T+1 延迟)
        if in_range and signals_today:
            _execute_at_close(
                signals_today, data_map, data_map_signal,
                today, calendar, broker, position_mgr,
                signal_engine, gc, market_filter)

        # 4. 记录净值
        if in_range:
            values[today] = position_mgr.total_value(prices_today)

    # ── 4. 最终净值 (确保持仓市值计入) ──
    valid_dates = [d for d in all_dates if d <= bt_end]
    if valid_dates:
        last_date = valid_dates[-1]
        prices_last = calendar.get_closing_prices(data_map, last_date)
        values[last_date] = position_mgr.total_value(prices_last)

    # ── 5. 计算指标 ──
    daily_values = pd.Series(values).sort_index()
    metrics = compute_metrics(
        daily_values=daily_values,
        trades=position_mgr.closed_trades,
        initial_capital=INITIAL_CAPITAL,
    )
    return metrics, position_mgr.closed_trades


def _execute_at_close(signals, data_map, data_map_signal,
                      signal_date, calendar, broker, position_mgr,
                      signal_engine, group_config, market_filter):
    """当日收盘价执行信号"""
    sorted_signals = sorted(
        signals.items(),
        key=lambda x: abs(x[1].score), reverse=True)

    for symbol, result in sorted_signals:
        df = data_map.get(symbol)  # 原始数据 (含真实收盘价)
        if df is None:
            continue

        idx = calendar.locate(symbol, signal_date)
        if idx is None or idx == 0:
            continue

        today_close = float(df.iloc[idx]["close"])
        prev_close = float(df.iloc[idx - 1]["close"])

        # 涨跌停检查 (收盘价 vs 前日收盘价)
        if not broker.can_trade(today_close, prev_close):
            continue

        if result.level.is_bullish:
            _buy_at_close(symbol, result, data_map_signal,
                          signal_date, today_close, idx,
                          broker, position_mgr, signal_engine,
                          group_config, market_filter)
        elif result.level.is_bearish:
            _sell_at_close(symbol, result, signal_date,
                           today_close, broker, position_mgr,
                           signal_engine)


def _buy_at_close(symbol, result, data_map_signal,
                  signal_date, close_price, idx,
                  broker, position_mgr, signal_engine,
                  group_config, market_filter):
    """收盘价买入"""
    market_bearish = market_filter.is_bearish(signal_date)

    # 冷却期检查
    if signal_engine.filter.is_in_cooldown(symbol, signal_date, True):
        return

    # ATR 从信号数据 (开盘价) 计算, 用于止损距离
    df_sig = data_map_signal.get(symbol)
    exec_atr = None
    if df_sig is not None and idx >= 60:
        exec_df = df_sig.iloc[:idx + 1].copy()
        try:
            exec_ind = signal_engine.pipeline.run(exec_df)
            exec_atr_ind = exec_ind.get("ATR")
            if exec_atr_ind:
                exec_atr = exec_atr_ind.values.get("atr")
        except Exception:
            pass

    if position_mgr.has_position(symbol):
        return

    buy_price = broker.buy_price(close_price)  # 收盘价 + 滑点
    signal_strength = 1.3 if "强买入" in result.level.label else 1.0
    group_boost = group_config.get_max_per_stock_boost(symbol)
    signal_strength *= group_boost
    group_stop_mult = group_config.get_atr_stop_mult(symbol)

    position_mgr.open_long(
        symbol=symbol,
        entry_date=signal_date,
        entry_price=buy_price,
        signal=f"{result.level.label} score={result.score:+.1f}",
        atr_value=exec_atr,
        bearish_market=market_bearish,
        signal_strength=signal_strength,
        atr_stop_mult=group_stop_mult,
    )


def _sell_at_close(symbol, result, signal_date,
                   close_price, broker, position_mgr, signal_engine):
    """收盘价卖出"""
    if not position_mgr.has_position(symbol):
        return

    sell_price = broker.sell_price(close_price)  # 收盘价 - 滑点
    closed = position_mgr.close_position(
        symbol=symbol,
        exit_date=signal_date,
        exit_price=sell_price,
        signal=f"{result.level.label} score={result.score:+.1f}",
    )
    if closed is not None:
        signal_engine.filter.record_exit(symbol, signal_date)
        if closed.pnl > 0:
            signal_engine.filter.record_win(symbol)
        else:
            signal_engine.filter.record_loss(symbol, signal_date)


# ══════════════════════════════════════════════════════════════
#  结果输出
# ══════════════════════════════════════════════════════════════

def print_comparison(metrics_a, trades_a, metrics_b, trades_b):
    """打印对比结果"""
    print("\n" + "=" * 72)
    print(f"  回测对比 ({START_DATE} ~ {END_DATE})")
    print("=" * 72)

    rows = [
        ("总收益率",   f"{metrics_a.total_return:.2%}",       f"{metrics_b.total_return:.2%}"),
        ("年化收益",   f"{metrics_a.annual_return:.2%}",      f"{metrics_b.annual_return:.2%}"),
        ("最大回撤",   f"{metrics_a.max_drawdown:.2%}",       f"{metrics_b.max_drawdown:.2%}"),
        ("夏普比率",   f"{metrics_a.sharpe_ratio:.2f}",       f"{metrics_b.sharpe_ratio:.2f}"),
        ("总交易数",   f"{metrics_a.trade_count}",            f"{metrics_b.trade_count}"),
        ("胜率",       f"{metrics_a.win_rate:.1%}",           f"{metrics_b.win_rate:.1%}"),
        ("平均盈利",   f"{metrics_a.avg_win_pct:.2%}",        f"{metrics_b.avg_win_pct:.2%}"),
        ("平均亏损",   f"{metrics_a.avg_loss_pct:.2%}",       f"{metrics_b.avg_loss_pct:.2%}"),
        ("盈亏比",     f"{metrics_a.profit_factor:.2f}",      f"{metrics_b.profit_factor:.2f}"),
        ("均持仓天",   f"{metrics_a.avg_holding_days:.0f}",   f"{metrics_b.avg_holding_days:.0f}"),
        ("总盈亏",     f"{metrics_a.total_pnl:+,.0f}",        f"{metrics_b.total_pnl:+,.0f}"),
        ("最终资产",   f"{metrics_a.final_value:,.0f}",       f"{metrics_b.final_value:,.0f}"),
    ]

    header = f"  {'指标':<10s}  {'A:收盘判断+T+1开盘':>20s}  {'B:开盘判断+当日收盘':>20s}"
    print(header)
    print("  " + "-" * 68)
    for label, va, vb in rows:
        print(f"  {label:<10s}  {va:>20s}  {vb:>20s}")

    # ── 分组统计对比 ──
    gc = GroupConfig()
    for label, trades in [("A (原策略)", trades_a), ("B (新策略)", trades_b)]:
        group_stats = defaultdict(
            lambda: {'wins': 0, 'losses': 0, 'total_pnl': 0, 'trades': []})
        for t in trades:
            g = gc.get_group(t.symbol)
            group_stats[g]['trades'].append(t)
            if t.pnl > 0:
                group_stats[g]['wins'] += 1
            else:
                group_stats[g]['losses'] += 1
            group_stats[g]['total_pnl'] += t.pnl

        print(f"\n  ── {label} 分组统计 ──")
        for g in sorted(group_stats.keys()):
            s = group_stats[g]
            total = s['wins'] + s['losses']
            if total == 0:
                continue
            wr = s['wins'] / total * 100
            print(f"  {g}: {total}笔 | 胜率{wr:.0f}% | "
                  f"PnL:{s['total_pnl']:+,.0f}")


def print_trade_details(label, trades):
    """打印详细交易记录"""
    gc = GroupConfig()
    group_stats = defaultdict(lambda: {'trades': []})
    for t in trades:
        g = gc.get_group(t.symbol)
        group_stats[g]['trades'].append(t)

    print(f"\n{'=' * 72}")
    print(f"  {label} — 详细交易记录 ({len(trades)}笔)")
    print(f"{'=' * 72}")

    for g in sorted(group_stats.keys()):
        sorted_trades = sorted(group_stats[g]['trades'], key=lambda t: t.entry_date)
        if not sorted_trades:
            continue
        print(f"\n  [{g}]")
        for i, t in enumerate(sorted_trades, 1):
            sig = t.entry_signal[:30] if t.entry_signal else ""
            result = "盈" if t.pnl > 0 else "亏"
            print(f"  {i:>2}. {t.symbol} | {t.entry_date} -> {t.exit_date} | {sig} | "
                  f"PnL:{t.pnl:+,.0f}({t.pnl_pct:+.2f}%) | 持{t.holding_days}d [{result}]")


def main():
    data_map = load_data()
    if not data_map:
        print("无可用数据, 请检查数据源")
        return

    print(f"\n回测区间: {START_DATE} ~ {END_DATE}")
    print(f"初始资金: {INITIAL_CAPITAL:,}")
    print()

    # ── 策略 A: 原策略 ──
    print(">>> 运行策略 A: 收盘价判断 + T+1 开盘价执行 ...")
    metrics_a, trades_a = run_original(data_map)

    # ── 策略 B: 新策略 ──
    print(">>> 运行策略 B: 开盘价判断 + 当日收盘价执行 ...")
    metrics_b, trades_b = run_open_signal(data_map)

    # ── 对比输出 ──
    print_comparison(metrics_a, trades_a, metrics_b, trades_b)

    # ── 详细交易记录 ──
    print_trade_details("A (原策略: 收盘判断+T+1开盘)", trades_a)
    print_trade_details("B (新策略: 开盘判断+当日收盘)", trades_b)


if __name__ == '__main__':
    main()
