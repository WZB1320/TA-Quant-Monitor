"""
绩效指标计算

从回测产生的日净值序列 + 交易记录中提取关键指标:
- 总收益率 / 年化收益率
- 最大回撤 (Max Drawdown)
- 夏普比率 (Sharpe Ratio)
- 胜率 / 盈亏比 / 平均持仓天数
- 基准对比 (超额收益 Alpha)
"""
from dataclasses import dataclass
from typing import List, Optional
import math
import numpy as np
import pandas as pd


@dataclass
class BacktestMetrics:
    """回测绩效指标"""
    # ── 收益指标 ──
    total_return: float = 0.0           # 总收益率
    annual_return: float = 0.0          # 年化收益率
    total_pnl: float = 0.0              # 总盈亏金额

    # ── 风险指标 ──
    max_drawdown: float = 0.0           # 最大回撤 (负值)
    max_drawdown_days: int = 0          # 最大回撤持续天数
    volatility: float = 0.0             # 年化波动率
    sharpe_ratio: float = 0.0           # 夏普比率

    # ── 交易指标 ──
    trade_count: int = 0                # 总交易次数
    win_count: int = 0                  # 盈利次数
    win_rate: float = 0.0               # 胜率
    avg_win_pct: float = 0.0            # 平均盈利%
    avg_loss_pct: float = 0.0           # 平均亏损%
    profit_factor: float = 0.0          # 盈亏比 (总盈利/总亏损绝对值)
    avg_holding_days: float = 0.0       # 平均持仓天数

    # ── 基准对比 ──
    benchmark_return: float = 0.0       # 基准总收益率
    alpha: float = 0.0                  # 超额收益

    # ── 资金 ──
    initial_capital: float = 0.0
    final_value: float = 0.0

    # ── 日频原始数据 (供绘图) ──
    daily_values: Optional[pd.Series] = None


def compute_metrics(
    daily_values: pd.Series,
    trades: List,
    initial_capital: float,
    benchmark_values: Optional[pd.Series] = None,
    risk_free_rate: float = 0.02,
) -> BacktestMetrics:
    """
    从日频净值序列计算全部指标

    Args:
        daily_values: 每日总资产序列 (日期索引)
        trades: 已平仓交易列表 (Trade 对象)
        initial_capital: 初始资金
        benchmark_values: 基准日净值序列 (与 daily_values 对齐)
        risk_free_rate: 无风险利率 (默认2%)

    Returns:
        BacktestMetrics
    """
    m = BacktestMetrics()
    m.initial_capital = initial_capital
    m.daily_values = daily_values

    if len(daily_values) < 2:
        return m

    # ── 收益 ──
    m.final_value = daily_values.iloc[-1]
    m.total_return = (m.final_value / initial_capital) - 1.0
    m.total_pnl = m.final_value - initial_capital

    # 年化收益
    years = (daily_values.index[-1] - daily_values.index[0]).days / 365.25
    if years > 0 and m.total_return > -1:
        m.annual_return = (1 + m.total_return) ** (1 / years) - 1

    # ── 日收益率 ──
    daily_returns = daily_values.pct_change().dropna()

    # ── 最大回撤 ──
    m.max_drawdown, m.max_drawdown_days = _max_drawdown(daily_values)

    # ── 波动率 ──
    m.volatility = daily_returns.std() * math.sqrt(252)

    # ── 夏普比率 ──
    if m.volatility > 0:
        excess = m.annual_return - risk_free_rate
        m.sharpe_ratio = excess / m.volatility

    # ── 交易统计 ──
    m.trade_count = len(trades)
    m.win_count = sum(1 for t in trades if t.pnl > 0)
    m.win_rate = m.win_count / m.trade_count if m.trade_count > 0 else 0.0

    wins = [t.pnl_pct for t in trades if t.pnl > 0]
    losses = [t.pnl_pct for t in trades if t.pnl <= 0]
    m.avg_win_pct = np.mean(wins) if wins else 0.0
    m.avg_loss_pct = np.mean(losses) if losses else 0.0

    total_win = sum(t.pnl for t in trades if t.pnl > 0)
    total_loss = abs(sum(t.pnl for t in trades if t.pnl <= 0))
    m.profit_factor = total_win / total_loss if total_loss > 0 else float('inf')

    closed = [t for t in trades if t.is_closed]
    m.avg_holding_days = (sum(t.holding_days for t in closed) / len(closed)
                          if closed else 0.0)

    # ── 基准对比 ──
    if benchmark_values is not None and len(benchmark_values) > 0:
        m.benchmark_return = (benchmark_values.iloc[-1] / benchmark_values.iloc[0]) - 1
        m.alpha = m.total_return - m.benchmark_return

    return m


def _max_drawdown(series: pd.Series) -> tuple:
    """计算最大回撤及持续天数"""
    if len(series) < 2:
        return 0.0, 0

    rolling_max = series.expanding().max()
    drawdown = series / rolling_max - 1.0
    max_dd = drawdown.min()

    # 回撤持续天数
    peak_idx = 0
    max_dd_days = 0
    for i in range(len(drawdown)):
        if drawdown.iloc[i] == 0:
            peak_idx = i
        else:
            days = i - peak_idx
            if days > max_dd_days:
                max_dd_days = days

    return max_dd, max_dd_days