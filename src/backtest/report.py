"""
回测报告生成 — 汇总 + 逐笔明细

输出:
1. 全局绩效摘要
2. 净值曲线 (ASCII art)
3. 逐笔交易明细
4. 对比基准
"""
from typing import List
import math
import pandas as pd

from .position import Trade
from .metrics import BacktestMetrics


def generate_report(metrics: BacktestMetrics, trades: List[Trade],
                    title: str = "回测报告") -> str:
    """生成完整的回测报告文本"""
    lines = []
    sep = "=" * 68
    sub = "-" * 68

    # ── 标题 ──
    lines.append(sep)
    lines.append(f"  {title}")
    lines.append(sep)

    # ── 全局绩效摘要 ──
    lines.append("")
    lines.append("  [1] 全局绩效摘要")
    lines.append(sub)

    label_w = 20
    def _row(label, value):
        lines.append(f"  {label:<{label_w}}: {value}")

    _row("初始资金",       f"{metrics.initial_capital:,.0f}")
    _row("最终资产",       f"{metrics.final_value:,.0f}")
    _row("总收益率",       _pct(metrics.total_return))
    _row("总盈亏",         f"{metrics.total_pnl:+,.0f}")
    _row("年化收益率",     _pct(metrics.annual_return))

    # 基准对比
    if metrics.benchmark_return != 0:
        _row("基准收益率",       _pct(metrics.benchmark_return))
        _row("超额收益 Alpha",   _pct(metrics.alpha))

    _row("最大回撤",       _pct(metrics.max_drawdown) +
         f"  (持续 {metrics.max_drawdown_days} 天)")
    _row("年化波动率",     _pct(metrics.volatility))
    _row("夏普比率",       f"{metrics.sharpe_ratio:.2f}")

    # ── 交易统计 ──
    lines.append("")
    lines.append("  [2] 交易统计")
    lines.append(sub)

    _row("总交易次数",     f"{metrics.trade_count}")
    _row("盈利次数",       f"{metrics.win_count}")
    _row("胜率",           _pct(metrics.win_rate))
    _row("平均盈利",       _pct(metrics.avg_win_pct))
    _row("平均亏损",       _pct(metrics.avg_loss_pct))
    _row("盈亏比",         f"{metrics.profit_factor:.2f}" if metrics.profit_factor != float('inf') else "N/A (无亏损)")
    _row("平均持仓天数",   f"{metrics.avg_holding_days:.1f} 天")

    # ── 净值曲线 ──
    if metrics.daily_values is not None and len(metrics.daily_values) > 0:
        lines.append("")
        lines.append("  [3] 净值曲线 (归一化)")
        lines.append(sub)
        lines.append(_ascii_chart(metrics.daily_values, width=60, height=12))

    # ── 逐笔交易明细 ──
    lines.append("")
    lines.append("  [4] 逐笔交易明细")
    lines.append(sub)

    if trades:
        # 表头
        header = f"  {'股票':<8} {'入场日':<12} {'出场日':<12} {'持仓天':>6} {'入场价':>8} {'出场价':>8} {'盈亏%':>8} {'盈亏':>10}"
        lines.append(header)
        lines.append("  " + "-" * (len(header) - 2))

        for t in trades:
            entry_d = str(t.entry_date)
            exit_d = str(t.exit_date) if t.exit_date else "(持仓中)"
            lines.append(
                f"  {t.symbol:<8} {entry_d:<12} {exit_d:<12} "
                f"{t.holding_days:>6} {t.entry_price:>8.2f} {t.exit_price:>8.2f} "
                f"{_pct(t.pnl_pct):>8} {t.pnl:>+10.0f}"
            )
    else:
        lines.append("  (无已平仓交易)")

    # ── 结尾 ──
    lines.append("")
    lines.append(sep)
    return "\n".join(lines)


def generate_summary(metrics: BacktestMetrics) -> str:
    """生成单行摘要 (用于批量结果)"""
    return (
        f"收益={_pct(metrics.total_return)}  "
        f"年化={_pct(metrics.annual_return)}  "
        f"夏普={metrics.sharpe_ratio:.2f}  "
        f"回撤={_pct(metrics.max_drawdown)}  "
        f"胜率={_pct(metrics.win_rate)}  "
        f"交易{metrics.trade_count}笔"
    )


def _pct(value: float) -> str:
    """格式化百分比"""
    sign = "+" if value >= 0 else ""
    return f"{sign}{value * 100:.2f}%"


def _ascii_chart(series: pd.Series, width: int = 60,
                 height: int = 12) -> str:
    """Generate ASCII line chart for a series."""
    if len(series) < 2:
        return "  (数据不足)"

    # 归一化到 0~1
    norm = (series - series.min()) / (series.max() - series.min() + 1e-10)
    rows = []

    for h in range(height - 1, -1, -1):
        line = ""
        for i in range(width):
            idx = int(len(norm) * i / width)
            val = norm.iloc[idx] * (height - 1)
            if abs(val - h) < 0.6:
                line += "#"
            else:
                line += " "
        rows.append(f"  |{line}|")

    # 底部标签
    label_line = "  +" + "-" * width + "+"
    rows.append(label_line)

    # 日期标签
    start_d = str(series.index[0])[:10]
    end_d = str(series.index[-1])[:10]
    label_text = f"    {start_d}"
    label_text += " " * (width - len(start_d) - len(end_d)) + end_d
    rows.append(label_text)

    # Y轴标注: 最高和最低
    min_v = series.min()
    max_v = series.max()
    rows[0] += f" {max_v:.2f}"
    rows[height - 1] += f" {min_v:.2f}"

    return "\n".join(rows)