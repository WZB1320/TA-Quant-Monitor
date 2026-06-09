"""资金曲线图 — 回测收益可视化"""
import os, sys
json_path = "data/signal_history.json"
if os.path.exists(json_path):
    os.remove(json_path)

from src.data_fetcher import DataManager, Watchlist
from src.backtest import BacktestEngine
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd

# ── 设置中文字体 ──
plt.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

# ── 加载数据 ──
dm = DataManager()
wl = Watchlist()
data_map = {}
for s in wl.get_all():
    code = s["code"]
    df = dm.get_daily_kline(code, start_date="2024-01-01")
    if df is not None and len(df) >= 120:
        data_map[code] = df
bench_df = dm.get_daily_kline("000001", start_date="2024-01-01")

# ── 运行回测 ──
engine = BacktestEngine(initial_capital=100000, lookback_days=120,
                        position_ratio=0.3, signal_dedup_days=5)
metrics = engine.run(data_map, benchmark_df=bench_df)

# ── 获取数据 ──
daily_values = metrics.daily_values
trades = engine.position_mgr.closed_trades

# ── 绘图 ──
fig, axes = plt.subplots(3, 1, figsize=(16, 12), gridspec_kw={"height_ratios": [3, 1, 1]})
fig.suptitle("资金曲线 & 回撤分析", fontsize=16, fontweight="bold", y=0.98)

# ── 图1: 资金曲线 ──
ax1 = axes[0]
ax1.plot(daily_values.index, daily_values.values, color="#2c7fb8", linewidth=1.5, label="Strategy")
ax1.axhline(y=100000, color="gray", linestyle="--", linewidth=0.8, alpha=0.6, label="Initial Capital")

# 标注每笔交易
for t in trades:
    # 买入点
    if t.entry_date in daily_values.index:
        v = daily_values.loc[t.entry_date]
        ax1.scatter(t.entry_date, v, s=30, color="limegreen", edgecolors="darkgreen",
                   linewidths=0.5, zorder=5)
    # 卖出点
    if t.exit_date in daily_values.index:
        v = daily_values.loc[t.exit_date]
        color = "red" if t.pnl <= 0 else "limegreen"
        ax1.scatter(t.exit_date, v, s=50, color=color, edgecolors="darkred",
                   linewidths=0.8, marker="X", zorder=5)

# 基准对比
if metrics.benchmark_return != 0:
    bench_start = daily_values.index[0]
    if bench_start in metrics.daily_values.index:
        base_val = metrics.daily_values.loc[bench_start]
        bench_ratio = 1 + metrics.benchmark_return
        bench_end = daily_values.index[-1]
        ax1.plot([bench_start, bench_end],
                 [base_val, base_val * bench_ratio],
                 color="orange", linewidth=1.5, linestyle="--", alpha=0.7, label="上证指数")

ax1.set_ylabel("Account Value (CNY)", fontsize=11)
ax1.legend(loc="upper left", fontsize=9)
ax1.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x:,.0f}"))
ax1.grid(True, alpha=0.3)

# 标注最终值
end_val = daily_values.iloc[-1]
ax1.annotate(f"{end_val:,.0f}  ({metrics.total_return*100:+.2f}%)",
             xy=(daily_values.index[-1], end_val),
             xytext=(15, 15), textcoords="offset points",
             fontsize=10, fontweight="bold", color="#2c7fb8",
             bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))

# ── 图2: 回撤曲线 ──
ax2 = axes[1]
rolling_max = daily_values.expanding().max()
drawdown = (daily_values / rolling_max - 1) * 100
ax2.fill_between(drawdown.index, 0, drawdown.values, color="red", alpha=0.3)
ax2.plot(drawdown.index, drawdown.values, color="darkred", linewidth=0.8)
ax2.set_ylabel("Drawdown (%)", fontsize=11)
ax2.axhline(y=0, color="black", linewidth=0.5)
ax2.grid(True, alpha=0.3)
max_dd = drawdown.min()
ax2.annotate(f"Max: {max_dd:.2f}%", xy=(drawdown.idxmin(), max_dd),
             xytext=(0, -20), textcoords="offset points",
             fontsize=10, color="darkred", fontweight="bold",
             arrowprops=dict(arrowstyle="->", color="darkred", lw=1))

# ── 图3: 持仓数量 ──
ax3 = axes[2]
# 从 daily_values 反推每日持仓数 (近似)
positions = {}
for t in trades:
    entry = t.entry_date
    exit_d = t.exit_date
    if entry not in positions:
        positions[entry] = 0
    positions[entry] += 1
    if exit_d not in positions:
        positions[exit_d] = 0
    positions[exit_d] -= 1

# 展开为日序列
pos_dates = sorted(positions.keys())
pos_series = {}
running = 0
for d in pos_dates:
    running += positions[d]
    pos_series[d] = running

# 对齐到 daily_values 的日期
aligned = []
for d in daily_values.index:
    closest = None
    for pdate in pos_dates:
        if pdate <= d:
            closest = pdate
    aligned.append(pos_series.get(closest, 0) if closest else 0)

ax3.fill_between(daily_values.index, 0, aligned, color="#2c7fb8", alpha=0.5)
ax3.plot(daily_values.index, aligned, color="#1a5276", linewidth=1)
ax3.set_ylabel("Positions", fontsize=11)
ax3.set_xlabel("Date", fontsize=11)
ax3.set_ylim(0, max(aligned) + 1 if aligned else 3)
ax3.yaxis.set_major_locator(mticker.MaxNLocator(integer=True))
ax3.grid(True, alpha=0.3)

# ── 性能指标标注 ──
stats_text = (
    f"Initial: 100,000  |  Final: {metrics.final_value:,.0f}  |  Return: {metrics.total_return*100:+.2f}%\n"
    f"Annual: {metrics.annual_return*100:+.2f}%  |  Sharpe: {metrics.sharpe_ratio:.2f}  |  MaxDD: {metrics.max_drawdown*100:.2f}%\n"
    f"Trades: {metrics.trade_count}  |  Win Rate: {metrics.win_rate*100:.1f}%  |  Profit Factor: {metrics.profit_factor:.2f}"
)
fig.text(0.5, 0.01, stats_text, ha="center", fontsize=10, fontfamily="monospace",
         bbox=dict(boxstyle="round,pad=0.5", facecolor="#f0f0f0", alpha=0.9))

plt.tight_layout(rect=[0, 0.06, 1, 0.95])
output_path = "equity_curve.png"
plt.savefig(output_path, dpi=150, bbox_inches="tight")
print(f"\nChart saved to: {output_path}")
print(f"\nKey stats:")
print(f"  Total Return: {metrics.total_return*100:+.2f}%")
print(f"  Annual Return: {metrics.annual_return*100:+.2f}%")
print(f"  Sharpe Ratio:  {metrics.sharpe_ratio:.2f}")
print(f"  Max Drawdown:  {metrics.max_drawdown*100:.2f}%")
print(f"  Win Rate:      {metrics.win_rate*100:.1f}%")
print(f"  Profit Factor: {metrics.profit_factor:.2f}")
print(f"  Total Trades:  {metrics.trade_count}")