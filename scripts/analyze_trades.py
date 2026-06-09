"""逐笔交易信号详情分析"""
import os, json
json_path = "data/signal_history.json"
if os.path.exists(json_path):
    os.remove(json_path)

from src.data_fetcher import DataManager, Watchlist
from src.backtest import BacktestEngine
from src.signal_engine import SignalEngine
from src.signal_engine.signals import SignalLevel
import pandas as pd

dm = DataManager()
wl = Watchlist()

print("=" * 75)
print("  逐笔交易信号详情")
print("=" * 75)

# 加载数据
data_map = {}
for s in wl.get_all():
    code = s["code"]
    df = dm.get_daily_kline(code, start_date="2024-01-01")
    if df is not None and len(df) >= 120:
        data_map[code] = df

bench_df = dm.get_daily_kline("000001", start_date="2024-01-01")

# 跑回测
engine = BacktestEngine(initial_capital=100000, lookback_days=120,
                        position_ratio=0.3, signal_dedup_days=5)
metrics = engine.run(data_map, benchmark_df=bench_df)

# ── 逐笔详情 ──
sig_engine = SignalEngine(dedup_days=5)  # 独立引擎查历史信号

for i, trade in enumerate(engine.position_mgr.closed_trades, 1):
    print(f"\n{'='*75}")
    print(f"  交易 #{i}: {trade.symbol}  {trade.side.value.upper()}")
    print(f"{'='*75}")
    print(f"  入场: {trade.entry_date}  @ {trade.entry_price:.2f}  ({trade.entry_signal})")
    print(f"  出场: {trade.exit_date}  @ {trade.exit_price:.2f}  ({trade.exit_signal})")
    print(f"  持仓: {trade.holding_days}天  盈亏: {trade.pnl:+.0f} ({trade.pnl_pct*100:+.2f}%)")

    # ── 入场信号详细分析 ──
    df = data_map.get(trade.symbol)
    if df is None:
        continue

    # 找到入场信号日期 (入场前一个交易日)
    entry_dt = trade.entry_date
    idx = None
    for j in range(len(df)):
        if df["date"].iloc[j] == entry_dt.strftime("%Y-%m-%d"):
            idx = j
            break

    if idx is None or idx < 1:
        print("  (无法定位入场日期)")
        continue

    signal_date_idx = idx - 1  # 信号在前一个交易日产生
    signal_date = df["date"].iloc[signal_date_idx]
    df_slice = df.iloc[:signal_date_idx + 1].copy()

    # 重新计算入场信号
    result = sig_engine.analyze(trade.symbol, df_slice)

    print(f"\n  [入场信号] 日期={signal_date}")
    print(f"  信号: {result.level.label}  得分: {result.score:+.1f}  置信度: {result.confidence:.0%}")
    print(f"  拦截: {'是 - ' + result.block_reason if result.hard_filter_blocked else '否'}")

    # 类别得分
    from src.signal_engine.scorer import Scorer
    scorer = Scorer()
    cat_scores = scorer.score_by_category(
        sig_engine.pipeline.run(df_slice)
    )
    print(f"\n  类别得分:")
    for cat, cs in cat_scores.items():
        bar = "#" * max(1, int(abs(cs) / 100 * 20)) if abs(cs) > 0 else "."
        sign = "+" if cs >= 0 else ""
        print(f"    {cat:<10}: {sign}{cs:6.1f}  {bar}")

    # 每项指标
    print(f"\n  指标明细:")
    ir = sig_engine.pipeline.run(df_slice)
    from src.signal_engine.scorer import WEIGHTS
    for name, (cat, w) in WEIGHTS.items():
        r = ir.get(name)
        if r:
            arrow = "+" if r.direction > 0 else "-" if r.direction < 0 else "0"
            print(f"    {arrow} {name:<12} [{r.strength:.1f}] w={w:.2f}  {r.description}")

    # 交叉验证
    validator_res = sig_engine.validator.summarize(ir)
    print(f"\n  交叉验证:")
    for cat, cs in validator_res.items():
        print(f"    {cat:<10}: dir={cs.direction:+d}  "
              f"看多{cs.consensus} vs 看空{cs.dissensus}")

    # ── 出场信号 ──
    exit_dt = trade.exit_date
    exit_idx = None
    for j in range(len(df)):
        if df["date"].iloc[j] == exit_dt.strftime("%Y-%m-%d"):
            exit_idx = j
            break

    if exit_idx is None or exit_idx < 1:
        print("  (无法定位出场日期)")
        continue

    exit_signal_idx = exit_idx - 1
    exit_signal_date = df["date"].iloc[exit_signal_idx]
    df_slice_exit = df.iloc[:exit_signal_idx + 1].copy()

    result_exit = sig_engine.analyze(trade.symbol, df_slice_exit)

    print(f"\n  [出场信号] 日期={exit_signal_date}")
    print(f"  信号: {result_exit.level.label}  得分: {result_exit.score:+.1f}  置信度: {result_exit.confidence:.0%}")
    print(f"  拦截: {'是 - ' + result_exit.block_reason if result_exit.hard_filter_blocked else '否'}")

    ir_exit = sig_engine.pipeline.run(df_slice_exit)
    print(f"\n  指标明细:")
    for name, (cat, w) in WEIGHTS.items():
        r = ir_exit.get(name)
        if r:
            arrow = "+" if r.direction > 0 else "-" if r.direction < 0 else "0"
            print(f"    {arrow} {name:<12} [{r.strength:.1f}]  {r.description}")

print(f"\n{'='*75}")
print(f"  汇总: {metrics.trade_count}笔 | 胜率{metrics.win_rate*100:.1f}% | "
      f"总盈亏{metrics.total_pnl:+,.0f} | Alpha{metrics.alpha*100:+.2f}%")
print("=" * 75)