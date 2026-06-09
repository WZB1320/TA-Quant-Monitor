"""仓位分配与资金明细"""
import os
json_path = "data/signal_history.json"
if os.path.exists(json_path):
    os.remove(json_path)

from src.data_fetcher import DataManager, Watchlist
from src.backtest import BacktestEngine
import pandas as pd

dm = DataManager()
wl = Watchlist()

data_map = {}
for s in wl.get_all():
    code = s["code"]
    df = dm.get_daily_kline(code, start_date="2024-01-01")
    if df is not None and len(df) >= 120:
        data_map[code] = df

bench_df = dm.get_daily_kline("000001", start_date="2024-01-01")

engine = BacktestEngine(initial_capital=100000, lookback_days=120,
                        position_ratio=0.3, signal_dedup_days=5)
metrics = engine.run(data_map, benchmark_df=bench_df)

print("=" * 75)
print("  仓位分配 & 资金明细")
print("=" * 75)

capital = 100000
print(f"\n初始资金: {capital:,.0f}")
print(f"规则: 单只股票最多占 {engine.position_ratio*100:.0f}% 仓位")
print(f"      买入 = 次日开盘价 * (1+滑点万1)")
print(f"      买入佣金 = max(成交额 * 万2.5, 5元)")
print(f"      卖出佣金 = 买入佣金 + 印花税(成交额 * 千1)")
print()

for i, t in enumerate(engine.position_mgr.closed_trades, 1):
    buy_cost = t.shares * t.entry_price       # 买入花费
    buy_comm = max(buy_cost * 0.00025, 5)      # 买入佣金
    sell_rev = t.shares * t.exit_price         # 卖回收回
    sell_comm = max(sell_rev * 0.00025, 5) + sell_rev * 0.001  # 卖出佣金+印花税
    net_cost = buy_cost + buy_comm             # 实际支出
    net_back = sell_rev - sell_comm            # 实际回收
    pnl = net_back - net_cost                  # 净盈亏
    pnl_pct = pnl / net_cost * 100
    pct_of_total = buy_cost / capital * 100

    print(f"--- 交易 #{i}: {t.symbol} ---")
    print(f"  信号: {t.entry_signal}")
    print(f"  买入日: {t.entry_date}  | 卖出日: {t.exit_date}  | 持仓: {t.holding_days}天")
    print(f"")
    print(f"  买入价: {t.entry_price:.2f} (含滑点)")
    print(f"  买入数量: {t.shares} 股")
    print(f"  买入金额: {t.shares} x {t.entry_price:.2f} = {buy_cost:,.0f}")
    print(f"  买入佣金: {buy_comm:,.0f}")
    print(f"  实际支出: {net_cost:,.0f}  (占总资金 {pct_of_total:.1f}%)")
    print(f"")
    print(f"  卖出价: {t.exit_price:.2f} (含滑点)")
    print(f"  卖出金额: {t.shares} x {t.exit_price:.2f} = {sell_rev:,.0f}")
    print(f"  卖出佣金+印花税: {sell_comm:,.0f}")
    print(f"  实际回收: {net_back:,.0f}")
    print(f"")
    print(f"  净盈亏: {net_back:,.0f} - {net_cost:,.0f} = {pnl:+,.0f}  ({pnl_pct:+.2f}%)")
    print()

print("=" * 75)
print("  资金汇总")
print("=" * 75)

total_invested = 0
total_returned = 0
total_comm = 0
for t in engine.position_mgr.closed_trades:
    buy_cost = t.shares * t.entry_price
    sell_rev = t.shares * t.exit_price
    total_invested += buy_cost
    total_returned += sell_rev
    total_comm += t.commission

print(f"")
print(f"  初始资金:        {capital:>12,.0f}")
print(f"  三笔总买入额:     {total_invested:>12,.0f}")
print(f"  三笔总卖出额:     {total_returned:>12,.0f}")
print(f"  总手续费:         {total_comm:>12,.0f}")
print(f"  总净盈亏:         {engine.position_mgr.total_pnl:>+12,.0f}")
print(f"  最终资产:         {engine.position_mgr.cash:>12,.0f}")
print(f"")

# 关键: 解释为什么胜率33%但总收益+11%
total_pnl = engine.position_mgr.total_pnl
print(f"  关键问题: 为什么只有3笔交易(2亏1赚)却总收益+11%?")
print(f"  {'-'*55}")
print(f"")

win_trade = None
loss_trades = []
for t in engine.position_mgr.closed_trades:
    if t.pnl > 0:
        win_trade = t
    else:
        loss_trades.append(t)

if win_trade:
    win_buy = win_trade.shares * win_trade.entry_price
    print(f"  盈利交易 ({win_trade.symbol}):")
    print(f"    投入 {win_buy:,.0f}, 赚 {win_trade.pnl:+,.0f} ({win_trade.pnl_pct*100:+.2f}%)")
    print(f"    占初始资金 {win_buy/capital*100:.1f}%")

for lt in loss_trades:
    lt_buy = lt.shares * lt.entry_price
    print(f"  亏损交易 ({lt.symbol}):")
    print(f"    投入 {lt_buy:,.0f}, 亏 {lt.pnl:+,.0f} ({lt.pnl_pct*100:+.2f}%)")
    print(f"    占初始资金 {lt_buy/capital*100:.1f}%")

print(f"")
print(f"  结论: 虽然胜率仅33%, 但盈利交易的仓位和涨幅都更大")
if win_trade:
    win_buy = win_trade.shares * win_trade.entry_price
    loss_buy_total = sum(t.shares * t.entry_price for t in loss_trades)
    print(f"    盈利仓投入 {win_buy:,.0f} vs 亏损仓合计投入 {loss_buy_total:,.0f}")
    print(f"    赚 {win_trade.pnl:+,.0f} vs 亏 {sum(t.pnl for t in loss_trades):+,.0f}")
    print(f"    盈亏比 = {abs(win_trade.pnl/sum(t.pnl for t in loss_trades)):.2f}")
print(f"")
print(f"  另外, 大部分时间资金闲置(现金), 只有少部分时间有持仓")
print(f"  闲置资金不对收益产生贡献, 但不产生亏损")
print("=" * 75)