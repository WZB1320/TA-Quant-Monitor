"""直接手动追踪: 不用回测引擎, 手动算每笔仓位"""
import os
json_path = "data/signal_history.json"
if os.path.exists(json_path):
    os.remove(json_path)

from datetime import date
from src.data_fetcher import DataManager, Watchlist
from src.signal_engine import SignalEngine
import pandas as pd

dm = DataManager()
wl = Watchlist()

# 只加载数据
data_map = {}
for s in wl.get_all():
    code = s["code"]
    df = dm.get_daily_kline(code, start_date="2024-01-01")
    if df is not None and len(df) >= 120:
        data_map[code] = df

# ── 手动模拟: 逐日算信号, T+1执行 ──
sig_engine = SignalEngine(dedup_days=5)

capital = 100000
position_ratio = 0.3
commission_rate = 0.00025
stamp_tax = 0.001
slippage = 0.0001

# 构建日历
all_dates = set()
for df in data_map.values():
    for d in df["date"]:
        all_dates.add(pd.Timestamp(d).date())
all_dates = sorted(all_dates)

positions = {}   # {symbol: {"entry_date": d, "entry_price": p, "shares": n}}
trades = []      # completed trades
cash = capital
daily_log = []   # (date, cash, mv, total)

for today in all_dates:
    # 检查是否有信号
    for symbol, df in data_map.items():
        idx = None
        for j in range(len(df)):
            if pd.Timestamp(df["date"].iloc[j]).date() == today:
                idx = j
                break
        if idx is None or idx < 120:
            continue

        df_slice = df.iloc[:idx + 1].copy()
        try:
            result = sig_engine.analyze(symbol, df_slice)
        except:
            continue

        if not result.level.is_actionable:
            continue

        # T+1: 找到明天
        next_idx = idx + 1
        if next_idx >= len(df):
            continue
        next_row = df.iloc[next_idx]
        next_date = pd.Timestamp(next_row["date"]).date()
        open_p = float(next_row["open"])

        if result.level.is_bullish:
            if symbol in positions:
                continue

            buy_p = open_p * (1 + slippage)
            max_cap = cash * position_ratio
            raw_shares = int(max_cap / buy_p / 100) * 100
            if raw_shares < 100:
                continue

            cost = raw_shares * buy_p
            comm = max(cost * commission_rate, 5)

            if cost + comm > cash:
                raw_shares -= 100
                if raw_shares < 100:
                    continue
                cost = raw_shares * buy_p
                comm = max(cost * commission_rate, 5)

            cash -= (cost + comm)
            positions[symbol] = {
                "entry_date": next_date,
                "entry_price": buy_p,
                "shares": raw_shares,
                "entry_comm": comm,
                "entry_signal": f"{result.level.label} score={result.score:+.1f}",
            }
            trades.append({
                "date": next_date, "symbol": symbol, "action": "BUY",
                "price": buy_p, "shares": raw_shares, "cost": cost,
                "comm": comm, "cash_after": cash,
                "note": f"{result.level.label} score={result.score:+.1f}",
            })

        elif result.level.is_bearish:
            pos = positions.get(symbol)
            if pos is None:
                continue

            sell_p = open_p * (1 - slippage)
            shares = pos["shares"]
            revenue = shares * sell_p
            comm = max(revenue * commission_rate, 5) + revenue * stamp_tax

            cash += (revenue - comm)

            pnl = revenue - pos["shares"] * pos["entry_price"] - pos["entry_comm"] - comm
            pnl_pct = pnl / (pos["shares"] * pos["entry_price"] + pos["entry_comm"])

            trades.append({
                "date": next_date, "symbol": symbol, "action": "SELL",
                "price": sell_p, "shares": shares, "revenue": revenue,
                "comm": comm, "cash_after": cash,
                "pnl": pnl, "pnl_pct": pnl_pct,
                "note": f"{result.level.label} score={result.score:+.1f}",
            })
            del positions[symbol]

    # 记录每日净值
    mv = 0
    for sym, pos in positions.items():
        df_s = data_map[sym]
        for j in range(len(df_s)):
            if pd.Timestamp(df_s["date"].iloc[j]).date() == today:
                mv += pos["shares"] * float(df_s["close"].iloc[j])
                break
    daily_log.append((today, cash, mv, cash + mv))

# ── 输出 ──
print("=" * 80)
print("  手动仓位追踪 (T+1开盘执行)")
print("=" * 80)
print(f"\n  初始资金: {capital:,.0f}")

for t in trades:
    if t["action"] == "BUY":
        print(f"\n  [{t['date']}] {t['symbol']} 买入")
        print(f"    价格: {t['price']:.2f} (含滑点万1)")
        print(f"    数量: {t['shares']}股")
        print(f"    金额: {t['cost']:,.0f} + 佣金{t['comm']:,.0f} = {t['cost']+t['comm']:,.0f}")
        print(f"    仓位: {(t['cost']+t['comm'])/capital*100:.1f}%")
        print(f"    买入后现金: {t['cash_after']:,.0f}")
        print(f"    信号: {t['note']}")
    else:
        print(f"\n  [{t['date']}] {t['symbol']} 卖出")
        print(f"    价格: {t['price']:.2f} (含滑点万1)")
        print(f"    数量: {t['shares']}股")
        print(f"    金额: {t['revenue']:,.0f} - 佣金印花税{t['comm']:,.0f} = {t['revenue']-t['comm']:,.0f}")
        print(f"    盈亏: {t['pnl']:+,.0f} ({t['pnl_pct']*100:+.2f}%)")
        print(f"    卖出后现金: {t['cash_after']:,.0f}")
        print(f"    信号: {t['note']}")

print(f"\n{'='*80}")
print(f"  汇总")
print(f"{'='*80}")

total_buy = sum(t['cost']+t['comm'] for t in trades if t['action']=='BUY')
total_sell = sum(t['revenue']-t['comm'] for t in trades if t['action']=='SELL')
total_comm = sum(t['comm'] for t in trades)

print(f"  总买入支出: {total_buy:,.0f}")
print(f"  总卖出收回: {total_sell:,.0f}")
print(f"  总手续费:   {total_comm:,.0f}")
print(f"  最终现金:   {cash:,.0f}")
print(f"  总收益率:   {(cash/capital-1)*100:+.2f}%")

# 持仓期间每日净值
print(f"\n  关键时刻净值:")
for d in trades:
    dt = d['date']
    for ld, lcash, lmv, ltotal in daily_log:
        if ld == dt:
            print(f"    {dt}  现金{lcash:,.0f} + 市值{lmv:,.0f} = {ltotal:,.0f}  ({ltotal/capital*100-100:+.2f}%)")
            break
# 最后一天
ld, lcash, lmv, ltotal = daily_log[-1]
print(f"    {ld}  现金{lcash:,.0f} + 市值{lmv:,.0f} = {ltotal:,.0f}  ({ltotal/capital*100-100:+.2f}%)")