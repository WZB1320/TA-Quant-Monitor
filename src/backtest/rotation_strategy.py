"""
动量轮动策略引擎 — 机械组专用

策略逻辑 (v1基线, 测试窗Alpha -9.97%, 最优版本):
  第一步: 选股层 (月度轮动)
    - 每月首个交易日计算股票的20日动量
    - 趋势过滤: MA60上方 + 动量为正
    - 选动量排名前N名

  第二步: 入场层
    - 调仓日买入新选入股票, 等权分配
    - 单股最大仓位30%

  第三步: 退出层
    - 固定止损-12% (给高波动强势股足够空间让利润奔跑)
    - MA60破位退出 (单日判断)
    - ATR trailing止盈: 盈利>10%启动, trail_mult=2.0
    - 月度调仓换出

集成方式:
    BacktestEngine路由层检测到 strategy_mode="rotation" 时,
    使用本类替代 BacktestEngine 运行该组回测.
"""
import numpy as np
import pandas as pd
from typing import Dict, List, Optional


# ── v1策略参数 ──
MOMENTUM_PERIOD = 20
TOP_N = 3
MA60_PERIOD = 60
MA20_PERIOD = 20
ATR_PERIOD = 14
TRAIL_START_PCT = 0.10
TRAIL_MULT = 2.0
HARD_STOP_PCT = -0.12
MAX_POSITION_RATIO = 0.30


def _calc_indicators(df):
    """计算策略所需指标"""
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"])
    df["ma60"] = df["close"].rolling(MA60_PERIOD).mean()
    df["ma20"] = df["close"].rolling(MA20_PERIOD).mean()
    df["momentum_20d"] = df["close"].pct_change(MOMENTUM_PERIOD)
    high_low = df["high"] - df["low"]
    high_close = (df["high"] - df["close"].shift(1)).abs()
    low_close = (df["low"] - df["close"].shift(1)).abs()
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    df["atr"] = tr.rolling(ATR_PERIOD).mean()
    df["atr_pct"] = df["atr"] / df["close"]
    return df


class RotationTrade:
    """轮动策略交易记录 — 适配BacktestEngine的Trade格式, 供路由层统一处理"""
    def __init__(self, symbol, entry_date, entry_price, exit_date, exit_price,
                 shares, pnl, pnl_pct, holding_days, exit_reason):
        self.symbol = symbol
        self.entry_date = entry_date
        self.entry_price = entry_price
        self.exit_date = exit_date
        self.exit_price = exit_price
        self.shares = shares
        self.pnl = pnl
        self.pnl_pct = pnl_pct
        self.holding_days = holding_days
        self.entry_signal = "动量轮动买入"
        self.exit_signal = exit_reason
        self.commission = 0.0
        self.highest_price = max(entry_price, exit_price)

    @property
    def is_closed(self) -> bool:
        return self.exit_date is not None


class RotationStrategy:
    """动量轮动 + 趋势过滤策略 (机械组专用)"""

    def __init__(self, initial_capital: float = 100000):
        self.capital = initial_capital
        self.cash = initial_capital
        self.positions: Dict = {}  # {code: {shares, avg_cost, entry_date, highest_price}}
        self.closed_trades: List[RotationTrade] = []
        self.daily_values: Optional[pd.Series] = None
        self.open_positions: Dict = {}

    def run(self, stock_data: Dict, bench_df: pd.DataFrame,
            start_date: str, end_date: str) -> dict:
        """运行回测

        Args:
            stock_data: {code: DataFrame} 股票日线数据
            bench_df: 基准日线数据 (用于提取交易日历)
            start_date: 回测开始日期
            end_date: 回测结束日期

        Returns:
            dict with keys: daily_values(pd.Series), closed_trades, open_positions
        """
        analyzed = {}
        for code, df in stock_data.items():
            analyzed[code] = _calc_indicators(df)

        bench_df = bench_df.copy()
        bench_df["date"] = pd.to_datetime(bench_df["date"])
        mask = (bench_df["date"] >= start_date) & (bench_df["date"] <= end_date)
        trading_days = bench_df[mask]["date"].tolist()

        current_holdings = set()
        last_rebalance_month = None
        daily_values_list = []

        for day in trading_days:
            day_ts = pd.Timestamp(day)

            # 判断是否调仓日(每月第一个交易日)
            month_key = (day_ts.year, day_ts.month)
            is_rebalance_day = (last_rebalance_month != month_key)
            if is_rebalance_day:
                last_rebalance_month = month_key

            # ── 第一步: 选股(调仓日) ──
            if is_rebalance_day:
                momentum_scores = {}
                for code, df in analyzed.items():
                    row = df[df["date"] == day_ts]
                    if row.empty:
                        continue
                    row = row.iloc[0]
                    if pd.isna(row["momentum_20d"]) or pd.isna(row["ma60"]):
                        continue
                    # 趋势过滤: MA60上方 + 动量为正
                    if row["close"] > row["ma60"] and row["momentum_20d"] > 0:
                        momentum_scores[code] = row["momentum_20d"]

                # 选动量前N名
                sorted_codes = sorted(momentum_scores.items(), key=lambda x: x[1], reverse=True)
                target_holdings = set([c for c, _ in sorted_codes[:TOP_N]])

                # 卖出落选股票
                to_sell = current_holdings - target_holdings
                for code in to_sell:
                    if code in self.positions:
                        row = analyzed[code][analyzed[code]["date"] == day_ts]
                        if not row.empty:
                            self._close_position(code, day_ts, float(row.iloc[0]["close"]), "月度调仓换出")

                current_holdings = target_holdings

            # ── 第二步: 退出检查(每日) ──
            for code in list(self.positions.keys()):
                df = analyzed[code]
                row = df[df["date"] == day_ts]
                if row.empty:
                    continue
                row = row.iloc[0]
                current_price = float(row["close"])
                pos = self.positions[code]

                # 更新最高价
                if current_price > pos["highest_price"]:
                    pos["highest_price"] = current_price

                avg_cost = pos["avg_cost"]

                # 1. 固定止损-12%
                if (current_price - avg_cost) / avg_cost <= HARD_STOP_PCT:
                    self._close_position(code, day_ts, current_price, f"固定止损{HARD_STOP_PCT*100:.0f}%")
                    current_holdings.discard(code)
                    continue

                # 2. MA60破位退出
                if not pd.isna(row["ma60"]) and current_price < row["ma60"]:
                    self._close_position(code, day_ts, current_price, "MA60破位退出")
                    current_holdings.discard(code)
                    continue

                # 3. ATR trailing stop (盈利>10%后启动)
                profit_pct = (current_price - avg_cost) / avg_cost
                if profit_pct > TRAIL_START_PCT and not pd.isna(row["atr"]):
                    trail_dist = row["atr"] * TRAIL_MULT
                    if current_price <= pos["highest_price"] - trail_dist:
                        self._close_position(code, day_ts, current_price,
                                           f"ATR trailing止盈(盈利{profit_pct*100:.1f}%)")
                        current_holdings.discard(code)
                        continue

            # ── 第三步: 入场(调仓日, 买入新选入股票) ──
            if is_rebalance_day:
                to_buy = current_holdings - set(self.positions.keys())
                n_holding = len(current_holdings)
                if n_holding > 0:
                    target_per_stock = min(self.capital / n_holding, self.capital * MAX_POSITION_RATIO)
                    for code in to_buy:
                        df = analyzed[code]
                        row = df[df["date"] == day_ts]
                        if row.empty:
                            continue
                        price = float(row.iloc[0]["close"])
                        target_value = min(target_per_stock, self.cash)
                        if target_value < 1000:
                            continue
                        shares = int(target_value / price / 100) * 100
                        if shares == 0:
                            continue
                        self._open_position(code, day_ts, price, shares)

            # 记录每日净值
            total_value = self.cash
            for code, pos in self.positions.items():
                df = analyzed[code]
                row = df[df["date"] == day_ts]
                if not row.empty:
                    total_value += pos["shares"] * float(row.iloc[0]["close"])
            daily_values_list.append({"date": day_ts, "value": total_value})

        # 转换daily_values为pd.Series
        if daily_values_list:
            dv_df = pd.DataFrame(daily_values_list).set_index("date")
            self.daily_values = dv_df["value"]

        # 未平仓持仓转为空dict (回测结束时清理)
        # 将剩余持仓按最后一天收盘价平仓记录
        if self.positions and daily_values_list:
            last_day = daily_values_list[-1]["date"]
            for code in list(self.positions.keys()):
                df = analyzed[code]
                row = df[df["date"] == last_day]
                if not row.empty:
                    self._close_position(code, last_day, float(row.iloc[0]["close"]), "回测结束平仓")

        return {
            "daily_values": self.daily_values,
            "closed_trades": self.closed_trades,
            "open_positions": {},
        }

    def _open_position(self, code, date, price, shares):
        """开仓"""
        cost = shares * price
        self.cash -= cost
        self.positions[code] = {
            "shares": shares,
            "avg_cost": price,
            "entry_date": date,
            "highest_price": price,
        }

    def _close_position(self, code, date, price, reason):
        """平仓"""
        pos = self.positions.pop(code)
        proceeds = pos["shares"] * price
        self.cash += proceeds
        pnl = (price - pos["avg_cost"]) * pos["shares"]
        pnl_pct = (price - pos["avg_cost"]) / pos["avg_cost"]
        holding_days = (date - pos["entry_date"]).days

        self.closed_trades.append(RotationTrade(
            symbol=code,
            entry_date=pos["entry_date"],
            entry_price=pos["avg_cost"],
            exit_date=date,
            exit_price=price,
            shares=pos["shares"],
            pnl=pnl,
            pnl_pct=pnl_pct,
            holding_days=holding_days,
            exit_reason=reason,
        ))
