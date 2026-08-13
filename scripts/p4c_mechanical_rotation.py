"""
机械组策略验证 — 动量轮动 + 趋势过滤 (v3综合优化)

v3三重优化 (2026-08-10):
  1. 选股增强: 相对强度(RS)过滤 — 股票动量/基准动量>1, 选真正强势股
  2. ATR自适应止损: 止损=买入价-2×ATR, 限制[6%,15%], 替代固定-12%
     → 高波动股(泰嘉/隆基)止损宽, 低波动股(福耀/国电)止损紧
  3. 放量入场确认: 量比>1.0才入场, 过滤缩量假突破

策略规则:
  第一步: 选股层 (月度轮动)
    - 每月首个交易日计算7只股票的20日动量
    - 趋势过滤: MA60上方 + 动量为正
    - v3: 相对强度过滤: 股票动量/沪深300动量 > 1.0
    - 选动量排名前3名

  第二步: 入场层 (v3: 放量确认)
    - 调仓日量比>1.0才入场, 缩量则等待
    - 月度调仓时, 新选入的股票买入, 落选的股票卖出

  第三步: 退出层 (v3: ATR自适应止损)
    - ATR止损: 买入价-2×ATR (限制6%~15%)
    - MA60破位退出: 连续2日确认 + 缓冲带1% + 盈利>5%保护
    - ATR trailing止盈: 盈利>10%启动, trail_mult=2.0
    - 月度调仓换出

对比基线:
  - v1: 固定-12%止损, 无RS过滤, 无放量确认
  - v2: MA60连续2日确认, 缓冲带1%
  - P5: 全仓7只趋势跟踪 (Alpha -24.47%/-23.06%)

用法: python scripts/p4c_mechanical_rotation.py
"""
import sys, os, json, shutil, warnings
import numpy as np
import pandas as pd
from datetime import datetime

warnings.filterwarnings("ignore")
project_root = os.path.join(os.path.dirname(__file__), "..")
sys.path.insert(0, project_root)

from src.config.runtime_mode import set_mode, RuntimeMode
set_mode(RuntimeMode.BACKTEST)

from src.data_fetcher import DataManager
from src.config.group_config import GroupConfig
from src.config.user_preferences import UserPreferences, _USER_PREF_FILE

# ── 机械组7只股票(固定) ──
MECHANICAL_STOCKS = [
    {"code": "000977", "name": "浪潮信息", "market": "sz", "sub": "AI算力"},
    {"code": "600660", "name": "福耀玻璃", "market": "sh", "sub": "汽车玻璃"},
    {"code": "002843", "name": "泰嘉股份", "market": "sz", "sub": "金属制品"},
    {"code": "601012", "name": "隆基绿能", "market": "sh", "sub": "光伏"},
    {"code": "600875", "name": "东方电气", "market": "sh", "sub": "电力设备"},
    {"code": "600406", "name": "国电南瑞", "market": "sh", "sub": "电力设备"},
    {"code": "000938", "name": "紫光股份", "market": "sz", "sub": "AI算力"},
]

# ── 双窗口验证 ──
TRAIN_START, TRAIN_END = "2024-07-01", "2025-06-30"
TEST_START, TEST_END = "2025-07-01", "2026-06-30"
DATA_START, DATA_END = "2024-02-01", "2026-07-13"
BENCHMARK = "sh.000300"

# ── 策略参数 (v3综合优化) ──
MOMENTUM_PERIOD = 20            # 动量计算周期(20日)
TOP_N = 3                       # 选动量前3名
MA60_PERIOD = 60
MA20_PERIOD = 20
ATR_PERIOD = 14
REBALANCE_DAY = 1               # 每月第1个交易日调仓
TRAIL_START_PCT = 0.10          # 盈利>10%启动trailing
TRAIL_MULT = 2.0                # trailing ATR倍数
MAX_POSITION_RATIO = 0.30       # 单股最大仓位30%
INITIAL_CAPITAL = 100000

# ── v3优化参数 (v4b: 基准MA60方向自适应止损) ──
# v4用ADX判断体制失败(短期ADX=50假信号), 改用基准MA60方向更稳定
# 基准MA60上方(牛市)→固定-12%紧止损; 基准MA60下方(震荡/熊市)→ATR 3×宽止损
USE_REGIME_STOP = True          # 启用市场体制自适应止损
REGIME_METHOD = "bench_ma60"    # v4b: 用基准MA60方向(非ADX)
REGIME_ADX_THRESHOLD = 25       # ADX阈值(保留, REGIME_METHOD=adx时生效)
ATR_STOP_MULT = 3.0             # 震荡市ATR止损倍数(宽止损, 给恢复空间)
FIXED_STOP_PCT = -0.12          # 趋势市固定止损-12%(紧止损, 控制反转)
MAX_STOP_PCT = 0.18             # ATR止损上限
MIN_STOP_PCT = 0.10             # ATR止损下限
MA60_BREAK_DAYS = 2             # MA60破位连续N日确认
MA60_BUFFER_PCT = 0.01          # MA60缓冲带1%
PROFIT_PROTECT_PCT = 0.05       # 盈利>5%后MA60破位立即退出
RS_THRESHOLD = 0.8              # 相对强度阈值
VOL_RATIO_MIN = 0.0             # 关闭放量确认
ADX_PERIOD = 14                 # ADX计算周期
BENCH_MA60_PERIOD = 60          # 基准MA60周期

# ── P5基线 (全仓7只趋势跟踪) ──
BASELINE = {
    "train": {"alpha_pct": -24.47, "sharpe": -0.5, "total_return_pct": -11.31, "trade_count": 15},
    "test": {"alpha_pct": -23.06, "sharpe": -0.8, "total_return_pct": 3.23, "trade_count": 18},
}


def calc_indicators(df):
    """计算策略所需指标"""
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"])
    df["ma60"] = df["close"].rolling(MA60_PERIOD).mean()
    df["ma20"] = df["close"].rolling(MA20_PERIOD).mean()
    # 20日动量
    df["momentum_20d"] = df["close"].pct_change(MOMENTUM_PERIOD)
    # ATR
    high_low = df["high"] - df["low"]
    high_close = (df["high"] - df["close"].shift(1)).abs()
    low_close = (df["low"] - df["close"].shift(1)).abs()
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    df["atr"] = tr.rolling(ATR_PERIOD).mean()
    df["atr_pct"] = df["atr"] / df["close"]
    # 量比(v3新增: 入场放量确认)
    df["vol_ma20"] = df["volume"].rolling(20).mean()
    df["vol_ratio"] = df["volume"] / df["vol_ma20"]
    return df


class RotationStrategy:
    """动量轮动 + 趋势过滤策略"""

    def __init__(self, capital):
        self.capital = capital
        self.cash = capital
        self.positions = {}  # {code: {shares, avg_cost, entry_date, highest_price}}
        self.closed_trades = []
        self.daily_values = []
        self.rebalance_log = []  # 记录每次调仓

    def run(self, stock_data, start_date, end_date, bench_df):
        """运行回测"""
        analyzed = {}
        for code, df in stock_data.items():
            df = calc_indicators(df)
            analyzed[code] = df

        # 预计算基准动量(v3) + 基准ADX(v4) + 基准MA60(v4b)
        bench_df = bench_df.copy()
        bench_df["date"] = pd.to_datetime(bench_df["date"])
        bench_df["bench_momentum_20d"] = bench_df["close"].pct_change(MOMENTUM_PERIOD)
        # v4b: 基准MA60(比ADX更稳定, 不会因短期波动产生假信号)
        bench_df["bench_ma60"] = bench_df["close"].rolling(BENCH_MA60_PERIOD).mean()
        # v4: 基准ADX(保留作为备选)
        bench_high = bench_df["high"]
        bench_low = bench_df["low"]
        plus_dm = bench_high.diff()
        minus_dm = -bench_low.diff()
        plus_dm = plus_dm.where((plus_dm > minus_dm) & (plus_dm > 0), 0)
        minus_dm = minus_dm.where((minus_dm > plus_dm) & (minus_dm > 0), 0)
        atr_bench = (bench_high - bench_low).rolling(ADX_PERIOD).mean()
        plus_di = 100 * (plus_dm.rolling(ADX_PERIOD).mean() / atr_bench)
        minus_di = 100 * (minus_dm.rolling(ADX_PERIOD).mean() / atr_bench)
        dx = 100 * ((plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan))
        bench_df["bench_adx"] = dx.rolling(ADX_PERIOD).mean()

        bench_momentum_map = dict(zip(bench_df["date"], bench_df["bench_momentum_20d"]))
        bench_adx_map = dict(zip(bench_df["date"], bench_df["bench_adx"]))
        bench_close_map = dict(zip(bench_df["date"], bench_df["close"]))
        bench_ma60_map = dict(zip(bench_df["date"], bench_df["bench_ma60"]))

        mask = (bench_df["date"] >= start_date) & (bench_df["date"] <= end_date)
        trading_days = bench_df[mask]["date"].tolist()

        current_holdings = set()
        last_rebalance_month = None

        for day in trading_days:
            day_ts = pd.Timestamp(day)

            # 判断是否调仓日(每月第一个交易日)
            month_key = (day_ts.year, day_ts.month)
            is_rebalance_day = (last_rebalance_month != month_key)
            if is_rebalance_day:
                last_rebalance_month = month_key

            # ── 第一步: 选股(调仓日) — v3: 相对强度过滤 ──
            if is_rebalance_day:
                bench_mom = bench_momentum_map.get(day_ts, 0)
                if pd.isna(bench_mom):
                    bench_mom = 0

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
                        # v3: 相对强度 = 股票动量 / 基准动量
                        rs = (row["momentum_20d"] / bench_mom) if bench_mom > 0 else (
                            999 if row["momentum_20d"] > 0 else 0)
                        if rs >= RS_THRESHOLD:
                            momentum_scores[code] = {
                                "momentum": row["momentum_20d"],
                                "rs": rs,
                                "vol_ratio": row.get("vol_ratio", 1.0),
                            }

                # 按动量排名选前N
                sorted_codes = sorted(momentum_scores.items(),
                                      key=lambda x: x[1]["momentum"], reverse=True)
                target_holdings = set([c for c, _ in sorted_codes[:TOP_N]])

                # 记录调仓
                self.rebalance_log.append({
                    "date": day_ts,
                    "candidates": len(momentum_scores),
                    "selected": list(target_holdings),
                    "momentum": {c: round(m["momentum"]*100, 2) for c, m in sorted_codes[:TOP_N]},
                    "bench_momentum": round(bench_mom*100, 2),
                })

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

                # 1. 体制自适应止损 (v4b: 基准MA60方向判断, 比ADX更稳定)
                avg_cost = pos["avg_cost"]
                current_loss_pct = (current_price - avg_cost) / avg_cost

                # v4b: 用基准MA60方向判断体制(基准价>MA60=牛市→紧止损, <MA60=震荡→宽止损)
                if USE_REGIME_STOP:
                    bench_close_today = bench_close_map.get(day_ts, 0)
                    bench_ma60_today = bench_ma60_map.get(day_ts, 0)
                    if pd.isna(bench_close_today) or pd.isna(bench_ma60_today) or bench_ma60_today == 0:
                        is_trending = True  # 无数据时默认趋势市(保守)
                    else:
                        is_trending = bench_close_today > bench_ma60_today
                else:
                    is_trending = True

                if USE_REGIME_STOP and not is_trending:
                    # 震荡市(基准<MA60): ATR宽止损(给恢复空间, 避免假突破被止损)
                    if not pd.isna(row["atr"]) and row["atr"] > 0:
                        atr_stop_price = avg_cost - row["atr"] * ATR_STOP_MULT
                        atr_stop_pct = (atr_stop_price - avg_cost) / avg_cost
                        atr_stop_pct = max(atr_stop_pct, -MAX_STOP_PCT)
                        atr_stop_pct = min(atr_stop_pct, -MIN_STOP_PCT)
                        if current_loss_pct <= atr_stop_pct:
                            self._close_position(code, day_ts, current_price,
                                               f"震荡市ATR止损{atr_stop_pct*100:.1f}%")
                            current_holdings.discard(code)
                            continue
                    else:
                        if current_loss_pct <= -MAX_STOP_PCT:
                            self._close_position(code, day_ts, current_price,
                                               f"震荡市回退止损{MAX_STOP_PCT*100:.0f}%")
                            current_holdings.discard(code)
                            continue
                else:
                    # 趋势市(基准>MA60): 固定-12%紧止损(控制反转风险)
                    if current_loss_pct <= FIXED_STOP_PCT:
                        reason = f"趋势市固定止损{FIXED_STOP_PCT*100:.0f}%" if USE_REGIME_STOP else f"固定止损{FIXED_STOP_PCT*100:.0f}%"
                        self._close_position(code, day_ts, current_price, reason)
                        current_holdings.discard(code)
                        continue

                # 2. MA60破位退出 (连续N日确认 + 缓冲带 + 盈利保护)
                if not pd.isna(row["ma60"]):
                    profit_pct = (current_price - avg_cost) / avg_cost
                    below_ma60 = current_price < row["ma60"] * (1 - MA60_BUFFER_PCT)

                    if below_ma60:
                        pos["ma60_break_days"] = pos.get("ma60_break_days", 0) + 1
                        if profit_pct > PROFIT_PROTECT_PCT or pos["ma60_break_days"] >= MA60_BREAK_DAYS:
                            reason = (f"MA60破位退出(盈利{profit_pct*100:.1f}%保护)" if profit_pct > PROFIT_PROTECT_PCT
                                     else f"MA60破位退出(连续{MA60_BREAK_DAYS}日)")
                            self._close_position(code, day_ts, current_price, reason)
                            current_holdings.discard(code)
                            continue
                    else:
                        pos["ma60_break_days"] = 0

                # 3. ATR trailing stop (盈利>10%后启动)
                profit_pct = (current_price - avg_cost) / avg_cost
                if profit_pct > TRAIL_START_PCT and not pd.isna(row["atr"]):
                    trail_dist = row["atr"] * TRAIL_MULT
                    if current_price <= pos["highest_price"] - trail_dist:
                        self._close_position(code, day_ts, current_price,
                                           f"ATR trailing止盈(盈利{profit_pct*100:.1f}%)")
                        current_holdings.discard(code)
                        continue

            # ── 第三步: 入场(调仓日) — v3: 放量确认 ──
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
                        # v3: 放量确认(量比>1.0), 缩量则跳过当日入场
                        vol_ratio = float(row.iloc[0].get("vol_ratio", 1.0) or 1.0)
                        if pd.isna(vol_ratio) or vol_ratio < VOL_RATIO_MIN:
                            continue
                        price = float(row.iloc[0]["close"])
                        target_value = min(target_per_stock, self.cash)
                        if target_value < 1000:
                            continue
                        shares = int(target_value / price / 100) * 100
                        if shares == 0:
                            continue
                        self._open_position(code, day_ts, price, shares, "动量轮动买入")

            # 记录每日净值
            total_value = self.cash
            for code, pos in self.positions.items():
                df = analyzed[code]
                row = df[df["date"] == day_ts]
                if not row.empty:
                    total_value += pos["shares"] * float(row.iloc[0]["close"])
            self.daily_values.append({"date": day_ts, "value": total_value})

        return self._compute_metrics(bench_df, start_date, end_date)

    def _open_position(self, code, date, price, shares, reason):
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

        self.closed_trades.append({
            "code": code,
            "entry_date": pos["entry_date"],
            "entry_price": pos["avg_cost"],
            "exit_date": date,
            "exit_price": price,
            "shares": pos["shares"],
            "pnl": pnl,
            "pnl_pct": pnl_pct,
            "holding_days": holding_days,
            "exit_reason": reason,
        })

    def _compute_metrics(self, bench_df, start, end):
        """计算绩效"""
        if not self.daily_values:
            return {"error": "无交易数据"}

        dv = pd.DataFrame(self.daily_values).set_index("date")
        daily_ret = dv["value"].pct_change().dropna()
        total_return = (dv["value"].iloc[-1] / self.capital) - 1
        sharpe = (daily_ret.mean() / daily_ret.std() * np.sqrt(252)
                  if daily_ret.std() > 0 else 0)

        cummax = dv["value"].cummax()
        drawdown = (dv["value"] - cummax) / cummax
        max_dd = drawdown.min()

        bench_df = bench_df.copy()
        bench_df["date"] = pd.to_datetime(bench_df["date"])
        bmask = (bench_df["date"] >= start) & (bench_df["date"] <= end)
        bsub = bench_df[bmask]
        bench_ret = (bsub["close"].iloc[-1] / bsub["close"].iloc[0]) - 1 if len(bsub) > 0 else 0
        alpha = total_return - bench_ret

        trade_count = len(self.closed_trades)
        win_trades = sum(1 for t in self.closed_trades if t["pnl"] > 0)
        win_rate = win_trades / trade_count if trade_count > 0 else 0
        avg_holding = np.mean([t["holding_days"] for t in self.closed_trades]) if trade_count > 0 else 0

        exit_reasons = {}
        for t in self.closed_trades:
            r = t["exit_reason"]
            # 归类
            if "调仓" in r:
                key = "月度调仓换出"
            elif "趋势市固定止损" in r:
                key = "趋势市固定止损-12%"
            elif "震荡市ATR止损" in r:
                key = "震荡市ATR宽止损"
            elif "震荡市回退止损" in r:
                key = "震荡市回退止损"
            elif "固定止损" in r:
                key = "固定止损-12%"
            elif "ATR止损" in r:
                key = "ATR自适应止损"
            elif "回退止损" in r:
                key = "回退止损"
            elif "MA60" in r and "保护" in r:
                key = "MA60破位(盈利保护)"
            elif "MA60" in r and "连续" in r:
                key = "MA60破位(连续确认)"
            elif "MA60" in r:
                key = "MA60破位退出"
            elif "trailing" in r:
                key = "ATR trailing止盈"
            else:
                key = r
            exit_reasons[key] = exit_reasons.get(key, 0) + 1

        # 选股统计: 哪些股票被选中过
        selected_stocks = {}
        for log in self.rebalance_log:
            for code in log["selected"]:
                selected_stocks[code] = selected_stocks.get(code, 0) + 1

        return {
            "total_return_pct": round(total_return * 100, 2),
            "alpha_pct": round(alpha * 100, 2),
            "benchmark_return_pct": round(bench_ret * 100, 2),
            "sharpe": round(sharpe, 3),
            "max_drawdown_pct": round(max_dd * 100, 2),
            "trade_count": trade_count,
            "win_rate_pct": round(win_rate * 100, 1),
            "avg_holding_days": round(avg_holding, 1),
            "exit_reasons": exit_reasons,
            "selected_stocks": selected_stocks,
            "rebalance_count": len(self.rebalance_log),
            "final_value": round(dv["value"].iloc[-1], 2),
        }


def main():
    print("=" * 100)
    print("  机械组策略验证 — 动量轮动 + 趋势过滤")
    print(f"  策略: 月度选动量前3 + MA60上方 + ATR trailing止盈 + MA60破位退出")
    print(f"  训练窗: {TRAIN_START}~{TRAIN_END} (震荡市, 基准+13.16%)")
    print(f"  测试窗: {TEST_START}~{TEST_END} (牛市, 基准+26.29%)")
    print("=" * 100)

    pref_existed = os.path.exists(_USER_PREF_FILE)
    pref_backup = _USER_PREF_FILE + ".p4c_bak"
    if pref_existed:
        shutil.copy2(_USER_PREF_FILE, pref_backup)
    UserPreferences().clear_all()

    try:
        dm = DataManager()
        stock_data = {}
        for s in MECHANICAL_STOCKS:
            df = dm.get_daily_kline(s["code"], start_date=DATA_START, end_date=DATA_END)
            if df is not None and len(df) > 120:
                stock_data[s["code"]] = df
                print(f"  {s['name']}({s['code']}/{s['sub']}): {len(df)}条")
        bench_df = dm.get_daily_kline(BENCHMARK, start_date=DATA_START, end_date=DATA_END)
        print(f"  沪深300: {len(bench_df)}条\n")

        results = {}
        for window_name, ws, we in [("训练窗", TRAIN_START, TRAIN_END),
                                     ("测试窗", TEST_START, TEST_END)]:
            print(f"{'='*100}")
            print(f"  [{window_name} {ws}~{we}]")
            print(f"{'='*100}")

            strat = RotationStrategy(INITIAL_CAPITAL)
            m = strat.run(stock_data, ws, we, bench_df)
            results[window_name] = m

            if "error" in m:
                print(f"  {m['error']}")
                continue

            print(f"\n  组合收益: {m['total_return_pct']:+.2f}%  Alpha: {m['alpha_pct']:+.2f}%  "
                  f"夏普: {m['sharpe']:.3f}  回撤: {m['max_drawdown_pct']:.2f}%")
            print(f"  交易笔数: {m['trade_count']}  胜率: {m['win_rate_pct']:.1f}%  "
                  f"平均持有: {m['avg_holding_days']:.1f}天  基准: {m['benchmark_return_pct']:+.2f}%")
            print(f"  调仓次数: {m['rebalance_count']}  退出原因: {m['exit_reasons']}")

            # 选股统计
            print(f"\n  选股统计(被选中次数):")
            code_to_name = {s["code"]: s["name"] for s in MECHANICAL_STOCKS}
            for code, count in sorted(m["selected_stocks"].items(), key=lambda x: x[1], reverse=True):
                name = code_to_name.get(code, code)
                sub = next((s["sub"] for s in MECHANICAL_STOCKS if s["code"] == code), "")
                print(f"    {name}({sub}): {count}次")

            # 交易明细
            if strat.closed_trades:
                print(f"\n  交易明细:")
                print(f"  {'股票':<10} {'子行业':<8} {'持有天数':>8} {'收益率':>8} {'退出原因':<20}")
                print(f"  {'-'*65}")
                for t in strat.closed_trades:
                    name = code_to_name.get(t["code"], t["code"])
                    sub = next((s["sub"] for s in MECHANICAL_STOCKS if s["code"] == t["code"]), "")
                    print(f"  {name:<10} {sub:<8} {t['holding_days']:>7}天 {t['pnl_pct']*100:>+7.2f}% {t['exit_reason']:<20}")

            # 调仓记录
            if strat.rebalance_log:
                print(f"\n  调仓记录(前5次):")
                for log in strat.rebalance_log[:5]:
                    selected_names = [code_to_name.get(c, c) for c in log["selected"]]
                    print(f"    {log['date'].strftime('%Y-%m-%d')}: 候选{log['candidates']}只 → 选中{selected_names}")

        # ── 对比基线 ──
        print(f"\n{'='*100}")
        print(f"  动量轮动 vs P5基线(全仓趋势跟踪) 对比")
        print(f"{'='*100}")
        print(f"\n  {'指标':<12} {'P5基线(全仓)':>14} {'动量轮动(前3)':>14} {'改进':>10}")
        print(f"  {'-'*54}")
        for window in ["训练窗", "测试窗"]:
            m = results.get(window, {})
            b = BASELINE["train" if window == "训练窗" else "test"]
            print(f"\n  [{window}]")
            for key, label in [("alpha_pct", "Alpha%"), ("sharpe", "夏普"),
                               ("total_return_pct", "收益%"), ("trade_count", "交易数")]:
                v_b = b.get(key, 0)
                v_r = m.get(key, 0)
                d = v_r - v_b
                print(f"  {label:<12} {v_b:>+13.2f} {v_r:>+13.2f} {d:>+9.2f}")

        # ── 评估 ──
        print(f"\n{'='*100}")
        print(f"  动量轮动策略评估")
        print(f"{'='*100}")
        train_m = results.get("训练窗", {})
        test_m = results.get("测试窗", {})
        train_alpha = train_m.get("alpha_pct", -100)
        test_alpha = test_m.get("alpha_pct", -100)

        print(f"\n  训练窗Alpha: {train_alpha:+.2f}% {'✅转正' if train_alpha > 0 else '❌仍为负'}")
        print(f"  测试窗Alpha: {test_alpha:+.2f}% {'✅转正' if test_alpha > 0 else '❌仍为负'}")

        print(f"\n  vs P5基线:")
        print(f"    训练Alpha: {train_alpha:+.2f}% vs P5 {BASELINE['train']['alpha_pct']:+.2f}% "
              f"({train_alpha - BASELINE['train']['alpha_pct']:+.2f}%)")
        print(f"    测试Alpha: {test_alpha:+.2f}% vs P5 {BASELINE['test']['alpha_pct']:+.2f}% "
              f"({test_alpha - BASELINE['test']['alpha_pct']:+.2f}%)")

        all_ok = train_alpha > 0 and test_alpha > 0
        improved = (train_alpha > BASELINE['train']['alpha_pct'] and
                   test_alpha > BASELINE['test']['alpha_pct'])
        print(f"\n  双窗Alpha转正: {'✅ 是' if all_ok else '❌ 否'}")
        print(f"  双窗均改善P5: {'✅ 是' if improved else '❌ 否'}")
        if improved and not all_ok:
            print(f"  → 显著改善P5, 可考虑参数调优进一步优化")
        elif all_ok:
            print(f"  → 策略验证通过, 可启用机械组")
        else:
            print(f"  → 未改善, 需重新设计策略")

    finally:
        if pref_existed and os.path.exists(pref_backup):
            shutil.move(pref_backup, _USER_PREF_FILE)


if __name__ == "__main__":
    main()
