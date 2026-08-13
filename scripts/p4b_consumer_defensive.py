"""
方案B 防御性持有+事件增强 — 消费组回测验证

策略规则 (基于消费组特性分析):
  1. 基础仓位: MA60上方 + RSI<50(估值低位) → 建仓30%资金
  2. 事件增强: 放量突破(vol_ratio>1.5 + 涨幅>2%) → 加仓20%资金
  3. 退出: MA60破位(收盘价<MA60) → 清仓, 不设目标价
  4. 止损: 硬止损-12% (与P5一致)
  5. 不做均值回归超卖反弹 (P2已证伪: 超卖后不反弹)
  6. 不做趋势跟踪频繁交易 (Hurst=0.334 趋势性弱)

对比基线:
  - P2均值回归: 训练Alpha-11~-12%, 测试Alpha-34~35%
  - P3+趋势跟踪: 训练Alpha-6.5%, 测试Alpha-35.3%

验证目标:
  - 能否在测试窗(牛市)减少损失 (消费股普跌, 方案B应低频少亏)
  - 能否在训练窗(震荡市)获得正Alpha
  - 交易频率应显著低于P2/P3 (低频持有)

用法: python scripts/p4b_consumer_defensive.py
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

# ── 消费组4只股票 ──
CONSUMER_STOCKS = [
    {"code": "600887", "name": "伊利股份", "market": "sh"},
    {"code": "603288", "name": "海天味业", "market": "sh"},
    {"code": "002507", "name": "涪陵榨菜", "market": "sz"},
    {"code": "300673", "name": "佩蒂股份", "market": "sz"},
]

# ── 双窗口验证 (与P2/P3一致, 便于横向对比) ──
TRAIN_START, TRAIN_END = "2024-07-01", "2025-06-30"
TEST_START, TEST_END = "2025-07-01", "2026-06-30"
DATA_START, DATA_END = "2024-02-01", "2026-07-13"
BENCHMARK = "sh.000300"

# ── 方案B 参数 ──
MA60_PERIOD = 60
MA20_PERIOD = 20
RSI_PERIOD = 14
RSI_LOW_THRESHOLD = 50          # 估值低位: RSI<50
VOL_MA_PERIOD = 20
VOL_RATIO_THRESHOLD = 1.5       # 放量: 量比>1.5
BREAKTHROUGH_PCT = 0.02         # 突破: 涨幅>2%
BASE_POSITION_RATIO = 0.30      # 基础仓位30%
EVENT_POSITION_RATIO = 0.20     # 事件加仓20%
MAX_POSITION_RATIO = 0.50       # 单股最大仓位50%
HARD_STOP_PCT = -0.12           # 硬止损-12%
INITIAL_CAPITAL = 100000        # 消费组独立资金10万

# ── P2/P3 基线对照 ──
BASELINE = {
    "train": {"alpha_pct": -6.5, "sharpe": 0.422, "total_return_pct": 6.7, "trade_count": 10},
    "test": {"alpha_pct": -35.3, "sharpe": -2.120, "total_return_pct": -9.0, "trade_count": 13},
    "p2_best_train": {"alpha_pct": -11.0, "sharpe": -0.5, "total_return_pct": 2.0, "trade_count": 8},
    "p2_best_test": {"alpha_pct": -34.0, "sharpe": -1.8, "total_return_pct": -12.0, "trade_count": 12},
}


def calc_indicators(df):
    """计算方案B所需的技术指标"""
    df = df.copy()
    df["ma60"] = df["close"].rolling(MA60_PERIOD).mean()
    df["ma20"] = df["close"].rolling(MA20_PERIOD).mean()

    # RSI
    delta = df["close"].diff()
    gain = delta.where(delta > 0, 0).rolling(RSI_PERIOD).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(RSI_PERIOD).mean()
    rs = gain / loss.replace(0, np.nan)
    df["rsi"] = 100 - (100 / (1 + rs))

    # 量比
    df["vol_ma"] = df["volume"].rolling(VOL_MA_PERIOD).mean()
    df["vol_ratio"] = df["volume"] / df["vol_ma"]

    # 涨幅
    df["pct_change"] = df["close"].pct_change()

    return df


class DefensiveHoldingStrategy:
    """方案B: 防御性持有+事件增强"""

    def __init__(self, capital):
        self.capital = capital
        self.cash = capital
        self.positions = {}  # {code: {shares, avg_cost, position_type, entry_date}}
        self.closed_trades = []
        self.daily_values = []
        self.trade_id = 0

    def run(self, stock_data, start_date, end_date, bench_df):
        """运行回测"""
        # 准备带指标的股票数据
        analyzed = {}
        for code, df in stock_data.items():
            df = calc_indicators(df)
            df["date"] = pd.to_datetime(df["date"])
            analyzed[code] = df

        # 构建交易日列表 (用基准日期)
        bench_df = bench_df.copy()
        bench_df["date"] = pd.to_datetime(bench_df["date"])
        mask = (bench_df["date"] >= start_date) & (bench_df["date"] <= end_date)
        trading_days = bench_df[mask]["date"].tolist()

        for day in trading_days:
            day_ts = pd.Timestamp(day)

            # 每日操作: 先检查退出, 再检查入场
            for code, df in analyzed.items():
                if code not in analyzed:
                    continue
                row = df[df["date"] == day_ts]
                if row.empty:
                    continue
                row = row.iloc[0]
                if pd.isna(row["ma60"]) or pd.isna(row["rsi"]):
                    continue

                current_price = float(row["close"])
                current_pos = self.positions.get(code)

                # ── 退出检查 ──
                if current_pos:
                    # 1. MA60破位退出 (核心退出逻辑)
                    if current_price < row["ma60"]:
                        self._close_position(code, day_ts, current_price, "MA60破位退出")
                        continue
                    # 2. 硬止损-12%
                    avg_cost = current_pos["avg_cost"]
                    if (current_price - avg_cost) / avg_cost <= HARD_STOP_PCT:
                        self._close_position(code, day_ts, current_price, f"硬止损{HARD_STOP_PCT*100:.0f}%")
                        continue

                # ── 入场检查 (只在无仓位时) ──
                if not current_pos:
                    # 1. 基础仓位: MA60上方 + RSI<50(估值低位)
                    if (current_price > row["ma60"] and
                        row["rsi"] < RSI_LOW_THRESHOLD and
                        row["pct_change"] > 0):  # 当日上涨才入场
                        target_ratio = BASE_POSITION_RATIO
                        self._open_position(code, day_ts, current_price, target_ratio, "基础仓位")

                    # 2. 事件增强: 放量突破 (独立于基础仓位, 可叠加)
                    elif (row["vol_ratio"] > VOL_RATIO_THRESHOLD and
                          row["pct_change"] > BREAKTHROUGH_PCT and
                          current_price > row["ma60"]):
                        target_ratio = EVENT_POSITION_RATIO
                        self._open_position(code, day_ts, current_price, target_ratio, "事件增强")

            # 记录每日净值
            total_value = self.cash
            for code, pos in self.positions.items():
                df = analyzed[code]
                row = df[df["date"] == day_ts]
                if not row.empty:
                    total_value += pos["shares"] * float(row.iloc[0]["close"])
            self.daily_values.append({"date": day_ts, "value": total_value})

        return self._compute_metrics(analyzed, bench_df, start_date, end_date)

    def _open_position(self, code, date, price, target_ratio, entry_type):
        """开仓"""
        target_value = self.capital * target_ratio
        available = min(target_value, self.cash)
        if available < 1000:
            return
        shares = int(available / price / 100) * 100  # 整手
        if shares == 0:
            return
        cost = shares * price
        self.cash -= cost

        if code in self.positions:
            # 加仓
            old = self.positions[code]
            total_shares = old["shares"] + shares
            new_avg = (old["avg_cost"] * old["shares"] + price * shares) / total_shares
            self.positions[code] = {
                "shares": total_shares,
                "avg_cost": new_avg,
                "entry_date": old["entry_date"],
                "position_type": old["position_type"] + "+" + entry_type,
            }
        else:
            self.positions[code] = {
                "shares": shares,
                "avg_cost": price,
                "entry_date": date,
                "position_type": entry_type,
            }
        self.trade_id += 1

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
            "entry_type": pos["position_type"],
            "exit_reason": reason,
        })

    def _compute_metrics(self, analyzed, bench_df, start, end):
        """计算绩效指标"""
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

        # 基准收益
        bench_df = bench_df.copy()
        bench_df["date"] = pd.to_datetime(bench_df["date"])
        bmask = (bench_df["date"] >= start) & (bench_df["date"] <= end)
        bsub = bench_df[bmask]
        bench_ret = (bsub["close"].iloc[-1] / bsub["close"].iloc[0]) - 1 if len(bsub) > 0 else 0
        alpha = total_return - bench_ret

        # 交易统计
        trade_count = len(self.closed_trades)
        win_trades = sum(1 for t in self.closed_trades if t["pnl"] > 0)
        win_rate = win_trades / trade_count if trade_count > 0 else 0
        avg_holding = np.mean([t["holding_days"] for t in self.closed_trades]) if trade_count > 0 else 0

        # 退出原因统计
        exit_reasons = {}
        for t in self.closed_trades:
            r = t["exit_reason"]
            exit_reasons[r] = exit_reasons.get(r, 0) + 1

        # 入场类型统计
        entry_types = {}
        for t in self.closed_trades:
            et = t["entry_type"]
            entry_types[et] = entry_types.get(et, 0) + 1

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
            "entry_types": entry_types,
            "final_value": round(dv["value"].iloc[-1], 2),
        }


def main():
    print("=" * 95)
    print("  方案B 防御性持有+事件增强 — 消费组回测验证")
    print(f"  策略: MA60上方+RSI<50建仓30% | 放量突破加仓20% | MA60破位退出 | 硬止损-12%")
    print(f"  训练窗: {TRAIN_START}~{TRAIN_END} (震荡市, 基准+13.16%)")
    print(f"  测试窗: {TEST_START}~{TEST_END} (牛市, 基准+26.29%)")
    print("=" * 95)

    pref_existed = os.path.exists(_USER_PREF_FILE)
    pref_backup = _USER_PREF_FILE + ".p4b_bak"
    if pref_existed:
        shutil.copy2(_USER_PREF_FILE, pref_backup)
    UserPreferences().clear_all()

    try:
        dm = DataManager()
        stock_data = {}
        for s in CONSUMER_STOCKS:
            df = dm.get_daily_kline(s["code"], start_date=DATA_START, end_date=DATA_END)
            if df is not None and len(df) > 120:
                stock_data[s["code"]] = df
                print(f"  {s['name']}({s['code']}): {len(df)}条")
        bench_df = dm.get_daily_kline(BENCHMARK, start_date=DATA_START, end_date=DATA_END)
        print(f"  沪深300: {len(bench_df)}条\n")

        results = {}
        for window_name, ws, we in [("训练窗", TRAIN_START, TRAIN_END),
                                     ("测试窗", TEST_START, TEST_END)]:
            print(f"{'='*95}")
            print(f"  [{window_name} {ws}~{we}]")
            print(f"{'='*95}")

            strat = DefensiveHoldingStrategy(INITIAL_CAPITAL)
            m = strat.run(stock_data, ws, we, bench_df)
            results[window_name] = m

            if "error" in m:
                print(f"  {m['error']}")
                continue

            print(f"\n  组合收益: {m['total_return_pct']:+.2f}%  Alpha: {m['alpha_pct']:+.2f}%  "
                  f"夏普: {m['sharpe']:.3f}  回撤: {m['max_drawdown_pct']:.2f}%")
            print(f"  交易笔数: {m['trade_count']}  胜率: {m['win_rate_pct']:.1f}%  "
                  f"平均持有: {m['avg_holding_days']:.1f}天  基准: {m['benchmark_return_pct']:+.2f}%")
            print(f"  入场类型: {m['entry_types']}")
            print(f"  退出原因: {m['exit_reasons']}")

            # 交易明细
            if strat.closed_trades:
                print(f"\n  {'股票':<8} {'入场类型':<12} {'持有天数':>8} {'收益率':>8} {'退出原因':<15}")
                print(f"  {'-'*60}")
                for t in strat.closed_trades:
                    code = t["code"]
                    name = next((s["name"] for s in CONSUMER_STOCKS if s["code"] == code), code)
                    print(f"  {name:<8} {t['entry_type']:<12} {t['holding_days']:>7}天 {t['pnl_pct']*100:>+7.2f}% {t['exit_reason']:<15}")

        # ── 对比基线 ──
        print(f"\n{'='*95}")
        print(f"  方案B vs P2/P3 基线对比")
        print(f"{'='*95}")
        print(f"\n  {'指标':<12} {'P3+趋势跟踪':>12} {'P2最优均值回归':>14} {'方案B防御持有':>14}")
        print(f"  {'-'*58}")
        for window, base_key, p2_key in [("训练窗", "train", "p2_best_train"),
                                          ("测试窗", "test", "p2_best_test")]:
            m = results.get(window, {})
            b = BASELINE[base_key]
            p2 = BASELINE[p2_key]
            print(f"\n  [{window}]")
            for key, label in [("alpha_pct", "Alpha%"), ("sharpe", "夏普"),
                               ("total_return_pct", "收益%"), ("trade_count", "交易数")]:
                v_b = b.get(key, 0)
                v_p2 = p2.get(key, 0)
                v_planb = m.get(key, 0)
                print(f"  {label:<12} {v_b:>+11.2f} {v_p2:>+13.2f} {v_planb:>+13.2f}")

        # ── 评估 ──
        print(f"\n{'='*95}")
        print(f"  方案B 评估")
        print(f"{'='*95}")
        train_m = results.get("训练窗", {})
        test_m = results.get("测试窗", {})
        train_alpha = train_m.get("alpha_pct", -100)
        test_alpha = test_m.get("alpha_pct", -100)
        train_trades = train_m.get("trade_count", 0)
        test_trades = test_m.get("trade_count", 0)

        print(f"\n  训练窗Alpha: {train_alpha:+.2f}% {'✅转正' if train_alpha > 0 else '❌仍为负'}")
        print(f"  测试窗Alpha: {test_alpha:+.2f}% {'✅转正' if test_alpha > 0 else '❌仍为负'}")
        print(f"  交易频率: 训练{train_trades}笔/测试{test_trades}笔 "
              f"({'✅低频' if max(train_trades, test_trades) < 10 else '⚠️偏高'})")
        print(f"\n  vs P2均值回归:")
        print(f"    训练Alpha: {train_alpha:+.2f}% vs P2 {BASELINE['p2_best_train']['alpha_pct']:+.2f}% "
              f"({train_alpha - BASELINE['p2_best_train']['alpha_pct']:+.2f}%)")
        print(f"    测试Alpha: {test_alpha:+.2f}% vs P2 {BASELINE['p2_best_test']['alpha_pct']:+.2f}% "
              f"({test_alpha - BASELINE['p2_best_test']['alpha_pct']:+.2f}%)")

        all_ok = train_alpha > 0 and test_alpha > 0
        print(f"\n  双窗Alpha转正: {'✅ 是' if all_ok else '❌ 否'}")
        if not all_ok and (train_alpha > BASELINE['p2_best_train']['alpha_pct'] or
                           test_alpha > BASELINE['p2_best_test']['alpha_pct']):
            print(f"  虽未双转正, 但相比P2有改善 → 可考虑参数调优")
        elif not all_ok:
            print(f"  未改善, 消费组标的问题难以通过策略解决 → 维持暂停")

    finally:
        if pref_existed and os.path.exists(pref_backup):
            shutil.move(pref_backup, _USER_PREF_FILE)


if __name__ == "__main__":
    main()
