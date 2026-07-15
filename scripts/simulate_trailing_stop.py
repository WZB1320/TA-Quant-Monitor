"""
京东方A(000725) 移动止盈历史触发模拟

读取真实K线数据，对历史上每次买入信号模拟移动止盈逻辑，
统计触发次数、触发价格、盈利情况等。
"""
import sys
import os
import json
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

project_root = os.path.join(os.path.dirname(__file__), "..")
sys.path.insert(0, project_root)

from src.data_fetcher import DataManager


def calc_atr(high, low, close, period=14):
    """计算 ATR (Wilder's smoothing) - 与项目 strength.py 一致"""
    n = len(close)
    tr = np.zeros(n)
    for i in range(1, n):
        hl = high[i] - low[i]
        hc = abs(high[i] - close[i - 1])
        lc = abs(low[i] - close[i - 1])
        tr[i] = max(hl, hc, lc)

    atr = np.zeros(n)
    atr[period] = np.mean(tr[1:period + 1])
    for i in range(period + 1, n):
        atr[i] = (atr[i - 1] * (period - 1) + tr[i]) / period
    return atr


def simulate_trailing_stop(df, entry_idx, entry_price, atr_at_entry,
                           atr_stop_mult=2.0, tier_mult_factors=None):
    """
    模拟移动止盈逻辑 (与 position.py check_stop_loss 一致)

    Args:
        tier_mult_factors: 可选，自定义各档位倍率系数
            默认 {"t1": 1.0, "t2": 0.8, "t3": 0.6}

    Returns:
        dict with exit_date, exit_price, exit_reason, highest_price,
               holding_days, pnl_pct, max_profit, trigger_tier
    """
    if tier_mult_factors is None:
        tier_mult_factors = {"t1": 1.0, "t2": 0.8, "t3": 0.6}
    highest_price = entry_price
    entry_date = df.iloc[entry_idx]["date"]

    for i in range(entry_idx, len(df)):
        row = df.iloc[i]
        current_price = float(row["close"])
        current_date = row["date"]

        # 更新最高价
        if current_price > highest_price:
            highest_price = current_price

        # 安全网硬止损: -10%
        if current_price <= entry_price * 0.90:
            pnl_pct = (current_price - entry_price) / entry_price
            return {
                "exit_date": current_date,
                "exit_price": round(current_price, 2),
                "exit_reason": "安全网硬止损",
                "highest_price": round(highest_price, 2),
                "holding_days": i - entry_idx,
                "pnl_pct": round(pnl_pct * 100, 1),
                "max_profit": round((highest_price - entry_price) / entry_price * 100, 1),
                "trigger_tier": "安全网",
            }

        profit_pct = (current_price - entry_price) / entry_price

        # ATR 硬止损
        stop_dist = atr_at_entry * atr_stop_mult
        if current_price <= entry_price - stop_dist:
            pnl_pct = (current_price - entry_price) / entry_price
            return {
                "exit_date": current_date,
                "exit_price": round(current_price, 2),
                "exit_reason": "ATR硬止损",
                "highest_price": round(highest_price, 2),
                "holding_days": i - entry_idx,
                "pnl_pct": round(pnl_pct * 100, 1),
                "max_profit": round((highest_price - entry_price) / entry_price * 100, 1),
                "trigger_tier": "安全网",
            }

        # 盈利自适应移动止盈倍率
        if profit_pct > 0.20:
            trailing_mult = atr_stop_mult * tier_mult_factors["t3"]
            tier = "T3 (>20%)"
        elif profit_pct > 0.10:
            trailing_mult = atr_stop_mult * tier_mult_factors["t2"]
            tier = "T2 (10~20%)"
        else:
            trailing_mult = atr_stop_mult * tier_mult_factors["t1"]
            tier = "T1 (<10%)"

        trailing_dist = atr_at_entry * trailing_mult

        # 移动止盈触发
        if highest_price > entry_price:
            if current_price <= highest_price - trailing_dist:
                drawdown = (current_price - highest_price) / highest_price
                return {
                    "exit_date": current_date,
                    "exit_price": round(current_price, 2),
                    "exit_reason": f"ATR移动止盈 (回撤{drawdown*100:.1f}%)",
                    "highest_price": round(highest_price, 2),
                    "holding_days": i - entry_idx,
                    "pnl_pct": round(profit_pct * 100, 1),
                    "max_profit": round((highest_price - entry_price) / entry_price * 100, 1),
                    "trigger_tier": tier,
                }

    # 到达数据末尾仍未触发
    last_row = df.iloc[-1]
    profit_pct = (float(last_row["close"]) - entry_price) / entry_price
    return {
        "exit_date": "未触发",
        "exit_price": round(float(last_row["close"]), 2),
        "exit_reason": "持仓中(未触发止盈)",
        "highest_price": round(highest_price, 2),
        "holding_days": len(df) - 1 - entry_idx,
        "pnl_pct": round(profit_pct * 100, 1),
        "max_profit": round((highest_price - entry_price) / entry_price * 100, 1),
        "trigger_tier": "未触发",
    }


def load_buy_signals():
    """从回测记忆文件加载 000725 的买入信号"""
    memory_dir = os.path.join(os.path.dirname(__file__), "..", "data", "backtest_memory")
    signals = []

    if not os.path.exists(memory_dir):
        return signals

    for fname in os.listdir(memory_dir):
        if not fname.endswith(".jsonl"):
            continue
        fpath = os.path.join(memory_dir, fname)
        with open(fpath, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    rec = json.loads(line.strip())
                    if (rec.get("symbol") == "000725"
                            and rec.get("record_type") == "signal"
                            and rec.get("level") in ("强买入", "买入")
                            and rec.get("executable", False)):
                        signals.append({
                            "date": rec["analysis_date"],
                            "level": rec["level"],
                            "score": rec["score"],
                            "price": rec.get("price_at_signal", 0),
                            "run_id": rec.get("run_id", ""),
                        })
                except json.JSONDecodeError:
                    continue

    # 去重 (同一日期多个run可能重复)
    seen = set()
    unique = []
    for s in signals:
        if s["date"] not in seen:
            seen.add(s["date"])
            unique.append(s)
    unique.sort(key=lambda x: x["date"])
    return unique


def run_simulation(df, buy_signals, tier_mult_factors, label, june_scenario=True):
    """跑一轮模拟并输出结果"""

    # 4. 逐个模拟
    print("\n" + "=" * 90)
    print(f"  {label}")
    print(f"  T1={tier_mult_factors['t1']}×  T2={tier_mult_factors['t2']}×  T3={tier_mult_factors['t3']}×")
    print("=" * 90)

    results = []
    for sig in buy_signals:
        sig_date = sig["date"]

        entry_idx = None
        for i, row in df.iterrows():
            if row["date"] > sig_date:
                entry_idx = i
                break

        if entry_idx is None or entry_idx >= len(df):
            continue

        entry_price = float(df.iloc[entry_idx]["open"])
        atr_idx = entry_idx - 1 if entry_idx > 0 else 0
        atr_at_entry = float(df.iloc[atr_idx]["atr"])
        if atr_at_entry <= 0 or np.isnan(atr_at_entry):
            atr_at_entry = 0.15

        result = simulate_trailing_stop(
            df, entry_idx, entry_price, atr_at_entry,
            atr_stop_mult=2.0,
            tier_mult_factors=tier_mult_factors,
        )

        result["entry_date"] = df.iloc[entry_idx]["date"]
        result["entry_price"] = round(entry_price, 2)
        result["signal_date"] = sig_date
        result["signal_level"] = sig["level"]
        result["atr_at_entry"] = round(atr_at_entry, 3)
        results.append(result)

        print(f"\n  ┌─ 信号日: {sig_date} ({sig['level']}, 得分{sig['score']:+.1f})")
        print(f"  │  入场日: {result['entry_date']} | 入场价: {result['entry_price']} | ATR: {result['atr_at_entry']}")
        print(f"  │  最高价: {result['highest_price']} | 最高盈利: +{result['max_profit']}%")
        print(f"  │  退出日: {result['exit_date']} | 退出价: {result['exit_price']}")
        print(f"  │  持仓天数: {result['holding_days']}天 | 最终盈利: {result['pnl_pct']:+.1f}%")
        print(f"  └─ 退出原因: {result['exit_reason']} | 触发档位: {result['trigger_tier']}")

    # 假设6月初以5.0元买入
    if june_scenario:
        print("\n  ── 假设场景: 6月初以5.0元买入 ──")

        june_idx = None
        for i, row in df.iterrows():
            if row["date"] >= "2026-06-01":
                june_idx = i
                break

        if june_idx is not None:
            atr_june = float(df.iloc[june_idx - 1]["atr"]) if june_idx > 0 else 0.15
            if atr_june <= 0 or np.isnan(atr_june):
                atr_june = 0.40

            june_result = simulate_trailing_stop(
                df, june_idx, 5.0, atr_june,
                atr_stop_mult=2.0,
                tier_mult_factors=tier_mult_factors,
            )
            print(f"  入场日: {df.iloc[june_idx]['date']} | 入场价: 5.00 | ATR: {atr_june:.3f}")
            print(f"  最高价: {june_result['highest_price']} | 最高盈利: +{june_result['max_profit']}%")
            print(f"  退出日: {june_result['exit_date']} | 退出价: {june_result['exit_price']}")
            print(f"  持仓天数: {june_result['holding_days']}天 | 最终盈利: {june_result['pnl_pct']:+.1f}%")
            print(f"  退出原因: {june_result['exit_reason']} | 触发档位: {june_result['trigger_tier']}")

    # 汇总统计
    print("\n  ── 汇总统计 ──")

    trigger_count = sum(1 for r in results if "移动止盈" in r["exit_reason"])
    hard_stop_count = sum(1 for r in results if "硬止损" in r["exit_reason"])
    safety_count = sum(1 for r in results if "安全网" in r["exit_reason"])
    still_open = sum(1 for r in results if "未触发" in r["exit_reason"])

    print(f"\n  历史买入信号总数:  {len(results)}")
    print(f"  移动止盈触发:      {trigger_count} 次")
    print(f"  ATR硬止损触发:     {hard_stop_count} 次")
    print(f"  安全网硬止损:      {safety_count} 次")
    print(f"  未触发(持仓中):    {still_open} 次")

    if trigger_count > 0:
        avg_pnl = np.mean([r["pnl_pct"] for r in results if "移动止盈" in r["exit_reason"]])
        avg_hold = np.mean([r["holding_days"] for r in results if "移动止盈" in r["exit_reason"]])
        avg_max = np.mean([r["max_profit"] for r in results if "移动止盈" in r["exit_reason"]])
        print(f"\n  移动止盈平均盈利:  {avg_pnl:+.1f}%")
        print(f"  移动止盈平均最高盈利: +{avg_max:.1f}%")
        print(f"  移动止盈平均持仓:  {avg_hold:.0f} 天")

        for tier in ["T1 (<10%)", "T2 (10~20%)", "T3 (>20%)"]:
            tier_results = [r for r in results if r["trigger_tier"] == tier]
            if tier_results:
                print(f"\n  {tier} 触发 {len(tier_results)} 次:")
                for r in tier_results:
                    print(f"    {r['signal_date']} → {r['exit_date']} | 盈利 {r['pnl_pct']:+.1f}% | 最高+{r['max_profit']}%")

    # 全部交易统计
    all_pnl = [r["pnl_pct"] for r in results]
    win_count = sum(1 for p in all_pnl if p > 0)
    print(f"\n  全部交易胜率:      {win_count}/{len(all_pnl)} = {win_count/len(all_pnl)*100:.0f}%")
    print(f"  全部交易平均盈利:  {np.mean(all_pnl):+.1f}%")
    print(f"  全部交易总盈利:    {sum(all_pnl):+.1f}%")

    return results


def main():
    print("=" * 90)
    print("  京东方A(000725) 移动止盈历史触发模拟 — A/B 对比")
    print("=" * 90)

    # 1. 获取K线数据
    print("\n[1] 获取K线数据...")
    dm = DataManager()
    df = dm.get_daily_kline("000725", start_date="2025-06-01", end_date="2026-07-13")

    if df is None or df.empty:
        print("  ✗ 获取数据失败")
        return

    if "date" not in df.columns:
        df = df.rename(columns={"date": "date"})

    df = df.sort_values("date").reset_index(drop=True)
    print(f"  ✓ 获取 {len(df)} 条日线数据 ({df.iloc[0]['date']} ~ {df.iloc[-1]['date']})")

    # 2. 计算 ATR
    high = df["high"].values.astype(np.float64)
    low = df["low"].values.astype(np.float64)
    close = df["close"].values.astype(np.float64)
    atr_arr = calc_atr(high, low, close, period=14)
    df["atr"] = atr_arr

    # 3. 加载历史买入信号
    print("\n[2] 加载历史买入信号...")
    buy_signals = load_buy_signals()

    if not buy_signals:
        print("  ✗ 未找到 000725 的可执行买入信号")
        return

    print(f"  ✓ 找到 {len(buy_signals)} 个可执行买入信号:")
    for s in buy_signals:
        print(f"    {s['date']} | {s['level']} | 得分 {s['score']:+.1f} | 信号价 {s['price']:.2f}")

    # ── A组: 默认参数 (T1=1.0, T2=0.8, T3=0.6) ──
    default_factors = {"t1": 1.0, "t2": 0.8, "t3": 0.6}
    results_a = run_simulation(df, buy_signals, default_factors,
                               "A组: 默认参数 (T1=1.0× → 回撤~7%)")

    # ── B组: T1倍率翻倍 (T1=2.0, T2=0.8, T3=0.6) ──
    adjusted_factors = {"t1": 2.0, "t2": 0.8, "t3": 0.6}
    results_b = run_simulation(df, buy_signals, adjusted_factors,
                               "B组: T1档位放宽 (T1=2.0× → 回撤~15%)")

    # ── A/B 对比 ──
    print("\n" + "=" * 90)
    print("  A/B 对比总结")
    print("=" * 90)

    print(f"\n  {'信号日':<14} {'A组退出日':<14} {'A组盈利':>8} {'A组档位':<14} │ {'B组退出日':<14} {'B组盈利':>8} {'B组档位':<14} │ {'差异'}")
    print(f"  {'─'*14} {'─'*14} {'─'*8} {'─'*14} │ {'─'*14} {'─'*8} {'─'*14} │ {'─'*6}")

    for ra, rb in zip(results_a, results_b):
        diff = rb["pnl_pct"] - ra["pnl_pct"]
        diff_str = f"{diff:+.1f}%" if diff != 0 else "相同"
        print(f"  {ra['signal_date']:<14} {ra['exit_date']:<14} {ra['pnl_pct']:>+7.1f}% {ra['trigger_tier']:<14} │ {rb['exit_date']:<14} {rb['pnl_pct']:>+7.1f}% {rb['trigger_tier']:<14} │ {diff_str}")

    # 对比统计
    print(f"\n  {'指标':<22} {'A组(默认 T1=1.0×)':>20} {'B组(T1=2.0×)':>20} {'变化':>12}")
    print(f"  {'─'*22} {'─'*20} {'─'*20} {'─'*12}")

    a_trail = sum(1 for r in results_a if "移动止盈" in r["exit_reason"])
    b_trail = sum(1 for r in results_b if "移动止盈" in r["exit_reason"])
    a_hard = sum(1 for r in results_a if "硬止损" in r["exit_reason"])
    b_hard = sum(1 for r in results_b if "硬止损" in r["exit_reason"])
    a_pnl = sum(r["pnl_pct"] for r in results_a)
    b_pnl = sum(r["pnl_pct"] for r in results_b)
    a_avg = np.mean([r["pnl_pct"] for r in results_a])
    b_avg = np.mean([r["pnl_pct"] for r in results_b])
    a_hold = np.mean([r["holding_days"] for r in results_a])
    b_hold = np.mean([r["holding_days"] for r in results_b])
    a_win = sum(1 for r in results_a if r["pnl_pct"] > 0)
    b_win = sum(1 for r in results_b if r["pnl_pct"] > 0)

    print(f"  {'移动止盈触发':<20} {a_trail:>18}次 {b_trail:>18}次 {f'{b_trail-a_trail:+d}次':>12}")
    print(f"  {'硬止损触发':<20} {a_hard:>18}次 {b_hard:>18}次 {f'{b_hard-a_hard:+d}次':>12}")
    print(f"  {'胜率':<20} {f'{a_win}/{len(results_a)} = {a_win/len(results_a)*100:.0f}%':>20} {f'{b_win}/{len(results_b)} = {b_win/len(results_b)*100:.0f}%':>20} {'':>12}")
    print(f"  {'总盈利':<20} {a_pnl:>+19.1f}% {b_pnl:>+19.1f}% {f'{b_pnl-a_pnl:+.1f}%':>12}")
    print(f"  {'平均盈利':<20} {a_avg:>+19.1f}% {b_avg:>+19.1f}% {f'{b_avg-a_avg:+.1f}%':>12}")
    print(f"  {'平均持仓天数':<18} {a_hold:>18.0f}天 {b_hold:>18.0f}天 {f'{b_hold-a_hold:+.0f}天':>12}")

    print()


if __name__ == "__main__":
    main()
