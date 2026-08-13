"""
诊断2 — 训练窗交易盈亏来源分析 (止损踏空 vs 信号方向错)

诊断1确认策略跑输等权持有16%, 但损耗来源未明. 两种机制开方不同:
  机制甲(止损踏空): 震荡市频繁假突破止损, 卖飞后不追回 → 优化止损能双赢(改善Alpha+不放大回撤)
  机制乙(信号方向错): 趋势信号在震荡市方向反了 → 需改信号逻辑

诊断方法 (训练窗, P2配置):
  1. 平仓原因分布: 止损率/止盈率/其他
  2. 止损后踏空率: 止损平仓后20天股价是否反弹超过exit_price (卖飞了)
  3. 信号方向准确率: 买入后盈亏 (pnl>0占比)
  4. 止盈交易是否赚够: 止盈时盈利幅度 + 止盈后是否继续涨(过早止盈)

判断:
  - 止损率高 + 止损后踏空率高 → 机制甲, 优化止损
  - 止损率不高但胜率低 → 机制乙, 改信号
  - 止盈后继续大涨 → 过早止盈, 放宽止盈

用法: python scripts/stop_loss_diagnosis.py
"""
import sys
import os
import json
import shutil
import warnings
import numpy as np
import pandas as pd
from datetime import datetime
from collections import Counter

warnings.filterwarnings("ignore")
project_root = os.path.join(os.path.dirname(__file__), "..")
sys.path.insert(0, project_root)

from src.config.runtime_mode import set_mode, RuntimeMode
set_mode(RuntimeMode.BACKTEST)

from src.backtest.engine import BacktestEngine
from src.backtest.regime_detector import RegimeDetector
from src.data_fetcher import DataManager
from src.config.group_config import GroupConfig
from src.config.user_preferences import UserPreferences, _USER_PREF_FILE

TRAIN_START, TRAIN_END = "2024-07-01", "2025-06-30"
DATA_START, DATA_END = "2024-02-01", "2026-07-13"
BENCHMARK = "sh.000300"
TOTAL_CAPITAL = 1000000

WEIGHTS = {"科技成长型": 0.40, "消费稳健型": 0.10, "周期资源型": 0.425,
           "医药创新型": 0.0, "机械制造型": 0.0}
REGIMES_CFG = {"周期资源型": {"trending"}}
ATR_OVERRIDE = {"科技成长型": 1.8}

REPORT_MD = os.path.join(project_root, "data", "stop_loss_diagnosis_report.md")


def load_watchlist():
    cfg_path = os.path.join(project_root, "config", "strategy_config.json")
    with open(cfg_path, "r", encoding="utf-8") as f:
        cfg = json.load(f)
    return {g: [s["code"] for s in stocks]
            for g, stocks in cfg["strategy_config"]["watchlist"].items()
            if not g.startswith("_")}


def classify_exit(exit_signal):
    """平仓原因分类: 止损/止盈/其他."""
    sig = exit_signal or ""
    if "硬止损" in sig:
        return "止损"
    if "移动止盈" in sig:
        return "止盈"
    return "其他"


def compute_regime_series(benchmark_df):
    detector = RegimeDetector()
    df = benchmark_df.copy()
    if "date" not in df.columns:
        df = df.reset_index()
    df["date"] = pd.to_datetime(df["date"]).dt.date
    return {d: detector.detect(benchmark_df, d) for d in df["date"].tolist()}


def get_price_after(df, exit_date, n_days):
    """获取 exit_date 后 n_days 个交易日的最高价/收盘价序列.
    返回 (max_close_after, prices_after_list)."""
    if df is None or exit_date is None:
        return None, []
    d = df.copy()
    if "date" not in d.columns:
        d = d.reset_index()
    d["date"] = pd.to_datetime(d["date"]).dt.date
    exit_d = exit_date if not isinstance(exit_date, pd.Timestamp) else exit_date.date()
    after = d[d["date"] > exit_d].head(n_days)
    if len(after) == 0:
        return None, []
    prices = after["close"].astype(float).tolist()
    return max(prices), prices


def run_group_with_trades(data_map, benchmark_df, group_codes, group_capital,
                          trade_regimes, atr_mult, start, end):
    """跑回测, 返回 closed_trades."""
    if group_capital < 1000:
        return []
    GroupConfig._instance = None
    GroupConfig._config = None
    engine = BacktestEngine(
        initial_capital=group_capital, lookback_days=120, position_ratio=0.3,
        commission_rate=0.00025, stamp_tax=0.001, slippage=0.0001,
        signal_dedup_days=5, risk_per_trade=0.05, atr_stop_mult=atr_mult,
        forced_regime=None, trade_regimes=trade_regimes,
    )
    sub_map = {c: data_map[c] for c in group_codes if c in data_map}
    engine.run(sub_map, benchmark_df=benchmark_df, start_date=start, end_date=end)
    return engine.position_mgr.closed_trades


def analyze_trades(trades, data_map, regime_map):
    """分析每笔交易."""
    rows = []
    for t in trades:
        reason = classify_exit(t.exit_signal)
        # 踏空分析: 平仓后20天最高价
        max_after_20, prices_20 = get_price_after(data_map.get(t.symbol), t.exit_date, 20)
        max_after_10, prices_10 = get_price_after(data_map.get(t.symbol), t.exit_date, 10)
        exit_p = t.exit_price or 0
        # 踏空幅度 = (平仓后最高价 - 平仓价)/平仓价
        miss_20 = (max_after_20 - exit_p) / exit_p if (max_after_20 and exit_p) else None
        miss_10 = (max_after_10 - exit_p) / exit_p if (max_after_10 and exit_p) else None
        # 平仓日 regime
        exit_d = t.exit_date
        if isinstance(exit_d, pd.Timestamp):
            exit_d = exit_d.date()
        exit_regime = regime_map.get(exit_d, "unknown")
        # 买入后是否曾大涨 (最高价 vs 入场价)
        max_gain = (t.highest_price - t.entry_price) / t.entry_price if t.entry_price else 0

        rows.append({
            "symbol": t.symbol, "reason": reason, "pnl_pct": t.pnl_pct * 100,
            "holding_days": t.holding_days, "exit_regime": exit_regime,
            "miss_20_pct": miss_20 * 100 if miss_20 is not None else None,
            "miss_10_pct": miss_10 * 100 if miss_10 is not None else None,
            "max_gain_pct": max_gain * 100,
            "entry_price": t.entry_price, "exit_price": exit_p,
            "exit_signal": t.exit_signal,
        })
    return rows


def main():
    run_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print("=" * 70)
    print("  诊断2 — 训练窗交易盈亏来源分析")
    print(f"  训练窗(震荡市): {TRAIN_START}~{TRAIN_END}")
    print(f"  区分: 机制甲(止损踏空) vs 机制乙(信号方向错)")
    print("=" * 70)

    pref_existed = os.path.exists(_USER_PREF_FILE)
    pref_backup = _USER_PREF_FILE + ".sl_bak"
    if pref_existed:
        shutil.copy2(_USER_PREF_FILE, pref_backup)
    UserPreferences().clear_all()

    try:
        watchlist = load_watchlist()
        dm = DataManager()
        print("\n拉取数据...")
        all_codes = [c for codes in watchlist.values() for c in codes]
        data_map = {}
        for code in all_codes:
            df = dm.get_daily_kline(code, start_date=DATA_START, end_date=DATA_END)
            if df is not None and len(df) > 80:
                data_map[code] = df
        benchmark_df = dm.get_daily_kline(BENCHMARK, start_date=DATA_START, end_date=DATA_END)
        print(f"  股票 {len(data_map)}/{len(all_codes)}, 基准 {len(benchmark_df)}条")

        regime_map = compute_regime_series(benchmark_df)

        # 跑各组回测, 收集交易
        all_rows = []
        group_stats = []
        for g, codes in watchlist.items():
            if g not in WEIGHTS or WEIGHTS[g] == 0:
                continue
            g_codes = [c for c in codes if c in data_map]
            if len(g_codes) < 2:
                continue
            capital = TOTAL_CAPITAL * WEIGHTS[g]
            regimes = REGIMES_CFG.get(g)
            atr_mult = ATR_OVERRIDE.get(g, 2.0)
            trades = run_group_with_trades(data_map, benchmark_df, g_codes, capital,
                                           regimes, atr_mult, TRAIN_START, TRAIN_END)
            rows = analyze_trades(trades, data_map, regime_map)
            for r in rows:
                r["group"] = g
            all_rows.extend(rows)
            print(f"  {g:12s}: {len(trades)}笔交易")
            group_stats.append({"group": g, "trade_count": len(trades), "rows": rows})

        if not all_rows:
            print("\n无交易数据, 退出.")
            return

        df = pd.DataFrame(all_rows)
        print(f"\n总交易数: {len(df)}")

        # ── 1. 平仓原因分布 ──
        print("\n" + "=" * 70)
        print("  1. 平仓原因分布")
        print("=" * 70)
        reason_cnt = df["reason"].value_counts()
        reason_rate = reason_cnt / len(df) * 100
        for r in ["止损", "止盈", "其他"]:
            if r in reason_cnt:
                print(f"  {r}: {reason_cnt[r]}笔 ({reason_rate[r]:.1f}%)")

        # ── 2. 止损 vs 止盈 盈亏对比 ──
        print("\n" + "=" * 70)
        print("  2. 止损 vs 止盈 盈亏对比")
        print("=" * 70)
        for r in ["止损", "止盈", "其他"]:
            sub = df[df["reason"] == r]
            if len(sub) == 0:
                continue
            win = (sub["pnl_pct"] > 0).sum()
            print(f"  {r}: {len(sub)}笔, 胜率{win/len(sub)*100:.1f}%, "
                  f"平均pnl{sub['pnl_pct'].mean():+.2f}%, "
                  f"持仓{sub['holding_days'].mean():.0f}天")

        # ── 3. 止损后踏空分析 (核心) ──
        print("\n" + "=" * 70)
        print("  3. 止损后踏空分析 (机制甲验证)")
        print("=" * 70)
        stop_trades = df[df["reason"] == "止损"]
        if len(stop_trades) > 0:
            miss20 = stop_trades["miss_20_pct"].dropna()
            miss10 = stop_trades["miss_10_pct"].dropna()
            # 踏空: 平仓后股价反弹>0
            miss20_pos = (miss20 > 0).sum()
            miss10_pos = (miss10 > 0).sum()
            print(f"  止损笔数: {len(stop_trades)}")
            print(f"  止损后10天股价反弹率: {miss10_pos}/{len(miss10)} ({miss10_pos/len(miss10)*100:.1f}%)")
            print(f"  止损后10天平均涨幅: {miss10.mean():+.2f}% (相对平仓价)")
            print(f"  止损后20天股价反弹率: {miss20_pos}/{len(miss20)} ({miss20_pos/len(miss20)*100:.1f}%)")
            print(f"  止损后20天平均涨幅: {miss20.mean():+.2f}% (相对平仓价)")
            # 严重踏空: 反弹>10%
            severe = (miss20 > 10).sum()
            print(f"  严重踏空(20天反弹>10%): {severe}/{len(miss20)} ({severe/len(miss20)*100:.1f}%)")

        # ── 4. 止盈后是否过早 (止盈后继续大涨) ──
        print("\n" + "=" * 70)
        print("  4. 止盈后是否过早 (止盈后继续涨)")
        print("=" * 70)
        tp_trades = df[df["reason"] == "止盈"]
        if len(tp_trades) > 0:
            miss20_tp = tp_trades["miss_20_pct"].dropna()
            tp_pos = (miss20_tp > 0).sum()
            print(f"  止盈笔数: {len(tp_trades)}")
            print(f"  止盈时平均盈利: {tp_trades['pnl_pct'].mean():+.2f}%")
            print(f"  止盈后20天继续涨率: {tp_pos}/{len(miss20_tp)} ({tp_pos/len(miss20_tp)*100:.1f}%)")
            print(f"  止盈后20天平均涨幅: {miss20_tp.mean():+.2f}% (相对平仓价)")
            severe_tp = (miss20_tp > 10).sum()
            print(f"  过早止盈(20天后继续涨>10%): {severe_tp}/{len(miss20_tp)} ({severe_tp/len(miss20_tp)*100:.1f}%)")

        # ── 5. 信号方向准确率 (机制乙验证) ──
        print("\n" + "=" * 70)
        print("  5. 信号方向准确率 (机制乙验证)")
        print("=" * 70)
        # 买入后曾大涨的交易占比 (最高价涨幅>10%)
        big_gain = (df["max_gain_pct"] > 10).sum()
        print(f"  买入后曾大涨(最高涨>10%)的交易: {big_gain}/{len(df)} ({big_gain/len(df)*100:.1f}%)")
        print(f"  买入后平均最高涨幅: {df['max_gain_pct'].mean():+.2f}%")
        # 但最终盈利的交易
        win_all = (df["pnl_pct"] > 0).sum()
        print(f"  最终盈利交易: {win_all}/{len(df)} ({win_all/len(df)*100:.1f}%)")
        # 买入后大涨但最终亏损(被止损打出来) → 典型踏空
        if len(df) > 0:
            gain_but_loss = df[(df["max_gain_pct"] > 10) & (df["pnl_pct"] < 0)]
            print(f"  曾大涨>10%但最终亏损(被止损打飞): {len(gain_but_loss)}/{len(df)} ({len(gain_but_loss)/len(df)*100:.1f}%)")

        # ── 6. 按 regime 分 (震荡市 vs 趋势市) ──
        print("\n" + "=" * 70)
        print("  6. 按平仓日 regime 分 (震荡市 vs 趋势市)")
        print("=" * 70)
        for reg in ["trending", "transition", "ranging"]:
            sub = df[df["exit_regime"] == reg]
            if len(sub) == 0:
                continue
            stop_rate = (sub["reason"] == "止损").sum() / len(sub) * 100
            win_rate = (sub["pnl_pct"] > 0).sum() / len(sub) * 100
            print(f"  {reg:12s}: {len(sub)}笔, 止损率{stop_rate:.1f}%, 胜率{win_rate:.1f}%, "
                  f"平均pnl{sub['pnl_pct'].mean():+.2f}%")

        # ── 7. 分组明细 ──
        print("\n" + "=" * 70)
        print("  7. 分组明细")
        print("=" * 70)
        for gs in group_stats:
            g = gs["group"]
            gr = pd.DataFrame(gs["rows"])
            if len(gr) == 0:
                continue
            stop_r = (gr["reason"] == "止损").sum() / len(gr) * 100
            stop_sub = gr[gr["reason"] == "止损"]
            miss20_g = stop_sub["miss_20_pct"].dropna()
            miss_rate = (miss20_g > 0).sum() / len(miss20_g) * 100 if len(miss20_g) > 0 else 0
            print(f"  {g:12s}: {len(gr)}笔, 止损率{stop_r:.1f}%, "
                  f"止损后20天踏空率{miss_rate:.1f}%, "
                  f"平均pnl{gr['pnl_pct'].mean():+.2f}%")

        # ── 结论 ──
        print("\n" + "=" * 70)
        print("  诊断结论")
        print("=" * 70)
        stop_rate_all = (df["reason"] == "止损").sum() / len(df) * 100
        if len(stop_trades) > 0:
            miss20_all = stop_trades["miss_20_pct"].dropna()
            miss_rate_all = (miss20_all > 0).sum() / len(miss20_all) * 100 if len(miss20_all) > 0 else 0
            miss_mean = miss20_all.mean() if len(miss20_all) > 0 else 0
        else:
            miss_rate_all = 0
            miss_mean = 0

        print(f"  止损率: {stop_rate_all:.1f}%")
        print(f"  止损后20天踏空率: {miss_rate_all:.1f}%")
        print(f"  止损后20天平均反弹: {miss_mean:+.2f}%")

        if stop_rate_all > 40 and miss_rate_all > 60:
            verdict = "机制甲(止损踏空)"
            print(f"\n  → 根因: 【机制甲·止损踏空】止损率高({stop_rate_all:.1f}%)且止损后股价反弹率高({miss_rate_all:.1f}%)")
            print("    震荡市频繁假突破止损, 卖飞后不追回, 踏空涨幅.")
            print("    建议: 震荡市放宽止损(降低止损触发)或减少交易, 可同时改善Alpha和回撤(双赢).")
        elif win_all / len(df) < 0.4:
            verdict = "机制乙(信号方向错)"
            print(f"\n  → 根因: 【机制乙·信号方向错】胜率低({win_all/len(df)*100:.1f}%), 信号方向在震荡市失效.")
            print("    建议: 震荡市改用反转/均值回归信号.")
        else:
            verdict = "混合"
            print(f"\n  → 根因: 【混合】止损率{stop_rate_all:.1f}%, 踏空率{miss_rate_all:.1f}%, 需组合优化.")

        # 报告
        report = generate_report(run_time, df, group_stats, verdict,
                                  stop_rate_all, miss_rate_all, miss_mean)
        with open(REPORT_MD, "w", encoding="utf-8") as f:
            f.write(report)
        print(f"\n✓ 报告 → {REPORT_MD}")

    finally:
        if pref_existed and os.path.exists(pref_backup):
            shutil.move(pref_backup, _USER_PREF_FILE)


def generate_report(run_time, df, group_stats, verdict, stop_rate, miss_rate, miss_mean):
    L = []
    L.append("# 诊断2 — 训练窗交易盈亏来源分析报告")
    L.append(f"\n**运行时间**: {run_time}")
    L.append(f"**窗口**: 训练窗(震荡市) {TRAIN_START}~{TRAIN_END}")
    L.append(f"**配置**: P2 (科技40/消费10/周期42.5, 周期trending过滤, 科技ATR1.8)\n")
    L.append("## 诊断目标\n")
    L.append("区分两种损耗机制:\n")
    L.append("- **机制甲(止损踏空)**: 震荡市频繁假突破止损, 卖飞后不追回 → 优化止损能双赢")
    L.append("- **机制乙(信号方向错)**: 趋势信号在震荡市方向反了 → 需改信号\n")

    L.append("## 1. 平仓原因分布\n")
    L.append("| 原因 | 笔数 | 占比% |")
    L.append("|------|------|-------|")
    reason_cnt = df["reason"].value_counts()
    for r in ["止损", "止盈", "其他"]:
        if r in reason_cnt:
            L.append(f"| {r} | {reason_cnt[r]} | {reason_cnt[r]/len(df)*100:.1f} |")
    L.append("")

    L.append("## 2. 止损 vs 止盈 盈亏对比\n")
    L.append("| 原因 | 笔数 | 胜率% | 平均pnl% | 平均持仓天 |")
    L.append("|------|------|-------|---------|-----------|")
    for r in ["止损", "止盈", "其他"]:
        sub = df[df["reason"] == r]
        if len(sub) == 0:
            continue
        win = (sub["pnl_pct"] > 0).sum()
        L.append(f"| {r} | {len(sub)} | {win/len(sub)*100:.1f} | {sub['pnl_pct'].mean():+.2f} | {sub['holding_days'].mean():.0f} |")
    L.append("")

    L.append("## 3. 止损后踏空分析 (机制甲核心证据)\n")
    stop_trades = df[df["reason"] == "止损"]
    if len(stop_trades) > 0:
        miss10 = stop_trades["miss_10_pct"].dropna()
        miss20 = stop_trades["miss_20_pct"].dropna()
        L.append("| 指标 | 值 |")
        L.append("|------|-----|")
        L.append(f"| 止损笔数 | {len(stop_trades)} |")
        L.append(f"| 止损后10天反弹率 | {(miss10>0).sum()}/{len(miss10)} ({(miss10>0).sum()/len(miss10)*100:.1f}%) |")
        L.append(f"| 止损后10天平均涨幅 | {miss10.mean():+.2f}% |")
        L.append(f"| 止损后20天反弹率 | {(miss20>0).sum()}/{len(miss20)} ({(miss20>0).sum()/len(miss20)*100:.1f}%) |")
        L.append(f"| 止损后20天平均涨幅 | {miss20.mean():+.2f}% |")
        L.append(f"| 严重踏空(20天反弹>10%) | {(miss20>10).sum()}/{len(miss20)} ({(miss20>10).sum()/len(miss20)*100:.1f}%) |")
    L.append("")

    L.append("## 4. 止盈后是否过早\n")
    tp = df[df["reason"] == "止盈"]
    if len(tp) > 0:
        miss20_tp = tp["miss_20_pct"].dropna()
        L.append("| 指标 | 值 |")
        L.append("|------|-----|")
        L.append(f"| 止盈笔数 | {len(tp)} |")
        L.append(f"| 止盈时平均盈利 | {tp['pnl_pct'].mean():+.2f}% |")
        L.append(f"| 止盈后20天继续涨率 | {(miss20_tp>0).sum()}/{len(miss20_tp)} ({(miss20_tp>0).sum()/len(miss20_tp)*100:.1f}%) |")
        L.append(f"| 止盈后20天平均涨幅 | {miss20_tp.mean():+.2f}% |")
        L.append(f"| 过早止盈(20天继续涨>10%) | {(miss20_tp>10).sum()}/{len(miss20_tp)} ({(miss20_tp>10).sum()/len(miss20_tp)*100:.1f}%) |")
    L.append("")

    L.append("## 5. 信号方向准确率 (机制乙验证)\n")
    big_gain = (df["max_gain_pct"] > 10).sum()
    win_all = (df["pnl_pct"] > 0).sum()
    gain_but_loss = df[(df["max_gain_pct"] > 10) & (df["pnl_pct"] < 0)]
    L.append("| 指标 | 值 |")
    L.append("|------|-----|")
    L.append(f"| 买入后曾大涨(>10%)占比 | {big_gain}/{len(df)} ({big_gain/len(df)*100:.1f}%) |")
    L.append(f"| 买入后平均最高涨幅 | {df['max_gain_pct'].mean():+.2f}% |")
    L.append(f"| 最终盈利占比 | {win_all}/{len(df)} ({win_all/len(df)*100:.1f}%) |")
    L.append(f"| 曾大涨>10%但最终亏损(被打飞) | {len(gain_but_loss)}/{len(df)} ({len(gain_but_loss)/len(df)*100:.1f}%) |")
    L.append("")

    L.append("## 6. 按平仓日 regime 分\n")
    L.append("| regime | 笔数 | 止损率% | 胜率% | 平均pnl% |")
    L.append("|--------|------|--------|-------|---------|")
    for reg in ["trending", "transition", "ranging"]:
        sub = df[df["exit_regime"] == reg]
        if len(sub) == 0:
            continue
        stop_r = (sub["reason"] == "止损").sum() / len(sub) * 100
        win_r = (sub["pnl_pct"] > 0).sum() / len(sub) * 100
        L.append(f"| {reg} | {len(sub)} | {stop_r:.1f} | {win_r:.1f} | {sub['pnl_pct'].mean():+.2f} |")
    L.append("")

    L.append("## 7. 分组明细\n")
    L.append("| 分组 | 笔数 | 止损率% | 止损后20天踏空率% | 平均pnl% |")
    L.append("|------|------|--------|------------------|---------|")
    for gs in group_stats:
        gr = pd.DataFrame(gs["rows"])
        if len(gr) == 0:
            continue
        stop_r = (gr["reason"] == "止损").sum() / len(gr) * 100
        stop_sub = gr[gr["reason"] == "止损"]
        miss20_g = stop_sub["miss_20_pct"].dropna()
        miss_rate = (miss20_g > 0).sum() / len(miss20_g) * 100 if len(miss20_g) > 0 else 0
        L.append(f"| {gs['group']} | {len(gr)} | {stop_r:.1f} | {miss_rate:.1f} | {gr['pnl_pct'].mean():+.2f} |")
    L.append("")

    L.append("## 诊断结论\n")
    L.append(f"**根因判断**: {verdict}\n")
    L.append(f"- 止损率: {stop_rate:.1f}%")
    L.append(f"- 止损后20天踏空率: {miss_rate:.1f}%")
    L.append(f"- 止损后20天平均反弹: {miss_mean:+.2f}%\n")
    if "甲" in verdict:
        L.append("**结论**: 震荡市频繁假突破止损+踏空, 是Alpha损耗主因. "
                 "优化止损(震荡市放宽止损/减少触发)可同时改善Alpha和回撤, 是双赢方向.")
    elif "乙" in verdict:
        L.append("**结论**: 信号方向在震荡市失效, 需改用反转/均值回归信号.")
    else:
        L.append("**结论**: 混合根因, 需组合优化.")
    return "\n".join(L)


if __name__ == "__main__":
    main()
