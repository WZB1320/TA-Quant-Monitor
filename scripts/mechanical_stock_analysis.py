"""
机械组7只股票特性分析 — 为策略选型提供数据支撑

机械组覆盖多个子行业:
  - AI算力: 浪潮信息(000977)、紫光股份(000938)
  - 汽车玻璃: 福耀玻璃(600660)
  - 金属制品: 泰嘉股份(002843)
  - 光伏: 隆基绿能(601012)
  - 电力设备: 东方电气(600875)、国电南瑞(600406)

分析维度:
  1. 波动率特性 (年化波动率/ATR占比/日内振幅)
  2. 趋势性 vs 均值回归性 (Hurst指数/自相关)
  3. 回撤与反弹特性 (最大回撤/反弹幅度/反弹周期)
  4. RSI超卖触发频率
  5. 量价关系 (OBV趋势/放量与涨幅关联)
  6. 股票间相关性 (板块内分化程度)
  7. 与沪深300相关性 (Alpha来源分析)

用法: python scripts/mechanical_stock_analysis.py
"""
import sys, os, warnings
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
project_root = os.path.join(os.path.dirname(__file__), "..")
sys.path.insert(0, project_root)

from src.data_fetcher import DataManager

MECHANICAL_STOCKS = [
    {"code": "000977", "name": "浪潮信息", "market": "sz", "sub": "AI算力"},
    {"code": "600660", "name": "福耀玻璃", "market": "sh", "sub": "汽车玻璃"},
    {"code": "002843", "name": "泰嘉股份", "market": "sz", "sub": "金属制品"},
    {"code": "601012", "name": "隆基绿能", "market": "sh", "sub": "光伏"},
    {"code": "600875", "name": "东方电气", "market": "sh", "sub": "电力设备"},
    {"code": "600406", "name": "国电南瑞", "market": "sh", "sub": "电力设备"},
    {"code": "000938", "name": "紫光股份", "market": "sz", "sub": "AI算力"},
]

WINDOWS = [
    ("训练窗(2024-07~2025-06 震荡市)", "2024-07-01", "2025-06-30"),
    ("测试窗(2025-07~2026-06 牛市)", "2025-07-01", "2026-06-30"),
]
DATA_START, DATA_END = "2024-02-01", "2026-07-13"
BENCHMARK = "sh.000300"


def calc_hurst(series, max_lag=100):
    series = np.log(series.dropna())
    lags = range(2, min(max_lag, len(series)//2))
    tau = []
    for lag in lags:
        diff = series.diff(lag).dropna()
        tau.append(np.sqrt(np.std(diff)))
    if len(tau) < 5:
        return 0.5
    poly = np.polyfit(np.log(list(lags)), np.log(tau), 1)
    return poly[0] * 2.0


def calc_autocorr(returns, lag=1):
    if len(returns) < 30:
        return 0
    return returns.autocorr(lag=lag)


def analyze_stock(df, name, sub, start, end):
    mask = (df["date"] >= start) & (df["date"] <= end)
    sub_df = df[mask].copy()
    if len(sub_df) < 30:
        return None

    close = sub_df["close"].astype(float)
    high = sub_df["high"].astype(float)
    low = sub_df["low"].astype(float)
    volume = sub_df["volume"].astype(float)

    daily_ret = close.pct_change().dropna()
    ann_vol = daily_ret.std() * np.sqrt(252)
    atr = (high - low).rolling(14).mean().mean()
    atr_ratio = atr / close.mean()
    intraday_range = ((high - low) / close).mean()

    hurst = calc_hurst(close)
    autocorr = calc_autocorr(daily_ret, lag=1)
    autocorr5 = calc_autocorr(daily_ret, lag=5)

    ma60 = close.rolling(60).mean()
    above_ma60 = (close > ma60).sum() / len(close) if len(ma60.dropna()) > 0 else 0

    cummax = close.cummax()
    drawdown = (close - cummax) / cummax
    max_dd = drawdown.min()
    dd_min_pos = drawdown.values.tolist().index(drawdown.min())
    remaining = close.iloc[dd_min_pos:]
    rebound = (remaining.iloc[-1] / remaining.iloc[0]) - 1 if len(remaining) > 5 else 0

    # RSI
    delta = close.diff()
    gain = delta.where(delta > 0, 0).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
    rs = gain / loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    rsi_oversold_30 = (rsi < 30).sum() / len(rsi.dropna()) if len(rsi.dropna()) > 0 else 0
    rsi_oversold_40 = (rsi < 40).sum() / len(rsi.dropna()) if len(rsi.dropna()) > 0 else 0
    rsi_overbought_70 = (rsi > 70).sum() / len(rsi.dropna()) if len(rsi.dropna()) > 0 else 0

    # 量价
    vol_ma = volume.rolling(20).mean()
    vol_ratio = volume / vol_ma
    high_vol_days = (vol_ratio > 1.5).sum() / len(vol_ratio.dropna()) if len(vol_ratio.dropna()) > 0 else 0
    vol_mask = vol_ratio > 1.5
    high_vol_ret = daily_ret[vol_mask.shift(1).fillna(False)].mean() if vol_mask.sum() > 0 else 0

    total_return = (close.iloc[-1] / close.iloc[0]) - 1
    sharpe = (daily_ret.mean() / daily_ret.std() * np.sqrt(252)) if daily_ret.std() > 0 else 0

    return {
        "name": name, "sub": sub, "days": len(sub_df),
        "total_return_pct": round(total_return * 100, 2),
        "ann_vol_pct": round(ann_vol * 100, 2),
        "sharpe": round(sharpe, 3),
        "atr_ratio_pct": round(atr_ratio * 100, 2),
        "intraday_range_pct": round(intraday_range * 100, 2),
        "hurst": round(hurst, 3),
        "autocorr_1d": round(autocorr, 3),
        "autocorr_5d": round(autocorr5, 3),
        "above_ma60_pct": round(above_ma60 * 100, 1),
        "max_dd_pct": round(max_dd * 100, 2),
        "rebound_pct": round(rebound * 100, 2),
        "rsi_oversold_30_pct": round(rsi_oversold_30 * 100, 1),
        "rsi_oversold_40_pct": round(rsi_oversold_40 * 100, 1),
        "rsi_overbought_70_pct": round(rsi_overbought_70 * 100, 1),
        "high_vol_days_pct": round(high_vol_days * 100, 1),
        "high_vol_ret_pct": round(high_vol_ret * 100, 2),
    }


def calc_correlation(stock_data, start, end):
    """计算股票间相关性矩阵"""
    returns = {}
    for code, df in stock_data.items():
        mask = (df["date"] >= start) & (df["date"] <= end)
        sub = df[mask].copy()
        if len(sub) > 30:
            returns[code] = sub["close"].astype(float).pct_change().dropna()

    if len(returns) < 2:
        return None

    ret_df = pd.DataFrame(returns)
    return ret_df.corr()


def main():
    print("=" * 110)
    print("  机械组7只股票特性分析 — 策略选型数据支撑")
    print("  覆盖子行业: AI算力(浪潮/紫光) | 汽车玻璃(福耀) | 金属制品(泰嘉) | 光伏(隆基) | 电力设备(东方/国电)")
    print("=" * 110)

    dm = DataManager()
    stock_data = {}
    name_map = {}
    for s in MECHANICAL_STOCKS:
        df = dm.get_daily_kline(s["code"], start_date=DATA_START, end_date=DATA_END)
        if df is not None and len(df) > 80:
            stock_data[s["code"]] = df
            name_map[s["code"]] = f"{s['name']}({s['sub']})"
            print(f"  {s['name']}({s['code']}/{s['sub']}): {len(df)}条")

    bench_df = dm.get_daily_kline(BENCHMARK, start_date=DATA_START, end_date=DATA_END)
    print(f"  沪深300: {len(bench_df)}条")

    # 股票间相关性
    print(f"\n{'='*110}")
    print(f"  股票间相关性矩阵 (全周期)")
    print(f"{'='*110}")
    corr_all = calc_correlation(stock_data, DATA_START, DATA_END)
    if corr_all is not None:
        code_to_name = {s["code"]: s["name"] for s in MECHANICAL_STOCKS}
        corr_all.index = [code_to_name.get(c, c) for c in corr_all.index]
        corr_all.columns = [code_to_name.get(c, c) for c in corr_all.columns]
        print(corr_all.round(2).to_string())
        avg_corr = (corr_all.values[np.triu_indices_from(corr_all.values, 1)].mean())
        print(f"\n  平均相关系数: {avg_corr:.3f}")
        print(f"  {'板块内高度相关(适合同板块策略)' if avg_corr > 0.6 else '板块内分化明显(适合多策略或选股)' if avg_corr < 0.4 else '中等相关'}")

    for window_name, start, end in WINDOWS:
        print(f"\n{'='*110}")
        print(f"  {window_name}")
        print(f"{'='*110}")

        bench_mask = (bench_df["date"] >= start) & (bench_df["date"] <= end)
        bench_sub = bench_df[bench_mask]
        bench_ret = (bench_sub["close"].iloc[-1] / bench_sub["close"].iloc[0]) - 1
        print(f"\n  沪深300基准收益: {bench_ret*100:+.2f}%")

        print(f"\n  {'股票':<10} {'子行业':<8} {'收益%':>8} {'年化波动':>8} {'夏普':>6} {'ATR占比':>8} {'Hurst':>7} {'自相关1d':>9}")
        print(f"  {'-'*78}")
        results = []
        for s in MECHANICAL_STOCKS:
            if s["code"] not in stock_data:
                continue
            r = analyze_stock(stock_data[s["code"]], s["name"], s["sub"], start, end)
            if r:
                results.append(r)
                print(f"  {r['name']:<10} {r['sub']:<8} {r['total_return_pct']:>+8.2f} {r['ann_vol_pct']:>7.2f}% {r['sharpe']:>6.3f} "
                      f"{r['atr_ratio_pct']:>7.2f}% {r['hurst']:>7.3f} {r['autocorr_1d']:>+9.3f}")

        print(f"\n  {'股票':<10} {'AboveMA60':>10} {'最大回撤':>8} {'反弹幅度':>8} {'RSI<30':>7} {'RSI<40':>7} {'RSI>70':>7} {'放量日':>7} {'放量涨跌':>8}")
        print(f"  {'-'*85}")
        for r in results:
            print(f"  {r['name']:<10} {r['above_ma60_pct']:>9.1f}% {r['max_dd_pct']:>+7.2f}% {r['rebound_pct']:>+7.2f}% "
                  f"{r['rsi_oversold_30_pct']:>6.1f}% {r['rsi_oversold_40_pct']:>6.1f}% {r['rsi_overbought_70_pct']:>6.1f}% "
                  f"{r['high_vol_days_pct']:>6.1f}% {r['high_vol_ret_pct']:>+7.2f}%")

        if results:
            avg_hurst = np.mean([r["hurst"] for r in results])
            avg_autocorr = np.mean([r["autocorr_1d"] for r in results])
            avg_rsi30 = np.mean([r["rsi_oversold_30_pct"] for r in results])
            avg_rsi40 = np.mean([r["rsi_oversold_40_pct"] for r in results])
            avg_vol = np.mean([r["ann_vol_pct"] for r in results])
            avg_dd = np.mean([r["max_dd_pct"] for r in results])
            avg_rebound = np.mean([r["rebound_pct"] for r in results])
            avg_highvol = np.mean([r["high_vol_days_pct"] for r in results])
            avg_alpha = np.mean([r["total_return_pct"] - bench_ret * 100 for r in results])
            # 分化度: 收益极差
            ret_range = max(r["total_return_pct"] for r in results) - min(r["total_return_pct"] for r in results)

            print(f"\n  ── 机械组平均特性 ──")
            print(f"  Hurst指数: {avg_hurst:.3f} ({'均值回归' if avg_hurst < 0.5 else '趋势性' if avg_hurst > 0.55 else '随机游走'})")
            print(f"  1日自相关: {avg_autocorr:+.3f} ({'均值回归(负)' if avg_autocorr < -0.05 else '趋势延续(正)' if avg_autocorr > 0.05 else '无明显规律'})")
            print(f"  RSI<30超卖频率: {avg_rsi30:.1f}%")
            print(f"  RSI<40超卖频率: {avg_rsi40:.1f}%")
            print(f"  年化波动率: {avg_vol:.2f}%")
            print(f"  最大回撤: {avg_dd:.2f}%")
            print(f"  低点反弹幅度: {avg_rebound:+.2f}%")
            print(f"  放量日占比: {avg_highvol:.1f}%")
            print(f"  平均Alpha: {avg_alpha:+.2f}%")
            print(f"  收益极差(分化度): {ret_range:.2f}%")

    # 策略选型建议
    print(f"\n{'='*110}")
    print(f"  策略选型数据支撑总结")
    print(f"{'='*110}")

    all_results = []
    for _, start, end in WINDOWS:
        for s in MECHANICAL_STOCKS:
            if s["code"] not in stock_data:
                continue
            r = analyze_stock(stock_data[s["code"]], s["name"], s["sub"], start, end)
            if r:
                all_results.append(r)

    if all_results:
        avg_hurst = np.mean([r["hurst"] for r in all_results])
        avg_autocorr = np.mean([r["autocorr_1d"] for r in all_results])
        avg_rsi30 = np.mean([r["rsi_oversold_30_pct"] for r in all_results])
        avg_rsi40 = np.mean([r["rsi_oversold_40_pct"] for r in all_results])
        avg_vol = np.mean([r["ann_vol_pct"] for r in all_results])
        avg_atr_ratio = np.mean([r["atr_ratio_pct"] for r in all_results])
        avg_intraday = np.mean([r["intraday_range_pct"] for r in all_results])
        avg_dd = np.mean([r["max_dd_pct"] for r in all_results])
        avg_rebound = np.mean([r["rebound_pct"] for r in all_results])
        avg_highvol = np.mean([r["high_vol_days_pct"] for r in all_results])

        print(f"\n  机械组7只股票两窗口综合特性:")
        print(f"  ├─ Hurst指数: {avg_hurst:.3f}")
        print(f"  ├─ 1日自相关: {avg_autocorr:+.3f}")
        print(f"  ├─ 年化波动率: {avg_vol:.2f}%")
        print(f"  ├─ ATR/价格占比: {avg_atr_ratio:.2f}%")
        print(f"  ├─ 日内振幅: {avg_intraday:.2f}%")
        print(f"  ├─ RSI<30超卖频率: {avg_rsi30:.1f}%")
        print(f"  ├─ RSI<40超卖频率: {avg_rsi40:.1f}%")
        print(f"  ├─ 最大回撤: {avg_dd:.2f}%")
        print(f"  ├─ 低点反弹幅度: {avg_rebound:+.2f}%")
        print(f"  └─ 放量日占比: {avg_highvol:.1f}%")

        print(f"\n  策略适配性评估:")
        print(f"  ├─ 趋势跟踪(MA60方向): {'适配' if avg_hurst > 0.5 else '不适配' if avg_hurst < 0.45 else '看数据'} (Hurst={avg_hurst:.3f})")
        print(f"  ├─ 均值回归(RSI超卖反弹): {'不适配' if avg_rsi30 < 5 else '勉强适配' if avg_rsi30 < 15 else '适配'} (RSI<30={avg_rsi30:.1f}%, 反弹={avg_rebound:+.1f}%)")
        print(f"  ├─ 网格交易(低波动高振幅): {'适配' if avg_vol < 25 and avg_intraday > 2 else '看数据'} (波动{avg_vol:.1f}%/振幅{avg_intraday:.1f}%)")
        print(f"  ├─ 事件驱动(放量突破): {'适配' if avg_highvol > 10 else '不适配'} (放量日{avg_highvol:.1f}%)")
        print(f"  └─ 动量轮动(子行业分化): {'板块分化适合轮动' if avg_corr < 0.5 else '板块同质化'} (平均相关{avg_corr:.3f})")

        # 分子行业分析
        print(f"\n  分子行业表现对比:")
        sub_groups = {}
        for r in all_results:
            sub = r["sub"]
            if sub not in sub_groups:
                sub_groups[sub] = []
            sub_groups[sub].append(r)

        for sub, rs in sub_groups.items():
            avg_ret = np.mean([r["total_return_pct"] for r in rs])
            avg_sharpe = np.mean([r["sharpe"] for r in rs])
            avg_h = np.mean([r["hurst"] for r in rs])
            print(f"  ├─ {sub}: 收益{avg_ret:+.2f}% 夏普{avg_sharpe:.3f} Hurst{avg_h:.3f}")


if __name__ == "__main__":
    main()
