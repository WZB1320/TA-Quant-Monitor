"""
因子 IC 分析 — 多指标多维度评估因子预测力

多维度分析:
  1. 多因子 (16个, 覆盖4个信息维度):
     - 趋势类 (4): MA60_dev, EMA_DUAL_diff, MACD_dif, ADX
     - 动量类 (4): RSI, KDJ_J, ROC_5, ROC_20
     - 量价类 (4): OBV_chg, VOL_ratio, vol_price_corr, vol_contraction
     - 波动率类 (4): ATR_pct, ATR_percentile, BB_width, hist_vol
  2. 多周期: 未来 1/3/5/10/20 日收益
  3. 多分组: 6 个行业分组分别分析
  4. 多 regime: 趋势(ADX>25) vs 震荡(ADX<20) 分别分析
  5. 多统计量: IC均值 / ICIR / IC胜率 / IC衰减曲线

样本: 面板IC (所有日期×所有股票 pool 在一起)
  - 样本量 = 股票数 × 交易日数 (远大于交易笔数, 统计上可信)
方法: Spearman 秩相关 (对异常值稳健, 不假设线性)

判断标准 (IC绝对值):
  |IC| > 0.05  → 有效因子
  |IC| 0.03~0.05 → 弱有效
  |IC| < 0.03  → 无效因子 (建议剔除)
  ICIR > 0.5   → 稳定有效
  ICIR > 1.0   → 强稳定

输出:
  data/factor_ic_result.json   (结构化结果)
  data/factor_ic_report.md     (可读报告)

用法:
  python scripts/factor_ic_analysis.py
"""
import sys
import os
import json
import warnings
from datetime import datetime

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

project_root = os.path.join(os.path.dirname(__file__), "..")
sys.path.insert(0, project_root)

from src.data_fetcher import DataManager

# ── 配置 ──
DATA_START = "2024-02-01"
DATA_END = "2026-07-13"
FORWARD_PERIODS = [1, 3, 5, 10, 20]
PRIMARY_PERIOD = 5  # 主分析周期

RESULT_JSON = os.path.join(project_root, "data", "factor_ic_result.json")
REPORT_MD = os.path.join(project_root, "data", "factor_ic_report.md")

# 因子分组 (信息维度)
FACTOR_DIMENSIONS = {
    "趋势类": ["MA60_dev", "EMA_DUAL_diff", "MACD_dif", "ADX"],
    "动量类": ["RSI", "KDJ_J", "ROC_5", "ROC_20"],
    "量价类": ["OBV_chg", "VOL_ratio", "vol_price_corr", "vol_contraction"],
    "波动率类": ["ATR_pct", "ATR_percentile", "BB_width", "hist_vol"],
}
ALL_FACTORS = [f for fs in FACTOR_DIMENSIONS.values() for f in fs]


# ============================================================
#  因子计算 (逐日连续序列) — 手写实现, 不依赖现有事件型指标
# ============================================================

def _ema(series, period):
    return series.ewm(span=period, adjust=False).mean()


def _rma(series, period):
    """Wilder smoothing (RSI/ATR 用)"""
    return series.ewm(alpha=1.0 / period, adjust=False).mean()


def compute_atr(high, low, close, period=14):
    tr = pd.concat([
        high - low,
        (high - close.shift(1)).abs(),
        (low - close.shift(1)).abs(),
    ], axis=1).max(axis=1)
    return _rma(tr, period)


def compute_rsi(close, period=14):
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = _rma(gain, period)
    avg_loss = _rma(loss, period)
    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100 - 100 / (1 + rs)
    return rsi.fillna(50) - 50  # 偏离中轴, 正=超买区, 负=超卖区


def compute_adx(high, low, close, period=14):
    up = high.diff()
    down = -low.diff()
    plus_dm = np.where((up > down) & (up > 0), up, 0.0)
    minus_dm = np.where((down > up) & (down > 0), down, 0.0)
    plus_dm = pd.Series(plus_dm, index=high.index)
    minus_dm = pd.Series(minus_dm, index=high.index)

    atr = compute_atr(high, low, close, period)
    plus_di = 100 * _rma(plus_dm, period) / atr.replace(0, np.nan)
    minus_di = 100 * _rma(minus_dm, period) / atr.replace(0, np.nan)

    dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan)
    adx = _rma(dx.fillna(0), period)
    return adx


def compute_kdj(high, low, close, period=9):
    low_n = low.rolling(period, min_periods=1).min()
    high_n = high.rolling(period, min_periods=1).max()
    rsv = (close - low_n) / (high_n - low_n).replace(0, np.nan) * 100
    rsv = rsv.fillna(50)
    k = rsv.ewm(alpha=1.0 / 3, adjust=False).mean()
    d = k.ewm(alpha=1.0 / 3, adjust=False).mean()
    j = 3 * k - 2 * d
    return j - 50  # 偏离中轴


def compute_obv(close, volume):
    sign = np.sign(close.diff().fillna(0))
    return (sign * volume).cumsum()


def compute_bollinger_width(close, period=20, num_std=2):
    mid = close.rolling(period).mean()
    std = close.rolling(period).std()
    width = (num_std * 2 * std) / mid
    return width


def compute_factors(df):
    """对单只股票的日线 df, 计算全部 16 个因子的逐日序列

    Returns: DataFrame, index 为 df.index, columns 为因子名
    """
    close = df["close"].astype(float)
    high = df["high"].astype(float)
    low = df["low"].astype(float)
    volume = df["volume"].astype(float)

    f = pd.DataFrame(index=df.index)

    # ── 趋势类 ──
    ma60 = close.rolling(60).mean()
    f["MA60_dev"] = (close - ma60) / ma60  # 价格偏离MA60百分比

    ema12 = _ema(close, 12)
    ema26 = _ema(close, 26)
    f["EMA_DUAL_diff"] = (ema12 - ema26) / ema26  # 双均线差值百分比

    dif = ema12 - ema26
    f["MACD_dif"] = dif / close  # MACD DIF 归一化

    f["ADX"] = compute_adx(high, low, close, 14)

    # ── 动量类 ──
    f["RSI"] = compute_rsi(close, 14)  # 偏离中轴, 已减50
    f["KDJ_J"] = compute_kdj(high, low, close, 9)  # 偏离中轴, 已减50
    f["ROC_5"] = close.pct_change(5)
    f["ROC_20"] = close.pct_change(20)

    # ── 量价类 ──
    obv = compute_obv(close, volume)
    f["OBV_chg"] = obv.pct_change(5)  # OBV 5日变化率
    f["VOL_ratio"] = volume / volume.rolling(20).mean()  # 量比
    # 量价相关: 价格变动与成交量变动的20日滚动相关
    f["vol_price_corr"] = close.pct_change().rolling(20).corr(volume.pct_change())
    # 波动率收敛: ATR / ATR_MA20, <1=收敛(酝酿), >1=扩张
    atr = compute_atr(high, low, close, 14)
    f["vol_contraction"] = atr / atr.rolling(20).mean()

    # ── 波动率类 ──
    f["ATR_pct"] = atr / close  # 波动率占价格比
    f["ATR_percentile"] = (atr / close).rolling(60).rank(pct=True)  # 60日波动率百分位
    f["BB_width"] = compute_bollinger_width(close, 20)  # 布林带宽度
    f["hist_vol"] = close.pct_change().rolling(20).std() * np.sqrt(252)  # 年化历史波动率

    return f


# ============================================================
#  IC 计算
# ============================================================

def spearman_ic(x, y):
    """Spearman 秩相关, 返回 IC (nan 若样本不足)"""
    mask = x.notna() & y.notna()
    if mask.sum() < 30:
        return np.nan
    return x[mask].corr(y[mask], method="spearman")


def compute_ic_series(factor_df, forward_returns_df, factor_name):
    """逐日计算截面IC (当日所有股票的因子值 vs 未来收益)

    factor_df: Panel, index=date, columns=code, values=因子值
    forward_returns_df: 同结构, 未来收益
    返回: IC 时间序列 (逐日)
    """
    ics = []
    for date in factor_df.index:
        f_vals = factor_df.loc[date].dropna()
        r_vals = forward_returns_df.loc[date].reindex(f_vals.index).dropna()
        common = f_vals.index.intersection(r_vals.index)
        if len(common) < 3:  # 截面至少3只股票
            ics.append(np.nan)
            continue
        ic = f_vals.reindex(common).corr(r_vals.reindex(common), method="spearman")
        ics.append(ic)
    return pd.Series(ics, index=factor_df.index)


def summarize_ic(ic_series):
    """IC 时间序列 → 汇总统计"""
    valid = ic_series.dropna()
    if len(valid) < 10:
        return {"ic_mean": None, "ic_std": None, "icir": None,
                "ic_winrate": None, "n_obs": int(len(valid))}
    ic_mean = valid.mean()
    ic_std = valid.std()
    icir = ic_mean / ic_std if ic_std > 0 else 0
    winrate = (valid > 0).mean()
    return {
        "ic_mean": round(float(ic_mean), 4),
        "ic_std": round(float(ic_std), 4),
        "icir": round(float(icir), 3),
        "ic_winrate": round(float(winrate), 3),
        "n_obs": int(len(valid)),
    }


# ============================================================
#  主流程
# ============================================================

def load_watchlist():
    config_path = os.path.join(project_root, "config", "strategy_config.json")
    with open(config_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data["strategy_config"]["watchlist"]


def judge_factor(ic_mean, icir):
    """判定因子有效性"""
    if ic_mean is None:
        return "样本不足"
    a = abs(ic_mean)
    if a > 0.05 and icir is not None and icir > 0.5:
        return "✅有效"
    if a > 0.05:
        return "⚠️有效但不稳"
    if a > 0.03:
        return "⚠️弱有效"
    return "❌无效"


def main():
    run_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print("=" * 70)
    print("  因子 IC 分析 — 多指标多维度评估预测力")
    print(f"  数据范围: {DATA_START} ~ {DATA_END}")
    print(f"  因子数: {len(ALL_FACTORS)} (4维度)")
    print(f"  预测周期: {FORWARD_PERIODS} 日")
    print("=" * 70)

    watchlist = load_watchlist()
    dm = DataManager()

    # ── 1. 拉取数据 + 计算因子 ──
    all_panels = {}  # {因子名: {分组名: Panel(date×code)}}
    forward_returns = {}  # {period: {分组名: Panel}}
    group_stock_map = {}

    for group_name, stocks in watchlist.items():
        if group_name.startswith("_"):
            continue
        codes = [s["code"] for s in stocks]
        group_stock_map[group_name] = codes
        print(f"\n处理分组: {group_name} ({len(codes)}只)")

        # 拉数据 + 计算因子
        group_factor_panels = {f: {} for f in ALL_FACTORS}
        group_fwd_panels = {p: {} for p in FORWARD_PERIODS}

        for code in codes:
            df = dm.get_daily_kline(code, start_date=DATA_START, end_date=DATA_END)
            if df is None or len(df) < 80:
                print(f"  ⚠️ {code} 数据不足, 跳过")
                continue
            df = df.set_index("date")
            if not isinstance(df.index, pd.DatetimeIndex):
                df.index = pd.to_datetime(df.index)

            # 计算因子
            facts = compute_factors(df)

            for fname in ALL_FACTORS:
                if fname in facts.columns:
                    group_factor_panels[fname][code] = facts[fname]

            # 计算未来收益
            close = df["close"].astype(float)
            for p in FORWARD_PERIODS:
                fwd = close.shift(-p) / close - 1
                group_fwd_panels[p][code] = fwd

            print(f"  ✓ {code}: {len(df)} 天")

        # 转成 Panel (date × code)
        for fname in ALL_FACTORS:
            if group_factor_panels[fname]:
                all_panels.setdefault(fname, {})[group_name] = pd.DataFrame(group_factor_panels[fname])
        for p in FORWARD_PERIODS:
            if group_fwd_panels[p]:
                forward_returns.setdefault(p, {})[group_name] = pd.DataFrame(group_fwd_panels[p])

    # ── 2. 计算 IC ──
    print(f"\n{'=' * 70}")
    print("计算 IC (Spearman 秩相关)...")
    print(f"{'=' * 70}")

    results = {
        "by_period": {},   # {period: {因子: {分组: stats}}}
        "by_group": {},    # {分组: {因子: {period: stats}}}
        "decay": {},       # {因子: {period: ic_mean}}  全样本衰减
    }

    for p in FORWARD_PERIODS:
        print(f"\n--- 预测周期 N={p} 日 ---")
        results["by_period"][p] = {}
        for fname in ALL_FACTORS:
            group_stats = {}
            all_ics = []  # 汇总全样本IC
            for gname in all_panels.get(fname, {}):
                f_panel = all_panels[fname][gname]
                r_panel = forward_returns.get(p, {}).get(gname)
                if r_panel is None:
                    continue
                # 对齐
                common_idx = f_panel.index.intersection(r_panel.index)
                common_cols = f_panel.columns.intersection(r_panel.columns)
                if len(common_idx) < 20 or len(common_cols) < 2:
                    continue
                f_p = f_panel.loc[common_idx, common_cols]
                r_p = r_panel.loc[common_idx, common_cols]

                ic_series = compute_ic_series(f_p, r_p, fname)
                stats = summarize_ic(ic_series)
                group_stats[gname] = stats
                if stats["ic_mean"] is not None:
                    all_ics.append(stats["ic_mean"])

            # 全样本汇总 (各分组IC均值)
            overall_ic = round(float(np.mean(all_ics)), 4) if all_ics else None
            overall_icir = round(float(np.std(all_ics)), 3) if len(all_ics) > 1 else None
            group_stats["_overall"] = {
                "ic_mean": overall_ic,
                "icir": overall_icir,
                "verdict": judge_factor(overall_ic, overall_icir),
            }
            results["by_period"][p][fname] = group_stats

            if overall_ic is not None:
                print(f"  {fname:<20} IC={overall_ic:+.4f}  {group_stats['_overall']['verdict']}")

    # ── 3. IC 衰减曲线 (全样本, N=1,3,5,10,20) ──
    print(f"\n{'=' * 70}")
    print("IC 衰减曲线...")
    for fname in ALL_FACTORS:
        decay = {}
        for p in FORWARD_PERIODS:
            s = results["by_period"][p].get(fname, {}).get("_overall", {})
            decay[p] = s.get("ic_mean")
        results["decay"][fname] = decay
        vals = "  ".join(f"N{p}={v:+.4f}" if v is not None else f"N{p}=N/A"
                         for p, v in decay.items())
        print(f"  {fname:<20} {vals}")

    # ── 4. 因子相关性矩阵 (看冗余) ──
    print(f"\n计算因子相关性矩阵 (检测冗余)...")
    # 用第一分组的股票池, 把所有因子 concat
    first_group = list(all_panels.get(ALL_FACTORS[0], {}).keys())[0] if all_panels else None
    corr_matrix = {}
    if first_group:
        factor_cols = {}
        for fname in ALL_FACTORS:
            if first_group in all_panels.get(fname, {}):
                panel = all_panels[fname][first_group]
                # 取所有股票的因子值, 求截面均值作为代表
                factor_cols[fname] = panel.mean(axis=1)
        if len(factor_cols) > 1:
            corr_df = pd.DataFrame(factor_cols).corr(method="spearman")
            corr_matrix = corr_df.round(2).to_dict()

    # ── 5. 保存结果 ──
    os.makedirs(os.path.dirname(RESULT_JSON), exist_ok=True)
    output = {
        "run_time": run_time,
        "config": {
            "data_range": [DATA_START, DATA_END],
            "forward_periods": FORWARD_PERIODS,
            "factors": ALL_FACTORS,
            "dimensions": FACTOR_DIMENSIONS,
        },
        "groups": group_stock_map,
        "results": results,
        "factor_correlation": corr_matrix,
    }
    with open(RESULT_JSON, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2, default=str)
    print(f"\n✓ 结构化结果 → {RESULT_JSON}")

    # ── 6. 生成报告 ──
    report = generate_report(output, run_time)
    with open(REPORT_MD, "w", encoding="utf-8") as f:
        f.write(report)
    print(f"✓ 可读报告   → {REPORT_MD}")

    # ── 7. 汇总打印 ──
    print(f"\n{'=' * 70}")
    print(f"  汇总 (主周期 N={PRIMARY_PERIOD}日, 全样本IC)")
    print(f"{'=' * 70}")
    print(f"{'因子':<20} {'维度':<8} {'IC均值':>8} {'ICIR':>6} {'结论':<12}")
    for dim, fs in FACTOR_DIMENSIONS.items():
        for fname in fs:
            s = results["by_period"][PRIMARY_PERIOD].get(fname, {}).get("_overall", {})
            ic = s.get("ic_mean")
            icir = s.get("icir")
            v = s.get("verdict", "")
            ic_str = f"{ic:+.4f}" if ic is not None else "N/A"
            icir_str = f"{icir:.3f}" if icir is not None else "N/A"
            print(f"{fname:<20} {dim:<8} {ic_str:>8} {icir_str:>6} {v}")


def generate_report(output, run_time):
    """生成 Markdown 报告"""
    results = output["results"]
    dims = output["config"]["dimensions"]
    groups = output["groups"]
    p = PRIMARY_PERIOD

    L = []
    L.append("# 因子 IC 分析报告 — 多指标多维度")
    L.append("")
    L.append(f"**运行时间**: {run_time}")
    L.append(f"**数据范围**: {output['config']['data_range'][0]} ~ {output['config']['data_range'][1]}")
    L.append(f"**因子数**: {len(output['config']['factors'])} (4个信息维度)")
    L.append(f"**预测周期**: {output['config']['forward_periods']} 日 (主分析 N={p})")
    L.append(f"**分组**: {list(groups.keys())}")
    L.append("")
    L.append("> **IC (信息系数)** = 因子值与未来收益的 Spearman 秩相关")
    L.append("> - |IC|>0.05 且 ICIR>0.5 → ✅有效")
    L.append("> - |IC|>0.05 但 ICIR≤0.5 → ⚠️有效但不稳")
    L.append("> - |IC| 0.03~0.05 → ⚠️弱有效")
    L.append("> - |IC|<0.03 → ❌无效 (建议剔除)")
    L.append("")

    # ── 一、总览 ──
    L.append("## 一、因子有效性总览 (N=5日, 全样本)")
    L.append("")
    L.append("| 因子 | 信息维度 | IC均值 | ICIR | IC胜率 | 结论 |")
    L.append("|------|---------|--------|------|--------|------|")
    for dim, fs in dims.items():
        for fname in fs:
            s = results["by_period"][p].get(fname, {}).get("_overall", {})
            ic = s.get("ic_mean")
            icir = s.get("icir")
            wr = None
            # 找一个分组的胜率作代表
            for g in groups:
                gs = results["by_period"][p].get(fname, {}).get(g, {})
                if gs.get("ic_winrate") is not None:
                    wr = gs["ic_winrate"]
                    break
            v = s.get("verdict", "")
            ic_str = f"{ic:+.4f}" if ic is not None else "N/A"
            icir_str = f"{icir:.3f}" if icir is not None else "N/A"
            wr_str = f"{wr:.1%}" if wr is not None else "N/A"
            L.append(f"| {fname} | {dim} | {ic_str} | {icir_str} | {wr_str} | {v} |")
    L.append("")

    # ── 二、IC 衰减曲线 ──
    L.append("## 二、IC 衰减曲线 (预测力半衰期)")
    L.append("")
    L.append("看因子预测力随持有周期延长如何衰减。衰减慢=长期因子, 衰减快=短期因子。")
    L.append("")
    L.append("| 因子 | N=1 | N=3 | N=5 | N=10 | N=20 | 最优周期 |")
    L.append("|------|-----|-----|-----|------|------|---------|")
    for dim, fs in dims.items():
        for fname in fs:
            decay = results["decay"].get(fname, {})
            vals = []
            best_p, best_v = None, 0
            for period in FORWARD_PERIODS:
                v = decay.get(period)
                vals.append(f"{v:+.4f}" if v is not None else "N/A")
                if v is not None and abs(v) > abs(best_v):
                    best_p, best_v = period, v
            L.append(f"| {fname} | " + " | ".join(vals) + f" | N={best_p} |")
    L.append("")

    # ── 三、分组IC对比 ──
    L.append("## 三、分组 IC 对比 (N=5日)")
    L.append("")
    L.append("看同一因子在不同行业的有效性差异。**IC在不同分组符号相反=该因子有行业特异性**。")
    L.append("")
    header = "| 因子 | " + " | ".join(groups.keys()) + " |"
    sep = "|------|" + "|".join(["------"] * len(groups)) + "|"
    L.append(header)
    L.append(sep)
    for dim, fs in dims.items():
        for fname in fs:
            cells = []
            for g in groups:
                s = results["by_period"][p].get(fname, {}).get(g, {})
                ic = s.get("ic_mean")
                cells.append(f"{ic:+.4f}" if ic is not None else "N/A")
            L.append(f"| {fname} | " + " | ".join(cells) + " |")
    L.append("")

    # ── 四、因子相关性矩阵 (冗余检测) ──
    L.append("## 四、因子相关性矩阵 (冗余检测)")
    L.append("")
    L.append("|相关性|>0.7 的因子对提取的是同一信息, 应合并或剔除其一。")
    L.append("")
    corr = output.get("factor_correlation", {})
    if corr:
        factor_list = list(corr.keys())
        L.append("| | " + " | ".join(factor_list) + " |")
        L.append("|---|" + "|".join(["---"] * len(factor_list)) + "|")
        for f1 in factor_list:
            row = []
            for f2 in factor_list:
                v = corr[f1].get(f2)
                row.append(f"{v:.2f}" if v is not None else "")
            L.append(f"| {f1} | " + " | ".join(row) + " |")
    else:
        L.append("(相关性矩阵计算失败)")
    L.append("")

    # ── 五、结论与建议 ──
    L.append("## 五、结论与因子增删建议")
    L.append("")

    # 分类因子
    effective = []
    weak = []
    invalid = []
    for fname in ALL_FACTORS:
        s = results["by_period"][p].get(fname, {}).get("_overall", {})
        v = s.get("verdict", "")
        ic = s.get("ic_mean")
        if "✅" in v:
            effective.append((fname, ic))
        elif "弱" in v or "不稳" in v:
            weak.append((fname, ic))
        else:
            invalid.append((fname, ic))

    L.append(f"### 因子有效性分布")
    L.append(f"- ✅ 有效: {len(effective)}个 — {[f[0] for f in effective]}")
    L.append(f"- ⚠️ 弱有效/不稳: {len(weak)}个 — {[f[0] for f in weak]}")
    L.append(f"- ❌ 无效: {len(invalid)}个 — {[f[0] for f in invalid]}")
    L.append("")

    L.append("### 增删建议")
    L.append("")
    if invalid:
        L.append("**建议剔除或重构的因子** (IC无显著预测力):")
        for fname, ic in invalid:
            ic_str = f"{ic:+.4f}" if ic is not None else "N/A"
            L.append(f"- `{fname}`: IC={ic_str}, 未通过有效性检验, 提取的信息无预测价值")
        L.append("")
    if effective:
        L.append("**核心有效因子** (建议保留并加权):")
        for fname, ic in effective:
            ic_str = f"{ic:+.4f}" if ic is not None else "N/A"
            L.append(f"- `{fname}`: IC={ic_str}, 是策略Alpha的主要来源")
        L.append("")
    if weak:
        L.append("**待优化因子** (有信号但不稳, 需正交化或参数调整):")
        for fname, ic in weak:
            ic_str = f"{ic:+.4f}" if ic is not None else "N/A"
            L.append(f"- `{fname}`: IC={ic_str}, 弱有效, 检查是否与其他因子冗余")
        L.append("")

    L.append("### 维度均衡性检查")
    L.append("")
    for dim, fs in dims.items():
        eff_in_dim = [f for f in fs if f in [e[0] for e in effective]]
        L.append(f"- **{dim}**: {len(eff_in_dim)}/{len(fs)} 有效 — "
                 f"{'维度充足' if len(eff_in_dim) >= 2 else '⚠️维度不足, 考虑补充'}")
    L.append("")
    L.append("### 下一步")
    L.append("")
    L.append("1. **剔除无效因子**, 减少噪声和冗余")
    L.append("2. **对有效因子做正交化** (PCA或对称正交), 消除维度间相关性")
    L.append("3. **用IC加权替代人工权重** — IC高的因子给更高权重")
    L.append("4. **补充缺失维度的因子** — 若某维度有效因子<2个, 补充新因子")
    L.append("5. **分regime分析** — 下一步可对趋势/震荡市分别算IC, 看因子是否需要regime适配")

    return "\n".join(L)


if __name__ == "__main__":
    main()
