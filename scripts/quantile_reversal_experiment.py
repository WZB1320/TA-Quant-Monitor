"""
分位数反转实验 — 绕过 classifier/filter, 直接测反转策略的真实夏普

背景:
  反转实验(简单反转direction)因只做多框架约束导致0交易。
  本实验绕过现有 SignalEngine/classifier/scorer, 直接用因子截面分位数选股:
    - 正常组: 买入复合因子值最高 20% (趋势跟随, 模拟现有策略逻辑)
    - 反转组: 买入复合因子值最低 20% (反转, 买入弱势股等反弹)
    - 基准:   等权持有全组股票 (buy and hold)

  这样能真正测出"反转 vs 趋势跟随"的夏普差异, 不被框架约束干扰。

复合因子构建 (4维度正交, 避免趋势三胞胎重复加权):
  - 趋势: MACD_dif      (IC=-0.049, 趋势类代表)
  - 动量: RSI           (IC=-0.039, 动量类代表)
  - 量价: vol_contraction (IC=-0.032, 量价类代表)
  - 波动率: hist_vol    (IC=-0.022, 波动率类代表)
  每个因子做截面 z-score 标准化后等权相加 = 复合因子值

  高复合因子值 = 趋势强+动量强+波动收敛+高波动 (现有策略看多的股票)
  低复合因子值 = 趋势弱+动量弱+波动扩张+低波动 (现有策略看空的股票)

回测规则 (简化但严谨):
  - 调仓频率 = 持有期 = 5 天 (固定调仓, 不重叠)
  - 每个调仓日: 按复合因子值排序, 选 top/bottom 20%, 等权买入
  - 持有 5 天后全部卖出, 重新选股
  - 交易成本: 每次调仓扣 0.3% (佣金万2.5×2 + 印花税千1 + 滑点万一×2)
  - 初始资金 10 万, 满仓等权

输出:
  data/quantile_reversal_result.json
  data/quantile_reversal_report.md

用法:
  python scripts/quantile_reversal_experiment.py
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

# 复用 IC 分析的因子计算
from scripts.factor_ic_analysis import compute_factors, load_watchlist  # noqa
from src.data_fetcher import DataManager

# ── 配置 ──
TRAIN_START, TRAIN_END = "2024-07-01", "2025-06-30"
TEST_START, TEST_END = "2025-07-01", "2026-06-30"
DATA_START, DATA_END = "2024-02-01", "2026-07-13"
HOLD_DAYS = 5                # 持有期 = 调仓频率
QUANTILE_PCT = 0.20          # 选股分位数 (top/bottom 20%)
COST_PER_REBAL = 0.003       # 每次调仓交易成本 0.3%
INITIAL_CAPITAL = 100000

# 复合因子: 4 维度各选 1 个代表 (避免趋势三胞胎冗余)
COMPOSITE_FACTORS = ["MACD_dif", "RSI", "vol_contraction", "hist_vol"]

RESULT_JSON = os.path.join(project_root, "data", "quantile_reversal_result.json")
REPORT_MD = os.path.join(project_root, "data", "quantile_reversal_report.md")


def zscore_cross_section(panel):
    """截面 z-score 标准化: 每一行(日期)对所有股票做 z-score

    panel: DataFrame, index=date, columns=code
    """
    mean = panel.mean(axis=1)
    std = panel.std(axis=1).replace(0, np.nan)
    return panel.sub(mean, axis=0).div(std, axis=0)


def build_composite_factor(factor_panels):
    """构建复合因子: 4 个因子截面 z-score 后等权相加

    factor_panels: {因子名: DataFrame(date×code)}
    Returns: DataFrame(date×code), 复合因子值
    """
    zscored = []
    for fname in COMPOSITE_FACTORS:
        if fname not in factor_panels:
            continue
        z = zscore_cross_section(factor_panels[fname])
        zscored.append(z)
    if not zscored:
        return None
    # 等权相加
    composite = sum(zscored) / len(zscored)
    return composite


def quantile_select(composite, date, n_select, side):
    """单日分位数选股

    composite: DataFrame(date×code)
    date: 调仓日
    n_select: 选几只
    side: 'top' (趋势跟随) or 'bottom' (反转)
    """
    row = composite.loc[date].dropna()
    if len(row) < 1:
        return []
    n = min(n_select, len(row))
    if side == "top":
        return row.nlargest(n).index.tolist()
    else:
        return row.nsmallest(n).index.tolist()


def run_quantile_backtest(composite, daily_returns, start_date, end_date,
                           side, n_select):
    """分位数选股回测

    composite: DataFrame(date×code), 复合因子值
    daily_returns: DataFrame(date×code), 日收益率
    start_date/end_date: 回测区间
    side: 'top' or 'bottom'
    n_select: 每次选几只

    Returns: dict with metrics
    """
    # 对齐日期
    dates = composite.loc[start_date:end_date].index
    dates = dates.intersection(daily_returns.index)
    if len(dates) < HOLD_DAYS * 4:
        return _empty_metrics()

    # 调仓日: 每 HOLD_DAYS 天
    rebalance_dates = []
    d = dates[0]
    while d <= dates[-1]:
        if d in composite.index:
            rebalance_dates.append(d)
        # 找下一个调仓日 (HOLD_DAYS 天后)
        idx = dates.get_loc(d)
        next_idx = idx + HOLD_DAYS
        if next_idx >= len(dates):
            break
        d = dates[next_idx]

    if len(rebalance_dates) < 3:
        return _empty_metrics()

    # 回测: 每个调仓日选股, 持有 HOLD_DAYS 天
    portfolio_returns = []  # 每个持有期的组合收益率
    trade_count = 0

    for i, reb_date in enumerate(rebalance_dates[:-1]):
        # 选股
        selected = quantile_select(composite, reb_date, n_select, side)
        if not selected:
            continue
        trade_count += len(selected)

        # 持有期
        reb_idx = dates.get_loc(reb_date)
        end_idx = min(reb_idx + HOLD_DAYS, len(dates) - 1)
        hold_dates = dates[reb_idx:end_idx + 1]
        if len(hold_dates) < 2:
            continue

        # 等权组合的日收益率
        stock_returns = daily_returns.loc[hold_dates, selected]
        port_daily = stock_returns.mean(axis=1)

        # 扣交易成本 (买入时)
        port_daily.iloc[0] -= COST_PER_REBAL / len(selected)

        portfolio_returns.extend(port_daily.values.tolist())

    if len(portfolio_returns) < 10:
        return _empty_metrics()

    returns = pd.Series(portfolio_returns)
    return _compute_metrics(returns, trade_count)


def run_benchmark(daily_returns, start_date, end_date, codes):
    """基准: 等权持有全组股票 buy and hold"""
    dates = daily_returns.loc[start_date:end_date].index
    valid_codes = [c for c in codes if c in daily_returns.columns]
    if not valid_codes:
        return _empty_metrics()
    bench = daily_returns.loc[dates, valid_codes].mean(axis=1).dropna()
    return _compute_metrics(bench, len(valid_codes))


def _compute_metrics(returns, trade_count):
    """计算夏普等指标"""
    if len(returns) < 2 or returns.std() == 0:
        return _empty_metrics()
    total_return = (1 + returns).prod() - 1
    n_days = len(returns)
    annual_return = (1 + total_return) ** (252 / n_days) - 1 if n_days > 0 else 0
    sharpe = (returns.mean() / returns.std() * np.sqrt(252)) if returns.std() > 0 else 0
    # 最大回撤
    cum = (1 + returns).cumprod()
    peak = cum.cummax()
    drawdown = (cum - peak) / peak
    max_dd = drawdown.min()
    vol = returns.std() * np.sqrt(252)
    win_rate = (returns > 0).mean()
    return {
        "total_return_pct": round(total_return * 100, 2),
        "annual_return_pct": round(annual_return * 100, 2),
        "sharpe_ratio": round(sharpe, 3),
        "max_drawdown_pct": round(max_dd * 100, 2),
        "volatility_pct": round(vol * 100, 2),
        "trade_count": trade_count,
        "win_rate_pct": round(win_rate * 100, 1),
        "n_days": n_days,
    }


def _empty_metrics():
    return {
        "total_return_pct": 0.0, "annual_return_pct": 0.0, "sharpe_ratio": 0.0,
        "max_drawdown_pct": 0.0, "volatility_pct": 0.0, "trade_count": 0,
        "win_rate_pct": 0.0, "n_days": 0,
    }


def main():
    run_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print("=" * 70)
    print("  分位数反转实验 — 绕过 classifier 直接测夏普")
    print(f"  训练窗: {TRAIN_START} ~ {TRAIN_END}")
    print(f"  测试窗: {TEST_START} ~ {TEST_END}")
    print(f"  持有期: {HOLD_DAYS} 天 | 选股: top/bottom {int(QUANTILE_PCT*100)}%")
    print(f"  复合因子: {COMPOSITE_FACTORS}")
    print("=" * 70)

    watchlist = load_watchlist()
    dm = DataManager()
    results = {}

    for group_name, stocks in watchlist.items():
        if group_name.startswith("_"):
            continue
        codes = [s["code"] for s in stocks]
        print(f"\n{'─' * 60}")
        print(f"分组: {group_name} ({len(codes)}只)")
        print(f"{'─' * 60}")

        # 拉数据 + 算因子
        factor_panels = {}
        daily_returns = {}
        for code in codes:
            df = dm.get_daily_kline(code, start_date=DATA_START, end_date=DATA_END)
            if df is None or len(df) < 80:
                continue
            df = df.set_index("date")
            if not isinstance(df.index, pd.DatetimeIndex):
                df.index = pd.to_datetime(df.index)

            facts = compute_factors(df)
            for fname in COMPOSITE_FACTORS:
                if fname in facts.columns:
                    factor_panels.setdefault(fname, {})[code] = facts[fname]
            daily_returns[code] = df["close"].astype(float).pct_change()

        if not factor_panels:
            print("  ⚠️ 数据不足, 跳过")
            continue

        # 转 Panel
        fp = {f: pd.DataFrame(d) for f, d in factor_panels.items()}
        ret_panel = pd.DataFrame(daily_returns)

        # 构建复合因子
        composite = build_composite_factor(fp)
        if composite is None:
            print("  ⚠️ 复合因子构建失败, 跳过")
            continue

        n_stocks = len(ret_panel.columns)
        n_select = max(1, int(np.ceil(n_stocks * QUANTILE_PCT)))
        print(f"  组内 {n_stocks} 只, 每次选 {n_select} 只")

        group_result = {"n_stocks": n_stocks, "n_select": n_select}

        for window, start, end in [("train", TRAIN_START, TRAIN_END),
                                    ("test", TEST_START, TEST_END)]:
            wlabel = "训练" if window == "train" else "测试"
            print(f"\n  [{wlabel}窗]")

            # 正常 (top = 趋势跟随)
            m_top = run_quantile_backtest(composite, ret_panel, start, end,
                                           "top", n_select)
            print(f"    趋势跟随(top):  夏普={m_top['sharpe_ratio']:+.3f} "
                  f"收益={m_top['total_return_pct']:+.2f}% 交易={m_top['trade_count']}笔")

            # 反转 (bottom = 买入弱势股)
            m_bot = run_quantile_backtest(composite, ret_panel, start, end,
                                           "bottom", n_select)
            print(f"    反转(bottom):   夏普={m_bot['sharpe_ratio']:+.3f} "
                  f"收益={m_bot['total_return_pct']:+.2f}% 交易={m_bot['trade_count']}笔")

            # 基准 (等权持有)
            m_bench = run_benchmark(ret_panel, start, end, ret_panel.columns.tolist())
            print(f"    基准(等权持有):  夏普={m_bench['sharpe_ratio']:+.3f} "
                  f"收益={m_bench['total_return_pct']:+.2f}%")

            delta = m_bot["sharpe_ratio"] - m_top["sharpe_ratio"]
            arrow = "↑" if delta > 0.1 else ("↓" if delta < -0.1 else "→")
            print(f"    反转-趋势夏普差: {arrow} {abs(delta):.3f}")

            group_result[window] = {
                "trend_follow": m_top,
                "reversal": m_bot,
                "benchmark": m_bench,
                "sharpe_delta_reversal_vs_trend": round(delta, 3),
            }

        results[group_name] = group_result

    # ── 保存结果 ──
    os.makedirs(os.path.dirname(RESULT_JSON), exist_ok=True)
    output = {
        "run_time": run_time,
        "config": {
            "hold_days": HOLD_DAYS,
            "quantile_pct": QUANTILE_PCT,
            "composite_factors": COMPOSITE_FACTORS,
            "cost_per_rebal": COST_PER_REBAL,
            "train": [TRAIN_START, TRAIN_END],
            "test": [TEST_START, TEST_END],
        },
        "results": results,
    }
    with open(RESULT_JSON, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)
    print(f"\n✓ 结构化结果 → {RESULT_JSON}")

    # ── 生成报告 ──
    report = generate_report(output, run_time)
    with open(REPORT_MD, "w", encoding="utf-8") as f:
        f.write(report)
    print(f"✓ 可读报告   → {REPORT_MD}")

    # ── 汇总 ──
    print(f"\n{'=' * 70}")
    print("  汇总: 趋势跟随 vs 反转 vs 基准 (夏普)")
    print(f"{'=' * 70}")
    print(f"{'分组':<12} {'训练(趋势/反转/基准)':>30} {'测试(趋势/反转/基准)':>30}")
    for g, r in results.items():
        tt, tr, tb = (r["train"]["trend_follow"]["sharpe_ratio"],
                      r["train"]["reversal"]["sharpe_ratio"],
                      r["train"]["benchmark"]["sharpe_ratio"])
        et, er, eb = (r["test"]["trend_follow"]["sharpe_ratio"],
                      r["test"]["reversal"]["sharpe_ratio"],
                      r["test"]["benchmark"]["sharpe_ratio"])
        print(f"{g:<12} {tt:>7.3f}/{tr:>7.3f}/{tb:>7.3f}   "
              f"{et:>7.3f}/{er:>7.3f}/{eb:>7.3f}")


def generate_report(output, run_time):
    """生成报告"""
    results = output["results"]
    cfg = output["config"]

    L = []
    L.append("# 分位数反转实验报告 — 绕过 classifier 直接测夏普")
    L.append("")
    L.append(f"**运行时间**: {run_time}")
    L.append(f"**训练窗**: {cfg['train'][0]} ~ {cfg['train'][1]}")
    L.append(f"**测试窗**: {cfg['test'][0]} ~ {cfg['test'][1]}")
    L.append(f"**持有期/调仓**: {cfg['hold_days']}天 | **选股**: top/bottom {int(cfg['quantile_pct']*100)}%")
    L.append(f"**复合因子**: {cfg['composite_factors']} (4维度正交, z-score等权)")
    L.append(f"**交易成本**: 每次调仓 {cfg['cost_per_rebal']*100:.1f}%")
    L.append("")
    L.append("> **实验设计**: 绕过现有 classifier/filter/scorer, 直接用复合因子截面分位数选股。")
    L.append("> - 趋势跟随(top): 买入复合因子值最高20% (模拟现有策略逻辑)")
    L.append("> - 反转(bottom): 买入复合因子值最低20% (买入弱势股等反弹)")
    L.append("> - 基准: 等权持有全组 (buy and hold, 剔除选股看市场beta)")
    L.append("")

    # ── 一、夏普对比总览 ──
    L.append("## 一、夏普对比总览 (趋势跟随 vs 反转 vs 基准)")
    L.append("")
    L.append("| 分组 | 窗口 | 趋势跟随 | 反转 | 基准 | 反转-趋势 | 反转-基准 |")
    L.append("|------|------|---------|------|------|----------|----------|")
    for g, r in results.items():
        for window in ["train", "test"]:
            w = r[window]
            tf, rv, bm = w["trend_follow"]["sharpe_ratio"], w["reversal"]["sharpe_ratio"], w["benchmark"]["sharpe_ratio"]
            d1 = round(rv - tf, 3)
            d2 = round(rv - bm, 3)
            a1 = "↑" if d1 > 0.1 else ("↓" if d1 < -0.1 else "→")
            a2 = "↑" if d2 > 0.1 else ("↓" if d2 < -0.1 else "→")
            wl = "训练" if window == "train" else "测试"
            L.append(f"| {g} | {wl} | {tf:+.3f} | {rv:+.3f} | {bm:+.3f} | {a1}{abs(d1):.3f} | {a2}{abs(d2):.3f} |")
    L.append("")

    # ── 二、详细指标对比 ──
    L.append("## 二、详细指标对比 (测试窗)")
    L.append("")
    for g, r in results.items():
        L.append(f"### {g}")
        L.append("")
        L.append("| 指标 | 趋势跟随 | 反转 | 基准 |")
        L.append("|------|---------|------|------|")
        t = r["test"]
        for metric in ["total_return_pct", "annual_return_pct", "sharpe_ratio",
                        "max_drawdown_pct", "volatility_pct", "win_rate_pct", "trade_count"]:
            label = {"total_return_pct": "总收益%", "annual_return_pct": "年化%",
                     "sharpe_ratio": "夏普", "max_drawdown_pct": "最大回撤%",
                     "volatility_pct": "波动率%", "win_rate_pct": "胜率%",
                     "trade_count": "交易次数"}.get(metric, metric)
            L.append(f"| {label} | {t['trend_follow'][metric]} | {t['reversal'][metric]} | {t['benchmark'][metric]} |")
        L.append("")

    # ── 三、结论 ──
    L.append("## 三、结论")
    L.append("")
    # 统计
    rev_beats_trend_train = sum(1 for r in results.values()
                                if r["train"]["sharpe_delta_reversal_vs_trend"] > 0.1)
    rev_beats_trend_test = sum(1 for r in results.values()
                               if r["test"]["sharpe_delta_reversal_vs_trend"] > 0.1)
    rev_beats_bench_test = sum(1 for r in results.values()
                               if r["test"]["reversal"]["sharpe_ratio"] - r["test"]["benchmark"]["sharpe_ratio"] > 0.1)
    total = len(results)

    L.append(f"- 反转 > 趋势跟随 (训练窗): {rev_beats_trend_train}/{total} 分组")
    L.append(f"- 反转 > 趋势跟随 (测试窗): {rev_beats_trend_test}/{total} 分组")
    L.append(f"- 反转 > 基准(等权持有) (测试窗): {rev_beats_bench_test}/{total} 分组")
    L.append("")

    if rev_beats_trend_test >= total / 2:
        L.append("✅ **反转假设成立**: 过半分组反转夏普 > 趋势跟随,")
        L.append("   证实因子方向确实反了 — A股在样本期呈反转效应, 买入弱势股优于追强势股。")
        L.append("   建议: 正式引入反转选股逻辑, 或对现有趋势打分取反使用。")
    elif rev_beats_bench_test >= total / 2:
        L.append("⚠️ **反转有Alpha但不稳定**: 反转能跑赢等权基准, 但未必跑赢趋势跟随。")
        L.append("   说明反转效应存在但弱于趋势(在趋势牛市), 或因子复合方式需优化。")
        L.append("   建议: 分组配置 — 反转有效的分组用反转, 趋势有效的用趋势。")
    else:
        L.append("❌ **反转未跑赢基准**: 反转策略既未跑赢趋势跟随也未跑赢等权持有。")
        L.append("   说明: 因子虽负IC, 但简单的分位数反转无法转化为夏普。")
        L.append("   可能原因: ①反转需要更精细的择时(不是固定持有5天) ②弱势股反弹幅度不够覆盖成本 ③持有期不对")
        L.append("   建议: Alpha在执行层(止损止盈/regime), 而非单纯因子方向反转。")
    L.append("")
    L.append("### 实验说明")
    L.append("")
    L.append("- 本实验绕过现有 classifier/filter/scorer, 直接用复合因子截面分位数选股")
    L.append("- 复合因子 = 4维度代表因子(MACD_dif/RSI/vol_contraction/hist_vol)的截面z-score等权相加")
    L.append("- 固定持有5天调仓, 等权选股, 扣0.3%交易成本")
    L.append("- 与现有策略的差异: 无止损止盈/无regime切换/无硬过滤, 纯因子分位数选股")
    L.append("- 若反转在此纯因子框架下跑赢趋势, 说明因子方向是关键; 若未跑赢, 说明现有策略Alpha来自执行层")

    return "\n".join(L)


if __name__ == "__main__":
    main()
