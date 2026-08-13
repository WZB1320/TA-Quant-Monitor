"""
P0 组合回测 — 周期组trending过滤 + 消费组降权10%

差异化配置:
  周期资源型: trade_regimes={trending} (仅趋势市交易, 已验证Alpha+89%)
  消费稳健型: 资金权重10% (降权止血, 原本占20%)
  其他分组:   正常权重, 基线参数

资金分配 (总100万):
  科技30% / 消费10% / 周期30% / 医药15% / 机械15%
  (消费从~20%降到10%, 释放的10%分给周期+科技各5%)

组合指标计算:
  分组独立跑回测 → 取每日净值(daily_values) → 按资金权重加权合并 → 算组合级夏普/收益/回撤/Alpha

对比:
  A. 基线组合: 5组都默认参数, 等权20%
  B. P0组合:   周期+trending过滤, 消费降权10%, 其他默认

用法:
  python scripts/p0_portfolio_backtest.py
"""
import sys
import os
import json
import shutil
import warnings
import numpy as np
import pandas as pd
from datetime import datetime

warnings.filterwarnings("ignore")
project_root = os.path.join(os.path.dirname(__file__), "..")
sys.path.insert(0, project_root)

from src.config.runtime_mode import set_mode, RuntimeMode
set_mode(RuntimeMode.BACKTEST)

from src.backtest.engine import BacktestEngine
from src.data_fetcher import DataManager
from src.config.group_config import GroupConfig
from src.config.user_preferences import UserPreferences, _USER_PREF_FILE

TRAIN_START, TRAIN_END = "2024-07-01", "2025-06-30"
TEST_START, TEST_END = "2025-07-01", "2026-06-30"
DATA_START, DATA_END = "2024-02-01", "2026-07-13"
BENCHMARK = "sh.000300"
TOTAL_CAPITAL = 1000000

# P0 资金权重分配
P0_WEIGHTS = {
    "科技成长型": 0.30,
    "消费稳健型": 0.10,   # 降权: 从基线20%→10%
    "周期资源型": 0.30,
    "医药创新型": 0.15,
    "机械制造型": 0.15,
}
BASELINE_WEIGHT = 0.20  # 基线等权20%

RESULT_JSON = os.path.join(project_root, "data", "p0_portfolio_result.json")
REPORT_MD = os.path.join(project_root, "data", "p0_portfolio_report.md")


def load_watchlist():
    cfg_path = os.path.join(project_root, "config", "strategy_config.json")
    with open(cfg_path, "r", encoding="utf-8") as f:
        cfg = json.load(f)
    return {g: [s["code"] for s in stocks]
            for g, stocks in cfg["strategy_config"]["watchlist"].items()
            if not g.startswith("_")}


def run_group(data_map, benchmark_df, group_codes, group_capital,
              trade_regimes, start, end):
    """跑单分组回测, 返回metrics + daily_values"""
    GroupConfig._instance = None
    GroupConfig._config = None
    try:
        engine = BacktestEngine(
            initial_capital=group_capital, lookback_days=120, position_ratio=0.3,
            commission_rate=0.00025, stamp_tax=0.001, slippage=0.0001,
            signal_dedup_days=5, risk_per_trade=0.05, atr_stop_mult=2.0,
            forced_regime=None, trade_regimes=trade_regimes,
        )
        sub_map = {c: data_map[c] for c in group_codes if c in data_map}
        m = engine.run(sub_map, benchmark_df=benchmark_df, start_date=start, end_date=end)
        return {
            "sharpe": getattr(m, "sharpe_ratio", 0) or 0,
            "total_return": getattr(m, "total_return", 0) or 0,
            "alpha": getattr(m, "alpha", 0) or 0,
            "max_drawdown": getattr(m, "max_drawdown", 0) or 0,
            "trade_count": getattr(m, "trade_count", 0) or 0,
            "win_rate": getattr(m, "win_rate", 0) or 0,
            "final_value": getattr(m, "final_value", group_capital) or group_capital,
            "daily_values": engine.daily_values.copy() if engine.daily_values is not None else None,
        }
    except Exception as e:
        return {"error": str(e)}


def compute_portfolio_metrics(group_results, weights, benchmark_df, start, end):
    """合并各分组daily_values, 算组合级指标"""
    # 加权合并每日净值
    # 注意: group_capital = TOTAL_CAPITAL × weight, daily_values已按该capital跑
    # 所以直接相加即可, 不再乘weight (否则double-count)
    portfolio_nav = None
    total_trades = 0
    for g, r in group_results.items():
        if "error" in r or r["daily_values"] is None:
            continue
        nav = r["daily_values"]  # 已按 group_capital 缩放
        if portfolio_nav is None:
            portfolio_nav = nav.copy()
        else:
            portfolio_nav = portfolio_nav.add(nav, fill_value=0)
        total_trades += r["trade_count"]

    if portfolio_nav is None or len(portfolio_nav) < 10:
        return {"error": "无有效净值数据"}

    # 裁剪到回测窗口 (统一把索引转成 datetime 比较)
    start_ts = pd.Timestamp(start)
    end_ts = pd.Timestamp(end)
    nav_idx = pd.to_datetime(portfolio_nav.index)
    portfolio_nav = pd.Series(portfolio_nav.values, index=nav_idx)
    mask = (portfolio_nav.index >= start_ts) & (portfolio_nav.index <= end_ts)
    portfolio_nav = portfolio_nav[mask]
    total_capital = sum(weights[g] * TOTAL_CAPITAL for g in group_results if "error" not in group_results[g])

    # 日收益率
    daily_returns = portfolio_nav.pct_change().dropna()

    # 组合指标
    total_return = (portfolio_nav.iloc[-1] / total_capital) - 1
    if daily_returns.std() > 0:
        sharpe = daily_returns.mean() / daily_returns.std() * np.sqrt(252)
    else:
        sharpe = 0.0

    # 最大回撤
    cummax = portfolio_nav.cummax()
    drawdown = (portfolio_nav - cummax) / cummax
    max_drawdown = drawdown.min()

    # 基准收益 (统一转datetime)
    bench_df = benchmark_df.set_index("date")["close"]
    bench_idx = pd.to_datetime(bench_df.index)
    bench_df = pd.Series(bench_df.values, index=bench_idx)
    bmask = (bench_df.index >= start_ts) & (bench_df.index <= end_ts)
    bench = bench_df[bmask]
    bench_return = (bench.iloc[-1] / bench.iloc[0]) - 1 if len(bench) > 0 else 0

    alpha = total_return - bench_return

    return {
        "sharpe": round(sharpe, 3),
        "total_return_pct": round(total_return * 100, 2),
        "alpha_pct": round(alpha * 100, 2),
        "benchmark_return_pct": round(bench_return * 100, 2),
        "max_drawdown_pct": round(max_drawdown * 100, 2),
        "trade_count": total_trades,
        "final_value": round(portfolio_nav.iloc[-1], 0),
    }


def main():
    run_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print("=" * 70)
    print("  P0 组合回测 — 周期trending过滤 + 消费降权10%")
    print(f"  训练:{TRAIN_START}~{TRAIN_END} | 测试:{TEST_START}~{TEST_END}")
    print(f"  总资金:{TOTAL_CAPITAL} | P0权重:{P0_WEIGHTS}")
    print("=" * 70)

    pref_existed = os.path.exists(_USER_PREF_FILE)
    pref_backup = _USER_PREF_FILE + ".p0_bak"
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

        all_results = {}

        for version, weights, regime_config in [
            ("baseline", {g: BASELINE_WEIGHT for g in watchlist if not g.startswith("_")}, {}),
            ("p0", P0_WEIGHTS, {"周期资源型": {"trending"}}),
        ]:
            print(f"\n{'='*60}\n  版本: {version.upper()}\n{'='*60}")
            version_results = {}
            for window, start, end in [("train", TRAIN_START, TRAIN_END),
                                        ("test", TEST_START, TEST_END)]:
                wl = "训练" if window == "train" else "测试"
                print(f"\n  [{wl}窗]")
                window_groups = {}
                for group_name, codes in watchlist.items():
                    if group_name.startswith("_"):
                        continue
                    group_codes = [c for c in codes if c in data_map]
                    if len(group_codes) < 2:
                        continue
                    capital = TOTAL_CAPITAL * weights[group_name]
                    regimes = regime_config.get(group_name)
                    r = run_group(data_map, benchmark_df, group_codes, capital,
                                  regimes, start, end)
                    if "error" not in r:
                        print(f"    {group_name:12s}: 夏普={r['sharpe']:+.3f} "
                              f"收益={r['total_return']*100:+.1f}% 交易={r['trade_count']}笔")
                    window_groups[group_name] = r

                # 组合级指标
                portfolio = compute_portfolio_metrics(window_groups, weights,
                                                       benchmark_df, start, end)
                if "error" not in portfolio:
                    print(f"    {'组合(加权)':12s}: 夏普={portfolio['sharpe']:+.3f} "
                          f"收益={portfolio['total_return_pct']:+.1f}% "
                          f"Alpha={portfolio['alpha_pct']:+.1f}% "
                          f"回撤={portfolio['max_drawdown_pct']:.1f}% "
                          f"交易={portfolio['trade_count']}笔")
                version_results[window] = {"groups": {
                    g: {k: v for k, v in r.items() if k != "daily_values"}
                    for g, r in window_groups.items()
                }, "portfolio": portfolio}
            all_results[version] = version_results

        # 保存
        os.makedirs(os.path.dirname(RESULT_JSON), exist_ok=True)
        output = {
            "run_time": run_time,
            "config": {
                "total_capital": TOTAL_CAPITAL,
                "p0_weights": P0_WEIGHTS,
                "baseline_weight": BASELINE_WEIGHT,
            },
            "results": all_results,
        }
        with open(RESULT_JSON, "w", encoding="utf-8") as f:
            json.dump(output, f, ensure_ascii=False, indent=2, default=str)
        print(f"\n✓ 结果 → {RESULT_JSON}")

        report = generate_report(output, run_time)
        with open(REPORT_MD, "w", encoding="utf-8") as f:
            f.write(report)
        print(f"✓ 报告 → {REPORT_MD}")

        # 汇总
        print(f"\n{'='*70}")
        print("  P0 vs 基线 组合级对比")
        print(f"{'='*70}")
        print(f"{'窗口':<6} {'版本':<8} {'夏普':>8} {'收益%':>8} {'Alpha%':>8} {'回撤%':>8} {'交易':>6}")
        for window, wl in [("train","训练"),("test","测试")]:
            for version, ver in [("baseline","基线"),("p0","P0")]:
                p = all_results[version][window]["portfolio"]
                if "error" in p: continue
                print(f"{wl:<6} {ver:<8} {p['sharpe']:>8.3f} {p['total_return_pct']:>8.1f} "
                      f"{p['alpha_pct']:>8.1f} {p['max_drawdown_pct']:>8.1f} {p['trade_count']:>6}")

    finally:
        if pref_existed and os.path.exists(pref_backup):
            shutil.move(pref_backup, _USER_PREF_FILE)


def generate_report(output, run_time):
    L = []
    L.append("# P0 组合回测报告 — 周期trending过滤 + 消费降权10%")
    L.append("")
    L.append(f"**运行时间**: {run_time}")
    L.append(f"**总资金**: {output['config']['total_capital']}")
    L.append(f"**P0权重**: {output['config']['p0_weights']}")
    L.append(f"**基线权重**: 各组{int(output['config']['baseline_weight']*100)}%等权")
    L.append("")
    L.append("**P0差异化配置**:")
    L.append("- 周期资源型: trade_regimes={trending} (仅趋势市交易)")
    L.append("- 消费稳健型: 资金权重10% (降权止血)")
    L.append("- 其他分组: 默认参数")
    L.append("")

    for window, wl in [("train","训练窗(震荡市)"),("test","测试窗(牛市)")]:
        L.append(f"## {wl}")
        L.append("")
        L.append("### 分组明细")
        L.append("")
        L.append("| 分组 | 版本 | 夏普 | 收益% | Alpha% | 交易数 |")
        L.append("|------|------|------|-------|--------|--------|")
        for version in ["baseline", "p0"]:
            groups = output["results"][version][window]["groups"]
            for g, r in groups.items():
                if "error" in r: continue
                ver = "基线" if version == "baseline" else "P0"
                L.append(f"| {g} | {ver} | {r['sharpe']:+.3f} | "
                         f"{r['total_return']*100:+.1f} | {r['alpha']*100:+.1f} | {r['trade_count']} |")
        L.append("")
        L.append("### 组合级指标 (加权合并)")
        L.append("")
        L.append("| 版本 | 夏普 | 收益% | Alpha% | 基准% | 回撤% | 交易数 |")
        L.append("|------|------|-------|--------|-------|-------|--------|")
        for version, ver in [("baseline","基线"),("p0","P0")]:
            p = output["results"][version][window]["portfolio"]
            if "error" in p: continue
            L.append(f"| {ver} | {p['sharpe']:+.3f} | {p['total_return_pct']:+.1f} | "
                     f"{p['alpha_pct']:+.1f} | {p['benchmark_return_pct']:+.1f} | "
                     f"{p['max_drawdown_pct']:.1f} | {p['trade_count']} |")
        L.append("")

    # 变化
    L.append("## P0 vs 基线 变化")
    L.append("")
    L.append("| 窗口 | Δ夏普 | Δ收益% | ΔAlpha% | Δ回撤% | Δ交易 |")
    L.append("|------|-------|--------|---------|--------|-------|")
    for window, wl in [("train","训练"),("test","测试")]:
        b = output["results"]["baseline"][window]["portfolio"]
        p = output["results"]["p0"][window]["portfolio"]
        if "error" in b or "error" in p: continue
        ds = round(p["sharpe"] - b["sharpe"], 3)
        dr = round(p["total_return_pct"] - b["total_return_pct"], 1)
        da = round(p["alpha_pct"] - b["alpha_pct"], 1)
        dd = round(p["max_drawdown_pct"] - b["max_drawdown_pct"], 1)
        dt = p["trade_count"] - b["trade_count"]
        L.append(f"| {wl} | {ds:+.3f} | {dr:+.1f} | {da:+.1f} | {dd:+.1f} | {dt} |")
    L.append("")

    # 结论
    L.append("## 结论")
    L.append("")
    bt = output["results"]["baseline"]["test"]["portfolio"]
    pt = output["results"]["p0"]["test"]["portfolio"]
    btr = output["results"]["baseline"]["train"]["portfolio"]
    ptr = output["results"]["p0"]["train"]["portfolio"]
    if "error" not in bt and "error" not in pt:
        test_sharpe_up = pt["sharpe"] > bt["sharpe"]
        test_alpha_up = pt["alpha_pct"] > bt["alpha_pct"]
        test_dd_down = pt["max_drawdown_pct"] > bt["max_drawdown_pct"]  # 回撤负值, >表示更小绝对值
        L.append(f"**测试窗(牛市)**: 夏普{'↑' if test_sharpe_up else '↓'} "
                 f"({bt['sharpe']:.3f}→{pt['sharpe']:.3f}), "
                 f"Alpha{'↑' if test_alpha_up else '↓'} "
                 f"({bt['alpha_pct']:+.1f}%→{pt['alpha_pct']:+.1f}%), "
                 f"回撤{'↓' if test_dd_down else '↑'}")
        L.append("")
    if "error" not in btr and "error" not in ptr:
        train_alpha_up = ptr["alpha_pct"] > btr["alpha_pct"]
        L.append(f"**训练窗(震荡市)**: Alpha{'↑' if train_alpha_up else '↓'} "
                 f"({btr['alpha_pct']:+.1f}%→{ptr['alpha_pct']:+.1f}%)")
        L.append("")

    L.append("**落地建议**:")
    if pt.get("sharpe", 0) > bt.get("sharpe", 0) and test_dd_down:
        L.append("- ✅ P0组合在测试窗夏普提升且回撤缩小, 方向正确")
        L.append("- 建议实盘采用: 周期组trending过滤 + 消费组降权")
    else:
        L.append("- ⚠️ P0效果需进一步评估, 可能需要调整权重或过滤策略")

    return "\n".join(L)


if __name__ == "__main__":
    main()
