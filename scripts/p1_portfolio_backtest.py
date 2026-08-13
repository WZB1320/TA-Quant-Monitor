"""
P1 组合回测 — 在P0基础上: 科技组止损收紧ATR1.8 + 医药组暂停

P0(已完成): 周期trending过滤 + 消费降权10% → 测试窗Alpha +2.4%→+20.5%
P1(本次)增量:
  - 科技组: atr_stop_mult 2.0→1.8 (收紧止损, 目标回撤-8.2%→-6%)
  - 医药组: 权重15%→0% (暂停, 两期Alpha都为负, 止血)
  - 释放的15%分给科技+周期各7.5% (正Alpha贡献者)

三版本对比:
  baseline: 等权20%, 默认参数
  P0:       周期+trending, 消费10%
  P1:       P0 + 科技ATR1.8, 医药0%, 科技37.5%/周期37.5%

用法:
  python scripts/p1_portfolio_backtest.py
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

# 三版本配置
VERSIONS = {
    "baseline": {
        "weights": {"科技成长型": 0.20, "消费稳健型": 0.20, "周期资源型": 0.20,
                     "医药创新型": 0.20, "机械制造型": 0.20},
        "regimes": {},
        "atr_override": {},
    },
    "p0": {
        "weights": {"科技成长型": 0.30, "消费稳健型": 0.10, "周期资源型": 0.30,
                     "医药创新型": 0.15, "机械制造型": 0.15},
        "regimes": {"周期资源型": {"trending"}},
        "atr_override": {},
    },
    "p1": {
        "weights": {"科技成长型": 0.375, "消费稳健型": 0.10, "周期资源型": 0.375,
                     "医药创新型": 0.0, "机械制造型": 0.15},
        "regimes": {"周期资源型": {"trending"}},
        "atr_override": {"科技成长型": 1.8},  # ATR 2.0→1.8
    },
}

RESULT_JSON = os.path.join(project_root, "data", "p1_portfolio_result.json")
REPORT_MD = os.path.join(project_root, "data", "p1_portfolio_report.md")


def load_watchlist():
    cfg_path = os.path.join(project_root, "config", "strategy_config.json")
    with open(cfg_path, "r", encoding="utf-8") as f:
        cfg = json.load(f)
    return {g: [s["code"] for s in stocks]
            for g, stocks in cfg["strategy_config"]["watchlist"].items()
            if not g.startswith("_")}


def run_group(data_map, benchmark_df, group_codes, group_capital,
              trade_regimes, atr_mult, start, end):
    """跑单分组回测, 返回metrics + daily_values"""
    if group_capital < 1000:  # 权重0%的组跳过
        return {"sharpe": 0, "total_return": 0, "alpha": 0, "max_drawdown": 0,
                "trade_count": 0, "win_rate": 0, "final_value": group_capital,
                "daily_values": None, "skipped": True}
    GroupConfig._instance = None
    GroupConfig._config = None
    try:
        engine = BacktestEngine(
            initial_capital=group_capital, lookback_days=120, position_ratio=0.3,
            commission_rate=0.00025, stamp_tax=0.001, slippage=0.0001,
            signal_dedup_days=5, risk_per_trade=0.05, atr_stop_mult=atr_mult,
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
    portfolio_nav = None
    total_trades = 0
    for g, r in group_results.items():
        if "error" in r or r.get("daily_values") is None:
            continue
        nav = r["daily_values"]
        if portfolio_nav is None:
            portfolio_nav = nav.copy()
        else:
            portfolio_nav = portfolio_nav.add(nav, fill_value=0)
        total_trades += r["trade_count"]

    if portfolio_nav is None or len(portfolio_nav) < 10:
        return {"error": "无有效净值数据"}

    start_ts = pd.Timestamp(start)
    end_ts = pd.Timestamp(end)
    nav_idx = pd.to_datetime(portfolio_nav.index)
    portfolio_nav = pd.Series(portfolio_nav.values, index=nav_idx)
    mask = (portfolio_nav.index >= start_ts) & (portfolio_nav.index <= end_ts)
    portfolio_nav = portfolio_nav[mask]
    total_capital = sum(weights[g] * TOTAL_CAPITAL for g in group_results
                        if "error" not in group_results[g] and not group_results[g].get("skipped"))

    daily_returns = portfolio_nav.pct_change().dropna()
    total_return = (portfolio_nav.iloc[-1] / total_capital) - 1
    sharpe = (daily_returns.mean() / daily_returns.std() * np.sqrt(252)
              if daily_returns.std() > 0 else 0.0)
    cummax = portfolio_nav.cummax()
    drawdown = (portfolio_nav - cummax) / cummax
    max_drawdown = drawdown.min()

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
    print("  P1 组合回测 — 科技ATR1.8 + 医药暂停 (在P0基础上)")
    print(f"  训练:{TRAIN_START}~{TRAIN_END} | 测试:{TEST_START}~{TEST_END}")
    print(f"  P1权重:{VERSIONS['p1']['weights']}")
    print(f"  P1科技ATR:{VERSIONS['p1']['atr_override']}")
    print("=" * 70)

    pref_existed = os.path.exists(_USER_PREF_FILE)
    pref_backup = _USER_PREF_FILE + ".p1_bak"
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

        for version, config in VERSIONS.items():
            print(f"\n{'='*60}\n  版本: {version.upper()}\n{'='*60}")
            weights = config["weights"]
            regimes_cfg = config["regimes"]
            atr_cfg = config["atr_override"]
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
                    regimes = regimes_cfg.get(group_name)
                    atr_mult = atr_cfg.get(group_name, 2.0)
                    r = run_group(data_map, benchmark_df, group_codes, capital,
                                  regimes, atr_mult, start, end)
                    if "error" not in r and not r.get("skipped"):
                        print(f"    {group_name:12s}: 夏普={r['sharpe']:+.3f} "
                              f"收益={r['total_return']*100:+.1f}% 交易={r['trade_count']}笔 "
                              f"ATR={atr_mult}")
                    elif r.get("skipped"):
                        print(f"    {group_name:12s}: [跳过-权重0%]")
                    window_groups[group_name] = r

                portfolio = compute_portfolio_metrics(window_groups, weights,
                                                       benchmark_df, start, end)
                if "error" not in portfolio:
                    print(f"    {'组合(加权)':12s}: 夏普={portfolio['sharpe']:+.3f} "
                          f"收益={portfolio['total_return_pct']:+.1f}% "
                          f"Alpha={portfolio['alpha_pct']:+.1f}% "
                          f"回撤={portfolio['max_drawdown_pct']:.1f}% "
                          f"交易={portfolio['trade_count']}笔")
                version_results[window] = {
                    "groups": {g: {k: v for k, v in r.items() if k != "daily_values"}
                               for g, r in window_groups.items()},
                    "portfolio": portfolio,
                }
            all_results[version] = version_results

        # 保存
        os.makedirs(os.path.dirname(RESULT_JSON), exist_ok=True)
        output = {
            "run_time": run_time,
            "config": {"versions": {k: {"weights": v["weights"],
                                         "atr_override": v["atr_override"]}
                                     for k, v in VERSIONS.items()}},
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
        print("  三版本组合级对比")
        print(f"{'='*70}")
        print(f"{'窗口':<6} {'版本':<10} {'夏普':>8} {'收益%':>8} {'Alpha%':>8} {'回撤%':>8} {'交易':>6}")
        for window, wl in [("train","训练"),("test","测试")]:
            for version in ["baseline","p0","p1"]:
                p = all_results[version][window]["portfolio"]
                if "error" in p: continue
                print(f"{wl:<6} {version:<10} {p['sharpe']:>8.3f} {p['total_return_pct']:>8.1f} "
                      f"{p['alpha_pct']:>8.1f} {p['max_drawdown_pct']:>8.1f} {p['trade_count']:>6}")

    finally:
        if pref_existed and os.path.exists(pref_backup):
            shutil.move(pref_backup, _USER_PREF_FILE)


def generate_report(output, run_time):
    L = []
    L.append("# P1 组合回测报告 — 科技ATR1.8 + 医药暂停")
    L.append("")
    L.append(f"**运行时间**: {run_time}")
    L.append("")
    L.append("**三版本配置**:")
    L.append("- baseline: 等权20%, 默认参数")
    L.append("- P0: 周期+trending过滤, 消费降权10%")
    L.append("- P1: P0 + 科技ATR1.8, 医药暂停(0%), 科技37.5%/周期37.5%")
    L.append("")

    for window, wl in [("train","训练窗(震荡市)"),("test","测试窗(牛市)")]:
        L.append(f"## {wl}")
        L.append("")
        L.append("### 分组明细")
        L.append("")
        L.append("| 分组 | 版本 | 夏普 | 收益% | Alpha% | 交易数 |")
        L.append("|------|------|------|-------|--------|--------|")
        for version in ["baseline", "p0", "p1"]:
            groups = output["results"][version][window]["groups"]
            for g, r in groups.items():
                if "error" in r: continue
                if r.get("skipped"):
                    L.append(f"| {g} | {version} | — | — | — | [暂停] |")
                    continue
                L.append(f"| {g} | {version} | {r['sharpe']:+.3f} | "
                         f"{r['total_return']*100:+.1f} | {r['alpha']*100:+.1f} | {r['trade_count']} |")
        L.append("")
        L.append("### 组合级指标")
        L.append("")
        L.append("| 版本 | 夏普 | 收益% | Alpha% | 基准% | 回撤% | 交易数 |")
        L.append("|------|------|-------|--------|-------|-------|--------|")
        for version, ver in [("baseline","baseline"),("p0","P0"),("p1","P1")]:
            p = output["results"][version][window]["portfolio"]
            if "error" in p: continue
            L.append(f"| {ver} | {p['sharpe']:+.3f} | {p['total_return_pct']:+.1f} | "
                     f"{p['alpha_pct']:+.1f} | {p['benchmark_return_pct']:+.1f} | "
                     f"{p['max_drawdown_pct']:.1f} | {p['trade_count']} |")
        L.append("")

    # 变化对比
    L.append("## P1 vs P0 vs baseline 变化")
    L.append("")
    L.append("| 窗口 | baseline→P0 ΔAlpha | P0→P1 ΔAlpha | baseline→P1 ΔAlpha | P1回撤% |")
    L.append("|------|-------------------|--------------|-------------------|---------|")
    for window, wl in [("train","训练"),("test","测试")]:
        b = output["results"]["baseline"][window]["portfolio"]
        p0 = output["results"]["p0"][window]["portfolio"]
        p1 = output["results"]["p1"][window]["portfolio"]
        if "error" in b or "error" in p0 or "error" in p1: continue
        d_b_p0 = round(p0["alpha_pct"] - b["alpha_pct"], 1)
        d_p0_p1 = round(p1["alpha_pct"] - p0["alpha_pct"], 1)
        d_b_p1 = round(p1["alpha_pct"] - b["alpha_pct"], 1)
        L.append(f"| {wl} | {d_b_p0:+.1f} | {d_p0_p1:+.1f} | {d_b_p1:+.1f} | {p1['max_drawdown_pct']:.1f} |")
    L.append("")

    # 结论
    L.append("## 结论")
    L.append("")
    bt = output["results"]["baseline"]["test"]["portfolio"]
    p0t = output["results"]["p0"]["test"]["portfolio"]
    p1t = output["results"]["p1"]["test"]["portfolio"]
    p1tr = output["results"]["p1"]["train"]["portfolio"]

    if "error" not in p1t:
        L.append(f"**测试窗(牛市) Alpha 演进**: baseline {bt['alpha_pct']:+.1f}% → "
                 f"P0 {p0t['alpha_pct']:+.1f}% → P1 {p1t['alpha_pct']:+.1f}%")
        L.append("")
        L.append(f"**测试窗(牛市) 夏普 演进**: {bt['sharpe']:.3f} → {p0t['sharpe']:.3f} → {p1t['sharpe']:.3f}")
        L.append("")
        L.append(f"**测试窗(牛市) 回撤 演进**: {bt['max_drawdown_pct']:.1f}% → "
                 f"{p0t['max_drawdown_pct']:.1f}% → {p1t['max_drawdown_pct']:.1f}%")
        L.append("")

    if "error" not in p1tr:
        L.append(f"**训练窗(震荡市) Alpha**: P1 {p1tr['alpha_pct']:+.1f}% (夏普 {p1tr['sharpe']:.3f})")
        L.append("")

    # 评估
    L.append("**稳定-收益-回撤 三角评估 (P1)**:")
    L.append("")
    if "error" not in p1t and "error" not in p1tr:
        test_ok = p1t["sharpe"] > 1.0 and p1t["alpha_pct"] > 0 and p1t["max_drawdown_pct"] > -10
        train_ok = p1tr["sharpe"] > 1.0
        L.append(f"| 维度 | 标准 | 测试窗 | 训练窗 |")
        L.append(f"|------|------|--------|--------|")
        L.append(f"| 稳定 | 夏普>1.0 | {p1t['sharpe']:.3f}{'✅' if p1t['sharpe']>1 else '❌'} | "
                 f"{p1tr['sharpe']:.3f}{'✅' if p1tr['sharpe']>1 else '⚠️'} |")
        L.append(f"| 收益 | Alpha>0 | {p1t['alpha_pct']:+.1f}%{'✅' if p1t['alpha_pct']>0 else '❌'} | "
                 f"{p1tr['alpha_pct']:+.1f}%{'❌' if p1tr['alpha_pct']<0 else '✅'} |")
        L.append(f"| 回撤 | <10% | {p1t['max_drawdown_pct']:.1f}%{'✅' if p1t['max_drawdown_pct']>-10 else '❌'} | "
                 f"{p1tr['max_drawdown_pct']:.1f}%{'✅' if p1tr['max_drawdown_pct']>-10 else '❌'} |")
        L.append("")

    return "\n".join(L)


if __name__ == "__main__":
    main()
