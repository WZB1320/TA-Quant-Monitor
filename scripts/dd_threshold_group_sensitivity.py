"""组级回撤保护阈值敏感性扫描 — 找最优平衡点

三场景 × 五档阈值:
  场景: 训练窗(2024-07~2025-06, 震荡市) / 测试窗(2025-07~2026-06, 牛市) / 2022熊市
  阈值: 无保护 / 8% / 10% / 12% / 15% (recovery=阈值/2, 降仓50%)

最优判断标准:
  1. 2022熊市: 能触发并减损 (回撤改善) — 证明保护起作用
  2. 训练窗: Alpha保持正 (不过度保护) — 证明温和行情不踏空
  3. 测试窗: Alpha接近无保护 (少踏空)

用法: python scripts/dd_threshold_group_sensitivity.py
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
BEAR_START, BEAR_END = "2022-01-01", "2022-12-31"
BEAR_DATA_START, BEAR_DATA_END = "2021-09-01", "2022-12-31"
BENCHMARK = "sh.000300"
TOTAL_CAPITAL = 1000000

WEIGHTS = {"科技成长型": 0.40, "消费稳健型": 0.10, "周期资源型": 0.425,
           "医药创新型": 0.0, "机械制造型": 0.0}
REGIMES_CFG = {"周期资源型": {"trending"}}
ATR_OVERRIDE = {"科技成长型": 1.8}

# 五档阈值 (threshold, recovery, label)
DD_CONFIGS = [
    ("无保护", None),
    ("8%", {"threshold": -0.08, "recovery": -0.04, "reduced_ratio": 0.5}),
    ("10%", {"threshold": -0.10, "recovery": -0.05, "reduced_ratio": 0.5}),
    ("12%", {"threshold": -0.12, "recovery": -0.06, "reduced_ratio": 0.5}),
    ("15%", {"threshold": -0.15, "recovery": -0.075, "reduced_ratio": 0.5}),
]

REPORT_MD = os.path.join(project_root, "data", "dd_threshold_group_sensitivity_report.md")


def load_watchlist():
    cfg_path = os.path.join(project_root, "config", "strategy_config.json")
    with open(cfg_path, "r", encoding="utf-8") as f:
        cfg = json.load(f)
    return {g: [s["code"] for s in stocks]
            for g, stocks in cfg["strategy_config"]["watchlist"].items()
            if not g.startswith("_")}


def run_group(data_map, benchmark_df, group_codes, group_capital,
              trade_regimes, atr_mult, start, end, dd_config):
    if group_capital < 1000:
        return {"skipped": True, "daily_values": None, "trade_count": 0,
                "dd_triggers": 0, "dd_reduce_trades": 0}
    GroupConfig._instance = None
    GroupConfig._config = None
    engine = BacktestEngine(
        initial_capital=group_capital, lookback_days=120, position_ratio=0.3,
        commission_rate=0.00025, stamp_tax=0.001, slippage=0.0001,
        signal_dedup_days=5, risk_per_trade=0.05, atr_stop_mult=atr_mult,
        forced_regime=None, trade_regimes=trade_regimes,
        dd_protection_config=dd_config,
    )
    sub_map = {c: data_map[c] for c in group_codes if c in data_map}
    m = engine.run(sub_map, benchmark_df=benchmark_df, start_date=start, end_date=end)
    dd_reduce_trades = sum(1 for t in engine.position_mgr.closed_trades
                           if "回撤保护" in (t.exit_signal or ""))
    return {
        "trade_count": getattr(m, "trade_count", 0) or 0,
        "daily_values": engine.daily_values.copy() if engine.daily_values is not None else None,
        "dd_triggers": engine.dd_protection_stats.get("triggers", 0),
        "dd_reduce_trades": dd_reduce_trades,
        "skipped": False,
    }


def compute_portfolio(group_results, benchmark_df, start, end):
    portfolio_nav = None
    total_trades = 0
    total_dd_trigs = 0
    total_dd_reduce_trades = 0
    for g, r in group_results.items():
        if r.get("skipped") or r.get("daily_values") is None:
            continue
        nav = r["daily_values"]
        portfolio_nav = nav if portfolio_nav is None else portfolio_nav.add(nav, fill_value=0)
        total_trades += r["trade_count"]
        total_dd_trigs += r["dd_triggers"]
        total_dd_reduce_trades += r["dd_reduce_trades"]
    if portfolio_nav is None or len(portfolio_nav) < 10:
        return {"error": "无有效净值"}

    start_ts = pd.Timestamp(start)
    end_ts = pd.Timestamp(end)
    nav_idx = pd.to_datetime(portfolio_nav.index)
    portfolio_nav = pd.Series(portfolio_nav.values, index=nav_idx)
    mask = (portfolio_nav.index >= start_ts) & (portfolio_nav.index <= end_ts)
    portfolio_nav = portfolio_nav[mask]
    invested = sum(w * TOTAL_CAPITAL for g, w in WEIGHTS.items() if w > 0)
    cash = TOTAL_CAPITAL - invested
    portfolio_nav = portfolio_nav + cash

    daily_ret = portfolio_nav.pct_change().dropna()
    total_return = (portfolio_nav.iloc[-1] / TOTAL_CAPITAL) - 1
    sharpe = (daily_ret.mean() / daily_ret.std() * np.sqrt(252)
              if daily_ret.std() > 0 else 0.0)
    cummax = portfolio_nav.cummax()
    max_dd = ((portfolio_nav - cummax) / cummax).min()

    bench = benchmark_df.copy()
    if "date" not in bench.columns:
        bench = bench.reset_index()
    bench["date"] = pd.to_datetime(bench["date"])
    bench_s = bench.set_index("date")["close"].astype(float)
    bench_s = bench_s[(bench_s.index >= start_ts) & (bench_s.index <= end_ts)]
    bench_ret = (bench_s.iloc[-1] / bench_s.iloc[0]) - 1 if len(bench_s) > 0 else 0
    alpha = total_return - bench_ret

    return {
        "sharpe": round(sharpe, 3),
        "total_return_pct": round(total_return * 100, 2),
        "alpha_pct": round(alpha * 100, 2),
        "benchmark_return_pct": round(bench_ret * 100, 2),
        "max_drawdown_pct": round(max_dd * 100, 2),
        "trade_count": total_trades,
        "dd_triggers": total_dd_trigs,
        "dd_reduce_trades": total_dd_reduce_trades,
    }


def run_window(data_map, benchmark_df, watchlist, start, end, dd_config):
    group_results = {}
    for g, codes in watchlist.items():
        if g not in WEIGHTS or WEIGHTS[g] == 0:
            group_results[g] = {"skipped": True, "daily_values": None,
                                "trade_count": 0, "dd_triggers": 0, "dd_reduce_trades": 0}
            continue
        g_codes = [c for c in codes if c in data_map]
        if len(g_codes) < 2:
            continue
        capital = TOTAL_CAPITAL * WEIGHTS[g]
        regimes = REGIMES_CFG.get(g)
        atr_mult = ATR_OVERRIDE.get(g, 2.0)
        r = run_group(data_map, benchmark_df, g_codes, capital,
                      regimes, atr_mult, start, end, dd_config)
        group_results[g] = r
    return compute_portfolio(group_results, benchmark_df, start, end)


def main():
    run_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print("=" * 95)
    print("  组级回撤保护阈值敏感性扫描 (8%/10%/12%/15%)")
    print("  三场景: 训练窗 / 测试窗 / 2022熊市")
    print("=" * 95)

    pref_existed = os.path.exists(_USER_PREF_FILE)
    pref_backup = _USER_PREF_FILE + ".ddgs_bak"
    if pref_existed:
        shutil.copy2(_USER_PREF_FILE, pref_backup)
    UserPreferences().clear_all()

    try:
        watchlist = load_watchlist()
        dm = DataManager()

        # 标准窗口数据
        print("\n拉取标准窗口数据...")
        all_codes = [c for codes in watchlist.values() for c in codes]
        data_map = {}
        for code in all_codes:
            df = dm.get_daily_kline(code, start_date=DATA_START, end_date=DATA_END)
            if df is not None and len(df) > 80:
                data_map[code] = df
        benchmark_df = dm.get_daily_kline(BENCHMARK, start_date=DATA_START, end_date=DATA_END)
        print(f"  股票 {len(data_map)}/{len(all_codes)}, 基准 {len(benchmark_df)}条")

        # 2022熊市数据
        print("拉取2022熊市数据...")
        bear_map = {}
        for code in all_codes:
            df = dm.get_daily_kline(code, start_date=BEAR_DATA_START, end_date=BEAR_DATA_END)
            if df is not None and len(df) > 80:
                bear_map[code] = df
        bear_bench = dm.get_daily_kline(BENCHMARK, start_date=BEAR_DATA_START, end_date=BEAR_DATA_END)
        print(f"  股票 {len(bear_map)}/{len(all_codes)}, 基准 {len(bear_bench)}条")

        # 扫描: 三场景 × 五档
        scenarios = [
            ("训练窗", data_map, benchmark_df, TRAIN_START, TRAIN_END),
            ("测试窗", data_map, benchmark_df, TEST_START, TEST_END),
            ("2022熊市", bear_map, bear_bench, BEAR_START, BEAR_END),
        ]
        results = {}  # {(scenario, dd_label): metrics}
        for sc_name, dmap, bdf, ws, we in scenarios:
            for dd_label, dd_cfg in DD_CONFIGS:
                key = (sc_name, dd_label)
                print(f"  [{sc_name} | {dd_label}] ...", end="", flush=True)
                m = run_window(dmap, bdf, watchlist, ws, we, dd_cfg)
                results[key] = m
                print(f" 收益{m['total_return_pct']:+.2f}% Alpha{m['alpha_pct']:+.2f}% "
                      f"夏普{m['sharpe']:.3f} 回撤{m['max_drawdown_pct']:.1f}% "
                      f"触发{m['dd_triggers']}次 降仓{m['dd_reduce_trades']}笔")

        # ── 输出对比表 ──
        print(f"\n{'='*95}")
        print(f"  扫描结果汇总")
        print(f"{'='*95}")
        print(f"\n{'场景':<10} {'阈值':<8} {'收益%':>8} {'Alpha%':>8} {'夏普':>7} {'回撤%':>7} {'触发':>5} {'降仓笔':>6}")
        print("-" * 70)
        for sc_name, _, _, _, _ in scenarios:
            for dd_label, _ in DD_CONFIGS:
                m = results[(sc_name, dd_label)]
                print(f"{sc_name:<10} {dd_label:<8} {m['total_return_pct']:>+8.2f} {m['alpha_pct']:>+8.2f} "
                      f"{m['sharpe']:>7.3f} {m['max_drawdown_pct']:>7.1f} {m['dd_triggers']:>5} {m['dd_reduce_trades']:>6}")
            print("-" * 70)

        # ── 最优判断 ──
        # 标准: 2022能减损(触发>0且回撤改善) + 训练Alpha>0 + 测试Alpha接近无保护
        none_train = results[("训练窗", "无保护")]
        none_test = results[("测试窗", "无保护")]
        none_bear = results[("2022熊市", "无保护")]

        print(f"\n{'='*95}")
        print(f"  最优平衡点判断")
        print(f"{'='*95}")
        print(f"\n  无保护基线: 训练Alpha{none_train['alpha_pct']:+.2f}% / 测试Alpha{none_test['alpha_pct']:+.2f}% / 2022回撤{none_bear['max_drawdown_pct']:.1f}%")
        print(f"\n{'阈值':<8} {'2022触发':>8} {'2022回撤改善':>12} {'训练Alpha':>10} {'训练达标':>8} {'测试Alpha':>10} {'综合':>6}")
        print("-" * 75)
        best_label = None
        best_score = -999
        for dd_label, _ in DD_CONFIGS:
            if dd_label == "无保护":
                continue
            t = results[("训练窗", dd_label)]
            s = results[("测试窗", dd_label)]
            b = results[("2022熊市", dd_label)]
            bear_dd_improve = b["max_drawdown_pct"] - none_bear["max_drawdown_pct"]  # 正=改善
            train_alpha_ok = t["alpha_pct"] > 0
            bear_triggered = b["dd_triggers"] > 0
            # 综合评分: 2022减损有效(+2) + 训练Alpha正(+1) + 测试Alpha接近无保护(差距小+1)
            test_alpha_cost = none_test["alpha_pct"] - s["alpha_pct"]  # 踏空成本
            score = 0
            if bear_triggered and bear_dd_improve >= 0:
                score += 2
            if train_alpha_ok:
                score += 1
            if test_alpha_cost < 2.0:  # 测试踏空<2%
                score += 1
            ok_t = "✅" if train_alpha_ok else "❌"
            print(f"{dd_label:<8} {b['dd_triggers']:>8} {bear_dd_improve:>+12.1f} {t['alpha_pct']:>+10.2f} {ok_t:>8} {s['alpha_pct']:>+10.2f} {score:>6}")
            if score > best_score:
                best_score = score
                best_label = dd_label

        print(f"\n  → 综合最优: {best_label} (评分{best_score}/4)")

        # 报告
        report = generate_report(run_time, results, none_train, none_test, none_bear, best_label)
        with open(REPORT_MD, "w", encoding="utf-8") as f:
            f.write(report)
        print(f"\n✓ 报告 → {REPORT_MD}")

    finally:
        if pref_existed and os.path.exists(pref_backup):
            shutil.move(pref_backup, _USER_PREF_FILE)


def generate_report(run_time, results, none_train, none_test, none_bear, best_label):
    L = []
    L.append("# 组级回撤保护阈值敏感性扫描报告\n")
    L.append(f"**运行时间**: {run_time}")
    L.append(f"**机制**: 组级真实降仓 (每组engine独立判断本组回撤>threshold → reduce_position真实部分平仓, 单向)")
    L.append(f"**扫描**: 三场景 × 五档阈值(无保护/8/10/12/15%), recovery=阈值/2, 降仓50%\n")

    L.append("## 无保护基线\n")
    L.append(f"- 训练窗: Alpha{none_train['alpha_pct']:+.2f}% / 夏普{none_train['sharpe']:.3f} / 回撤{none_train['max_drawdown_pct']:.1f}%")
    L.append(f"- 测试窗: Alpha{none_test['alpha_pct']:+.2f}% / 夏普{none_test['sharpe']:.3f} / 回撤{none_test['max_drawdown_pct']:.1f}%")
    L.append(f"- 2022熊市: 回撤{none_bear['max_drawdown_pct']:.1f}%\n")

    for sc_name in ["训练窗", "测试窗", "2022熊市"]:
        L.append(f"## {sc_name}\n")
        L.append("| 阈值 | 收益% | Alpha% | 夏普 | 回撤% | 触发次数 | 降仓笔数 |")
        L.append("|------|-------|--------|------|-------|---------|---------|")
        for dd_label, _ in DD_CONFIGS:
            m = results[(sc_name, dd_label)]
            L.append(f"| {dd_label} | {m['total_return_pct']:+.2f} | {m['alpha_pct']:+.2f} | "
                     f"{m['sharpe']:.3f} | {m['max_drawdown_pct']:.1f} | {m['dd_triggers']} | {m['dd_reduce_trades']} |")
        L.append("")

    L.append("## 最优平衡点判断\n")
    L.append("标准: 2022熊市能触发并减损(2分) + 训练窗Alpha>0(1分) + 测试窗踏空<2%(1分), 满分4分\n")
    L.append("| 阈值 | 2022触发 | 2022回撤改善 | 训练Alpha | 训练达标 | 测试Alpha | 综合评分 |")
    L.append("|------|---------|-------------|-----------|---------|-----------|---------|")
    for dd_label, _ in DD_CONFIGS:
        if dd_label == "无保护":
            continue
        t = results[("训练窗", dd_label)]
        s = results[("测试窗", dd_label)]
        b = results[("2022熊市", dd_label)]
        bear_dd_improve = b["max_drawdown_pct"] - none_bear["max_drawdown_pct"]
        train_alpha_ok = t["alpha_pct"] > 0
        bear_triggered = b["dd_triggers"] > 0
        test_alpha_cost = none_test["alpha_pct"] - s["alpha_pct"]
        score = 0
        if bear_triggered and bear_dd_improve >= 0:
            score += 2
        if train_alpha_ok:
            score += 1
        if test_alpha_cost < 2.0:
            score += 1
        ok_t = "✅" if train_alpha_ok else "❌"
        L.append(f"| {dd_label} | {b['dd_triggers']} | {bear_dd_improve:+.1f}% | {t['alpha_pct']:+.2f}% | {ok_t} | {s['alpha_pct']:+.2f}% | {score}/4 |")
    L.append(f"\n**综合最优: {best_label}**\n")
    return "\n".join(L)


if __name__ == "__main__":
    main()
