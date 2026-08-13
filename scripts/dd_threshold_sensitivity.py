"""回撤保护阈值敏感性分析 — 2025年以来 (2025-01-01~2026-06-30)

复用 ytd_2025_group_comparison.py 的各组回测, 只跑一次拿到组合净值,
然后扫描不同回撤保护参数 (threshold/recovery/reduced_ratio) 对收益/夏普/回撤的影响.

回撤保护模型 (与 p2/p3 一致, 事后净值降仓):
  - 正常: 收益 = daily_return
  - 保护 (回撤 < threshold): 收益 = daily_return × reduced_ratio
  - 恢复 (回撤 > recovery): 回到正常

用法:
  python scripts/dd_threshold_sensitivity.py
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
from src.backtest.engine import BacktestEngine
from src.data_fetcher import DataManager
from src.config.group_config import GroupConfig
from src.config.user_preferences import UserPreferences, _USER_PREF_FILE

START, END = "2025-01-01", "2026-06-30"
DATA_START, DATA_END = "2024-08-01", "2026-07-13"
BENCHMARK = "sh.000300"
TOTAL_CAPITAL = 1000000

WEIGHTS = {"科技成长型": 0.40, "消费稳健型": 0.10, "周期资源型": 0.425,
           "医药创新型": 0.0, "机械制造型": 0.0}
REGIMES_CFG = {"周期资源型": {"trending"}}
ATR_OVERRIDE = {"科技成长型": 1.8}


def load_watchlist():
    cfg = json.load(open(os.path.join(project_root, "config", "strategy_config.json"), "r", encoding="utf-8"))
    return {g: [s["code"] for s in stocks]
            for g, stocks in cfg["strategy_config"]["watchlist"].items()
            if not g.startswith("_")}


def run_group(data_map, benchmark_df, codes, capital, trade_regimes, atr_mult):
    if capital < 1000:
        return None
    GroupConfig._instance = None; GroupConfig._config = None
    engine = BacktestEngine(initial_capital=capital, lookback_days=120, position_ratio=0.3,
        commission_rate=0.00025, stamp_tax=0.001, slippage=0.0001,
        signal_dedup_days=5, risk_per_trade=0.05, atr_stop_mult=atr_mult,
        forced_regime=None, trade_regimes=trade_regimes)
    sub = {c: data_map[c] for c in codes if c in data_map}
    m = engine.run(sub, benchmark_df=benchmark_df, start_date=START, end_date=END)
    return engine, m


def apply_protection(portfolio_nav, threshold, recovery, reduced_ratio):
    """组合级回撤保护 (与 p2/p3 一致的事后净值降仓模型)."""
    cummax = portfolio_nav.cummax()
    drawdown = (portfolio_nav - cummax) / cummax
    in_prot = False
    prot_nav = [portfolio_nav.iloc[0]]
    p_days = 0; p_trigs = 0
    for i in range(1, len(portfolio_nav)):
        dd = drawdown.iloc[i]
        if dd < threshold and not in_prot:
            in_prot = True; p_trigs += 1
        elif dd > recovery and in_prot:
            in_prot = False
        r = portfolio_nav.iloc[i] / portfolio_nav.iloc[i-1] - 1
        a = r * reduced_ratio if in_prot else r
        prot_nav.append(prot_nav[-1] * (1 + a))
        if in_prot: p_days += 1
    return pd.Series(prot_nav, index=portfolio_nav.index), p_days, p_trigs


def metrics(nav, bench_nav, capital):
    daily_ret = nav.pct_change().dropna()
    total_ret = (nav.iloc[-1] / capital) - 1
    sharpe = (daily_ret.mean() / daily_ret.std() * np.sqrt(252)) if daily_ret.std() > 0 else 0
    cummax = nav.cummax()
    max_dd = ((nav - cummax) / cummax).min()
    bench_ret = (bench_nav.iloc[-1] / bench_nav.iloc[0]) - 1
    alpha = total_ret - bench_ret
    return {"return_pct": round(total_ret*100, 2), "sharpe": round(sharpe, 3),
            "max_dd_pct": round(max_dd*100, 2), "alpha_pct": round(alpha*100, 2),
            "bench_pct": round(bench_ret*100, 2)}


def main():
    run_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print("=" * 95)
    print(f"  回撤保护阈值敏感性分析 — P3配置 ({START}~{END})")
    print("=" * 95)

    pref_existed = os.path.exists(_USER_PREF_FILE)
    if pref_existed: shutil.copy2(_USER_PREF_FILE, _USER_PREF_FILE + ".bak")
    UserPreferences().clear_all()
    try:
        watchlist = load_watchlist()
        dm = DataManager()
        print("\n拉取数据...")
        all_codes = [c for codes in watchlist.values() for c in codes]
        data_map = {}
        for c in all_codes:
            df = dm.get_daily_kline(c, start_date=DATA_START, end_date=DATA_END)
            if df is not None and len(df) > 80: data_map[c] = df
        bench_df = dm.get_daily_kline(BENCHMARK, start_date=DATA_START, end_date=DATA_END)
        print(f"  股票 {len(data_map)}/{len(all_codes)}, 基准 {len(bench_df)}条")

        # 基准净值
        bench = bench_df.copy()
        if "date" not in bench.columns: bench = bench.reset_index()
        bench["date"] = pd.to_datetime(bench["date"])
        bench_nav = bench.set_index("date")["close"].astype(float)
        bench_nav = bench_nav[(bench_nav.index >= pd.Timestamp(START)) & (bench_nav.index <= pd.Timestamp(END))]
        bench_nav = bench_nav / bench_nav.iloc[0]
        bench_m = metrics(bench_nav * TOTAL_CAPITAL, bench_nav, TOTAL_CAPITAL)

        # 各组回测 (只跑一次)
        print("\n跑各组回测...")
        group_navs = {}
        for g, codes in watchlist.items():
            if g not in WEIGHTS or WEIGHTS[g] == 0:
                print(f"  {g}: [暂停]")
                continue
            g_codes = [c for c in codes if c in data_map]
            if len(g_codes) < 2: continue
            capital = TOTAL_CAPITAL * WEIGHTS[g]
            regimes = REGIMES_CFG.get(g)
            atr = ATR_OVERRIDE.get(g, 2.0)
            engine, m = run_group(data_map, bench_df, g_codes, capital, regimes, atr)
            dv = engine.daily_values.copy()
            dv_idx = pd.to_datetime(dv.index)
            nav = pd.Series(dv.values, index=dv_idx)
            group_navs[g] = nav
            print(f"  {g}: 收益{getattr(m,'total_return',0)*100:+.1f}%")

        # 组合净值 (未保护)
        port_nav_raw = None
        for g, nav in group_navs.items():
            port_nav_raw = nav.copy() if port_nav_raw is None else port_nav_raw.add(nav, fill_value=0)
        invested = sum(w * TOTAL_CAPITAL for g, w in WEIGHTS.items() if w > 0)
        cash = TOTAL_CAPITAL - invested
        port_nav_raw = port_nav_raw + cash

        # ── 扫描配置 ──
        # 1) 阈值扫描 (recovery=threshold/2, ratio=0.5)
        threshold_sweep = [
            ("无保护", None, None, None),
            ("12%", -0.12, -0.06, 0.5),
            ("10%", -0.10, -0.05, 0.5),
            ("8%(基线)", -0.08, -0.04, 0.5),
            ("6%", -0.06, -0.03, 0.5),
            ("5%", -0.05, -0.025, 0.5),
            ("4%", -0.04, -0.02, 0.5),
            ("3%", -0.03, -0.015, 0.5),
        ]
        # 2) 降仓比例扫描 (threshold=-0.06, recovery=-0.03)
        ratio_sweep = [
            ("30%仓", -0.06, -0.03, 0.3),
            ("40%仓", -0.06, -0.03, 0.4),
            ("50%仓(基线)", -0.06, -0.03, 0.5),
            ("60%仓", -0.06, -0.03, 0.6),
            ("70%仓", -0.06, -0.03, 0.7),
        ]

        def run_sweep(name, configs):
            rows = []
            for label, th, rec, ratio in configs:
                if th is None:
                    nav_p = port_nav_raw
                    p_days, p_trigs = 0, 0
                else:
                    nav_p, p_days, p_trigs = apply_protection(port_nav_raw, th, rec, ratio)
                m = metrics(nav_p, bench_nav, TOTAL_CAPITAL)
                rows.append({**m, "label": label, "trigs": p_trigs, "p_days": p_days,
                             "th": th, "rec": rec, "ratio": ratio})
            return rows

        print("\n扫描回撤阈值...")
        th_rows = run_sweep("threshold", threshold_sweep)
        print("扫描降仓比例...")
        rt_rows = run_sweep("ratio", ratio_sweep)

        # ── 输出 ──
        print(f"\n{'='*95}")
        print(f"  扫描结果 (基准沪深300: +{bench_m['bench_pct']:.2f}%)")
        print(f"{'='*95}")

        print(f"\n[A] 回撤阈值影响 (recovery=阈值/2, 降仓至50%)")
        print(f"{'阈值':<14} {'收益%':>8} {'Alpha%':>8} {'夏普':>7} {'回撤%':>7} {'触发':>5} {'降仓天':>6}")
        print("-" * 65)
        for r in th_rows:
            print(f"{r['label']:<14} {r['return_pct']:>+8.2f} {r['alpha_pct']:>+8.2f} {r['sharpe']:>7.3f} {r['max_dd_pct']:>7.1f} {r['trigs']:>5} {r['p_days']:>6}")

        print(f"\n[B] 降仓比例影响 (阈值6%, recovery3%)")
        print(f"{'保护期仓位':<16} {'收益%':>8} {'Alpha%':>8} {'夏普':>7} {'回撤%':>7} {'触发':>5} {'降仓天':>6}")
        print("-" * 70)
        for r in rt_rows:
            print(f"{r['label']:<16} {r['return_pct']:>+8.2f} {r['alpha_pct']:>+8.2f} {r['sharpe']:>7.3f} {r['max_dd_pct']:>7.1f} {r['trigs']:>5} {r['p_days']:>6}")

        # 报告
        report = f"""# 回撤保护阈值敏感性分析报告

**运行时间**: {run_time}
**窗口**: {START}~{END}
**基准沪深300**: +{bench_m['bench_pct']:.2f}%
**未保护组合**: 收益{th_rows[0]['return_pct']:+.2f}% / 夏普{th_rows[0]['sharpe']:.3f} / 回撤{th_rows[0]['max_dd_pct']:.1f}%

## [A] 回撤阈值影响 (recovery=阈值/2, 降仓至50%)

| 阈值 | 收益% | Alpha% | 夏普 | 回撤% | 触发次数 | 降仓天数 |
|------|-------|--------|------|-------|---------|---------|
"""
        for r in th_rows:
            report += f"| {r['label']} | {r['return_pct']:+.2f} | {r['alpha_pct']:+.2f} | {r['sharpe']:.3f} | {r['max_dd_pct']:.1f} | {r['trigs']} | {r['p_days']} |\n"

        report += f"""
## [B] 降仓比例影响 (阈值6%, recovery3%)

| 保护期仓位 | 收益% | Alpha% | 夏普 | 回撤% | 触发次数 | 降仓天数 |
|-----------|-------|--------|------|-------|---------|---------|
"""
        for r in rt_rows:
            report += f"| {r['label']} | {r['return_pct']:+.2f} | {r['alpha_pct']:+.2f} | {r['sharpe']:.3f} | {r['max_dd_pct']:.1f} | {r['trigs']} | {r['p_days']} |\n"

        # 自动结论
        report += "\n## 关键发现\n\n"
        # 阈值影响: 找夏普最高
        best_sharpe = max(th_rows, key=lambda x: x['sharpe'])
        best_return = th_rows[0]  # 无保护收益最高
        best_dd = min(th_rows[1:], key=lambda x: x['max_dd_pct']) if len(th_rows) > 1 else th_rows[0]
        report += f"- **收益最高**: {best_return['label']} (收益{best_return['return_pct']:+.2f}%, 回撤{best_return['max_dd_pct']:.1f}%)\n"
        report += f"- **夏普最高**: {best_sharpe['label']} (夏普{best_sharpe['sharpe']:.3f}, 收益{best_sharpe['return_pct']:+.2f}%, 回撤{best_sharpe['max_dd_pct']:.1f}%)\n"
        report += f"- **回撤最小**: {best_dd['label']} (回撤{best_dd['max_dd_pct']:.1f}%, 收益{best_dd['return_pct']:+.2f}%)\n"

        report_path = os.path.join(project_root, "data", "dd_threshold_sensitivity_report.md")
        with open(report_path, "w", encoding="utf-8") as f:
            f.write(report)
        print(f"\n✓ 报告 → {report_path}")

    finally:
        if pref_existed and os.path.exists(_USER_PREF_FILE + ".bak"):
            shutil.move(_USER_PREF_FILE + ".bak", _USER_PREF_FILE)


if __name__ == "__main__":
    main()
