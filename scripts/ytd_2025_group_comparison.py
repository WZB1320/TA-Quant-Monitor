"""2025年以来全组回测对比 — P3配置, 2025-01-01~2026-06-30"""
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
DATA_START = "2024-08-01"
DATA_END = "2026-07-13"
BENCHMARK = "sh.000300"
TOTAL_CAPITAL = 1000000

WEIGHTS = {"科技成长型": 0.40, "消费稳健型": 0.10, "周期资源型": 0.425,
           "医药创新型": 0.0, "机械制造型": 0.0}
REGIMES_CFG = {"周期资源型": {"trending"}}
ATR_OVERRIDE = {"科技成长型": 1.8}
DD_THRESHOLD = -0.08
DD_RECOVERY = -0.04
DD_REDUCED_RATIO = 0.5

def load_watchlist():
    cfg = json.load(open(os.path.join(project_root, "config", "strategy_config.json"), "r", encoding="utf-8"))
    return {g: [s["code"] for s in stocks]
            for g, stocks in cfg["strategy_config"]["watchlist"].items()
            if not g.startswith("_")}

def load_names():
    cfg = json.load(open(os.path.join(project_root, "config", "strategy_config.json"), "r", encoding="utf-8"))
    names = {}
    for g, stocks in cfg["strategy_config"]["watchlist"].items():
        if g.startswith("_"):
            continue
        if not isinstance(stocks, list):
            continue
        for s in stocks: names[s["code"]] = s["name"]
    return names

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

def apply_dd_protection(nav):
    cummax = nav.cummax()
    dd = (nav - cummax) / cummax
    in_prot = False
    prot_nav = [nav.iloc[0]]
    p_days = 0; p_trigs = 0
    for i in range(1, len(nav)):
        if dd.iloc[i] < DD_THRESHOLD and not in_prot:
            in_prot = True; p_trigs += 1
        elif dd.iloc[i] > DD_RECOVERY and in_prot:
            in_prot = False
        r = nav.iloc[i] / nav.iloc[i-1] - 1
        a = r * DD_REDUCED_RATIO if in_prot else r
        prot_nav.append(prot_nav[-1] * (1 + a))
        if in_prot: p_days += 1
    return pd.Series(prot_nav, index=nav.index), p_days, p_trigs

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
    print("=" * 90)
    print(f"  2025年以来全组回测对比 — P3配置 ({START}~{END})")
    print("=" * 90)

    pref_existed = os.path.exists(_USER_PREF_FILE)
    if pref_existed: shutil.copy2(_USER_PREF_FILE, _USER_PREF_FILE + ".bak")
    UserPreferences().clear_all()
    try:
        watchlist = load_watchlist()
        names = load_names()
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

        # 各组回测
        group_results = {}
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
            group_results[g] = {
                "sharpe": round(getattr(m, "sharpe_ratio", 0) or 0, 3),
                "return_pct": round(getattr(m, "total_return", 0)*100, 2),
                "alpha_pct": round(getattr(m, "alpha", 0)*100, 2),
                "max_dd_pct": round(getattr(m, "max_drawdown", 0)*100, 2),
                "trades": getattr(m, "trade_count", 0),
                "win_rate": round(getattr(m, "win_rate", 0)*100, 1),
            }

        # 等权持有基准 (各组自选股等权买入持有)
        print("\n计算等权持有基准...")
        eq_hold_navs = {}
        for g, codes in watchlist.items():
            if g not in WEIGHTS or WEIGHTS[g] == 0: continue
            g_codes = [c for c in codes if c in data_map]
            navs = []
            for c in g_codes:
                df = data_map[c].copy()
                if "date" not in df.columns: df = df.reset_index()
                df["date"] = pd.to_datetime(df["date"])
                s = df.set_index("date")["close"].astype(float)
                s = s[(s.index >= pd.Timestamp(START)) & (s.index <= pd.Timestamp(END))]
                if len(s) < 10: continue
                s = s / s.iloc[0]
                navs.append(s)
            if navs:
                eq = pd.concat(navs, axis=1).mean(axis=1)
                eq_hold_navs[g] = eq

        # 组合净值
        port_nav = None
        for g, nav in group_navs.items():
            port_nav = nav.copy() if port_nav is None else port_nav.add(nav, fill_value=0)
        invested = sum(w * TOTAL_CAPITAL for g, w in WEIGHTS.items() if w > 0)
        cash = TOTAL_CAPITAL - invested
        port_nav = port_nav + cash
        # 回撤保护
        port_nav_prot, p_days, p_trigs = apply_dd_protection(port_nav)
        port_m = metrics(port_nav_prot, bench_nav, TOTAL_CAPITAL)

        # 等权持有组合净值
        eq_port_nav = None
        for g, eq in eq_hold_navs.items():
            scaled = eq * (WEIGHTS[g] * TOTAL_CAPITAL)
            eq_port_nav = scaled.copy() if eq_port_nav is None else eq_port_nav.add(scaled, fill_value=0)
        eq_port_nav = eq_port_nav + cash
        eq_port_m = metrics(eq_port_nav, bench_nav, TOTAL_CAPITAL)

        # ── 输出 ──
        print(f"\n{'='*90}")
        print(f"  回测结果 ({START}~{END})")
        print(f"{'='*90}")

        print(f"\n{'类别':<16} {'收益%':>8} {'Alpha%':>8} {'夏普':>8} {'回撤%':>8} {'交易数':>6} {'胜率%':>6}")
        print("-" * 70)
        print(f"{'沪深300(基准)':<16} {bench_m['return_pct']:>+8.2f} {'—':>8} {'—':>8} {'—':>8} {'—':>6} {'—':>6}")
        for g, r in group_results.items():
            print(f"{g:<16} {r['return_pct']:>+8.2f} {r['alpha_pct']:>+8.2f} {r['sharpe']:>8.3f} {r['max_dd_pct']:>8.1f} {r['trades']:>6} {r['win_rate']:>6.1f}")
        print("-" * 70)
        print(f"{'P3组合(策略)':<16} {port_m['return_pct']:>+8.2f} {port_m['alpha_pct']:>+8.2f} {port_m['sharpe']:>8.3f} {port_m['max_dd_pct']:>8.1f} {'—':>6} {'—':>6}")
        print(f"{'等权持有(对照)':<16} {eq_port_m['return_pct']:>+8.2f} {eq_port_m['alpha_pct']:>+8.2f} {eq_port_m['sharpe']:>8.3f} {eq_port_m['max_dd_pct']:>8.1f} {'—':>6} {'—':>6}")

        print(f"\n  回撤保护: {p_trigs}次触发, {p_days}天降仓")

        # 各组 策略 vs 等权持有 对比
        print(f"\n{'='*90}")
        print(f"  各组: 策略 vs 等权持有")
        print(f"{'='*90}")
        print(f"{'分组':<16} {'策略收益%':>10} {'持有收益%':>10} {'差值%':>8} {'策略胜?':>8}")
        print("-" * 60)
        for g in group_results:
            s_ret = group_results[g]["return_pct"]
            h_ret = metrics(eq_hold_navs[g] * (WEIGHTS[g]*TOTAL_CAPITAL), bench_nav, WEIGHTS[g]*TOTAL_CAPITAL)["return_pct"]
            diff = s_ret - h_ret
            win = "✅" if diff > 0 else "❌"
            print(f"{g:<16} {s_ret:>+10.2f} {h_ret:>+10.2f} {diff:>+8.2f} {win:>8}")

        # 报告
        report = f"""# 2025年以来全组回测对比报告

**运行时间**: {run_time}
**窗口**: {START}~{END}
**配置**: P3 (trail_mult[2.0/1.5/1.0], hard_stop 0.12, 震荡市禁用trailing)

## 组合级对比

| 类别 | 收益% | Alpha% | 夏普 | 回撤% |
|------|-------|--------|------|-------|
| 沪深300(基准) | {bench_m['return_pct']:+.2f} | — | — | — |
| P3组合(策略) | {port_m['return_pct']:+.2f} | {port_m['alpha_pct']:+.2f} | {port_m['sharpe']:.3f} | {port_m['max_dd_pct']:.1f} |
| 等权持有(对照) | {eq_port_m['return_pct']:+.2f} | {eq_port_m['alpha_pct']:+.2f} | {eq_port_m['sharpe']:.3f} | {eq_port_m['max_dd_pct']:.1f} |

## 各组明细

| 分组 | 权重% | 策略收益% | 策略Alpha% | 夏普 | 回撤% | 交易数 | 胜率% |
|------|-------|----------|-----------|------|-------|--------|-------|
"""
        for g, r in group_results.items():
            report += f"| {g} | {WEIGHTS[g]*100:.0f} | {r['return_pct']:+.2f} | {r['alpha_pct']:+.2f} | {r['sharpe']:.3f} | {r['max_dd_pct']:.1f} | {r['trades']} | {r['win_rate']:.1f} |\n"

        report += f"""
## 各组: 策略 vs 等权持有

| 分组 | 策略收益% | 持有收益% | 差值% | 策略胜? |
|------|----------|----------|-------|---------|
"""
        for g in group_results:
            s_ret = group_results[g]["return_pct"]
            h_ret = metrics(eq_hold_navs[g] * (WEIGHTS[g]*TOTAL_CAPITAL), bench_nav, WEIGHTS[g]*TOTAL_CAPITAL)["return_pct"]
            diff = s_ret - h_ret
            win = "✅" if diff > 0 else "❌"
            report += f"| {g} | {s_ret:+.2f} | {h_ret:+.2f} | {diff:+.2f} | {win} |\n"

        report += f"\n**回撤保护**: {p_trigs}次触发, {p_days}天降仓\n"

        report_path = os.path.join(project_root, "data", "ytd_2025_group_comparison_report.md")
        with open(report_path, "w", encoding="utf-8") as f:
            f.write(report)
        print(f"\n✓ 报告 → {report_path}")

    finally:
        if pref_existed and os.path.exists(_USER_PREF_FILE + ".bak"):
            shutil.move(_USER_PREF_FILE + ".bak", _USER_PREF_FILE)

if __name__ == "__main__":
    main()
