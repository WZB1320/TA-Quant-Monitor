"""
震荡市买入持有 — hybrid regime 切换可行性验证

根因诊断确认: 训练窗策略+8.27%跑输等权持有+24.20%, 趋势择时在震荡市拖累收益.
本实验验证"震荡市买入持有"方向是否可行:

hybrid 逻辑 (逐日按 regime 切换收益源):
  - trending 日  → 用原P2策略当日收益率
  - 震荡日(ranging/transition) → 用等权持有本组自选股当日收益率

注意: 这是"理想上限"验证 — 假设 regime 已知、无切换成本、持仓瞬时衔接.
  目的: 确认方向可行(Alpha能否转正), 验证通过后再做引擎级真实实现.

对比四条曲线 (训练窗):
  - pure_strategy : 纯P2策略
  - pure_hold     : 纯等权持有 (理论上限)
  - hybrid        : regime切换 (本次验证)
  - 沪深300        : 基准

用法: python scripts/range_hybrid_experiment.py
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
from src.backtest.regime_detector import RegimeDetector
from src.data_fetcher import DataManager
from src.config.group_config import GroupConfig
from src.config.user_preferences import UserPreferences, _USER_PREF_FILE

TRAIN_START, TRAIN_END = "2024-07-01", "2025-06-30"
DATA_START, DATA_END = "2024-02-01", "2026-07-13"
BENCHMARK_CSI300 = "sh.000300"
TOTAL_CAPITAL = 1000000

# P2 配置
WEIGHTS = {"科技成长型": 0.40, "消费稳健型": 0.10, "周期资源型": 0.425,
           "医药创新型": 0.0, "机械制造型": 0.0}
REGIMES_CFG = {"周期资源型": {"trending"}}
ATR_OVERRIDE = {"科技成长型": 1.8}

# hybrid 切换: 哪些 regime 用"持有"
HOLD_REGIMES = {"transition", "ranging"}  # 震荡日买入持有; trending 用策略

REPORT_MD = os.path.join(project_root, "data", "range_hybrid_experiment_report.md")


def load_watchlist():
    cfg_path = os.path.join(project_root, "config", "strategy_config.json")
    with open(cfg_path, "r", encoding="utf-8") as f:
        cfg = json.load(f)
    return {g: [s["code"] for s in stocks]
            for g, stocks in cfg["strategy_config"]["watchlist"].items()
            if not g.startswith("_")}


def compute_regime_series(benchmark_df):
    detector = RegimeDetector()
    df = benchmark_df.copy()
    if "date" not in df.columns:
        df = df.reset_index()
    df["date"] = pd.to_datetime(df["date"]).dt.date
    regimes = {}
    for d in df["date"].tolist():
        regimes[d] = detector.detect(benchmark_df, d)
    return regimes


def build_equal_weight_nav(data_map, codes, start, end):
    """等权买入持有净值 (归一化, 起点=1)."""
    start_ts = pd.Timestamp(start)
    end_ts = pd.Timestamp(end)
    navs = []
    for c in codes:
        if c not in data_map:
            continue
        df = data_map[c].copy()
        if "date" not in df.columns:
            df = df.reset_index()
        df["date"] = pd.to_datetime(df["date"])
        s = df.set_index("date")["close"].astype(float)
        s = s[(s.index >= start_ts) & (s.index <= end_ts)]
        if len(s) < 10:
            continue
        s = s / s.iloc[0]
        navs.append(s)
    if not navs:
        return None
    return pd.concat(navs, axis=1).mean(axis=1)


def run_group_strategy(data_map, benchmark_df, group_codes, group_capital,
                       trade_regimes, atr_mult, start, end):
    """跑P2策略, 返回 daily_values (基于group_capital)."""
    if group_capital < 1000:
        return None
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
    if engine.daily_values is None:
        return None
    dv = engine.daily_values.copy()
    dv_idx = pd.to_datetime(dv.index)
    return pd.Series(dv.values, index=dv_idx)


def build_hybrid_nav(strategy_nav, hold_nav, regime_map, group_capital, hold_regimes,
                     hold_ratio=1.0):
    """逐日按 regime 切换收益源, 构建 hybrid 净值 (基于group_capital).
    - regime in hold_regimes → 用 hold_nav 当日收益率 × hold_ratio (1.0=满仓持有, 0.5=半仓)
    - 否则(trending) → 用 strategy_nav 当日收益率
    hold_ratio<1 时, 震荡日仅 hold_ratio 仓位持有, 余下现金(0收益), 降低回撤.
    """
    # 日收益率
    strat_ret = strategy_nav.pct_change().dropna()
    hold_ret = hold_nav.pct_change().dropna()
    # 共同交易日
    common = strat_ret.index.intersection(hold_ret.index)
    strat_ret = strat_ret.loc[common]
    hold_ret = hold_ret.loc[common]

    # regime 对齐
    regimes = []
    for ts in common:
        d = ts.date() if hasattr(ts, "date") else ts
        regimes.append(regime_map.get(d, "transition"))

    # 逐日切换 (持有日按 hold_ratio 缩放收益, 模拟部分仓位持有+部分现金)
    hybrid_ret = pd.Series(index=common, dtype=float)
    hold_days = 0
    strat_days = 0
    for i in range(len(common)):
        if regimes[i] in hold_regimes:
            hybrid_ret.iloc[i] = hold_ret.iloc[i] * hold_ratio
            hold_days += 1
        else:
            hybrid_ret.iloc[i] = strat_ret.iloc[i]
            strat_days += 1

    # 累乘得净值 (起点=group_capital)
    hybrid_nav = group_capital * (1 + hybrid_ret).cumprod()
    return hybrid_nav, strat_days, hold_days


def metrics_from_nav(nav, total_capital, bench_nav):
    """算收益/夏普/回撤/Alpha."""
    daily_ret = nav.pct_change().dropna()
    total_return = (nav.iloc[-1] / total_capital) - 1 if nav.iloc[0] <= total_capital else (nav.iloc[-1] / nav.iloc[0]) - 1
    sharpe = (daily_ret.mean() / daily_ret.std() * np.sqrt(252)
              if daily_ret.std() > 0 else 0.0)
    cummax = nav.cummax()
    drawdown = (nav - cummax) / cummax
    max_dd = drawdown.min()
    bench_ret = (bench_nav.iloc[-1] / bench_nav.iloc[0]) - 1 if len(bench_nav) > 0 else 0
    alpha = total_return - bench_ret
    return {
        "total_return_pct": round(total_return * 100, 2),
        "sharpe": round(sharpe, 3),
        "max_drawdown_pct": round(max_dd * 100, 2),
        "alpha_pct": round(alpha * 100, 2),
        "benchmark_return_pct": round(bench_ret * 100, 2),
    }


def main():
    run_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print("=" * 70)
    print("  震荡市买入持有 — hybrid regime 切换可行性验证")
    print(f"  训练窗(震荡市): {TRAIN_START}~{TRAIN_END}")
    print(f"  切换规则: trending→P2策略, 震荡({HOLD_REGIMES})→等权持有")
    print("=" * 70)

    pref_existed = os.path.exists(_USER_PREF_FILE)
    pref_backup = _USER_PREF_FILE + ".hyb_bak"
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
        csi300_df = dm.get_daily_kline(BENCHMARK_CSI300, start_date=DATA_START, end_date=DATA_END)
        print(f"  股票 {len(data_map)}/{len(all_codes)}, 沪深300 {len(csi300_df)}条")

        # regime
        print("\n预计算 regime...")
        regime_map = compute_regime_series(csi300_df)
        rc = pd.Series(list(regime_map.values())).value_counts()
        print(f"  全期分布: {dict(rc)}")

        # 沪深300基准净值 (训练窗归一化)
        csi300 = csi300_df.copy()
        if "date" not in csi300.columns:
            csi300 = csi300.reset_index()
        csi300["date"] = pd.to_datetime(csi300["date"])
        csi300_nav = csi300.set_index("date")["close"].astype(float)
        csi300_nav = csi300_nav[(csi300_nav.index >= pd.Timestamp(TRAIN_START)) &
                                (csi300_nav.index <= pd.Timestamp(TRAIN_END))]
        csi300_nav = csi300_nav / csi300_nav.iloc[0]

        # 各组: 策略净值 / 持有净值 (hybrid按ratio扫描时再构建)
        print("\n跑各组策略 + 等权持有...")
        group_strat_nav = {}   # 基于group_capital
        group_hold_nav = {}    # 归一化(起点1)
        group_capitals = {}
        for g, codes in watchlist.items():
            if g not in WEIGHTS or WEIGHTS[g] == 0:
                continue
            g_codes = [c for c in codes if c in data_map]
            if len(g_codes) < 2:
                continue
            capital = TOTAL_CAPITAL * WEIGHTS[g]
            group_capitals[g] = capital
            regimes = REGIMES_CFG.get(g)
            atr_mult = ATR_OVERRIDE.get(g, 2.0)
            # 策略
            strat_nav = run_group_strategy(data_map, csi300_df, g_codes, capital,
                                           regimes, atr_mult, TRAIN_START, TRAIN_END)
            # 持有 (归一化)
            hold_nav = build_equal_weight_nav(data_map, g_codes, TRAIN_START, TRAIN_END)
            if strat_nav is None or hold_nav is None:
                continue
            group_strat_nav[g] = strat_nav
            group_hold_nav[g] = hold_nav
            print(f"  {g:12s}: 策略{strat_nav.iloc[-1]/capital*100-100:+.1f}% "
                  f"持有{hold_nav.iloc[-1]*100-100:+.1f}%")

        # ── 组合层面: 纯策略 / 纯持有 ──
        print("\n" + "=" * 70)
        print("  组合层面对比 (训练窗)")
        print("=" * 70)

        def combine(nav_dict, with_cash=True):
            """加权求和各组净值 + 现金."""
            port = None
            for g, nav in nav_dict.items():
                if port is None:
                    port = nav.copy()
                else:
                    port = port.add(nav, fill_value=0)
            if with_cash:
                invested = sum(group_capitals.values())
                cash = TOTAL_CAPITAL - invested
                port = port + cash
            return port

        port_strat = combine(group_strat_nav)
        port_hold = combine({g: v * group_capitals[g] for g, v in group_hold_nav.items()})
        m_strat = metrics_from_nav(port_strat, TOTAL_CAPITAL, csi300_nav)
        m_hold = metrics_from_nav(port_hold, TOTAL_CAPITAL, csi300_nav)

        # ── hold_ratio 扫描 ──
        HOLD_RATIOS = [1.0, 0.7, 0.5, 0.3]
        print(f"\n震荡日持有仓位比例扫描 (trending日始终用P2策略):")
        print(f"{'持有比例':<10} {'收益%':>8} {'Alpha%':>8} {'夏普':>8} {'回撤%':>8} {'达标?':>8}")
        scan_results = []
        best = None
        for ratio in HOLD_RATIOS:
            grp_hybrid = {}
            switch_stats = {}
            for g in group_strat_nav:
                hybrid_nav, s_days, h_days = build_hybrid_nav(
                    group_strat_nav[g], group_hold_nav[g], regime_map,
                    group_capitals[g], HOLD_REGIMES, hold_ratio=ratio)
                grp_hybrid[g] = hybrid_nav
                switch_stats[g] = {"strategy_days": s_days, "hold_days": h_days}
            port_hyb = combine(grp_hybrid)
            m = metrics_from_nav(port_hyb, TOTAL_CAPITAL, csi300_nav)
            # 达标: Alpha>0 且回撤>-10%
            ok = m["alpha_pct"] > 0 and m["max_drawdown_pct"] > -10
            tag = "✅" if ok else "❌"
            print(f"{ratio:<10.1f} {m['total_return_pct']:>+8.2f} {m['alpha_pct']:>+8.2f} "
                  f"{m['sharpe']:>8.3f} {m['max_drawdown_pct']:>8.1f} {tag:>8}")
            scan_results.append({"ratio": ratio, "metrics": m, "ok": ok,
                                  "group_navs": grp_hybrid, "switch_stats": switch_stats})
            if ok and (best is None or m["alpha_pct"] > best["metrics"]["alpha_pct"]):
                best = {"ratio": ratio, "metrics": m,
                        "group_navs": grp_hybrid, "switch_stats": switch_stats}

        print(f"\n纯策略(对照): 收益{m_strat['total_return_pct']:+.2f}% Alpha{m_strat['alpha_pct']:+.2f}% "
              f"夏普{m_strat['sharpe']:.3f} 回撤{m_strat['max_drawdown_pct']:.1f}%")
        print(f"纯持有(上限): 收益{m_hold['total_return_pct']:+.2f}% Alpha{m_hold['alpha_pct']:+.2f}% "
              f"夏普{m_hold['sharpe']:.3f} 回撤{m_hold['max_drawdown_pct']:.1f}%")

        # ── 分组差异化实验: 只在科技+周期启用震荡持有, 消费保持纯策略 ──
        # 消费组hybrid两头挨打(策略+7.4%/持有+14.7%/hybrid-0.13%), 切换损耗吃掉收益
        # 科技+周期持有收益远超策略, hybrid有效
        DIFF_CONFIGS = [
            {"name": "差异化A:科技0.6/周期0.5/消费0",
             "ratios": {"科技成长型": 0.6, "消费稳健型": 0.0, "周期资源型": 0.5}},
            {"name": "差异化B:科技0.7/周期0.5/消费0",
             "ratios": {"科技成长型": 0.7, "消费稳健型": 0.0, "周期资源型": 0.5}},
            {"name": "差异化C:科技0.6/周期0.6/消费0",
             "ratios": {"科技成长型": 0.6, "消费稳健型": 0.0, "周期资源型": 0.6}},
            {"name": "差异化D:科技0.5/周期0.5/消费0",
             "ratios": {"科技成长型": 0.5, "消费稳健型": 0.0, "周期资源型": 0.5}},
        ]
        print(f"\n分组差异化实验 (消费保持纯策略, 科技+周期启用震荡持有):")
        print(f"{'方案':<34} {'收益%':>8} {'Alpha%':>8} {'夏普':>8} {'回撤%':>8} {'达标?':>8}")
        diff_results = []
        for cfg in DIFF_CONFIGS:
            grp_hybrid = {}
            switch_stats = {}
            for g in group_strat_nav:
                ratio = cfg["ratios"].get(g, 0.0)
                if ratio == 0.0:
                    # 纯策略, 不切换
                    grp_hybrid[g] = group_strat_nav[g].copy()
                    switch_stats[g] = {"strategy_days": len(group_strat_nav[g]), "hold_days": 0}
                else:
                    hybrid_nav, s_days, h_days = build_hybrid_nav(
                        group_strat_nav[g], group_hold_nav[g], regime_map,
                        group_capitals[g], HOLD_REGIMES, hold_ratio=ratio)
                    grp_hybrid[g] = hybrid_nav
                    switch_stats[g] = {"strategy_days": s_days, "hold_days": h_days}
            port_diff = combine(grp_hybrid)
            m = metrics_from_nav(port_diff, TOTAL_CAPITAL, csi300_nav)
            ok = m["alpha_pct"] > 0 and m["max_drawdown_pct"] > -10
            tag = "✅" if ok else "❌"
            print(f"{cfg['name']:<34} {m['total_return_pct']:>+8.2f} {m['alpha_pct']:>+8.2f} "
                  f"{m['sharpe']:>8.3f} {m['max_drawdown_pct']:>8.1f} {tag:>8}")
            diff_results.append({"name": cfg["name"], "ratios": cfg["ratios"],
                                  "metrics": m, "ok": ok,
                                  "group_navs": grp_hybrid, "switch_stats": switch_stats})
            if ok and (best is None or m["alpha_pct"] > best["metrics"]["alpha_pct"]):
                best = {"ratio": cfg["name"], "metrics": m,
                        "group_navs": grp_hybrid, "switch_stats": switch_stats}

        # ── 差异化 + 回撤保护 (突破回撤10%瓶颈) ──
        def apply_dd_protect(nav, threshold=-0.08, recovery=-0.04, reduced=0.5):
            cummax = nav.cummax()
            dd = (nav - cummax) / cummax
            in_prot = False
            prot_nav = [nav.iloc[0]]
            prot_days = 0
            trig = 0
            for i in range(1, len(nav)):
                if dd.iloc[i] < threshold and not in_prot:
                    in_prot = True
                    trig += 1
                elif dd.iloc[i] > recovery and in_prot:
                    in_prot = False
                r = nav.iloc[i] / nav.iloc[i - 1] - 1
                if in_prot:
                    r *= reduced
                    prot_days += 1
                prot_nav.append(prot_nav[-1] * (1 + r))
            return pd.Series(prot_nav, index=nav.index), prot_days, trig

        print(f"\n差异化方案 + 回撤保护(>8%降仓50%):")
        print(f"{'方案':<34} {'收益%':>8} {'Alpha%':>8} {'夏普':>8} {'回撤%':>8} {'保护':>10} {'达标?':>8}")
        prot_best = None
        for dr in diff_results:
            if dr["metrics"]["alpha_pct"] <= 0:
                continue
            port_raw = combine(dr["group_navs"])
            port_prot, p_days, p_trig = apply_dd_protect(port_raw)
            m = metrics_from_nav(port_prot, TOTAL_CAPITAL, csi300_nav)
            ok = m["alpha_pct"] > 0 and m["max_drawdown_pct"] > -10
            tag = "✅" if ok else "❌"
            print(f"{dr['name']:<34} {m['total_return_pct']:>+8.2f} {m['alpha_pct']:>+8.2f} "
                  f"{m['sharpe']:>8.3f} {m['max_drawdown_pct']:>8.1f} {p_trig}次/{p_days}天 {tag:>6}")
            if ok and (prot_best is None or m["alpha_pct"] > prot_best["metrics"]["alpha_pct"]):
                prot_best = {"name": dr["name"], "metrics": m,
                             "group_navs": dr["group_navs"], "switch_stats": dr["switch_stats"]}
        if prot_best is not None:
            best = {"ratio": prot_best["name"] + "+保护", "metrics": prot_best["metrics"],
                    "group_navs": prot_best["group_navs"], "switch_stats": prot_best["switch_stats"]}

        # ── 最优方案的分组明细 ──
        group_detail = []
        if best is not None:
            print(f"\n最优持有比例={best['ratio']} (Alpha>0且回撤<10%) 分组明细:")
            print(f"{'分组':<12} {'策略收益%':>10} {'持有收益%':>10} {'hybrid收益%':>12} {'策略天':>8} {'持有天':>8}")
            for g in group_strat_nav:
                cap = group_capitals[g]
                s_ret = group_strat_nav[g].iloc[-1] / cap - 1
                h_ret = group_hold_nav[g].iloc[-1] - 1
                hy_ret = best["group_navs"][g].iloc[-1] / cap - 1
                ss = best["switch_stats"][g]
                print(f"{g:<12} {s_ret*100:>+10.2f} {h_ret*100:>+10.2f} {hy_ret*100:>+12.2f} "
                      f"{ss['strategy_days']:>8} {ss['hold_days']:>8}")
                group_detail.append({
                    "group": g, "strategy_return_pct": round(s_ret * 100, 2),
                    "hold_return_pct": round(h_ret * 100, 2),
                    "hybrid_return_pct": round(hy_ret * 100, 2),
                    "strategy_days": ss["strategy_days"], "hold_days": ss["hold_days"],
                })

        # ── 结论 ──
        print("\n" + "=" * 70)
        print("  结论")
        print("=" * 70)
        print(f"  纯策略 Alpha: {m_strat['alpha_pct']:+.2f}% 回撤{m_strat['max_drawdown_pct']:.1f}%")
        print(f"  纯持有 Alpha: {m_hold['alpha_pct']:+.2f}% 回撤{m_hold['max_drawdown_pct']:.1f}% (上限, 回撤失控)")
        if best is not None:
            bm = best["metrics"]
            print(f"  最优hybrid(持有{best['ratio']}): Alpha {bm['alpha_pct']:+.2f}% "
                  f"夏普{bm['sharpe']:.3f} 回撤{bm['max_drawdown_pct']:.1f}% ✅达标")
            print(f"\n  ✅ 找到 Alpha转正 且回撤<10% 的平衡点: 震荡日持有{best['ratio']}仓位")
            print(f"     下一步: 引擎级真实实现(处理持仓衔接+切换成本).")
        else:
            print(f"\n  ⚠️ 无ratio同时满足Alpha>0且回撤<10%, 需放宽约束或调其他参数.")
        print(f"\n  注意: 本实验为理想上限(regime已知/无切换成本), 真实实现会打折.")

        # 报告
        report = generate_report(run_time, m_strat, m_hold, scan_results, best,
                                  group_detail, dict(rc))
        with open(REPORT_MD, "w", encoding="utf-8") as f:
            f.write(report)
        print(f"\n✓ 报告 → {REPORT_MD}")

    finally:
        if pref_existed and os.path.exists(pref_backup):
            shutil.move(pref_backup, _USER_PREF_FILE)


def generate_report(run_time, m_strat, m_hold, scan_results, best, group_detail, regime_dist):
    L = []
    L.append("# 震荡市买入持有 — hybrid regime 切换 + 持仓比例扫描报告")
    L.append(f"\n**运行时间**: {run_time}")
    L.append(f"**窗口**: 训练窗(震荡市) {TRAIN_START}~{TRAIN_END}\n")
    L.append("## 实验设计\n")
    L.append("根因诊断确认趋势择时在震荡市拖累收益(策略+8.27% vs 等权持有+24.20%). "
             "本实验验证\"震荡市买入持有\"方向, 并扫描持仓比例找 Alpha转正且回撤<10% 的平衡点:\n")
    L.append(f"- **切换规则**: trending日→P2策略收益率; 震荡日({sorted(HOLD_REGIMES)})→等权持有×hold_ratio")
    L.append("- **hold_ratio**: 震荡日持有仓位 (1.0=满仓, 0.5=半仓持有+半仓现金), 用于控制回撤")
    L.append("- **性质**: 理想上限验证 (regime已知/无切换成本), 验证通过后再做引擎级实现\n")
    L.append(f"**全期regime分布**: {regime_dist}\n")

    L.append("## 持仓比例扫描 (训练窗, vs沪深300)\n")
    L.append("| 震荡日持有比例 | 收益% | Alpha% | 夏普 | 回撤% | 达标(Alpha>0&回撤<10%) |")
    L.append("|--------------|-------|--------|------|-------|----------------------|")
    for r in scan_results:
        m = r["metrics"]
        tag = "✅" if r["ok"] else "❌"
        L.append(f"| {r['ratio']:.1f} | {m['total_return_pct']:+.2f} | {m['alpha_pct']:+.2f} | "
                 f"{m['sharpe']:.3f} | {m['max_drawdown_pct']:.1f} | {tag} |")
    L.append(f"| 纯策略(对照) | {m_strat['total_return_pct']:+.2f} | {m_strat['alpha_pct']:+.2f} | "
             f"{m_strat['sharpe']:.3f} | {m_strat['max_drawdown_pct']:.1f} | — |")
    L.append(f"| 纯持有(上限) | {m_hold['total_return_pct']:+.2f} | {m_hold['alpha_pct']:+.2f} | "
             f"{m_hold['sharpe']:.3f} | {m_hold['max_drawdown_pct']:.1f} | ❌回撤失控 |")
    L.append("")

    L.append("## 最优方案分组明细\n")
    if best is not None:
        bm = best["metrics"]
        L.append(f"**最优持有比例={best['ratio']}** (Alpha {bm['alpha_pct']:+.2f}%, "
                 f"夏普{bm['sharpe']:.3f}, 回撤{bm['max_drawdown_pct']:.1f}%)\n")
        L.append("| 分组 | 策略收益% | 持有收益% | hybrid收益% | 策略天 | 持有天 |")
        L.append("|------|----------|----------|------------|--------|--------|")
        for r in group_detail:
            L.append(f"| {r['group']} | {r['strategy_return_pct']:+.2f} | {r['hold_return_pct']:+.2f} | "
                     f"{r['hybrid_return_pct']:+.2f} | {r['strategy_days']} | {r['hold_days']} |")
    else:
        L.append("无ratio同时满足Alpha>0且回撤<10%.")
    L.append("")

    L.append("## 结论\n")
    L.append(f"- 纯策略 Alpha: **{m_strat['alpha_pct']:+.2f}%** 回撤 {m_strat['max_drawdown_pct']:.1f}%")
    L.append(f"- 纯持有 Alpha: **{m_hold['alpha_pct']:+.2f}%** 回撤 {m_hold['max_drawdown_pct']:.1f}% (上限, 回撤失控)\n")
    if best is not None:
        L.append(f"**✅ 找到平衡点**: 震荡日持有{best['ratio']}仓位 → "
                 f"Alpha {bm['alpha_pct']:+.2f}%, 夏普{bm['sharpe']:.3f}, 回撤{bm['max_drawdown_pct']:.1f}%")
        L.append(f"训练窗 hybrid 收益 {bm['total_return_pct']:+.2f}% (纯策略 {m_strat['total_return_pct']:+.2f}%), "
                 "方向可行. 下一步做引擎级真实实现(处理持仓衔接+切换成本).\n")
    else:
        L.append("**⚠️ 无ratio同时满足Alpha>0且回撤<10%**, 需放宽约束或调其他参数.\n")
    L.append("**注意**: 本实验为理想上限(regime已知/无切换成本), 真实实现会打折.")
    return "\n".join(L)


if __name__ == "__main__":
    main()
