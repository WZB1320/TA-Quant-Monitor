"""
震荡市 Alpha 根因诊断 — 三基准对比

P2 训练窗(震荡市) Alpha -4.9%, 三组绝对收益为正但跑输沪深300(+13.2%).
根因两种可能:
  A. 基准不匹配: 自选股偏中小盘, 沪深300是大盘蓝筹, 震荡市大盘强于中小盘
     → 策略其实有 Alpha, 只是被大盘基准掩盖
  B. 策略无效: 趋势信号在震荡市择时拖累, 跑输"买入持有"
     → 需改策略

诊断方法: 用三种基准重测 P2 训练窗 Alpha
  基准A: 沪深300 (原基准, 大盘蓝筹)
  基准B: 等权全自选股买入持有 (选股能力基准)
  基准C: 等权本组自选股买入持有 (组内择时能力基准)

判断逻辑:
  - 若策略 Alpha vs 基准B/C 转正 → 根因是基准不匹配(A), 换基准即可
  - 若策略 Alpha vs 基准B/C 仍为负 → 根因是策略无效(B), 需改策略
  - 基准C 最严苛: 跑赢本组等权持有才算择时有 Alpha

只诊断训练窗(震荡市), 不重跑测试窗.
用法: python scripts/range_alpha_diagnosis.py
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
DATA_START, DATA_END = "2024-02-01", "2026-07-13"
BENCHMARK_CSI300 = "sh.000300"
TOTAL_CAPITAL = 1000000

# P2 配置
WEIGHTS = {"科技成长型": 0.40, "消费稳健型": 0.10, "周期资源型": 0.425,
           "医药创新型": 0.0, "机械制造型": 0.0}
REGIMES_CFG = {"周期资源型": {"trending"}}
ATR_OVERRIDE = {"科技成长型": 1.8}

REPORT_MD = os.path.join(project_root, "data", "range_alpha_diagnosis_report.md")


def load_watchlist():
    cfg_path = os.path.join(project_root, "config", "strategy_config.json")
    with open(cfg_path, "r", encoding="utf-8") as f:
        cfg = json.load(f)
    return {g: [s["code"] for s in stocks]
            for g, stocks in cfg["strategy_config"]["watchlist"].items()
            if not g.startswith("_")}


def build_equal_weight_benchmark(data_map, codes, start, end):
    """构建等权买入持有基准: 每只股票从start归一化为1, 等权平均.
    返回净值序列(index=date, value=净值, 起点≈1)."""
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
        # 从起点归一化
        s = s / s.iloc[0]
        navs.append(s)
    if not navs:
        return None
    eq = pd.concat(navs, axis=1).mean(axis=1)
    eq.name = "equal_weight_nav"
    return eq


def run_group_backtest(data_map, benchmark_df, group_codes, group_capital,
                       trade_regimes, atr_mult, start, end):
    """跑 P2 配置分组回测, 返回 daily_values."""
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
    return engine.daily_values.copy() if engine.daily_values is not None else None


def calc_return(nav_series):
    """从净值序列算区间收益率."""
    if nav_series is None or len(nav_series) < 2:
        return 0.0
    return (nav_series.iloc[-1] / nav_series.iloc[0]) - 1


def calc_alpha(strategy_nav, bench_nav, total_capital):
    """算 Alpha = 策略收益 - 基准收益. 策略净值基于total_capital, 基准净值归一化."""
    strat_ret = (strategy_nav.iloc[-1] / total_capital) - 1 if len(strategy_nav) > 0 else 0
    bench_ret = (bench_nav.iloc[-1] / bench_nav.iloc[0]) - 1 if len(bench_nav) > 0 else 0
    return strat_ret - bench_ret, strat_ret, bench_ret


def main():
    run_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print("=" * 70)
    print("  震荡市 Alpha 根因诊断 — 三基准对比")
    print(f"  训练窗(震荡市): {TRAIN_START}~{TRAIN_END}")
    print(f"  P2配置: {WEIGHTS} + 现金{(1-sum(WEIGHTS.values()))*100:.1f}%")
    print("=" * 70)

    pref_existed = os.path.exists(_USER_PREF_FILE)
    pref_backup = _USER_PREF_FILE + ".diag_bak"
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

        # ── 构建三种基准 ──
        print("\n构建基准...")
        # 基准A: 沪深300
        csi300 = csi300_df.copy()
        if "date" not in csi300.columns:
            csi300 = csi300.reset_index()
        csi300["date"] = pd.to_datetime(csi300["date"])
        csi300_nav = csi300.set_index("date")["close"].astype(float)
        csi300_nav = csi300_nav[(csi300_nav.index >= pd.Timestamp(TRAIN_START)) &
                                (csi300_nav.index <= pd.Timestamp(TRAIN_END))]
        csi300_nav = csi300_nav / csi300_nav.iloc[0]
        csi300_ret = calc_return(csi300_nav)
        print(f"  基准A 沪深300: 训练窗收益 {csi300_ret*100:+.2f}%")

        # 基准B: 等权全自选股买入持有
        all_watch_codes = [c for codes in watchlist.values() for c in codes if c in data_map]
        eq_all_nav = build_equal_weight_benchmark(data_map, all_watch_codes, TRAIN_START, TRAIN_END)
        eq_all_ret = calc_return(eq_all_nav)
        print(f"  基准B 等权全自选({len(all_watch_codes)}只): 训练窗收益 {eq_all_ret*100:+.2f}%")

        # 基准C: 等权本组自选股 (逐组构建)
        group_eq_navs = {}
        for g, codes in watchlist.items():
            g_codes = [c for c in codes if c in data_map]
            if len(g_codes) < 2:
                continue
            g_eq = build_equal_weight_benchmark(data_map, g_codes, TRAIN_START, TRAIN_END)
            group_eq_navs[g] = g_eq
            g_ret = calc_return(g_eq)
            print(f"  基准C 等权{g}({len(g_codes)}只): 训练窗收益 {g_ret*100:+.2f}%")

        # ── 跑 P2 分组回测 ──
        print("\n跑 P2 分组回测 (训练窗)...")
        group_strategy_navs = {}  # {group: daily_values (基于group_capital)}
        group_capitals = {}
        for group_name, codes in watchlist.items():
            if group_name not in WEIGHTS or WEIGHTS[group_name] == 0:
                continue
            group_codes = [c for c in codes if c in data_map]
            if len(group_codes) < 2:
                continue
            capital = TOTAL_CAPITAL * WEIGHTS[group_name]
            group_capitals[group_name] = capital
            regimes = REGIMES_CFG.get(group_name)
            atr_mult = ATR_OVERRIDE.get(group_name, 2.0)
            dv = run_group_backtest(data_map, csi300_df, group_codes, capital,
                                    regimes, atr_mult, TRAIN_START, TRAIN_END)
            if dv is not None:
                # 转成 datetime 索引
                dv_idx = pd.to_datetime(dv.index)
                dv = pd.Series(dv.values, index=dv_idx)
                group_strategy_navs[group_name] = dv
                strat_ret = calc_return(dv)
                print(f"  {group_name:12s}: 策略收益 {strat_ret*100:+.2f}% "
                      f"(资金{capital:.0f})")

        # ── 计算各组三基准 Alpha ──
        print("\n" + "=" * 70)
        print("  各组三基准 Alpha 对比 (训练窗)")
        print("=" * 70)
        print(f"{'分组':<12} {'策略收益%':>10} {'vs沪深300':>12} {'vs等权全自选':>14} {'vs等权本组':>12}")

        group_alpha_results = []
        for g, dv in group_strategy_navs.items():
            strat_ret = calc_return(dv)
            capital = group_capitals[g]
            # vs 沪深300
            alpha_a, _, _ = calc_alpha(dv, csi300_nav, capital)
            # vs 等权全自选
            alpha_b, _, _ = calc_alpha(dv, eq_all_nav, capital)
            # vs 等权本组
            g_eq = group_eq_navs.get(g)
            alpha_c, _, _ = calc_alpha(dv, g_eq, capital) if g_eq is not None else (None, None, None)

            alpha_a_pct = f"{alpha_a*100:+.2f}" if alpha_a is not None else "—"
            alpha_b_pct = f"{alpha_b*100:+.2f}" if alpha_b is not None else "—"
            alpha_c_pct = f"{alpha_c*100:+.2f}" if alpha_c is not None else "—"
            print(f"{g:<12} {strat_ret*100:>+10.2f} {alpha_a_pct:>12} {alpha_b_pct:>14} {alpha_c_pct:>12}")
            group_alpha_results.append({
                "group": g, "strategy_return_pct": round(strat_ret * 100, 2),
                "alpha_vs_csi300_pct": round(alpha_a * 100, 2) if alpha_a is not None else None,
                "alpha_vs_eqall_pct": round(alpha_b * 100, 2) if alpha_b is not None else None,
                "alpha_vs_eqgroup_pct": round(alpha_c * 100, 2) if alpha_c is not None else None,
            })

        # ── 组合层面三基准 Alpha ──
        print("\n" + "=" * 70)
        print("  组合层面三基准 Alpha (训练窗)")
        print("=" * 70)
        # 组合净值 = 各组daily_values求和 + 现金
        portfolio_nav = None
        for g, dv in group_strategy_navs.items():
            if portfolio_nav is None:
                portfolio_nav = dv.copy()
            else:
                portfolio_nav = portfolio_nav.add(dv, fill_value=0)
        invested = sum(group_capitals.values())
        cash = TOTAL_CAPITAL - invested
        portfolio_nav = portfolio_nav + cash

        port_ret = calc_return(portfolio_nav)
        # 注意 portfolio_nav 起点≈invested+cash=TOTAL_CAPITAL, 所以直接用 iloc[-1]/TOTAL_CAPITAL-1
        port_ret = (portfolio_nav.iloc[-1] / TOTAL_CAPITAL) - 1

        alpha_a, _, bench_a = calc_alpha(portfolio_nav, csi300_nav, TOTAL_CAPITAL)
        alpha_b, _, bench_b = calc_alpha(portfolio_nav, eq_all_nav, TOTAL_CAPITAL)
        # 组合 vs 等权全自选 (组合层面只用全自选做基准, 没有组合本组概念)

        print(f"  组合策略收益: {port_ret*100:+.2f}%")
        print(f"  基准A 沪深300收益: {bench_a*100:+.2f}%  → Alpha {alpha_a*100:+.2f}%")
        print(f"  基准B 等权全自选收益: {bench_b*100:+.2f}%  → Alpha {alpha_b*100:+.2f}%")

        portfolio_result = {
            "strategy_return_pct": round(port_ret * 100, 2),
            "csi300_return_pct": round(bench_a * 100, 2),
            "eqall_return_pct": round(bench_b * 100, 2),
            "alpha_vs_csi300_pct": round(alpha_a * 100, 2),
            "alpha_vs_eqall_pct": round(alpha_b * 100, 2),
        }

        # ── 诊断结论 ──
        print("\n" + "=" * 70)
        print("  诊断结论")
        print("=" * 70)
        # 各组 vs 等权本组(基准C) 的 Alpha 正负统计
        alpha_c_pos = sum(1 for r in group_alpha_results
                          if r["alpha_vs_eqgroup_pct"] is not None and r["alpha_vs_eqgroup_pct"] > 0)
        alpha_c_neg = sum(1 for r in group_alpha_results
                          if r["alpha_vs_eqgroup_pct"] is not None and r["alpha_vs_eqgroup_pct"] <= 0)
        alpha_b_pos = sum(1 for r in group_alpha_results
                          if r["alpha_vs_eqall_pct"] is not None and r["alpha_vs_eqall_pct"] > 0)

        print(f"  vs 等权全自选(基准B): Alpha转正 {alpha_b_pos}/{len(group_alpha_results)} 组")
        print(f"  vs 等权本组(基准C, 最严苛): Alpha转正 {alpha_c_pos}/{len(group_alpha_results)} 组, "
              f"为负 {alpha_c_neg} 组")
        print(f"  组合 vs 等权全自选: Alpha {alpha_b*100:+.2f}%")

        if alpha_b > 0 and alpha_c_pos >= alpha_c_neg:
            verdict = "A_基准不匹配"
            print("\n  → 根因判断: 【基准不匹配】策略跑赢等权持有, 本身有 Alpha,")
            print("    沪深300(大盘蓝筹)在震荡市强于中小盘自选股, 掩盖了策略 Alpha.")
            print("    建议: 换等权自选股为基准, Alpha 可转正, 无需改策略.")
        elif alpha_c_neg > alpha_c_pos:
            verdict = "B_策略无效"
            print("\n  → 根因判断: 【策略无效】策略跑输本组等权持有, 择时在震荡市拖累收益.")
            print("    建议: 震荡市改用反转/均值回归信号, 或震荡市降低交易频率/降仓.")
        else:
            verdict = "混合"
            print("\n  → 根因判断: 【混合】部分组基准不匹配, 部分组策略无效, 需分组处理.")

        # 保存报告
        report = generate_report(run_time, group_alpha_results, portfolio_result,
                                  group_eq_navs, csi300_ret, eq_all_ret,
                                  {"csi300": csi300_ret, "eq_all": eq_all_ret,
                                   "groups": {g: calc_return(v) for g, v in group_eq_navs.items()}},
                                  verdict)
        with open(REPORT_MD, "w", encoding="utf-8") as f:
            f.write(report)
        print(f"\n✓ 报告 → {REPORT_MD}")

    finally:
        if pref_existed and os.path.exists(pref_backup):
            shutil.move(pref_backup, _USER_PREF_FILE)


def generate_report(run_time, group_results, portfolio_result, group_eq_navs,
                    csi300_ret, eq_all_ret, bench_returns, verdict):
    L = []
    L.append("# 震荡市 Alpha 根因诊断报告 — 三基准对比")
    L.append(f"\n**运行时间**: {run_time}")
    L.append(f"**窗口**: 训练窗(震荡市) {TRAIN_START}~{TRAIN_END}")
    L.append(f"**P2配置**: {WEIGHTS} + 现金{(1-sum(WEIGHTS.values()))*100:.1f}%\n")

    L.append("## 诊断思路\n")
    L.append("P2 训练窗 Alpha -4.9% (vs 沪深300), 三组绝对收益为正但跑输沪深300. "
             "用三种基准区分根因:\n")
    L.append("- **基准A 沪深300**: 原基准, 大盘蓝筹 (震荡市大盘强于中小盘)")
    L.append("- **基准B 等权全自选**: 33只自选股等权买入持有 (选股能力基准)")
    L.append("- **基准C 等权本组**: 本组自选股等权买入持有 (组内择时能力基准, 最严苛)\n")
    L.append("**判断**: 策略跑赢等权持有 → 基准不匹配(换基准即可); "
             "跑输等权持有 → 策略无效(需改策略)\n")

    L.append("## 三基准训练窗收益对比\n")
    L.append("| 基准 | 收益% | 说明 |")
    L.append("|------|-------|------|")
    L.append(f"| 沪深300 | {csi300_ret*100:+.2f} | 大盘蓝筹 (原基准) |")
    L.append(f"| 等权全自选(33只) | {eq_all_ret*100:+.2f} | 中小盘为主 |")
    for g, r in bench_returns["groups"].items():
        L.append(f"| 等权{g} | {r*100:+.2f} | 组内买入持有 |")
    L.append("")

    L.append("## 各组三基准 Alpha 对比\n")
    L.append("| 分组 | 策略收益% | vs沪深300 | vs等权全自选 | vs等权本组 |")
    L.append("|------|----------|----------|-------------|-----------|")
    for r in group_results:
        a = r["alpha_vs_csi300_pct"]
        b = r["alpha_vs_eqall_pct"]
        c = r["alpha_vs_eqgroup_pct"]
        a_s = f"{a:+.2f}" if a is not None else "—"
        b_s = f"{b:+.2f}" if b is not None else "—"
        c_s = f"{c:+.2f}" if c is not None else "—"
        L.append(f"| {r['group']} | {r['strategy_return_pct']:+.2f} | {a_s} | {b_s} | {c_s} |")
    L.append("")

    L.append("## 组合层面 Alpha\n")
    L.append("| 策略收益% | vs沪深300 Alpha% | vs等权全自选 Alpha% |")
    L.append("|----------|-----------------|--------------------|")
    L.append(f"| {portfolio_result['strategy_return_pct']:+.2f} | "
             f"{portfolio_result['alpha_vs_csi300_pct']:+.2f} | "
             f"{portfolio_result['alpha_vs_eqall_pct']:+.2f} |")
    L.append("")

    L.append("## 诊断结论\n")
    L.append(f"**根因判断**: {verdict}\n")
    L.append(f"组合 vs 沪深300 Alpha: {portfolio_result['alpha_vs_csi300_pct']:+.2f}%")
    L.append(f"组合 vs 等权全自选 Alpha: {portfolio_result['alpha_vs_eqall_pct']:+.2f}%\n")

    if "A" in verdict:
        L.append("**结论**: 策略跑赢等权持有, Alpha 被沪深300大盘基准掩盖. "
                 "换等权自选股基准后 Alpha 可转正, 无需改策略逻辑.")
    elif "B" in verdict:
        L.append("**结论**: 策略跑输等权持有, 趋势信号在震荡市择时拖累收益. "
                 "需对震荡市改用反转/均值回归信号或降仓降频.")
    else:
        L.append("**结论**: 混合根因, 需分组处理.")

    return "\n".join(L)


if __name__ == "__main__":
    main()
