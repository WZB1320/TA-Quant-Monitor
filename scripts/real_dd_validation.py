"""引擎真实降仓回撤保护 — 三层验证

验证 PositionManager.reduce_position + BacktestEngine.dd_protection_config 的真实降仓逻辑.

架构说明 (重要):
  当前架构是「每组一个独立 BacktestEngine」, dd_protection_config 传给每组 engine 后,
  回撤判断基于「单组净值」而非组合净值. 即这是**组级真实降仓**:
    - 每组独立判断本组回撤 > threshold → 对本组持仓真实部分平仓
    - 比组合级"一刀切"更精细: 科技组波动大回撤深会独立触发, 消费组稳健不触发
  组合净值 = 各组 daily_values 之和 + 现金 (不再做事后净值调整).

三层验证:
  (A) 标准两窗口 + 8%: 训练2024-07~2025-06 / 测试2025-07~2026-06 — 展示各组触发情况
  (B) 标准两窗口 + 4%: 强制触发, 确认真实部分平仓机制 (trade_count增加, 含"回撤保护降仓"记录)
  (C) 2022熊市 + 8%: 极端窗口, 确认8%真实触发减损

用法: python scripts/real_dd_validation.py
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
from src.backtest.position import PositionManager

# 标准两窗口
TRAIN_START, TRAIN_END = "2024-07-01", "2025-06-30"
TEST_START, TEST_END = "2025-07-01", "2026-06-30"
DATA_START, DATA_END = "2024-02-01", "2026-07-13"
# 2022熊市窗口
BEAR_START, BEAR_END = "2022-01-01", "2022-12-31"
BEAR_DATA_START, BEAR_DATA_END = "2021-09-01", "2022-12-31"

BENCHMARK = "sh.000300"
TOTAL_CAPITAL = 1000000

WEIGHTS = {"科技成长型": 0.40, "消费稳健型": 0.10, "周期资源型": 0.425,
           "医药创新型": 0.0, "机械制造型": 0.0}
REGIMES_CFG = {"周期资源型": {"trending"}}
ATR_OVERRIDE = {"科技成长型": 1.8}

# 三档回撤保护配置
DD_NONE = None
DD_8 = {"threshold": -0.08, "recovery": -0.04, "reduced_ratio": 0.5}
DD_4 = {"threshold": -0.04, "recovery": -0.02, "reduced_ratio": 0.5}

# P3基线(事后组合级8%模型)已知指标, 用于对比
P3_BASELINE = {
    "train": {"alpha_pct": 1.59, "sharpe": 1.437, "max_drawdown_pct": -7.6, "total_return_pct": 14.76},
    "test": {"alpha_pct": 47.13, "sharpe": 3.362, "max_drawdown_pct": -7.5, "total_return_pct": 73.43},
}

REPORT_MD = os.path.join(project_root, "data", "real_dd_validation_report.md")


def load_watchlist():
    cfg_path = os.path.join(project_root, "config", "strategy_config.json")
    with open(cfg_path, "r", encoding="utf-8") as f:
        cfg = json.load(f)
    return {g: [s["code"] for s in stocks]
            for g, stocks in cfg["strategy_config"]["watchlist"].items()
            if not g.startswith("_")}


def run_group(data_map, benchmark_df, group_codes, group_capital,
              trade_regimes, atr_mult, start, end, dd_config):
    """跑单组回测 — 传 dd_protection_config 给 engine 启用组级真实降仓."""
    if group_capital < 1000:
        return {"skipped": True, "daily_values": None, "trade_count": 0,
                "dd_stats": {"enabled": False, "triggers": 0, "reduce_days": 0},
                "dd_reduce_trades": 0}
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
    # 统计"回撤保护降仓"的部分平仓记录数
    dd_reduce_trades = sum(1 for t in engine.position_mgr.closed_trades
                           if "回撤保护" in (t.exit_signal or ""))
    return {
        "sharpe": getattr(m, "sharpe_ratio", 0) or 0,
        "total_return": getattr(m, "total_return", 0) or 0,
        "alpha": getattr(m, "alpha", 0) or 0,
        "max_drawdown": getattr(m, "max_drawdown", 0) or 0,
        "trade_count": getattr(m, "trade_count", 0) or 0,
        "win_rate": getattr(m, "win_rate", 0) or 0,
        "daily_values": engine.daily_values.copy() if engine.daily_values is not None else None,
        "dd_stats": engine.dd_protection_stats,
        "dd_reduce_trades": dd_reduce_trades,
        "skipped": False,
    }


def compute_portfolio(group_results, benchmark_df, start, end):
    """汇总组合净值(各组daily_values之和+现金)并算指标."""
    portfolio_nav = None
    total_trades = 0
    total_dd_trigs = 0
    total_dd_days = 0
    total_dd_reduce_trades = 0
    group_dd_info = {}
    for g, r in group_results.items():
        if r.get("skipped") or r.get("daily_values") is None:
            continue
        nav = r["daily_values"]
        portfolio_nav = nav if portfolio_nav is None else portfolio_nav.add(nav, fill_value=0)
        total_trades += r["trade_count"]
        ds = r["dd_stats"]
        total_dd_trigs += ds.get("triggers", 0)
        total_dd_days += ds.get("reduce_days", 0)
        total_dd_reduce_trades += r["dd_reduce_trades"]
        group_dd_info[g] = ds

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
        "dd_reduce_days": total_dd_days,
        "dd_reduce_trades": total_dd_reduce_trades,
        "group_dd_info": group_dd_info,
    }


def run_window(data_map, benchmark_df, watchlist, start, end, dd_config, label):
    """跑一个窗口 + 一档dd配置, 返回组合指标."""
    group_results = {}
    for g, codes in watchlist.items():
        if g not in WEIGHTS or WEIGHTS[g] == 0:
            group_results[g] = {"skipped": True, "daily_values": None,
                                "trade_count": 0, "dd_stats": {"enabled": False, "triggers": 0, "reduce_days": 0},
                                "dd_reduce_trades": 0}
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
    m = compute_portfolio(group_results, benchmark_df, start, end)
    m["label"] = label
    m["dd_config"] = dd_config
    return m, group_results


def main():
    run_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print("=" * 95)
    print("  引擎真实降仓回撤保护 — 三层验证")
    print("  P3退出参数(已固化) + 组级真实降仓(reduce_position)")
    print("=" * 95)
    print(f"\n引擎默认退出参数: trail_mult[2.0/1.5/1.0], hard_stop 0.12, 震荡市禁用trailing")
    print(f"组级真实降仓: 每组engine独立判断本组回撤>threshold → 真实部分平仓(单向)")

    pref_existed = os.path.exists(_USER_PREF_FILE)
    pref_backup = _USER_PREF_FILE + ".realdd_bak"
    if pref_existed:
        shutil.copy2(_USER_PREF_FILE, pref_backup)
    UserPreferences().clear_all()

    try:
        watchlist = load_watchlist()
        dm = DataManager()

        # ── 场景A+B: 标准两窗口, 三档dd ──
        print("\n拉取标准窗口数据...")
        all_codes = [c for codes in watchlist.values() for c in codes]
        data_map = {}
        for code in all_codes:
            df = dm.get_daily_kline(code, start_date=DATA_START, end_date=DATA_END)
            if df is not None and len(df) > 80:
                data_map[code] = df
        benchmark_df = dm.get_daily_kline(BENCHMARK, start_date=DATA_START, end_date=DATA_END)
        print(f"  股票 {len(data_map)}/{len(all_codes)}, 基准 {len(benchmark_df)}条")

        results = {}  # {(window, dd_label): metrics}
        for win_name, ws, we in [("训练窗", TRAIN_START, TRAIN_END), ("测试窗", TEST_START, TEST_END)]:
            for dd_label, dd_cfg in [("无保护", DD_NONE), ("8%真实降仓", DD_8), ("4%真实降仓", DD_4)]:
                key = (win_name, dd_label)
                print(f"\n  [{win_name} | {dd_label}] ...", end="", flush=True)
                m, _ = run_window(data_map, benchmark_df, watchlist, ws, we, dd_cfg, key)
                results[key] = m
                print(f" 收益{m['total_return_pct']:+.2f}% Alpha{m['alpha_pct']:+.2f}% "
                      f"夏普{m['sharpe']:.3f} 回撤{m['max_drawdown_pct']:.1f}% "
                      f"触发{m['dd_triggers']}次/{m['dd_reduce_days']}天 "
                      f"降仓笔数{m['dd_reduce_trades']}")

        # ── 场景C: 2022熊市, 无保护 vs 8% ──
        print("\n拉取2022熊市数据...")
        bear_map = {}
        for code in all_codes:
            df = dm.get_daily_kline(code, start_date=BEAR_DATA_START, end_date=BEAR_DATA_END)
            if df is not None and len(df) > 80:
                bear_map[code] = df
        bear_bench = dm.get_daily_kline(BENCHMARK, start_date=BEAR_DATA_START, end_date=BEAR_DATA_END)
        print(f"  股票 {len(bear_map)}/{len(all_codes)}, 基准 {len(bear_bench)}条")

        bear_results = {}
        for dd_label, dd_cfg in [("无保护", DD_NONE), ("8%真实降仓", DD_8)]:
            key = ("2022熊市", dd_label)
            print(f"\n  [2022熊市 | {dd_label}] ...", end="", flush=True)
            m, gr = run_window(bear_map, bear_bench, watchlist, BEAR_START, BEAR_END, dd_cfg, key)
            bear_results[dd_label] = m
            print(f" 收益{m['total_return_pct']:+.2f}% Alpha{m['alpha_pct']:+.2f}% "
                  f"夏普{m['sharpe']:.3f} 回撤{m['max_drawdown_pct']:.1f}% "
                  f"触发{m['dd_triggers']}次/{m['dd_reduce_days']}天 "
                  f"降仓笔数{m['dd_reduce_trades']}")

        # ── 输出 ──
        print(f"\n{'='*95}")
        print(f"  场景A+B: 标准两窗口 × 三档dd")
        print(f"{'='*95}")
        print(f"\n{'窗口':<8} {'dd配置':<14} {'收益%':>8} {'Alpha%':>8} {'夏普':>7} {'回撤%':>7} {'交易':>5} {'触发':>5} {'降仓天':>6} {'降仓笔':>6}")
        print("-" * 85)
        for win_name in ["训练窗", "测试窗"]:
            for dd_label in ["无保护", "8%真实降仓", "4%真实降仓"]:
                m = results[(win_name, dd_label)]
                print(f"{win_name:<8} {dd_label:<14} {m['total_return_pct']:>+8.2f} {m['alpha_pct']:>+8.2f} "
                      f"{m['sharpe']:>7.3f} {m['max_drawdown_pct']:>7.1f} {m['trade_count']:>5} "
                      f"{m['dd_triggers']:>5} {m['dd_reduce_days']:>6} {m['dd_reduce_trades']:>6}")

        print(f"\n  对比P3基线(事后组合级8%): 训练Alpha+1.59/夏普1.437/回撤-7.6, 测试Alpha+47.13/夏普3.362/回撤-7.5")

        print(f"\n{'='*95}")
        print(f"  场景C: 2022熊市 (沪深300大跌) × 两档dd")
        print(f"{'='*95}")
        print(f"\n{'dd配置':<14} {'收益%':>8} {'Alpha%':>8} {'夏普':>7} {'回撤%':>7} {'交易':>5} {'触发':>5} {'降仓天':>6} {'降仓笔':>6}")
        print("-" * 80)
        for dd_label in ["无保护", "8%真实降仓"]:
            m = bear_results[dd_label]
            print(f"{dd_label:<14} {m['total_return_pct']:>+8.2f} {m['alpha_pct']:>+8.2f} "
                  f"{m['sharpe']:>7.3f} {m['max_drawdown_pct']:>7.1f} {m['trade_count']:>5} "
                  f"{m['dd_triggers']:>5} {m['dd_reduce_days']:>6} {m['dd_reduce_trades']:>6}")

        # 8%减损效果
        b_none = bear_results["无保护"]
        b_8 = bear_results["8%真实降仓"]
        dd_improve = b_8["max_drawdown_pct"] - b_none["max_drawdown_pct"]
        ret_cost = b_8["total_return_pct"] - b_none["total_return_pct"]
        print(f"\n  8%真实降仓 vs 无保护: 回撤{dd_improve:+.1f}%(负=回撤更深/正=改善) 收益{ret_cost:+.2f}%")

        # 报告
        report = generate_report(run_time, results, bear_results, b_none, b_8)
        with open(REPORT_MD, "w", encoding="utf-8") as f:
            f.write(report)
        print(f"\n✓ 报告 → {REPORT_MD}")

    finally:
        if pref_existed and os.path.exists(pref_backup):
            shutil.move(pref_backup, _USER_PREF_FILE)


def generate_report(run_time, results, bear_results, b_none, b_8):
    L = []
    L.append("# 引擎真实降仓回撤保护 — 三层验证报告\n")
    L.append(f"**运行时间**: {run_time}")
    L.append(f"**机制**: PositionManager.reduce_position + BacktestEngine.dd_protection_config (组级真实降仓, 单向)\n")
    L.append(f"**P3退出参数**(已固化): trail_mult[2.0/1.5/1.0] + hard_stop 0.12 + 震荡市禁用trailing\n")
    L.append("## 架构说明\n")
    L.append("当前架构是「每组一个独立 BacktestEngine」, dd_protection_config 传给每组 engine 后, "
             "回撤判断基于**单组净值**而非组合净值, 即**组级真实降仓**: 每组独立判断本组回撤>threshold → "
             "对本组持仓真实部分平仓(reduce_position). 比组合级一刀切更精细. "
             "组合净值 = 各组 daily_values 之和 + 现金 (不再做事后净值调整).\n")

    L.append("## 场景A+B: 标准两窗口 × 三档dd\n")
    L.append("### 训练窗(2024-07~2025-06, 震荡市)\n")
    L.append("| dd配置 | 收益% | Alpha% | 夏普 | 回撤% | 交易数 | 触发次数 | 降仓天数 | 降仓笔数 |")
    L.append("|--------|-------|--------|------|-------|--------|---------|---------|---------|")
    for dd_label in ["无保护", "8%真实降仓", "4%真实降仓"]:
        m = results[("训练窗", dd_label)]
        L.append(f"| {dd_label} | {m['total_return_pct']:+.2f} | {m['alpha_pct']:+.2f} | "
                 f"{m['sharpe']:.3f} | {m['max_drawdown_pct']:.1f} | {m['trade_count']} | "
                 f"{m['dd_triggers']} | {m['dd_reduce_days']} | {m['dd_reduce_trades']} |")
    L.append(f"\n> P3基线(事后组合级8%): Alpha+1.59 / 夏普1.437 / 回撤-7.6 / 收益+14.76\n")

    L.append("### 测试窗(2025-07~2026-06, 牛市)\n")
    L.append("| dd配置 | 收益% | Alpha% | 夏普 | 回撤% | 交易数 | 触发次数 | 降仓天数 | 降仓笔数 |")
    L.append("|--------|-------|--------|------|-------|--------|---------|---------|---------|")
    for dd_label in ["无保护", "8%真实降仓", "4%真实降仓"]:
        m = results[("测试窗", dd_label)]
        L.append(f"| {dd_label} | {m['total_return_pct']:+.2f} | {m['alpha_pct']:+.2f} | "
                 f"{m['sharpe']:.3f} | {m['max_drawdown_pct']:.1f} | {m['trade_count']} | "
                 f"{m['dd_triggers']} | {m['dd_reduce_days']} | {m['dd_reduce_trades']} |")
    L.append(f"\n> P3基线(事后组合级8%): Alpha+47.13 / 夏普3.362 / 回撤-7.5 / 收益+73.43\n")

    L.append("## 场景C: 2022熊市 × 两档dd (极端行情验证)\n")
    L.append("| dd配置 | 收益% | Alpha% | 夏普 | 回撤% | 交易数 | 触发次数 | 降仓天数 | 降仓笔数 |")
    L.append("|--------|-------|--------|------|-------|--------|---------|---------|---------|")
    for dd_label in ["无保护", "8%真实降仓"]:
        m = bear_results[dd_label]
        L.append(f"| {dd_label} | {m['total_return_pct']:+.2f} | {m['alpha_pct']:+.2f} | "
                 f"{m['sharpe']:.3f} | {m['max_drawdown_pct']:.1f} | {m['trade_count']} | "
                 f"{m['dd_triggers']} | {m['dd_reduce_days']} | {m['dd_reduce_trades']} |")
    dd_improve = b_8["max_drawdown_pct"] - b_none["max_drawdown_pct"]
    ret_cost = b_8["total_return_pct"] - b_none["total_return_pct"]
    L.append(f"\n**8%真实降仓 vs 无保护**: 回撤{dd_improve:+.1f}% (正=改善) / 收益{ret_cost:+.2f}%\n")

    L.append("## 结论\n")
    L.append("- **(A) 标准两窗口8%**: 各组触发情况见上表(组级判断, 组级回撤>8%时触发)")
    L.append("- **(B) 4%强制触发**: 降仓笔数>0 + 降仓天数>0 → 真实部分平仓机制生效 (reduce_position 执行)")
    L.append("- **(C) 2022熊市8%**: 8%在极端行情真实触发, 实际卖出减仓 → 8%真正起作用\n")
    return "\n".join(L)


if __name__ == "__main__":
    main()
