"""
P2 消费组均值回归验证 — 完整实现 (信号层+退出层+配置层)

P2 改动 (2026-08-10):
  与P4不同, P2真正落地代码改动, 三层协同:
    1. 信号层: strategy_mode="mean_reversion" → filter/classifier跳过MA60方向约束
       (RSI/KDJ原始逻辑已符合均值回归: 超卖拐头→看多, 超买拐头→看空, 无需反转)
    2. 退出层: mean_reversion_config → target_profit_pct=6%目标止盈 + 全体制禁用trailing
       + hard_stop_pct=8%收紧止损 (P4失败根因: 只改信号没改退出, trailing不适配均值回归)
    3. 配置层: RSI权重0.18→0.28, KDJ权重0.08→0.15, MA60权重0.18→0.10
       RSI oversold 35→30, KDJ oversold_j 30→15
       strength: RSI×1.4, KDJ×1.3, MA60×0.8
       atr_stop_mult 1.8→1.2 (消费股低波动)

P4 失败教训:
  P4脚本声称改了5项, 但strategy_mode字段在源码中零匹配, 实际跑的还是趋势跟踪.
  P2在源码层面真正实现: GroupConfig返回strategy_mode/mean_reversion_exit,
  filter/classifier读取strategy_mode跳过MA60约束, position.py新增target_profit_pct分支,
  BacktestEngine接收mean_reversion_config合并到stop_loss_params+regime_exit_config.

P3+ 基线 (对照, 消费组趋势跟踪):
  训练窗: 组合Alpha+1.59%/夏普1.437/回撤-7.6%
          消费组: 夏普+0.422/收益+6.7%/Alpha-6.5%/10笔
  测试窗: 组合Alpha+46.11%/夏普3.341/回撤-7.5%
          消费组: 夏普-2.120/收益-9.0%/Alpha-35.3%/13笔

用法: python scripts/p2_consumer_mr_validation.py
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

WEIGHTS = {"科技成长型": 0.40, "消费稳健型": 0.10, "周期资源型": 0.425,
           "医药创新型": 0.0, "机械制造型": 0.0}
REGIMES_CFG = {"周期资源型": {"trending"}}
ATR_OVERRIDE = {"科技成长型": 1.8}
DD_CONFIG = BacktestEngine.DEFAULT_DD_PROTECTION_CONFIG

# P3+ 基线 (消费组趋势跟踪, 对照)
P3PLUS_BASELINE = {
    "train": {"alpha_pct": 1.59, "sharpe": 1.437, "max_drawdown_pct": -7.6,
              "total_return_pct": 14.76},
    "test": {"alpha_pct": 46.11, "sharpe": 3.341, "max_drawdown_pct": -7.5,
             "total_return_pct": 72.40},
    "consumer_train": {"sharpe": 0.422, "total_return": 0.067, "alpha": -0.065,
                       "trade_count": 10},
    "consumer_test": {"sharpe": -2.120, "total_return": -0.090, "alpha": -0.353,
                      "trade_count": 13},
}

REPORT_MD = os.path.join(project_root, "data", "p2_consumer_mr_validation_report.md")
RESULT_JSON = os.path.join(project_root, "data", "p2_consumer_mr_validation_result.json")


def load_watchlist():
    cfg_path = os.path.join(project_root, "config", "strategy_config.json")
    with open(cfg_path, "r", encoding="utf-8") as f:
        cfg = json.load(f)
    return {g: [s["code"] for s in stocks]
            for g, stocks in cfg["strategy_config"]["watchlist"].items()
            if not g.startswith("_")}


def run_group(data_map, benchmark_df, group_codes, group_capital,
              trade_regimes, atr_mult, start, end, dd_config,
              group_name=None):
    """跑单组回测.

    消费组(group_name="消费稳健型")自动走均值回归:
      - strategy_mode="mean_reversion" 由 GroupConfig 从 strategy_config.json 读取
      - mean_reversion_config 由本函数从配置中提取并传给 BacktestEngine
    其他组完全不变.
    """
    if group_capital < 1000:
        return {"skipped": True, "daily_values": None, "trade_count": 0,
                "sharpe": 0, "total_return": 0, "alpha": 0, "max_drawdown": 0,
                "win_rate": 0, "final_value": group_capital,
                "dd_stats": {"enabled": False, "triggers": 0, "reduce_days": 0},
                "dd_reduce_trades": 0}

    GroupConfig._instance = None
    GroupConfig._config = None

    # 提取均值回归退出配置 (仅消费组有)
    mean_reversion_config = None
    if group_name == "消费稳健型":
        gc = GroupConfig()
        gc._load()
        consumer_cfg = gc._groups.get("消费稳健型", {})
        mean_reversion_config = consumer_cfg.get("mean_reversion_exit")
        strategy_mode = consumer_cfg.get("strategy_mode", "trend_following")
        print(f"    [消费组] strategy_mode={strategy_mode}, "
              f"mean_reversion_exit={mean_reversion_config}")

    engine = BacktestEngine(
        initial_capital=group_capital, lookback_days=120, position_ratio=0.3,
        commission_rate=0.00025, stamp_tax=0.001, slippage=0.0001,
        signal_dedup_days=5, risk_per_trade=0.05, atr_stop_mult=atr_mult,
        forced_regime=None, trade_regimes=trade_regimes,
        dd_protection_config=dd_config,
        mean_reversion_config=mean_reversion_config,
    )
    sub_map = {c: data_map[c] for c in group_codes if c in data_map}
    m = engine.run(sub_map, benchmark_df=benchmark_df, start_date=start, end_date=end)
    dd_reduce_trades = sum(1 for t in engine.position_mgr.closed_trades
                           if "回撤保护" in (t.exit_signal or ""))

    # 统计退出原因分布
    exit_reasons = {}
    for t in engine.position_mgr.closed_trades:
        sig = t.exit_signal or ""
        if "均值回归止盈" in sig:
            exit_reasons["均值回归止盈"] = exit_reasons.get("均值回归止盈", 0) + 1
        elif "ATR移动止盈" in sig or "移动止盈" in sig:
            exit_reasons["trailing止盈"] = exit_reasons.get("trailing止盈", 0) + 1
        elif "ATR硬止损" in sig or "安全网" in sig or "硬止损" in sig:
            exit_reasons["止损"] = exit_reasons.get("止损", 0) + 1
        elif "score=" in sig:
            exit_reasons["信号退出"] = exit_reasons.get("信号退出", 0) + 1
        elif "回撤保护" in sig:
            exit_reasons["回撤保护"] = exit_reasons.get("回撤保护", 0) + 1
        else:
            exit_reasons["其他"] = exit_reasons.get("其他", 0) + 1

    return {
        "sharpe": getattr(m, "sharpe_ratio", 0) or 0,
        "total_return": getattr(m, "total_return", 0) or 0,
        "alpha": getattr(m, "alpha", 0) or 0,
        "max_drawdown": getattr(m, "max_drawdown", 0) or 0,
        "trade_count": getattr(m, "trade_count", 0) or 0,
        "win_rate": getattr(m, "win_rate", 0) or 0,
        "final_value": getattr(m, "final_value", group_capital) or group_capital,
        "daily_values": engine.daily_values.copy() if engine.daily_values is not None else None,
        "dd_stats": engine.dd_protection_stats,
        "dd_reduce_trades": dd_reduce_trades,
        "exit_reasons": exit_reasons,
        "skipped": False,
    }


def compute_portfolio_metrics(group_results, benchmark_df, start, end):
    portfolio_nav = None
    total_trades = 0
    total_dd_triggers = 0
    total_dd_reduce_trades = 0
    for g, r in group_results.items():
        if r.get("skipped") or r.get("daily_values") is None:
            continue
        nav = r["daily_values"]
        portfolio_nav = nav if portfolio_nav is None else portfolio_nav.add(nav, fill_value=0)
        total_trades += r["trade_count"]
        ds = r.get("dd_stats", {})
        total_dd_triggers += ds.get("triggers", 0)
        total_dd_reduce_trades += r.get("dd_reduce_trades", 0)

    if portfolio_nav is None or len(portfolio_nav) < 10:
        return {"error": "无有效净值数据"}

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
    drawdown = (portfolio_nav - cummax) / cummax
    max_dd = drawdown.min()

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
        "dd_triggers": total_dd_triggers,
        "dd_reduce_trades": total_dd_reduce_trades,
    }


def main():
    run_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print("=" * 95)
    print("  P2 消费组均值回归验证 — 完整实现 (信号层+退出层+配置层)")
    print(f"  信号层: strategy_mode=mean_reversion → filter/classifier跳过MA60方向约束")
    print(f"  退出层: target_profit_pct=6%目标止盈 + 全体制禁用trailing + hard_stop=8%")
    print(f"  配置层: RSI权重0.28/KDJ0.15/MA60降到0.10, oversold收紧")
    print(f"  训练窗: {TRAIN_START}~{TRAIN_END} (震荡市)")
    print(f"  测试窗: {TEST_START}~{TEST_END} (牛市)")
    print("=" * 95)

    # 确认消费组配置已生效
    gc = GroupConfig()
    gc._load()
    consumer_cfg = gc._groups.get("消费稳健型", {})
    print(f"\n消费组策略模式: {consumer_cfg.get('strategy_mode', 'trend_following')}")
    print(f"消费组RSI权重: {consumer_cfg.get('indicator_weights', {}).get('RSI')}")
    print(f"消费组KDJ权重: {consumer_cfg.get('indicator_weights', {}).get('KDJ')}")
    print(f"消费组MA60权重: {consumer_cfg.get('indicator_weights', {}).get('MA60')}")
    print(f"消费组RSI oversold: {consumer_cfg.get('indicator_params', {}).get('RSI', {}).get('oversold')}")
    print(f"消费组KDJ oversold_j: {consumer_cfg.get('indicator_params', {}).get('KDJ', {}).get('oversold_j')}")
    print(f"消费组mean_reversion_exit: {consumer_cfg.get('mean_reversion_exit')}")

    pref_existed = os.path.exists(_USER_PREF_FILE)
    pref_backup = _USER_PREF_FILE + ".p2val_bak"
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

        for window_name, ws, we in [("训练窗", TRAIN_START, TRAIN_END),
                                     ("测试窗", TEST_START, TEST_END)]:
            print(f"\n{'='*95}")
            print(f"  [{window_name} {ws}~{we}]")
            print(f"{'='*95}")
            group_results = {}
            for g, codes in watchlist.items():
                if g not in WEIGHTS or WEIGHTS[g] == 0:
                    print(f"  {g:12s}: [跳过-权重0%]")
                    group_results[g] = {"skipped": True}
                    continue
                g_codes = [c for c in codes if c in data_map]
                if len(g_codes) < 2:
                    continue
                capital = TOTAL_CAPITAL * WEIGHTS[g]
                regimes = REGIMES_CFG.get(g)
                atr_mult = ATR_OVERRIDE.get(g, 2.0)
                # 消费组使用配置中的atr_stop_mult(1.2), 不用ATR_OVERRIDE默认值
                if g == "消费稳健型":
                    atr_mult = consumer_cfg.get("atr_stop_mult", 1.2)
                r = run_group(data_map, benchmark_df, g_codes, capital,
                              regimes, atr_mult, ws, we, DD_CONFIG, group_name=g)
                group_results[g] = r
                if not r.get("skipped"):
                    tag = " ← 均值回归" if g == "消费稳健型" else ""
                    print(f"  {g:12s}: 夏普{r['sharpe']:+.3f} 收益{r['total_return']*100:+.1f}% "
                          f"Alpha{r['alpha']*100:+.1f}% 交易{r['trade_count']}笔 "
                          f"胜率{r['win_rate']*100:.0f}%{tag}")
                    if r.get("exit_reasons"):
                        print(f"                退出: {r['exit_reasons']}")

            m = compute_portfolio_metrics(group_results, benchmark_df, ws, we)
            if window_name == "训练窗":
                train_m = m
                train_groups = group_results
            else:
                test_m = m
                test_groups = group_results

            print(f"\n  组合: 夏普{m['sharpe']:.3f} 收益{m['total_return_pct']:+.2f}% "
                  f"Alpha{m['alpha_pct']:+.2f}% 回撤{m['max_drawdown_pct']:.1f}%")

        # ── 汇总对比 ──
        print(f"\n{'='*95}")
        print(f"  P2 消费组均值回归 vs P3+ 基线 对比")
        print(f"{'='*95}")

        print(f"\n  {'指标':<14} {'P3+基线':>10} {'P2均值回归':>12} {'改进':>10}")
        print(f"  {'-'*52}")
        for window, p3p, p2 in [("训练窗", P3PLUS_BASELINE["train"], train_m),
                                ("测试窗", P3PLUS_BASELINE["test"], test_m)]:
            print(f"\n  [{window} 组合]")
            for key, label in [("alpha_pct", "Alpha%"), ("sharpe", "夏普"),
                               ("max_drawdown_pct", "回撤%"), ("total_return_pct", "收益%")]:
                v3 = p3p.get(key, 0)
                v4 = p2.get(key, 0)
                d = v4 - v3
                print(f"  {label:<14} {v3:>+10.2f} {v4:>+12.2f} {d:>+10.2f}")

        # 消费组对比
        print(f"\n  [消费组明细对比]")
        for window, base_c, res_c in [("训练窗", P3PLUS_BASELINE["consumer_train"],
                                        train_groups.get("消费稳健型", {})),
                                       ("测试窗", P3PLUS_BASELINE["consumer_test"],
                                        test_groups.get("消费稳健型", {}))]:
            print(f"\n  [{window}]")
            for key, label in [("sharpe", "夏普"), ("total_return", "收益"),
                               ("alpha", "Alpha"), ("trade_count", "交易数")]:
                v3 = base_c.get(key, 0)
                v4 = res_c.get(key, 0)
                if key in ("total_return", "alpha"):
                    print(f"  {label:<14} {v3*100:>+10.2f}% {v4*100:>+11.2f}% {v4*100-v3*100:>+10.2f}%")
                else:
                    print(f"  {label:<14} {v3:>+10} {v4:>+12} {v4-v3:>+10.2f}")
            wr = res_c.get("win_rate", 0)
            print(f"  {'胜率':<14} {'—':>10} {wr*100:>11.0f}% {'':>10}")
            if res_c.get("exit_reasons"):
                print(f"  {'退出原因':<14} {res_c['exit_reasons']}")

        # 三角评估
        print(f"\n{'='*95}")
        print(f"  稳定-收益-回撤 三角评估 (P2)")
        print(f"{'='*95}")
        for window, m in [("训练窗(震荡市)", train_m), ("测试窗(牛市)", test_m)]:
            sharpe_ok = m["sharpe"] > 1.0
            alpha_ok = m["alpha_pct"] > 0
            dd_ok = m["max_drawdown_pct"] > -10
            print(f"  {window}: 夏普{m['sharpe']:.3f}{'✅' if sharpe_ok else '❌'} "
                  f"Alpha{m['alpha_pct']:+.2f}%{'✅' if alpha_ok else '❌'} "
                  f"回撤{m['max_drawdown_pct']:.1f}%{'✅' if dd_ok else '❌'}")
        all_ok = (train_m["sharpe"] > 1.0 and train_m["alpha_pct"] > 0 and
                  train_m["max_drawdown_pct"] > -10 and
                  test_m["sharpe"] > 1.0 and test_m["alpha_pct"] > 0 and
                  test_m["max_drawdown_pct"] > -10)
        print(f"\n  六项全达标: {'✅ 是' if all_ok else '❌ 否'}")

        # 消费组 Alpha 是否转正
        c_train_alpha = train_groups.get("消费稳健型", {}).get("alpha", 0)
        c_test_alpha = test_groups.get("消费稳健型", {}).get("alpha", 0)
        print(f"\n  消费组Alpha转正:")
        print(f"    训练窗: {P3PLUS_BASELINE['consumer_train']['alpha']*100:+.2f}% → {c_train_alpha*100:+.2f}% "
              f"{'✅转正' if c_train_alpha > 0 else '❌仍为负'}")
        print(f"    测试窗: {P3PLUS_BASELINE['consumer_test']['alpha']*100:+.2f}% → {c_test_alpha*100:+.2f}% "
              f"{'✅转正' if c_test_alpha > 0 else '❌仍为负'}")

        # 报告
        report = generate_report(run_time, train_m, test_m, train_groups, test_groups, all_ok)
        with open(REPORT_MD, "w", encoding="utf-8") as f:
            f.write(report)
        result = {"train": train_m, "test": test_m,
                  "train_groups": {g: {k: v for k, v in r.items() if k != "daily_values"}
                                   for g, r in train_groups.items()},
                  "test_groups": {g: {k: v for k, v in r.items() if k != "daily_values"}
                                  for g, r in test_groups.items()}}
        with open(RESULT_JSON, "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2, default=str)
        print(f"\n✓ 报告 → {REPORT_MD}")
        print(f"✓ 数据 → {RESULT_JSON}")

    finally:
        if pref_existed and os.path.exists(pref_backup):
            shutil.move(pref_backup, _USER_PREF_FILE)


def generate_report(run_time, train_m, test_m, train_groups, test_groups, all_ok):
    L = []
    L.append("# P2 消费组均值回归验证报告\n")
    L.append(f"**运行时间**: {run_time}")
    L.append(f"**版本**: P2 (完整实现 — 信号层+退出层+配置层)\n")

    L.append("## P2 改动 (三层协同)\n")
    L.append("### 与P4的区别: P2真正落地代码改动\n")
    L.append("| 层级 | P4(失败) | P2(完整) |")
    L.append("|------|---------|---------|")
    L.append("| 信号层 | strategy_mode字段未实现, 实际跑趋势跟踪 | filter/classifier跳过MA60方向约束 |")
    L.append("| 退出层 | 未改退出逻辑, trailing不适配 | target_profit_pct=6%目标止盈+禁trailing+hard_stop=8% |")
    L.append("| 配置层 | 配置未改 | RSI0.28/KDJ0.15/MA60降到0.10, oversold收紧 |")
    L.append("")
    L.append("### 消费组配置变更\n")
    L.append("| 参数 | 趋势跟踪(P3+) | 均值回归(P2) |")
    L.append("|------|---------------|-------------|")
    L.append("| strategy_mode | trend_following | mean_reversion |")
    L.append("| RSI权重 | 0.18 | 0.28 |")
    L.append("| KDJ权重 | 0.08 | 0.15 |")
    L.append("| MA60权重 | 0.18 | 0.10 |")
    L.append("| RSI oversold | 35 | 30 |")
    L.append("| KDJ oversold_j | 30 | 15 |")
    L.append("| strength RSI | 1.0 | 1.4 |")
    L.append("| strength KDJ | 1.0 | 1.3 |")
    L.append("| strength MA60 | 1.2 | 0.8 |")
    L.append("| atr_stop_mult | 1.8 | 1.2 |")
    L.append("| target_profit_pct | — | 6% |")
    L.append("| hard_stop_pct | 12% | 8% |")
    L.append("| trailing | 全体制启用 | 全体制禁用 |")
    L.append("")
    L.append("> 其他组(科技/周期)完全不变, 零影响.\n")

    L.append("## 组合级指标对比\n")
    for window, p3p, p2 in [("训练窗(震荡市)", P3PLUS_BASELINE["train"], train_m),
                            ("测试窗(牛市)", P3PLUS_BASELINE["test"], test_m)]:
        L.append(f"### {window}\n")
        L.append("| 指标 | P3+基线 | P2均值回归 | 改进 |")
        L.append("|------|---------|-----------|------|")
        for key, lbl in [("alpha_pct", "Alpha%"), ("sharpe", "夏普"),
                         ("max_drawdown_pct", "回撤%"), ("total_return_pct", "收益%")]:
            v3 = p3p.get(key, 0)
            v4 = p2.get(key, 0)
            d = v4 - v3
            L.append(f"| {lbl} | {v3} | {v4} | {d:+} |")
        L.append("")

    L.append("## 消费组明细对比\n")
    for window, base_c, res_c in [("训练窗", P3PLUS_BASELINE["consumer_train"],
                                    train_groups.get("消费稳健型", {})),
                                   ("测试窗", P3PLUS_BASELINE["consumer_test"],
                                    test_groups.get("消费稳健型", {}))]:
        L.append(f"### {window}\n")
        L.append("| 指标 | P3+趋势跟踪 | P2均值回归 | 改进 |")
        L.append("|------|------------|-----------|------|")
        for key, lbl in [("sharpe", "夏普"), ("total_return", "收益%"),
                         ("alpha", "Alpha%"), ("trade_count", "交易数")]:
            v3 = base_c.get(key, 0)
            v4 = res_c.get(key, 0)
            if key in ("total_return", "alpha"):
                d = (v4 - v3) * 100
                L.append(f"| {lbl} | {v3*100:+.2f} | {v4*100:+.2f} | {d:+.2f} |")
            else:
                d = v4 - v3
                L.append(f"| {lbl} | {v3} | {v4} | {d:+.2f} |")
        wr = res_c.get("win_rate", 0)
        L.append(f"| 胜率 | — | {wr*100:.0f}% | — |")
        if res_c.get("exit_reasons"):
            L.append(f"| 退出原因 | — | {res_c['exit_reasons']} | — |")
        L.append("")

    L.append("## 三角评估\n")
    L.append("| 维度 | 标准 | 训练窗 | 测试窗 |")
    L.append("|------|------|--------|--------|")
    sharpe_t = "✅" if train_m["sharpe"] > 1.0 else "❌"
    sharpe_s = "✅" if test_m["sharpe"] > 1.0 else "❌"
    alpha_t = "✅" if train_m["alpha_pct"] > 0 else "❌"
    alpha_s = "✅" if test_m["alpha_pct"] > 0 else "❌"
    dd_t = "✅" if train_m["max_drawdown_pct"] > -10 else "❌"
    dd_s = "✅" if test_m["max_drawdown_pct"] > -10 else "❌"
    L.append(f"| 稳定 | 夏普>1.0 | {train_m['sharpe']:.3f}{sharpe_t} | {test_m['sharpe']:.3f}{sharpe_s} |")
    L.append(f"| 收益 | Alpha>0 | {train_m['alpha_pct']:+.2f}%{alpha_t} | {test_m['alpha_pct']:+.2f}%{alpha_s} |")
    L.append(f"| 回撤 | <10% | {train_m['max_drawdown_pct']:.1f}%{dd_t} | {test_m['max_drawdown_pct']:.1f}%{dd_s} |")
    L.append("")

    L.append("## 结论\n")
    c_train_alpha = train_groups.get("消费稳健型", {}).get("alpha", 0)
    c_test_alpha = test_groups.get("消费稳健型", {}).get("alpha", 0)
    if all_ok:
        L.append("**✅ P2 全量验证通过! 六项三角指标全部达标!**\n")
    else:
        L.append("**⚠️ P2 部分指标未达标**, 详情见三角评估表.\n")
    L.append(f"- 消费组训练窗Alpha: {P3PLUS_BASELINE['consumer_train']['alpha']*100:+.2f}% → {c_train_alpha*100:+.2f}%")
    L.append(f"- 消费组测试窗Alpha: {P3PLUS_BASELINE['consumer_test']['alpha']*100:+.2f}% → {c_test_alpha*100:+.2f}%")
    L.append(f"- 组合训练窗Alpha: {P3PLUS_BASELINE['train']['alpha_pct']:+.2f}% → {train_m['alpha_pct']:+.2f}%")
    L.append(f"- 组合测试窗Alpha: {P3PLUS_BASELINE['test']['alpha_pct']:+.2f}% → {test_m['alpha_pct']:+.2f}%")
    return "\n".join(L)


if __name__ == "__main__":
    main()
