"""
信号质量过滤对比实验 — 验证「裕度>10过滤」能否让训练窗Alpha转正

原理:
  裕度 = score - score_threshold
  裕度>10过滤 ⟺ score > threshold + 10 ⟺ score_threshold 提高10分
  (信号脆弱性分析已证实: 49%信号裕度<5, 75%裕度<10, 是噪声)

对比:
  基线: 原始 score_threshold (当前实盘)
  过滤: score_threshold + 10 (仅裕度>10的稳健信号才买入)

指标: 夏普 / 总收益 / Alpha(vs沪深300) / 交易数 / 胜率 / 最大回撤
窗口: 训练窗(震荡市) + 测试窗(牛市)

预期:
  训练窗: 过滤后交易数大减(剔除噪声), Alpha转正, 夏普提升
  测试窗: 过滤后交易数略减, 夏普基本持平(牛市信号裕度本就大)

用法:
  python scripts/margin_filter_experiment.py
"""
import sys
import os
import json
import shutil
import warnings
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

# ── 配置 ──
TRAIN_START, TRAIN_END = "2024-07-01", "2025-06-30"
TEST_START, TEST_END = "2025-07-01", "2026-06-30"
DATA_START, DATA_END = "2024-02-01", "2026-07-13"
BENCHMARK = "sh.000300"
MARGIN_FILTER = 10  # 裕度过滤阈值 (score_threshold + 10)

RESULT_JSON = os.path.join(project_root, "data", "margin_filter_result.json")
REPORT_MD = os.path.join(project_root, "data", "margin_filter_report.md")


def load_watchlist():
    cfg_path = os.path.join(project_root, "config", "strategy_config.json")
    with open(cfg_path, "r", encoding="utf-8") as f:
        cfg = json.load(f)
    return {g: [s["code"] for s in stocks]
            for g, stocks in cfg["strategy_config"]["watchlist"].items()
            if not g.startswith("_")}


def run_backtest(data_map, benchmark_df, group_codes, threshold_delta,
                  start, end):
    """跑回测, threshold_delta=0基线, =10裕度过滤. 返回metrics摘要"""
    GroupConfig._instance = None
    GroupConfig._config = None

    original_get_all = GroupConfig.get_all_group_params

    def patched_get_all(self, code):
        params = original_get_all(self, code)
        if threshold_delta != 0:
            params["score_threshold"] = params.get("score_threshold", 25) + threshold_delta
        return params
    GroupConfig.get_all_group_params = patched_get_all

    try:
        engine = BacktestEngine(
            initial_capital=100000, lookback_days=120, position_ratio=0.3,
            commission_rate=0.00025, stamp_tax=0.001, slippage=0.0001,
            signal_dedup_days=5, risk_per_trade=0.05, atr_stop_mult=2.0,
            forced_regime=None,
        )
        sub_map = {c: data_map[c] for c in group_codes if c in data_map}
        m = engine.run(sub_map, benchmark_df=benchmark_df,
                        start_date=start, end_date=end)
        return {
            "sharpe": round(getattr(m, "sharpe_ratio", 0) or 0, 3),
            "total_return_pct": round(getattr(m, "total_return", 0) or 0, 2),
            "alpha_pct": round(getattr(m, "alpha", 0) or 0, 2),
            "benchmark_return_pct": round(getattr(m, "benchmark_return", 0) or 0, 2),
            "max_drawdown_pct": round(getattr(m, "max_drawdown", 0) or 0, 2),
            "trade_count": getattr(m, "trade_count", 0) or 0,
            "win_rate_pct": round((getattr(m, "win_rate", 0) or 0) * 100, 1),
        }
    except Exception as e:
        return {"error": str(e)}
    finally:
        GroupConfig.get_all_group_params = original_get_all


def main():
    run_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print("=" * 70)
    print("  信号质量过滤对比实验 — 裕度>10 过滤")
    print(f"  过滤: score_threshold + {MARGIN_FILTER} (仅裕度>10信号才买入)")
    print(f"  训练:{TRAIN_START}~{TRAIN_END} | 测试:{TEST_START}~{TEST_END}")
    print("=" * 70)

    # 清除用户偏好
    pref_existed = os.path.exists(_USER_PREF_FILE)
    pref_backup = _USER_PREF_FILE + ".margin_bak"
    if pref_existed:
        shutil.copy2(_USER_PREF_FILE, pref_backup)
    UserPreferences().clear_all()

    try:
        watchlist = load_watchlist()
        dm = DataManager()

        # 拉全量数据 + 基准
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

        for group_name, codes in watchlist.items():
            if group_name.startswith("_"):
                continue
            group_codes = [c for c in codes if c in data_map]
            if len(group_codes) < 2:
                continue

            print(f"\n{'─' * 60}")
            print(f"分组: {group_name} ({len(group_codes)}只)")
            print(f"{'─' * 60}")

            group_result = {}
            for window, start, end in [("train", TRAIN_START, TRAIN_END),
                                        ("test", TEST_START, TEST_END)]:
                wl = "训练" if window == "train" else "测试"
                print(f"\n  [{wl}窗]")
                window_result = {}
                for label, delta in [("baseline", 0), ("filtered", MARGIN_FILTER)]:
                    m = run_backtest(data_map, benchmark_df, group_codes,
                                      delta, start, end)
                    window_result[label] = m
                    if "error" not in m:
                        print(f"    {label:9s}: 夏普={m['sharpe']:+.3f} "
                              f"收益={m['total_return_pct']:+.2f}% "
                              f"Alpha={m['alpha_pct']:+.2f}% "
                              f"交易={m['trade_count']}笔 胜率={m['win_rate_pct']}%")
                    else:
                        print(f"    {label}: ERROR {m['error'][:50]}")
                group_result[window] = window_result
            all_results[group_name] = group_result

        # ── 保存 ──
        os.makedirs(os.path.dirname(RESULT_JSON), exist_ok=True)
        output = {
            "run_time": run_time,
            "config": {
                "margin_filter": MARGIN_FILTER,
                "train": [TRAIN_START, TRAIN_END],
                "test": [TEST_START, TEST_END],
                "benchmark": BENCHMARK,
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
        print(f"\n{'=' * 70}")
        print("  汇总: 训练窗 Alpha 变化 (基线 → 过滤)")
        print(f"{'=' * 70}")
        print(f"{'分组':<12} {'基线Alpha%':>12} {'过滤Alpha%':>12} {'Alpha变化':>10} "
              f"{'基线夏普':>10} {'过滤夏普':>10}")
        train_alpha_improved = 0
        train_sharpe_improved = 0
        for g, gr in all_results.items():
            b = gr["train"]["baseline"]
            f_ = gr["train"]["filtered"]
            if "error" in b or "error" in f_:
                continue
            d_alpha = f_["alpha_pct"] - b["alpha_pct"]
            d_sharpe = f_["sharpe"] - b["sharpe"]
            mark_a = "↑" if d_alpha > 0 else "↓"
            mark_s = "↑" if d_sharpe > 0 else "↓"
            print(f"{g:<12} {b['alpha_pct']:>12.2f} {f_['alpha_pct']:>12.2f} "
                  f"{mark_a}{abs(d_alpha):>8.2f}  {b['sharpe']:>10.3f} {f_['sharpe']:>10.3f}")
            if f_["alpha_pct"] > b["alpha_pct"]:
                train_alpha_improved += 1
            if f_["sharpe"] > b["sharpe"]:
                train_sharpe_improved += 1
        total = len(all_results)
        print(f"\n  训练窗: Alpha提升 {train_alpha_improved}/{total} 组, "
              f"夏普提升 {train_sharpe_improved}/{total} 组")

    finally:
        if pref_existed and os.path.exists(pref_backup):
            shutil.move(pref_backup, _USER_PREF_FILE)


def generate_report(output, run_time):
    L = []
    L.append("# 信号质量过滤对比实验报告 — 裕度>10 过滤")
    L.append("")
    L.append(f"**运行时间**: {run_time}")
    cfg = output["config"]
    L.append(f"**过滤规则**: score_threshold + {cfg['margin_filter']} (仅裕度>{cfg['margin_filter']}的信号才买入)")
    L.append(f"**训练窗**: {cfg['train'][0]} ~ {cfg['train'][1]} (震荡市)")
    L.append(f"**测试窗**: {cfg['test'][0]} ~ {cfg['test'][1]} (牛市)")
    L.append(f"**基准**: {cfg['benchmark']} (沪深300)")
    L.append("")
    L.append("> **假设**: 信号脆弱性分析发现训练窗49%信号裕度<5(噪声), 过滤后Alpha应转正")
    L.append("")

    # ── 一、训练窗对比 (核心) ──
    L.append("## 一、训练窗对比 (震荡市 — 核心验证)")
    L.append("")
    L.append("| 分组 | 版本 | 夏普 | 收益% | Alpha% | 基准% | 交易数 | 胜率% | 回撤% |")
    L.append("|------|------|------|-------|--------|-------|--------|-------|-------|")
    for g, gr in output["results"].items():
        for label in ["baseline", "filtered"]:
            r = gr["train"][label]
            if "error" in r:
                continue
            name = "基线" if label == "baseline" else "过滤(+10)"
            L.append(f"| {g} | {name} | {r['sharpe']:+.3f} | {r['total_return_pct']:+.2f} | "
                     f"{r['alpha_pct']:+.2f} | {r['benchmark_return_pct']:+.2f} | "
                     f"{r['trade_count']} | {r['win_rate_pct']} | {r['max_drawdown_pct']:.2f} |")
    L.append("")

    # ── 二、测试窗对比 ──
    L.append("## 二、测试窗对比 (牛市 — 稳健性验证)")
    L.append("")
    L.append("| 分组 | 版本 | 夏普 | 收益% | Alpha% | 交易数 | 胜率% |")
    L.append("|------|------|------|-------|--------|--------|-------|")
    for g, gr in output["results"].items():
        for label in ["baseline", "filtered"]:
            r = gr["test"][label]
            if "error" in r:
                continue
            name = "基线" if label == "baseline" else "过滤(+10)"
            L.append(f"| {g} | {name} | {r['sharpe']:+.3f} | {r['total_return_pct']:+.2f} | "
                     f"{r['alpha_pct']:+.2f} | {r['trade_count']} | {r['win_rate_pct']} |")
    L.append("")

    # ── 三、Alpha与夏普变化 ──
    L.append("## 三、变化幅度 (过滤 - 基线)")
    L.append("")
    L.append("| 分组 | 窗口 | ΔAlpha% | Δ夏普 | Δ交易数 | Alpha转正? |")
    L.append("|------|------|---------|-------|---------|-----------|")
    train_alpha_turned = 0
    train_alpha_improved = 0
    train_sharpe_improved = 0
    for g, gr in output["results"].items():
        for window, wl in [("train", "训练"), ("test", "测试")]:
            b = gr[window]["baseline"]
            f_ = gr[window]["filtered"]
            if "error" in b or "error" in f_:
                continue
            d_alpha = round(f_["alpha_pct"] - b["alpha_pct"], 2)
            d_sharpe = round(f_["sharpe"] - b["sharpe"], 3)
            d_trades = f_["trade_count"] - b["trade_count"]
            # Alpha转正: 基线Alpha<0 且 过滤后Alpha>0
            turned = "✅是" if (b["alpha_pct"] < 0 and f_["alpha_pct"] > 0) else ("—" if b["alpha_pct"] >= 0 else "否")
            L.append(f"| {g} | {wl} | {d_alpha:+.2f} | {d_sharpe:+.3f} | {d_trades} | {turned} |")
            if window == "train":
                if f_["alpha_pct"] > b["alpha_pct"]:
                    train_alpha_improved += 1
                if b["alpha_pct"] < 0 and f_["alpha_pct"] > 0:
                    train_alpha_turned += 1
                if f_["sharpe"] > b["sharpe"]:
                    train_sharpe_improved += 1
    L.append("")

    # ── 四、结论 ──
    L.append("## 四、结论")
    L.append("")
    total = len(output["results"])
    L.append(f"**训练窗(震荡市)统计**:")
    L.append(f"- Alpha提升: {train_alpha_improved}/{total} 组")
    L.append(f"- Alpha由负转正: {train_alpha_turned}/{total} 组")
    L.append(f"- 夏普提升: {train_sharpe_improved}/{total} 组")
    L.append("")

    if train_alpha_turned >= total * 0.6:
        L.append("✅ **假设成立**: 裕度>10过滤使多数分组训练窗Alpha转正")
        L.append("")
        L.append("信号脆弱性分析的根因诊断被验证: 震荡市Alpha缺失的核心原因是噪声信号(裕度<10)。")
        L.append("过滤掉这些信号后, 训练窗Alpha转正, 说明剩余的稳健信号有真实预测力。")
        L.append("")
        L.append("**落地建议**:")
        L.append("- 实盘增加裕度过滤: score - score_threshold > 10 才执行买入")
        L.append("- 或等价地: 各分组 score_threshold 提高10分")
        L.append("- 注意测试窗影响: 若测试窗夏普大幅下降, 需分组差异化过滤(仅震荡市过滤)")
    elif train_alpha_improved >= total * 0.6:
        L.append("⚠️ **部分成立**: 过滤后多数分组Alpha提升, 但未全部转正")
        L.append("")
        L.append("过滤方向正确(Alpha提升), 但力度不够或根因更复杂。")
        L.append("可能需要: 更大裕度阈值(>15) + 分组差异化 + regime配合。")
    else:
        L.append("❌ **假设不成立**: 过滤后Alpha未改善")
        L.append("")
        L.append("说明震荡市Alpha缺失根因不是信号脆弱, 而是其他因素:")
        L.append("- 可能是止损止盈在震荡市频繁假突破触发")
        L.append("- 可能是选股方向在震荡市本身无效")
        L.append("建议转向: 止损触发频率分析 + 假突破率统计")

    return "\n".join(L)


if __name__ == "__main__":
    main()
