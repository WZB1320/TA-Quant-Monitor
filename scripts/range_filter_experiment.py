"""
震荡市空仓实验 — ADX regime 识别 + 震荡市不交易

背景:
  裕度过滤实验已证伪「信号质量」假设: 过滤脆弱信号后 Alpha 更差。
  新假设: 震荡市所有趋势信号都无效 (无论强弱), 应整段不交易。
  regime 分布: 训练窗 trending43%/transition42%/ranging15%, 测试窗 trending51%/transition41%/ranging8%

对比 3 个级别:
  A. 基线: 不过滤 (所有 regime 都交易)
  B. 过滤震荡市: trade_regimes={trending, transition} (仅 ranging 不交易, 占15%)
  C. 仅趋势市: trade_regimes={trending} (ranging+transition 都不交易, 占57%)

指标: 夏普 / 收益 / Alpha(vs沪深300) / 交易数 / 胜率
窗口: 训练窗(震荡市) + 测试窗(牛市)

预期:
  训练窗: C(仅趋势市) Alpha 转正或接近0 (震荡+转换期不交易, 避免无效交易)
  测试窗: C 夏普基本持平 (牛市trending占51%, 过渡期交易也有效)

用法:
  python scripts/range_filter_experiment.py
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

TRAIN_START, TRAIN_END = "2024-07-01", "2025-06-30"
TEST_START, TEST_END = "2025-07-01", "2026-06-30"
DATA_START, DATA_END = "2024-02-01", "2026-07-13"
BENCHMARK = "sh.000300"

# 3个过滤级别
LEVELS = {
    "baseline": None,                          # 不过滤
    "filter_ranging": {"trending", "transition"},  # 仅过滤震荡市(ranging 15%)
    "trending_only": {"trending"},             # 仅趋势市(ranging+transition 57%不交易)
}

RESULT_JSON = os.path.join(project_root, "data", "range_filter_result.json")
REPORT_MD = os.path.join(project_root, "data", "range_filter_report.md")


def load_watchlist():
    cfg_path = os.path.join(project_root, "config", "strategy_config.json")
    with open(cfg_path, "r", encoding="utf-8") as f:
        cfg = json.load(f)
    return {g: [s["code"] for s in stocks]
            for g, stocks in cfg["strategy_config"]["watchlist"].items()
            if not g.startswith("_")}


def run_backtest(data_map, benchmark_df, group_codes, trade_regimes, start, end):
    GroupConfig._instance = None
    GroupConfig._config = None
    try:
        engine = BacktestEngine(
            initial_capital=100000, lookback_days=120, position_ratio=0.3,
            commission_rate=0.00025, stamp_tax=0.001, slippage=0.0001,
            signal_dedup_days=5, risk_per_trade=0.05, atr_stop_mult=2.0,
            forced_regime=None, trade_regimes=trade_regimes,
        )
        sub_map = {c: data_map[c] for c in group_codes if c in data_map}
        m = engine.run(sub_map, benchmark_df=benchmark_df, start_date=start, end_date=end)
        return {
            "sharpe": round(getattr(m, "sharpe_ratio", 0) or 0, 3),
            "total_return_pct": round((getattr(m, "total_return", 0) or 0) * 100, 2),
            "alpha_pct": round((getattr(m, "alpha", 0) or 0) * 100, 2),
            "max_drawdown_pct": round((getattr(m, "max_drawdown", 0) or 0) * 100, 2),
            "trade_count": getattr(m, "trade_count", 0) or 0,
            "win_rate_pct": round((getattr(m, "win_rate", 0) or 0) * 100, 1),
        }
    except Exception as e:
        return {"error": str(e)}


def main():
    run_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print("=" * 70)
    print("  震荡市空仓实验 — ADX regime 识别 + 震荡市不交易")
    print(f"  级别: A=基线 B=过滤ranging C=仅trending")
    print(f"  训练:{TRAIN_START}~{TRAIN_END} | 测试:{TEST_START}~{TEST_END}")
    print("=" * 70)

    pref_existed = os.path.exists(_USER_PREF_FILE)
    pref_backup = _USER_PREF_FILE + ".range_bak"
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
        for group_name, codes in watchlist.items():
            if group_name.startswith("_"):
                continue
            group_codes = [c for c in codes if c in data_map]
            if len(group_codes) < 2:
                continue
            print(f"\n{'─'*60}\n分组: {group_name} ({len(group_codes)}只)\n{'─'*60}")
            group_result = {}
            for window, start, end in [("train", TRAIN_START, TRAIN_END),
                                        ("test", TEST_START, TEST_END)]:
                wl = "训练" if window == "train" else "测试"
                print(f"\n  [{wl}窗]")
                window_result = {}
                for label, regimes in LEVELS.items():
                    m = run_backtest(data_map, benchmark_df, group_codes, regimes, start, end)
                    window_result[label] = m
                    if "error" not in m:
                        print(f"    {label:15s}: 夏普={m['sharpe']:+.3f} 收益={m['total_return_pct']:+.2f}% "
                              f"Alpha={m['alpha_pct']:+.2f}% 交易={m['trade_count']}笔 胜率={m['win_rate_pct']}%")
                group_result[window] = window_result
            all_results[group_name] = group_result

        os.makedirs(os.path.dirname(RESULT_JSON), exist_ok=True)
        output = {
            "run_time": run_time,
            "config": {"levels": {k: list(v) if v else None for k, v in LEVELS.items()},
                       "train": [TRAIN_START, TRAIN_END], "test": [TEST_START, TEST_END]},
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
        print(f"\n{'='*70}\n  汇总: 训练窗 Alpha (基线 → 过滤ranging → 仅trending)\n{'='*70}")
        print(f"{'分组':<12} {'基线Alpha':>10} {'过滤ranging':>12} {'仅trending':>12} {'基线夏普':>10} {'仅trend夏普':>12}")
        turned = 0; total = 0
        for g, gr in all_results.items():
            b = gr["train"]["baseline"]; r = gr["train"]["filter_ranging"]; t = gr["train"]["trending_only"]
            if "error" in b or "error" in t: continue
            total += 1
            mark = "✅" if (b["alpha_pct"] < 0 and t["alpha_pct"] >= 0) else ("—" if b["alpha_pct"] >= 0 else "❌")
            if b["alpha_pct"] < 0 and t["alpha_pct"] >= 0: turned += 1
            print(f"{g:<12} {b['alpha_pct']:>10.2f} {r['alpha_pct']:>12.2f} {t['alpha_pct']:>12.2f} "
                  f"{b['sharpe']:>10.3f} {t['sharpe']:>12.3f} {mark}")
        print(f"\n  训练窗 Alpha 由负转正: {turned}/{total} 组")
    finally:
        if pref_existed and os.path.exists(pref_backup):
            shutil.move(pref_backup, _USER_PREF_FILE)


def generate_report(output, run_time):
    L = []
    L.append("# 震荡市空仓实验报告 — ADX regime 识别 + 震荡市不交易")
    L.append("")
    L.append(f"**运行时间**: {run_time}")
    L.append(f"**训练窗**: {output['config']['train'][0]} ~ {output['config']['train'][1]} (震荡市)")
    L.append(f"**测试窗**: {output['config']['test'][0]} ~ {output['config']['test'][1]} (牛市)")
    L.append("")
    L.append("> **新假设**: 震荡市所有趋势信号都无效(裕度过滤已证伪信号质量假设), 应整段不交易")
    L.append("> regime分布: 训练窗 trending43%/transition42%/ranging15%")
    L.append("")
    L.append("**3个级别**:")
    L.append("- A.基线: 不过滤 (所有regime都交易)")
    L.append("- B.过滤ranging: 仅ranging(15%)不交易")
    L.append("- C.仅trending: ranging+transition(57%)都不交易, 仅趋势市交易")
    L.append("")

    L.append("## 一、训练窗对比 (震荡市 — 核心验证)")
    L.append("")
    L.append("| 分组 | 级别 | 夏普 | 收益% | Alpha% | 交易数 | 胜率% | 回撤% |")
    L.append("|------|------|------|-------|--------|--------|-------|-------|")
    for g, gr in output["results"].items():
        for label in ["baseline", "filter_ranging", "trending_only"]:
            r = gr["train"][label]
            if "error" in r: continue
            name = {"baseline":"A基线","filter_ranging":"B过滤ranging","trending_only":"C仅trending"}[label]
            L.append(f"| {g} | {name} | {r['sharpe']:+.3f} | {r['total_return_pct']:+.2f} | "
                     f"{r['alpha_pct']:+.2f} | {r['trade_count']} | {r['win_rate_pct']} | {r['max_drawdown_pct']:.2f} |")
    L.append("")

    L.append("## 二、测试窗对比 (牛市 — 稳健性验证)")
    L.append("")
    L.append("| 分组 | 级别 | 夏普 | 收益% | Alpha% | 交易数 | 胜率% |")
    L.append("|------|------|------|-------|--------|--------|-------|")
    for g, gr in output["results"].items():
        for label in ["baseline", "filter_ranging", "trending_only"]:
            r = gr["test"][label]
            if "error" in r: continue
            name = {"baseline":"A基线","filter_ranging":"B过滤ranging","trending_only":"C仅trending"}[label]
            L.append(f"| {g} | {name} | {r['sharpe']:+.3f} | {r['total_return_pct']:+.2f} | "
                     f"{r['alpha_pct']:+.2f} | {r['trade_count']} | {r['win_rate_pct']} |")
    L.append("")

    L.append("## 三、变化幅度 (C仅trending - A基线)")
    L.append("")
    L.append("| 分组 | 窗口 | ΔAlpha% | Δ夏普 | Δ交易数 | Alpha转正? |")
    L.append("|------|------|---------|-------|---------|-----------|")
    train_turned = 0; train_total = 0
    train_alpha_up = 0; train_sharpe_up = 0
    for g, gr in output["results"].items():
        for window, wl in [("train","训练"),("test","测试")]:
            b = gr[window]["baseline"]; t = gr[window]["trending_only"]
            if "error" in b or "error" in t: continue
            da = round(t["alpha_pct"] - b["alpha_pct"], 2)
            ds = round(t["sharpe"] - b["sharpe"], 3)
            dt = t["trade_count"] - b["trade_count"]
            turned = "✅是" if (b["alpha_pct"] < 0 and t["alpha_pct"] >= 0) else ("—" if b["alpha_pct"] >= 0 else "否")
            L.append(f"| {g} | {wl} | {da:+.2f} | {ds:+.3f} | {dt} | {turned} |")
            if window == "train":
                train_total += 1
                if b["alpha_pct"] < 0 and t["alpha_pct"] >= 0: train_turned += 1
                if t["alpha_pct"] > b["alpha_pct"]: train_alpha_up += 1
                if t["sharpe"] > b["sharpe"]: train_sharpe_up += 1
    L.append("")

    L.append("## 四、结论")
    L.append("")
    L.append(f"**训练窗(震荡市)统计 (C仅trending vs A基线)**:")
    L.append(f"- Alpha提升: {train_alpha_up}/{train_total} 组")
    L.append(f"- Alpha由负转正: {train_turned}/{train_total} 组")
    L.append(f"- 夏普提升: {train_sharpe_up}/{train_total} 组")
    L.append("")

    if train_turned >= train_total * 0.6:
        L.append("✅ **假设成立**: 仅趋势市交易使多数分组训练窗Alpha转正")
        L.append("")
        L.append("震荡市(含transition)不交易后, 训练窗Alpha转正, 证实:")
        L.append("趋势信号在非趋势市(regime≠trending)确实无效, 整段不交易优于过滤信号。")
        L.append("")
        L.append("**落地建议**:")
        L.append("- 实盘增加 trade_regimes={'trending'} 过滤, 仅趋势市开仓")
        L.append("- 注意测试窗影响: 若测试窗夏普大幅下降, 可放宽到 {trending, transition}")
    elif train_alpha_up >= train_total * 0.6:
        L.append("⚠️ **部分成立**: 多数分组Alpha提升但未全部转正")
        L.append("")
        L.append("仅趋势市交易方向正确(Alpha提升), 但力度或regime识别精度不足。")
        L.append("可能需要: regime识别升级(多维) + 过滤阈值调整 + 分组差异化。")
    else:
        L.append("❌ **假设不成立**: 仅趋势市交易后Alpha未改善")
        L.append("")
        L.append("说明训练窗Alpha缺失根因不是regime适配问题, 而是:")
        L.append("- 可能基准选择问题(沪深300不匹配中小盘选股)")
        L.append("- 可能趋势信号本身在2024-2025全期都无效(非regime问题)")
        L.append("建议: 换基准(等权自选股)重测Alpha + 分析trending期交易盈亏")

    return "\n".join(L)


if __name__ == "__main__":
    main()
