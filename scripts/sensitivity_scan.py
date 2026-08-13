"""
执行层参数敏感性扫描 (P0)

目标: 判断执行层关键参数是"平台"(稳健)还是"尖峰"(过拟合)
方法: 对每个参数在范围内取值, 跑回测, 看夏普/收益/回撤的变化曲线

扫描参数 (每个单独扫描, 其他用默认):
  1. atr_stop_mult:        1.5 / 2.0 / 2.5 / 3.0 / 3.5   (ATR止损倍率, 最重要)
  2. trail_mult_high:      0.4 / 0.5 / 0.6 / 0.7          (盈利>20%档收紧系数)
  3. hard_stop_pct:        0.08 / 0.10 / 0.12             (硬止损比例)

评判标准:
  - 平台(稳健): 参数±20%范围内夏普平稳变化 (波动<30%)
  - 尖峰(过拟合): 最优点夏普高, 邻域骤降 (波动>50%)

注: 用 forced_regime=trending (与实盘一致), 分组差异化 atr_stop_mult 在扫描时统一覆盖
"""
import sys
import os
import json
import copy
import warnings
from datetime import datetime

import pandas as pd

warnings.filterwarnings("ignore")
project_root = os.path.join(os.path.dirname(__file__), "..")
sys.path.insert(0, project_root)

from src.backtest.engine import BacktestEngine
from src.data_fetcher import DataManager
from src.config.group_config import GroupConfig  # 重置单例用
from src.config.runtime_mode import set_mode, RuntimeMode
from src.config.user_preferences import UserPreferences, _USER_PREF_FILE

# 必须在创建任何 SignalEngine/Filter 之前设置为回测模式,
# 否则 Filter 会从磁盘加载实时信号历史, 导致误去重 → 0交易
set_mode(RuntimeMode.BACKTEST)

# ── 配置 ──
TRAIN_START, TRAIN_END = "2024-07-01", "2025-06-30"
TEST_START, TEST_END = "2025-07-01", "2026-06-30"
DATA_START, DATA_END = "2024-02-01", "2026-07-13"
INITIAL_CAPITAL = 100000
FORCED_REGIME = None  # auto模式 (与walk_forward一致, 让ADX自动判体制, 能产生交易)

# 扫描参数网格
SCAN_GRID = {
    "atr_stop_mult": [1.5, 2.0, 2.5, 3.0, 3.5],
    "trail_mult_high": [0.4, 0.5, 0.6, 0.7],
    "hard_stop_pct": [0.08, 0.10, 0.12],
}
# 每个参数的"默认值"(当前实盘值)
DEFAULTS = {"atr_stop_mult": 2.5, "trail_mult_high": 0.6, "hard_stop_pct": 0.10}

RESULT_JSON = os.path.join(project_root, "data", "sensitivity_scan_result.json")
REPORT_MD = os.path.join(project_root, "data", "sensitivity_scan_report.md")


def load_watchlist():
    """从 strategy_config.json 读取分组股票池"""
    import json
    cfg_path = os.path.join(project_root, "config", "strategy_config.json")
    with open(cfg_path, "r", encoding="utf-8") as f:
        cfg = json.load(f)
    raw = cfg["strategy_config"]["watchlist"]
    groups = {}
    for g, stocks in raw.items():
        if g.startswith("_"):
            continue
        groups[g] = [s["code"] for s in stocks]
    return groups


def run_single_backtest(data_map, group_codes, atr_stop_mult, stop_loss_params,
                        start_date, end_date):
    """跑单次回测, 返回metrics摘要"""
    try:
        # 重置 GroupConfig 单例, 避免上次回测状态残留 (关键!)
        GroupConfig._instance = None
        GroupConfig._config = None

        # ⚠️ 关键: signal_executor 用 group_config.get_atr_stop_mult(symbol) 覆盖全局值,
        # 必须 patch 类方法让所有实例的分组专属值跟随扫描值, 否则扫描无效
        GroupConfig.get_atr_stop_mult = lambda self, symbol: atr_stop_mult

        engine = BacktestEngine(
            initial_capital=INITIAL_CAPITAL,
            lookback_days=120,
            position_ratio=0.3,
            commission_rate=0.00025,
            stamp_tax=0.001,
            slippage=0.0001,
            signal_dedup_days=5,
            risk_per_trade=0.05,
            atr_stop_mult=atr_stop_mult,
            forced_regime=FORCED_REGIME,
            stop_loss_params=stop_loss_params if stop_loss_params else None,
        )

        sub_map = {c: data_map[c] for c in group_codes if c in data_map}
        if len(sub_map) < 2:
            return None
        metrics = engine.run(sub_map, start_date=start_date, end_date=end_date)
        return {
            "sharpe": round(getattr(metrics, "sharpe_ratio", 0) or 0, 3),
            "total_return_pct": round(getattr(metrics, "total_return", 0) or 0, 2),
            "max_drawdown_pct": round(getattr(metrics, "max_drawdown", 0) or 0, 2),
            "trade_count": getattr(metrics, "trade_count", 0) or 0,
            "win_rate_pct": round((getattr(metrics, "win_rate", 0) or 0) * 100, 1),
        }
    except Exception as e:
        return {"error": str(e)}


def scan_param(param_name, values, data_map, group_codes, window_label,
               start, end):
    """扫描单个参数"""
    results = []
    for val in values:
        if param_name == "atr_stop_mult":
            m = run_single_backtest(data_map, group_codes, atr_stop_mult=val,
                                     stop_loss_params=None,
                                     start_date=start, end_date=end)
        else:
            sp = {param_name: val}
            m = run_single_backtest(data_map, group_codes,
                                     atr_stop_mult=DEFAULTS["atr_stop_mult"],
                                     stop_loss_params=sp,
                                     start_date=start, end_date=end)
        m["param_value"] = val
        m["param_name"] = param_name
        m["window"] = window_label
        results.append(m)
        if "error" not in m:
            print(f"    {param_name}={val}: 夏普={m['sharpe']:+.3f} "
                  f"收益={m['total_return_pct']:+.2f}% 回撤={m['max_drawdown_pct']:.2f}% "
                  f"交易={m['trade_count']}笔")
        else:
            print(f"    {param_name}={val}: ERROR {m['error'][:60]}")
    return results


def classify_stability(sharpes, default_idx):
    """判断参数是平台/尖峰/单调

    sharpes: 各参数值的夏普列表
    default_idx: 默认值在列表中的索引
    """
    if len(sharpes) < 3 or all(s == 0 for s in sharpes):
        return "unknown", 0
    default_sharpe = sharpes[default_idx]
    if default_sharpe == 0:
        return "unknown", 0
    # 邻域波动率: 相对于默认值的最大偏差
    neighbors = [s for i, s in enumerate(sharpes)
                 if abs(i - default_idx) == 1]
    if not neighbors:
        return "unknown", 0
    max_dev = max(abs(n - default_sharpe) for n in neighbors)
    volatility_pct = (max_dev / abs(default_sharpe)) * 100 if default_sharpe != 0 else 0

    # 单调性: 是否持续上升/下降
    diffs = [sharpes[i+1] - sharpes[i] for i in range(len(sharpes)-1)]
    same_sign = sum(1 for d in diffs if d != 0)
    pos = sum(1 for d in diffs if d > 0)
    neg = sum(1 for d in diffs if d < 0)
    is_monotone = (pos == same_sign or neg == same_sign) and same_sign > 0

    if volatility_pct > 50:
        return "spike(尖峰)", round(volatility_pct, 1)
    elif is_monotone and volatility_pct < 30:
        return "monotone(单调)", round(volatility_pct, 1)
    elif volatility_pct < 30:
        return "plateau(平台)", round(volatility_pct, 1)
    else:
        return "unstable(不稳定)", round(volatility_pct, 1)


def main():
    run_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print("=" * 70)
    print("  执行层参数敏感性扫描 (P0)")
    print(f"  forced_regime={FORCED_REGIME} | 训练:{TRAIN_START}~{TRAIN_END} | 测试:{TEST_START}~{TEST_END}")
    print(f"  扫描参数: {list(SCAN_GRID.keys())}")
    print("=" * 70)

    # 清除用户手动 regime 偏好 (避免 trending 偏好干扰, 与 walk_forward 一致)
    import shutil
    pref_existed = os.path.exists(_USER_PREF_FILE)
    pref_backup = _USER_PREF_FILE + ".sens_bak"
    if pref_existed:
        shutil.copy2(_USER_PREF_FILE, pref_backup)
    UserPreferences().clear_all()

    try:
        watchlist = load_watchlist()
        dm = DataManager()

        # 拉全量数据 (训练+测试)
        print("\n拉取数据...")
        all_codes = [c for codes in watchlist.values() for c in codes]
        data_map = {}
        for code in all_codes:
            df = dm.get_daily_kline(code, start_date=DATA_START, end_date=DATA_END)
            if df is not None and len(df) > 80:
                data_map[code] = df
        print(f"  拉取 {len(data_map)}/{len(all_codes)} 只股票")

        # 按分组扫描
        all_results = {}
        stability_summary = {}

        for group_name, codes in watchlist.items():
            if group_name.startswith("_"):
                continue
            group_codes = [c for c in codes if c in data_map]
            if len(group_codes) < 2:
                print(f"\n跳过 {group_name}: 股票不足")
                continue

            print(f"\n{'─' * 60}")
            print(f"分组: {group_name} ({len(group_codes)}只)")
            print(f"{'─' * 60}")

            # 不预切分数据, 传全量+日期给 engine.run
            group_result = {}
            for window, start, end in [("train", TRAIN_START, TRAIN_END),
                                        ("test", TEST_START, TEST_END)]:
                wl = "训练" if window == "train" else "测试"
                print(f"\n  [{wl}窗]")
                window_result = {}
                for param_name, values in SCAN_GRID.items():
                    print(f"  扫描 {param_name}:")
                    results = scan_param(param_name, values, data_map, group_codes, wl, start, end)
                    window_result[param_name] = results
                group_result[window] = window_result
            all_results[group_name] = group_result

            # 稳定性判断
            print(f"\n  稳定性判断 ({group_name}):")
            for param_name, values in SCAN_GRID.items():
                default_val = DEFAULTS[param_name]
                default_idx = values.index(default_val)
                stab_results = {}
                for window in ["train", "test"]:
                    sharpes = [r.get("sharpe", 0) for r in group_result[window][param_name]
                               if "error" not in r]
                    stab, vol = classify_stability(sharpes, default_idx)
                    stab_results[window] = {"stability": stab, "volatility_pct": vol,
                                             "sharpes": sharpes}
                stability_summary.setdefault(param_name, {})[group_name] = stab_results
                print(f"    {param_name}: 训练={stab_results['train']['stability']} "
                      f"测试={stab_results['test']['stability']}")

        # ── 保存结果 ──
        os.makedirs(os.path.dirname(RESULT_JSON), exist_ok=True)
        output = {
            "run_time": run_time,
            "config": {
                "forced_regime": FORCED_REGIME,
                "train": [TRAIN_START, TRAIN_END],
                "test": [TEST_START, TEST_END],
                "scan_grid": SCAN_GRID,
                "defaults": DEFAULTS,
            },
            "results": all_results,
            "stability": stability_summary,
        }
        with open(RESULT_JSON, "w", encoding="utf-8") as f:
            json.dump(output, f, ensure_ascii=False, indent=2, default=str)
        print(f"\n✓ 结果 → {RESULT_JSON}")

        # ── 生成报告 ──
        report = generate_report(output, run_time)
        with open(REPORT_MD, "w", encoding="utf-8") as f:
            f.write(report)
        print(f"✓ 报告 → {REPORT_MD}")

        # ── 汇总 ──
        print(f"\n{'=' * 70}")
        print("  汇总: 参数稳定性 (默认值邻域波动%)")
        print(f"{'=' * 70}")
        print(f"{'参数':<18} {'分组':<12} {'训练稳定性':<22} {'测试稳定性':<22}")
        for param_name in SCAN_GRID:
            for g in stability_summary.get(param_name, {}):
                s = stability_summary[param_name][g]
                print(f"{param_name:<18} {g:<12} "
                      f"{s['train']['stability']:<18}({s['train']['volatility_pct']}%)  "
                      f"{s['test']['stability']:<18}({s['test']['volatility_pct']}%)")
    finally:
        # 恢复用户偏好
        if pref_existed and os.path.exists(pref_backup):
            shutil.move(pref_backup, _USER_PREF_FILE)


def generate_report(output, run_time):
    L = []
    L.append("# 执行层参数敏感性扫描报告 (P0)")
    L.append("")
    L.append(f"**运行时间**: {run_time}")
    cfg = output["config"]
    L.append(f"**模式**: forced_regime={cfg['forced_regime']} (与实盘一致)")
    L.append(f"**训练窗**: {cfg['train'][0]} ~ {cfg['train'][1]}")
    L.append(f"**测试窗**: {cfg['test'][0]} ~ {cfg['test'][1]}")
    L.append(f"**默认值**: {cfg['defaults']}")
    L.append("")
    L.append("> **目标**: 判断执行层参数是「平台」(稳健)还是「尖峰」(过拟合)")
    L.append("> - 平台: 参数±20%范围内夏普平稳(波动<30%) → 稳健可用")
    L.append("> - 尖峰: 最优点夏普高, 邻域骤降(波动>50%) → 过拟合警告")
    L.append("")

    # ── 一、稳定性总览 ──
    L.append("## 一、稳定性总览")
    L.append("")
    L.append("| 参数 | 分组 | 训练稳定性 | 训练波动% | 测试稳定性 | 测试波动% |")
    L.append("|------|------|----------|----------|----------|----------|")
    for param_name in output["config"]["scan_grid"]:
        for g, s in output["stability"].get(param_name, {}).items():
            L.append(f"| {param_name} | {g} | {s['train']['stability']} | "
                     f"{s['train']['volatility_pct']} | {s['test']['stability']} | "
                     f"{s['test']['volatility_pct']} |")
    L.append("")

    # ── 二、各参数详细扫描曲线 ──
    L.append("## 二、各参数扫描曲线 (夏普)")
    L.append("")
    for param_name, values in output["config"]["scan_grid"].items():
        L.append(f"### {param_name} (默认={output['config']['defaults'][param_name]})")
        L.append("")
        L.append("| 分组 | 窗口 | " + " | ".join(f"{v}" for v in values) + " |")
        L.append("|------|------|" + "|".join(["---"] * len(values)) + "|")
        for g, gr in output["results"].items():
            for window in ["train", "test"]:
                sharpes = [r.get("sharpe", 0) for r in gr[window][param_name]]
                wl = "训练" if window == "train" else "测试"
                L.append(f"| {g} | {wl} | " + " | ".join(f"{s:+.3f}" for s in sharpes) + " |")
        L.append("")

    # ── 三、结论 ──
    L.append("## 三、结论与建议")
    L.append("")
    # 统计各参数稳定性
    for param_name in output["config"]["scan_grid"]:
        groups_data = output["stability"].get(param_name, {})
        plateau_test = sum(1 for g, s in groups_data.items()
                           if "平台" in s["test"]["stability"] or "单调" in s["test"]["stability"])
        spike_test = sum(1 for g, s in groups_data.items()
                         if "尖峰" in s["test"]["stability"])
        total = len(groups_data)
        L.append(f"**{param_name}** (默认={output['config']['defaults'][param_name]}):")
        L.append(f"- 测试窗: 平台/单调 {plateau_test}/{total} 组, 尖峰 {spike_test}/{total} 组")
        if spike_test > total / 2:
            L.append(f"- ⚠️ **过半分组为尖峰**: 当前默认值可能过拟合, 建议重选参数或加宽止损")
        elif plateau_test > total / 2:
            L.append(f"- ✅ **过半分组为平台**: 参数稳健, 邻域表现稳定")
        else:
            L.append(f"- ⚠️ **稳定性混合**: 部分组稳健部分过拟合, 建议分组差异化设置")
        L.append("")

    L.append("### 整体判断")
    L.append("")
    L.append("- 若 atr_stop_mult 多数分组为平台 → 执行层Alpha来源稳健, 可继续依赖")
    L.append("- 若 trail_mult_high 多数为尖峰 → 三档止盈的收紧系数过拟合, 建议简化")
    L.append("- 若 hard_stop_pct 多数为平台 → 硬止损10%是稳健安全网")
    L.append("- 若多数参数为尖峰 → 执行层Alpha可能是过拟合幻觉, 需重新审视")

    return "\n".join(L)


if __name__ == "__main__":
    main()
