"""
信号触发分析 — 统计训练窗震荡市"参数微调就消失"的脆弱信号占比

背景:
  执行层敏感性扫描发现: 训练窗(震荡市)参数极不稳定(atr_stop_mult 5/5组尖峰)。
  假设: 震荡市触发的信号大部分是"勉强踩过阈值"的脆弱信号, 参数微调就消失。
  若假设成立 → 震荡市Alpha缺失的根因是噪声驱动的脆弱信号。

方法 (重跑法, 准确):
  对每个分组, 用5个 score_threshold 值跑训练窗回测:
    T-10, T-5, T(原值), T+5, T+10
  记录每次的买入信号集合 (symbol, entry_date)
  脆弱信号 = T时触发, 但 T+5 就消失的信号 (裕度 < 5)
  极脆弱信号 = T时触发, 但 T+5 和 T+10 都消失 (裕度 < 5 且非单调)

输出:
  data/signal_fragility_result.json
  data/signal_fragility_report.md

用法:
  python scripts/signal_fragility_analysis.py
"""
import sys
import os
import json
import shutil
import re
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
DATA_START, DATA_END = "2024-02-01", "2025-06-30"
THRESHOLD_DELTAS = [-10, -5, 0, 5, 10]  # T-10, T-5, T, T+5, T+10

RESULT_JSON = os.path.join(project_root, "data", "signal_fragility_result.json")
REPORT_MD = os.path.join(project_root, "data", "signal_fragility_report.md")


def load_watchlist():
    cfg_path = os.path.join(project_root, "config", "strategy_config.json")
    with open(cfg_path, "r", encoding="utf-8") as f:
        cfg = json.load(f)
    raw = cfg["strategy_config"]["watchlist"]
    return {g: [s["code"] for s in stocks]
            for g, stocks in raw.items() if not g.startswith("_")}


def get_base_threshold(symbol):
    """获取分组基础 score_threshold (auto模式, 不含manual_regime_presets)"""
    gc = GroupConfig()
    return gc.get_score_threshold(symbol)


def extract_signals(position_mgr):
    """从已平仓+未平仓交易中提取买入信号集合 (symbol, entry_date_str, score)"""
    signals = []
    # 已平仓交易
    for t in position_mgr.closed_trades:
        score = _extract_score(t.entry_signal)
        signals.append({
            "symbol": t.symbol,
            "entry_date": str(t.entry_date),
            "score": score,
        })
    # 未平仓交易
    for symbol, t in position_mgr.open_positions.items():
        score = _extract_score(t.entry_signal)
        signals.append({
            "symbol": symbol,
            "entry_date": str(t.entry_date),
            "score": score,
        })
    return signals


def _extract_score(signal_str):
    """从 entry_signal 字符串提取 score 值 (格式: '强买入 score=+45.1')"""
    if not signal_str:
        return None
    m = re.search(r"score=([+-]?[\d.]+)", signal_str)
    return float(m.group(1)) if m else None


def run_backtest_with_threshold(data_map, group_codes, threshold_override,
                                 start, end):
    """用指定 score_threshold 跑回测, 返回信号列表"""
    # 重置单例
    GroupConfig._instance = None
    GroupConfig._config = None

    # monkey-patch: 覆盖 get_all_group_params 返回的 score_threshold
    # (signal_engine 用 get_all_group_params 获取分组参数, 不是 get_score_threshold)
    original_get_all = GroupConfig.get_all_group_params

    def patched_get_all(self, code):
        params = original_get_all(self, code)
        params["score_threshold"] = threshold_override
        return params
    GroupConfig.get_all_group_params = patched_get_all

    try:
        engine = BacktestEngine(
            initial_capital=100000, lookback_days=120, position_ratio=0.3,
            commission_rate=0.00025, stamp_tax=0.001, slippage=0.0001,
            signal_dedup_days=5, risk_per_trade=0.05, atr_stop_mult=2.0,
            forced_regime=None,
        )
        engine.run(data_map, start_date=start, end_date=end)
        return extract_signals(engine.position_mgr)
    except Exception as e:
        print(f"    ERROR: {e}")
        return []
    finally:
        GroupConfig.get_all_group_params = original_get_all


def signal_key(s):
    """信号唯一键: (symbol, entry_date)"""
    return (s["symbol"], s["entry_date"])


def main():
    run_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print("=" * 70)
    print("  信号触发分析 — 训练窗震荡市脆弱信号占比")
    print(f"  训练窗: {TRAIN_START} ~ {TRAIN_END} (震荡市)")
    print(f"  阈值微调: {THRESHOLD_DELTAS}")
    print("=" * 70)

    # 清除用户偏好
    pref_existed = os.path.exists(_USER_PREF_FILE)
    pref_backup = _USER_PREF_FILE + ".frag_bak"
    if pref_existed:
        shutil.copy2(_USER_PREF_FILE, pref_backup)
    UserPreferences().clear_all()

    try:
        watchlist = load_watchlist()
        dm = DataManager()
        all_results = {}

        for group_name, codes in watchlist.items():
            print(f"\n{'─' * 60}")
            print(f"分组: {group_name} ({len(codes)}只)")
            print(f"{'─' * 60}")

            # 拉数据
            data_map = {}
            for c in codes:
                df = dm.get_daily_kline(c, start_date=DATA_START, end_date=DATA_END)
                if df is not None and len(df) > 80:
                    data_map[c] = df
            if len(data_map) < 2:
                print(f"  跳过: 数据不足")
                continue

            # 获取基础阈值
            GroupConfig._instance = None
            GroupConfig._config = None
            base_T = get_base_threshold(codes[0])
            print(f"  基础 score_threshold = {base_T}")

            # 跑5个阈值
            threshold_signals = {}  # {阈值: [信号]}
            for delta in THRESHOLD_DELTAS:
                T = base_T + delta
                print(f"  跑 T{delta:+d}={T}...", end=" ", flush=True)
                sigs = run_backtest_with_threshold(data_map, codes, T,
                                                    TRAIN_START, TRAIN_END)
                threshold_signals[delta] = sigs
                print(f"{len(sigs)}个信号")

            # 基准(T=0)的信号集
            base_signals = threshold_signals[0]
            base_keys = {signal_key(s) for s in base_signals}

            if not base_signals:
                print(f"  ⚠️ 基准阈值无信号")
                all_results[group_name] = {
                    "base_threshold": base_T,
                    "base_signal_count": 0,
                    "fragile_counts": {d: 0 for d in THRESHOLD_DELTAS},
                    "fragile_pcts": {d: 0 for d in THRESHOLD_DELTAS},
                    "score_distribution": [],
                }
                continue

            # 统计: T+5/T+10 时消失的信号
            fragile_counts = {}
            for delta in [5, 10]:
                keys_delta = {signal_key(s) for s in threshold_signals[delta]}
                disappeared = base_keys - keys_delta
                fragile_counts[delta] = len(disappeared)

            # 脆弱信号占比
            fragile_pcts = {
                d: round(c / len(base_signals) * 100, 1)
                for d, c in fragile_counts.items()
            }

            # score裕度分布
            scores = [s["score"] for s in base_signals if s["score"] is not None]
            margins = [round(s - base_T, 1) for s in scores]
            margin_buckets = {"<5": 0, "5-10": 0, "10-20": 0, ">20": 0}
            for m in margins:
                if m < 5:
                    margin_buckets["<5"] += 1
                elif m < 10:
                    margin_buckets["5-10"] += 1
                elif m < 20:
                    margin_buckets["10-20"] += 1
                else:
                    margin_buckets[">20"] += 1

            all_results[group_name] = {
                "base_threshold": base_T,
                "base_signal_count": len(base_signals),
                "fragile_counts": fragile_counts,
                "fragile_pcts": fragile_pcts,
                "margin_buckets": margin_buckets,
                "score_margins": margins,
                "signal_details": base_signals,
            }

            print(f"\n  结果: 基准{len(base_signals)}个信号")
            print(f"    T+5消失(脆弱): {fragile_counts[5]}个 ({fragile_pcts[5]}%)")
            print(f"    T+10消失(极脆弱): {fragile_counts[10]}个 ({fragile_pcts[10]}%)")
            print(f"    裕度分布: <5分={margin_buckets['<5']}个, 5-10分={margin_buckets['5-10']}个, "
                  f"10-20分={margin_buckets['10-20']}个, >20分={margin_buckets['>20']}个")

        # ── 保存 ──
        os.makedirs(os.path.dirname(RESULT_JSON), exist_ok=True)
        output = {
            "run_time": run_time,
            "config": {
                "train": [TRAIN_START, TRAIN_END],
                "threshold_deltas": THRESHOLD_DELTAS,
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
        print("  汇总: 脆弱信号占比")
        print(f"{'=' * 70}")
        print(f"{'分组':<12} {'基准信号':>8} {'T+5消失%':>10} {'T+10消失%':>10} {'裕度<5占比':>12}")
        total_base = 0
        total_fragile5 = 0
        for g, r in all_results.items():
            base = r["base_signal_count"]
            f5 = r["fragile_pcts"].get(5, 0)
            f10 = r["fragile_pcts"].get(10, 0)
            mb = r["margin_buckets"]
            margin5_pct = round(mb["<5"] / base * 100, 1) if base > 0 else 0
            print(f"{g:<12} {base:>8} {f5:>10}% {f10:>10}% {margin5_pct:>12}%")
            total_base += base
            total_fragile5 += r["fragile_counts"].get(5, 0)
        if total_base > 0:
            print(f"{'合计':<12} {total_base:>8} {round(total_fragile5/total_base*100,1):>10}%")

    finally:
        if pref_existed and os.path.exists(pref_backup):
            shutil.move(pref_backup, _USER_PREF_FILE)


def generate_report(output, run_time):
    L = []
    L.append("# 信号触发分析报告 — 训练窗震荡市脆弱信号占比")
    L.append("")
    L.append(f"**运行时间**: {run_time}")
    cfg = output["config"]
    L.append(f"**训练窗**: {cfg['train'][0]} ~ {cfg['train'][1]} (震荡市)")
    L.append(f"**阈值微调**: {cfg['threshold_deltas']} (T=基础阈值)")
    L.append("")
    L.append("> **假设**: 震荡市触发的信号大部分是「勉强踩过阈值」的脆弱信号,")
    L.append("> 参数微调(score_threshold+5)就消失 → 噪声驱动, 非真实Alpha")
    L.append("> **若脆弱信号占比>50%**: 假设成立, 震荡市Alpha缺失根因是噪声信号")
    L.append("")

    # ── 一、脆弱信号总览 ──
    L.append("## 一、脆弱信号总览")
    L.append("")
    L.append("| 分组 | 基础阈值 | 基准信号数 | T+5消失(脆弱) | T+5占比 | T+10消失(极脆弱) | T+10占比 |")
    L.append("|------|---------|----------|-------------|---------|----------------|---------|")
    for g, r in output["results"].items():
        base = r["base_signal_count"]
        # JSON序列化后key变字符串, 兼容 int/str
        f5 = r["fragile_counts"].get(5, r["fragile_counts"].get("5", 0))
        f5p = r["fragile_pcts"].get(5, r["fragile_pcts"].get("5", 0))
        f10 = r["fragile_counts"].get(10, r["fragile_counts"].get("10", 0))
        f10p = r["fragile_pcts"].get(10, r["fragile_pcts"].get("10", 0))
        L.append(f"| {g} | {r['base_threshold']} | {base} | {f5} | {f5p}% | {f10} | {f10p}% |")
    L.append("")

    # ── 二、score裕度分布 ──
    L.append("## 二、score裕度分布 (score - 阈值)")
    L.append("")
    L.append("裕度越小越脆弱: 裕度<5 = 阈值+5就消失的脆弱信号")
    L.append("")
    L.append("| 分组 | 裕度<5(脆弱) | 裕度5-10 | 裕度10-20 | 裕度>20(稳健) |")
    L.append("|------|------------|---------|----------|-------------|")
    for g, r in output["results"].items():
        mb = r["margin_buckets"]
        L.append(f"| {g} | {mb['<5']} | {mb['5-10']} | {mb['10-20']} | {mb['>20']} |")
    L.append("")

    # ── 三、结论 ──
    L.append("## 三、结论")
    L.append("")
    total_base = sum(r["base_signal_count"] for r in output["results"].values())
    total_fragile5 = sum(r["fragile_counts"].get(5, r["fragile_counts"].get("5", 0))
                         for r in output["results"].values())
    total_margin5 = sum(r["margin_buckets"]["<5"] for r in output["results"].values())
    overall_fragile_pct = round(total_fragile5 / total_base * 100, 1) if total_base > 0 else 0
    overall_margin5_pct = round(total_margin5 / total_base * 100, 1) if total_base > 0 else 0
    L.append(f"**整体统计**:")
    L.append(f"- 训练窗(震荡市)基准信号总数: {total_base}")
    L.append(f"- T+5消失的脆弱信号: {total_fragile5} ({overall_fragile_pct}%)")
    L.append(f"- 裕度<5的信号占比: {overall_margin5_pct}%")
    L.append("")

    if overall_fragile_pct > 50:
        L.append("✅ **假设成立**: 脆弱信号占比 > 50%")
        L.append("")
        L.append("震荡市触发的信号过半是「勉强踩过阈值」的噪声信号——score_threshold微调5分就消失。")
        L.append("这证实了震荡市Alpha缺失的根因: **信号在阈值附近高度密集, 非真实预测力驱动**。")
        L.append("")
        L.append("**根因分析**:")
        L.append("- 震荡市股票打分普遍偏低且集中在阈值附近 → 大量「勉强触发」的信号")
        L.append("- 这些信号本质是噪声, 参数微调即消失 → 训练窗参数极不稳定的原因")
        L.append("- 执行层参数在训练窗的「尖峰」(敏感性扫描发现)正是因为信号脆弱")
        L.append("")
        L.append("**解决方向**:")
        L.append("1. 震荡市提高score_threshold(过滤脆弱信号) 或 直接降低换手/空仓观望")
        L.append("2. regime识别升级: 准确识别震荡市, 应用差异化策略")
        L.append("3. 信号质量过滤: 要求score裕度>10才执行(剔除踩线信号)")
    elif overall_fragile_pct > 30:
        L.append("⚠️ **部分成立**: 脆弱信号占比 30-50%")
        L.append("")
        L.append("震荡市有相当比例的脆弱信号, 但非主导。根因可能是混合的:")
        L.append("- 部分信号确实脆弱(噪声驱动)")
        L.append("- 部分信号有真实预测力但被震荡市消耗")
        L.append("建议: 信号质量过滤(裕度>10) + regime差异化")
    else:
        L.append("❌ **假设不成立**: 脆弱信号占比 < 30%")
        L.append("")
        L.append("震荡市信号多数是稳健的(裕度>10), 参数微调不消失。")
        L.append("说明震荡市Alpha缺失的根因不是信号脆弱, 而是其他因素:")
        L.append("- 可能是止损止盈在震荡市频繁触发(假突破多)")
        L.append("- 可能是选股方向在震荡市无效")
        L.append("建议: 转向分析止损触发频率和假突破率")

    return "\n".join(L)


if __name__ == "__main__":
    main()
