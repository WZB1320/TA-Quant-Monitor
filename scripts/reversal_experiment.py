"""
反转实验 — 验证"因子方向是否反了"

背景:
  IC 分析显示现有因子全部负 IC (反转效应), 但策略按趋势跟随使用。
  本实验把打分方向取反, 用 walk-forward 对比反转前后的夏普。

实验设计:
  对每个分组, 跑 4 个组合:
    1. 正常模式 + 训练窗 (2024-07 ~ 2025-06)
    2. 正常模式 + 测试窗 (2025-07 ~ 2026-06)
    3. 反转模式 + 训练窗
    4. 反转模式 + 测试窗

  核心对比:
    - 反转前后夏普变化 (夏普提升 = 方向反了的证据)
    - 反转前后收益/回撤/交易次数变化
    - 反转前后因子 IC (IC 取反, 数学性质: 反转后 IC = -原 IC)

日志:
  - DEBUG 级别记录每次打分的因子贡献 (log_detail=True)
  - 日志输出到 data/reversal_experiment.log
  - 控制台只打印汇总结果

输出:
  data/reversal_experiment_result.json   (结构化结果)
  data/reversal_experiment_report.md     (可读报告)
  data/reversal_experiment.log           (详细打分日志)

用法:
  python scripts/reversal_experiment.py
"""
import sys
import os
import json
import shutil
import logging
from datetime import datetime

project_root = os.path.join(os.path.dirname(__file__), "..")
sys.path.insert(0, project_root)

from src.data_fetcher import DataManager
from src.config.group_config import GroupConfig
from src.config.runtime_mode import set_mode, RuntimeMode
from src.config.user_preferences import UserPreferences, _USER_PREF_FILE
from src.backtest.engine import BacktestEngine

set_mode(RuntimeMode.BACKTEST)

# ── 配置 ──
TRAIN_START = "2024-07-01"
TRAIN_END = "2025-06-30"
TEST_START = "2025-07-01"
TEST_END = "2026-06-30"
DATA_START = "2024-02-01"
DATA_END = "2026-07-13"
BENCHMARK = "sh.000300"

PREF_BACKUP = _USER_PREF_FILE + ".reversal_bak"
LOG_FILE = os.path.join(project_root, "data", "reversal_experiment.log")
RESULT_JSON = os.path.join(project_root, "data", "reversal_experiment_result.json")
REPORT_MD = os.path.join(project_root, "data", "reversal_experiment_report.md")
IC_RESULT = os.path.join(project_root, "data", "factor_ic_result.json")


def setup_logging():
    """配置日志: DEBUG 到文件, INFO 到控制台"""
    os.makedirs(os.path.dirname(LOG_FILE), exist_ok=True)
    # 清空旧日志
    if os.path.exists(LOG_FILE):
        os.remove(LOG_FILE)

    root = logging.getLogger()
    root.setLevel(logging.DEBUG)

    # 文件: DEBUG (记录每次打分的因子贡献)
    fh = logging.FileHandler(LOG_FILE, encoding="utf-8")
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(logging.Formatter("%(asctime)s %(name)s %(levelname)s %(message)s"))
    root.addHandler(fh)

    # 控制台: INFO (只看汇总)
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    ch.setFormatter(logging.Formatter("%(message)s"))
    root.addHandler(ch)


def load_watchlist():
    config_path = os.path.join(project_root, "config", "strategy_config.json")
    with open(config_path, "r", encoding="utf-8") as f:
        return json.load(f)["strategy_config"]["watchlist"]


def run_backtest(data_map, benchmark_df, start, end, reverse_mode):
    """单次回测 — reverse_mode 控制是否反转因子方向

    log_detail=True 会通过 logging.DEBUG 记录每次打分的因子贡献到日志文件。
    """
    GroupConfig._instance = None
    GroupConfig._config = None

    engine = BacktestEngine(
        initial_capital=100000,
        lookback_days=120,
        position_ratio=0.3,
        commission_rate=0.00025,
        stamp_tax=0.001,
        slippage=0.0001,
        signal_dedup_days=5,
        risk_per_trade=0.05,
        atr_stop_mult=2.0,
        forced_regime=None,  # auto: 让 ADX 自动判断
        reverse_mode=reverse_mode,
        log_detail=True,     # 记录因子贡献到日志
    )
    metrics = engine.run(
        data_map=data_map,
        benchmark_df=benchmark_df,
        start_date=start,
        end_date=end,
    )
    return metrics


def metrics_to_dict(m):
    return {
        "total_return_pct": round(m.total_return * 100, 2),
        "annual_return_pct": round(m.annual_return * 100, 2),
        "sharpe_ratio": round(m.sharpe_ratio, 3),
        "max_drawdown_pct": round(m.max_drawdown * 100, 2),
        "volatility_pct": round(m.volatility * 100, 2),
        "trade_count": m.trade_count,
        "win_rate_pct": round(m.win_rate * 100, 1),
        "alpha_pct": round(m.alpha * 100, 2),
    }


def load_ic_results():
    """加载已有 IC 分析结果 (反转后 IC = -原 IC)"""
    if not os.path.exists(IC_RESULT):
        return None
    with open(IC_RESULT, "r", encoding="utf-8") as f:
        return json.load(f)


def generate_report(results, ic_data, run_time):
    """生成对比报告"""
    L = []
    L.append("# 反转实验报告 — 因子方向是否反了?")
    L.append("")
    L.append(f"**运行时间**: {run_time}")
    L.append(f"**训练窗**: {TRAIN_START} ~ {TRAIN_END}")
    L.append(f"**测试窗**: {TEST_START} ~ {TEST_END}")
    L.append(f"**体制模式**: auto (ADX 自动判断)")
    L.append(f"**反转方式**: Scorer 层 direction 取反 (价格>MA60 原本看多 → 反转后看空)")
    L.append(f"**详细日志**: {LOG_FILE}")
    L.append("")
    L.append("> **核心假设**: IC分析显示因子全部负IC(反转效应), 但策略按趋势跟随使用。")
    L.append("> 若反转后夏普显著提升 → 证实因子方向反了。")
    L.append("> 若反转后夏普无变化或更差 → 说明Alpha不在因子方向, 而在执行层(止损/regime/选股池)。")
    L.append("")

    # ── 一、夏普对比总览 ──
    L.append("## 一、夏普对比总览 (反转前 vs 反转后)")
    L.append("")
    L.append("| 分组 | 窗口 | 反转前夏普 | 反转后夏普 | 夏普变化 | 反转前收益 | 反转后收益 | 反转前交易 | 反转后交易 |")
    L.append("|------|------|----------|----------|---------|----------|----------|----------|----------|")
    for g, r in results.items():
        for window in ["train", "test"]:
            n = r[window]["normal"]
            rv = r[window]["reversed"]
            sharpe_delta = round(rv["sharpe_ratio"] - n["sharpe_ratio"], 3)
            arrow = "↑" if sharpe_delta > 0.1 else ("↓" if sharpe_delta < -0.1 else "→")
            wlabel = "训练" if window == "train" else "测试"
            L.append(
                f"| {g} | {wlabel} | {n['sharpe_ratio']} | {rv['sharpe_ratio']} | "
                f"{arrow}{abs(sharpe_delta):.3f} | "
                f"{n['total_return_pct']}% | {rv['total_return_pct']}% | "
                f"{n['trade_count']} | {rv['trade_count']} |"
            )
    L.append("")

    # ── 二、详细对比 ──
    L.append("## 二、分组详细对比")
    L.append("")
    for g, r in results.items():
        L.append(f"### {g}")
        L.append("")
        L.append("| 指标 | 训练窗(正常) | 训练窗(反转) | 测试窗(正常) | 测试窗(反转) |")
        L.append("|------|------------|------------|------------|------------|")
        nt, rt, ne, re_ = r["train"]["normal"], r["train"]["reversed"], r["test"]["normal"], r["test"]["reversed"]
        for metric in ["total_return_pct", "annual_return_pct", "sharpe_ratio",
                        "max_drawdown_pct", "volatility_pct", "trade_count",
                        "win_rate_pct", "alpha_pct"]:
            label = {"total_return_pct": "总收益%", "annual_return_pct": "年化%",
                     "sharpe_ratio": "夏普", "max_drawdown_pct": "最大回撤%",
                     "volatility_pct": "波动率%", "trade_count": "交易次数",
                     "win_rate_pct": "胜率%", "alpha_pct": "Alpha%"}.get(metric, metric)
            L.append(f"| {label} | {nt[metric]} | {rt[metric]} | {ne[metric]} | {re_[metric]} |")
        L.append("")

    # ── 三、因子 IC 对比 (反转后 IC = -原 IC) ──
    L.append("## 三、因子 IC 对比 (反转后 IC = -原 IC)")
    L.append("")
    L.append("> 数学性质: 因子方向取反后, IC 符号取反, 绝对值不变。")
    L.append("> 反转的意义在于: 若原 IC 为负(反转效应), 反转后 IC 变正 → 因子变成正向预测。")
    L.append("")
    if ic_data:
        results_ic = ic_data.get("results", {})
        by_period = results_ic.get("by_period", {})
        # N=5 的 IC
        p5 = by_period.get("5", {})
        L.append("### 主周期 N=5日 IC 对比")
        L.append("")
        L.append("| 因子 | 原IC | 反转后IC | 原判定 | 反转后判定 |")
        L.append("|------|------|---------|--------|----------|")
        for fname, gdata in p5.items():
            if fname == "_overall":
                continue
            ic = gdata.get("_overall", {}).get("ic_mean")
            verdict = gdata.get("_overall", {}).get("verdict", "")
            if ic is not None:
                rev_ic = round(-ic, 4)
                # 反转后判定
                a = abs(rev_ic)
                if a > 0.05:
                    rev_verdict = "✅有效(正)" if rev_ic > 0 else "❌有效(负)"
                elif a > 0.03:
                    rev_verdict = "⚠️弱有效(正)" if rev_ic > 0 else "⚠️弱有效(负)"
                else:
                    rev_verdict = "❌无效"
                L.append(f"| {fname} | {ic:+.4f} | {rev_ic:+.4f} | {verdict} | {rev_verdict} |")
        L.append("")
    else:
        L.append("(未找到 IC 分析结果, 请先运行 factor_ic_analysis.py)")
        L.append("")

    # ── 四、结论 ──
    L.append("## 四、结论与判断")
    L.append("")
    # 统计反转后夏普提升的分组数
    improved_train = sum(1 for r in results.values()
                         if r["train"]["reversed"]["sharpe_ratio"] - r["train"]["normal"]["sharpe_ratio"] > 0.2)
    improved_test = sum(1 for r in results.values()
                        if r["test"]["reversed"]["sharpe_ratio"] - r["test"]["normal"]["sharpe_ratio"] > 0.2)
    total = len(results)

    L.append(f"- 反转后训练窗夏普显著提升(>+0.2)的分组: {improved_train}/{total}")
    L.append(f"- 反转后测试窗夏普显著提升(>+0.2)的分组: {improved_test}/{total}")
    L.append("")

    if improved_train >= total / 2:
        L.append("✅ **反转假设成立**: 过半分组反转后训练窗夏普显著提升,")
        L.append("   证实因子方向确实反了 — 现有趋势跟随逻辑与A股反转规律相悖。")
        L.append("   建议: 正式切换为反转模式, 并重新做 walk-forward 寻优。")
    elif improved_train + improved_test >= total / 2:
        L.append("⚠️ **部分支持反转假设**: 反转在部分窗口/分组有效, 但不稳定。")
        L.append("   建议: 分组配置 — 对反转有效的分组用反转模式, 其他保持原样。")
    else:
        L.append("❌ **反转假设不成立**: 反转后夏普未普遍提升,")
        L.append("   说明策略的 Alpha 不在因子方向, 而在执行层(止损止盈/regime切换/选股池)。")
        L.append("   建议: 放弃方向调整, 转向执行层优化和因子正交化。")
    L.append("")
    L.append("### 详细打分日志")
    L.append("")
    L.append(f"每次打分的因子贡献明细已记录到: `{LOG_FILE}`")
    L.append("可用 `grep REVERSED {LOG_FILE}` 筛选反转模式的打分记录, 对比正常模式。")

    return "\n".join(L)


def main():
    run_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    setup_logging()
    log = logging.getLogger(__name__)

    log.info("=" * 70)
    log.info("  反转实验 — 验证因子方向是否反了")
    log.info(f"  训练窗: {TRAIN_START} ~ {TRAIN_END}")
    log.info(f"  测试窗: {TEST_START} ~ {TEST_END}")
    log.info(f"  因子贡献日志: {LOG_FILE}")
    log.info("=" * 70)

    # ── 备份并清除用户偏好 (强制 auto) ──
    pref_existed = os.path.exists(_USER_PREF_FILE)
    if pref_existed:
        shutil.copy2(_USER_PREF_FILE, PREF_BACKUP)
    UserPreferences().clear_all()
    log.info("✓ 已清除用户偏好 (强制 auto 体制)")

    try:
        watchlist = load_watchlist()
        dm = DataManager()

        # 基准
        log.info(f"\n拉取基准 {BENCHMARK}...")
        benchmark_df = dm.get_daily_kline(BENCHMARK, start_date=DATA_START, end_date=DATA_END)
        if benchmark_df is None or benchmark_df.empty:
            log.info("  ⚠️ 基准拉取失败, 继续 (alpha 将为0)")
            benchmark_df = None
        else:
            log.info(f"  ✓ {len(benchmark_df)} 条")

        results = {}

        for group_name, stocks in watchlist.items():
            if group_name.startswith("_"):
                continue
            codes = [s["code"] for s in stocks]
            log.info(f"\n{'─' * 70}")
            log.info(f"分组: {group_name} ({len(codes)}只: {', '.join(codes)})")
            log.info(f"{'─' * 70}")

            # 拉数据 (一次拉够, 4次回测共用)
            data_map = {}
            for code in codes:
                df = dm.get_daily_kline(code, start_date=DATA_START, end_date=DATA_END)
                if df is not None and not df.empty:
                    data_map[code] = df
            if len(data_map) < 2:
                log.info(f"  ⚠️ 数据不足, 跳过")
                continue
            log.info(f"  ✓ 拉取 {len(data_map)}/{len(codes)} 只")

            group_result = {"stocks": codes, "train": {}, "test": {}}

            # 4 个组合: 正常/反转 × 训练/测试
            for window, start, end in [("train", TRAIN_START, TRAIN_END),
                                        ("test", TEST_START, TEST_END)]:
                wlabel = "训练" if window == "train" else "测试"

                log.info(f"\n  ▶ {wlabel}窗 正常模式...")
                m_normal = run_backtest(data_map, benchmark_df, start, end, reverse_mode=False)
                log.info(f"    夏普={m_normal.sharpe_ratio:.3f} 收益={m_normal.total_return*100:.2f}% "
                         f"交易={m_normal.trade_count}笔")

                log.info(f"  ▶ {wlabel}窗 反转模式...")
                m_reversed = run_backtest(data_map, benchmark_df, start, end, reverse_mode=True)
                log.info(f"    夏普={m_reversed.sharpe_ratio:.3f} 收益={m_reversed.total_return*100:.2f}% "
                         f"交易={m_reversed.trade_count}笔")

                delta = m_reversed.sharpe_ratio - m_normal.sharpe_ratio
                arrow = "↑" if delta > 0.1 else ("↓" if delta < -0.1 else "→")
                log.info(f"  ➜ {wlabel}窗夏普变化: {arrow} {abs(delta):.3f}")

                group_result[window]["normal"] = metrics_to_dict(m_normal)
                group_result[window]["reversed"] = metrics_to_dict(m_reversed)

            results[group_name] = group_result

        # ── 保存结果 ──
        os.makedirs(os.path.dirname(RESULT_JSON), exist_ok=True)
        with open(RESULT_JSON, "w", encoding="utf-8") as f:
            json.dump({"run_time": run_time, "results": results}, f,
                      ensure_ascii=False, indent=2)
        log.info(f"\n✓ 结构化结果 → {RESULT_JSON}")

        # ── 生成报告 ──
        ic_data = load_ic_results()
        report = generate_report(results, ic_data, run_time)
        with open(REPORT_MD, "w", encoding="utf-8") as f:
            f.write(report)
        log.info(f"✓ 可读报告   → {REPORT_MD}")

        # ── 汇总 ──
        log.info(f"\n{'=' * 70}")
        log.info("  汇总: 反转前后夏普对比")
        log.info(f"{'=' * 70}")
        log.info(f"{'分组':<12} {'训练(正常→反转)':>20} {'测试(正常→反转)':>20}")
        for g, r in results.items():
            tn, tr = r["train"]["normal"]["sharpe_ratio"], r["train"]["reversed"]["sharpe_ratio"]
            en, er = r["test"]["normal"]["sharpe_ratio"], r["test"]["reversed"]["sharpe_ratio"]
            log.info(f"{g:<12} {tn:>8.3f} → {tr:>8.3f}   {en:>8.3f} → {er:>8.3f}")

    finally:
        # 恢复用户偏好
        if os.path.exists(PREF_BACKUP):
            shutil.move(PREF_BACKUP, _USER_PREF_FILE)
            log.info("\n✓ 已恢复用户偏好")
        elif os.path.exists(_USER_PREF_FILE):
            os.remove(_USER_PREF_FILE)


if __name__ == "__main__":
    main()
