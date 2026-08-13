"""
Walk-Forward 验证 — 检验现有参数是否过拟合

方法:
  - 训练窗: 2024-07-01 ~ 2025-06-30 (用现有 strategy_config.json 参数)
  - 测试窗: 2025-07-01 ~ 2026-06-30 (完全相同参数, 样本外)
  - 对比两窗口的夏普/收益/回撤/交易次数, 判断参数稳健性

遍历全部分组 (科技/消费/周期/医药/机械), 每分组独立回测。
体制模式: auto (清除用户手动 trending 偏好, 让 ADX 自动判断,
         否则两窗口都被强制为 trending, 掩盖体制自适应能力)。

判断标准:
  - 夏普衰减 < 30% 且测试夏普 > 0.8  → ✅ 稳健, 可继续优化
  - 夏普衰减 30~60% 或测试夏普 0.3~0.8 → ⚠️ 部分过拟合, 谨慎
  - 夏普衰减 > 60% 或测试夏普 < 0.3   → ❌ 严重过拟合, 参数不可信

输出:
  data/walk_forward_result.json   (结构化结果)
  data/walk_forward_report.md     (可读报告)

用法:
  python scripts/walk_forward_validation.py
"""
import sys
import os
import json
import shutil
from datetime import datetime

project_root = os.path.join(os.path.dirname(__file__), "..")
sys.path.insert(0, project_root)

from src.data_fetcher import DataManager
from src.config.group_config import GroupConfig
from src.config.runtime_mode import set_mode, RuntimeMode
from src.config.user_preferences import UserPreferences, _USER_PREF_FILE
from src.backtest.engine import BacktestEngine

# 必须在创建任何 SignalEngine/Filter 之前设置为回测模式,
# 否则 Filter 会从磁盘加载实时信号历史, 导致误去重
set_mode(RuntimeMode.BACKTEST)

# ── Walk-Forward 配置 ──
TRAIN_START = "2024-07-01"
TRAIN_END = "2025-06-30"
TEST_START = "2025-07-01"
TEST_END = "2026-06-30"
# 数据拉取范围: 含 120 天指标预热 (lookback_days)
DATA_START = "2024-02-01"
DATA_END = "2026-07-13"
# 沪深300指数: baostock 用 "sh.000300" (纯数字 000300 会被误格式化为 sz.000300)
BENCHMARK = "sh.000300"

PREF_BACKUP = _USER_PREF_FILE + ".walkforward_bak"

# 输出路径
RESULT_JSON = os.path.join(project_root, "data", "walk_forward_result.json")
REPORT_MD = os.path.join(project_root, "data", "walk_forward_report.md")


def load_watchlist():
    """从 strategy_config.json 读取全部分组和股票"""
    config_path = os.path.join(project_root, "config", "strategy_config.json")
    with open(config_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data["strategy_config"]["watchlist"]


def fetch_data(dm, codes):
    """拉取一组股票的全量数据 (含预热)"""
    data_map = {}
    for code in codes:
        df = dm.get_daily_kline(code, start_date=DATA_START, end_date=DATA_END)
        if df is not None and not df.empty:
            data_map[code] = df
        else:
            print(f"    ⚠️  {code} 数据拉取失败")
    return data_map


def run_backtest(data_map, benchmark_df, start, end):
    """单次回测 (forced_regime=None, 让 ADX 自动判断体制)

    每次重置 GroupConfig 单例, 避免上一次回测的状态残留。
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
        forced_regime=None,  # auto: 让 ADX 自动判断, 不强制 trending
    )
    metrics = engine.run(
        data_map=data_map,
        benchmark_df=benchmark_df,
        start_date=start,
        end_date=end,
    )
    return metrics, engine.position_mgr.closed_trades


def metrics_to_dict(m):
    """BacktestMetrics → 可序列化 dict"""
    return {
        "total_return_pct": round(m.total_return * 100, 2),
        "annual_return_pct": round(m.annual_return * 100, 2),
        "sharpe_ratio": round(m.sharpe_ratio, 3),
        "max_drawdown_pct": round(m.max_drawdown * 100, 2),
        "volatility_pct": round(m.volatility * 100, 2),
        "trade_count": m.trade_count,
        "win_rate_pct": round(m.win_rate * 100, 1),
        "profit_factor": round(m.profit_factor, 2) if m.profit_factor != float("inf") else None,
        "avg_holding_days": round(m.avg_holding_days, 1),
        "alpha_pct": round(m.alpha * 100, 2),
        "benchmark_return_pct": round(m.benchmark_return * 100, 2),
    }


def judge_robustness(train_sharpe, test_sharpe, train_alpha_pct=None, test_alpha_pct=None):
    """根据训练/测试夏普判断过拟合程度

    判断优先级:
      1. 训练窗夏普 < 0.5 → 训练窗无Alpha (参数在样本内就不行, 谈不上过拟合)
      2. 测试窗夏普 < 0.3 → 严重过拟合 (Alpha在样本外消失)
      3. 夏普衰减 > 60%   → 严重过拟合
      4. 夏普衰减 > 30% 或测试夏普 < 0.8 → 部分过拟合
      5. 否则 → 稳健

    Returns:
        (verdict, reason)
        verdict: "稳健" / "部分过拟合" / "严重过拟合" / "训练窗无Alpha" / "样本不足"
    """
    if train_sharpe is None or test_sharpe is None:
        return "样本不足", "夏普无法计算"

    # 1. 训练窗就没Alpha: 参数在样本内就不具备显著预测力
    if train_sharpe < 0.5:
        return ("训练窗无Alpha",
                f"训练窗夏普 {train_sharpe:.2f} < 0.5, 参数在样本内就无显著Alpha, "
                f"谈不上过拟合——是参数本身无效")

    # 2. 测试窗Alpha消失
    if test_sharpe < 0.3:
        return ("严重过拟合",
                f"训练夏普 {train_sharpe:.2f} 但测试夏普 {test_sharpe:.2f} < 0.3, "
                f"Alpha在样本外消失")

    # 3-4. 衰减分析
    decay = (1 - test_sharpe / train_sharpe) * 100

    if decay > 60:
        return ("严重过拟合",
                f"夏普衰减 {decay:.0f}% > 60%, 参数对历史过拟合")
    if decay > 30 or test_sharpe < 0.8:
        return ("部分过拟合",
                f"夏普衰减 {decay:.0f}%, 测试夏普 {test_sharpe:.2f}, 稳健性存疑")
    return ("稳健",
            f"夏普衰减 {decay:.0f}%, 测试夏普 {test_sharpe:.2f} > 0.8")


def generate_report(results, run_time):
    """生成 Markdown 报告"""
    verdict_icon = {
        "稳健": "✅", "部分过拟合": "⚠️", "严重过拟合": "❌",
        "训练窗无Alpha": "🚫", "样本不足": "❔",
    }
    lines = []
    lines.append("# Walk-Forward 验证报告")
    lines.append("")
    lines.append(f"**运行时间**: {run_time}")
    lines.append(f"**训练窗**: {TRAIN_START} ~ {TRAIN_END} (参数在此窗口『已经调好』)")
    lines.append(f"**测试窗**: {TEST_START} ~ {TEST_END} (完全相同参数, 样本外)")
    lines.append(f"**体制模式**: auto (ADX 自动判断, 未强制 trending)")
    lines.append(f"**分组**: 全部 {len(results)} 个分组独立回测")
    lines.append("")
    lines.append("> **核心判断逻辑**:")
    lines.append("> - 若训练窗夏普 < 0.5 → 参数在样本内就无显著Alpha (🚫训练窗无Alpha)")
    lines.append("> - 若训练夏普≥0.5 但测试夏普<0.3 → Alpha在样本外消失 (❌严重过拟合)")
    lines.append("> - 若夏普衰减<30% 且测试夏普>0.8 → 稳健 (✅)")
    lines.append("> - 测试窗高夏普但Alpha为负 → 是Beta行情, 非真Alpha (⚠️警惕)")
    lines.append("")

    # ── 总览表 ──
    lines.append("## 一、总览: 训练窗 vs 测试窗")
    lines.append("")
    lines.append("| 分组 | 训练夏普 | 测试夏普 | 训练Alpha | 测试Alpha | 训练交易 | 测试交易 | 结论 |")
    lines.append("|------|---------|---------|----------|----------|---------|---------|------|")
    for g, r in results.items():
        t, te = r["train"], r["test"]
        icon = verdict_icon.get(r["verdict"], "")
        lines.append(
            f"| {g} | {t['sharpe_ratio']} | {te['sharpe_ratio']} | "
            f"{t['alpha_pct']}% | {te['alpha_pct']}% | "
            f"{t['trade_count']} | {te['trade_count']} | {icon}{r['verdict']} |"
        )
    lines.append("")
    lines.append("*Alpha = 分组收益 - 沪深300收益; Alpha为负表示跑输基准, 高夏普可能只是Beta行情。*")
    lines.append("")

    # ── 详细分析 ──
    lines.append("## 二、分组详细分析")
    lines.append("")
    for g, r in results.items():
        t, te = r["train"], r["test"]
        lines.append(f"### {g}")
        lines.append(f"- 股票: {', '.join(r['stocks'])}")
        lines.append(f"- **结论**: {verdict_icon.get(r['verdict'],'')} {r['verdict']} — {r['reason']}")

        # 警示: 测试窗高夏普但alpha为负 → beta行情
        if te["sharpe_ratio"] > 1.0 and te["alpha_pct"] < 0:
            lines.append(f"- ⚠️ **警惕Beta行情**: 测试窗夏普 {te['sharpe_ratio']} 虽高, "
                         f"但 Alpha {te['alpha_pct']}% 为负, 说明跑输沪深300, "
                         f"高夏普来自市场整体上涨而非选股Alpha。")
        # 警示: 测试窗alpha远超训练窗 → 可能是运气
        if (te["alpha_pct"] > 20 and t["alpha_pct"] < 5
                and t["alpha_pct"] is not None):
            lines.append(f"- ⚠️ **Alpha突变可疑**: 训练窗Alpha {t['alpha_pct']}% → "
                         f"测试窗 {te['alpha_pct']}%, 跨度异常, 更可能是周期/行业beta而非稳定Alpha。")
        lines.append("")
        lines.append("| 指标 | 训练窗 | 测试窗 | 变化 |")
        lines.append("|------|--------|--------|------|")
        lines.append(f"| 总收益 | {t['total_return_pct']}% | {te['total_return_pct']}% | {round(te['total_return_pct']-t['total_return_pct'],2)}pp |")
        lines.append(f"| 年化收益 | {t['annual_return_pct']}% | {te['annual_return_pct']}% | {round(te['annual_return_pct']-t['annual_return_pct'],2)}pp |")
        lines.append(f"| 夏普比率 | {t['sharpe_ratio']} | {te['sharpe_ratio']} | {r['sharpe_decay_pct']}% |")
        lines.append(f"| 最大回撤 | {t['max_drawdown_pct']}% | {te['max_drawdown_pct']}% | {round(te['max_drawdown_pct']-t['max_drawdown_pct'],2)}pp |")
        lines.append(f"| 波动率 | {t['volatility_pct']}% | {te['volatility_pct']}% | {round(te['volatility_pct']-t['volatility_pct'],2)}pp |")
        lines.append(f"| 胜率 | {t['win_rate_pct']}% | {te['win_rate_pct']}% | {round(te['win_rate_pct']-t['win_rate_pct'],1)}pp |")
        lines.append(f"| 盈亏比 | {t['profit_factor']} | {te['profit_factor']} | - |")
        lines.append(f"| 交易次数 | {t['trade_count']} | {te['trade_count']} | {te['trade_count']-t['trade_count']} |")
        lines.append(f"| Alpha(vs沪深300) | {t['alpha_pct']}% | {te['alpha_pct']}% | {round(te['alpha_pct']-t['alpha_pct'],2)}pp |")
        lines.append(f"| 基准收益 | {t['benchmark_return_pct']}% | {te['benchmark_return_pct']}% | - |")
        lines.append("")

    # ── 总体结论 ──
    lines.append("## 三、总体结论与建议")
    lines.append("")
    verdicts = [r["verdict"] for r in results.values()]
    stable = verdicts.count("稳健")
    partial = verdicts.count("部分过拟合")
    severe = verdicts.count("严重过拟合")
    no_alpha = verdicts.count("训练窗无Alpha")
    insufficient = verdicts.count("样本不足")
    lines.append(f"统计: ✅稳健 {stable} / ⚠️部分过拟合 {partial} / ❌严重过拟合 {severe} "
                 f"/ 🚫训练窗无Alpha {no_alpha} / ❔样本不足 {insufficient}")
    lines.append("")

    if no_alpha >= len(results) / 2:
        lines.append("🚫 **过半分组在训练窗就无Alpha**: 现有参数在 auto 体制模式下,")
        lines.append("   样本内(2024-07~2025-06)夏普普遍 < 0.5, 且 Alpha 多为负(跑输沪深300)。")
        lines.append("   这说明此前展示的『漂亮业绩』高度依赖 `forced_regime=trending` 强制模式,")
        lines.append("   一旦交给 ADX 自动判断体制, 参数就失去预测力。")
        lines.append("")
        lines.append("   **根本问题不是过拟合, 而是参数本身缺乏稳健Alpha** —— 现有业绩更可能是")
        lines.append("   『trending 预设 + 2025下半年科技/周期 beta 行情』共同作用的结果。")
        lines.append("")
        lines.append("   **建议**:")
        lines.append("   1. 不要在现有参数上继续微调, 那是在优化一个不存在的Alpha")
        lines.append("   2. 直接进入阶段1: 对每个指标做 IC 检验, 找出哪些真有预测力")
        lines.append("   3. 重新审视 `forced_regime=trending` 是否应作为实盘默认 — 它掩盖了参数失效")
    elif severe >= len(results) / 2:
        lines.append("❌ **过半分组严重过拟合**: 训练窗有Alpha但样本外消失, 参数对历史拟合过度。")
        lines.append("   建议: 暂停参数微调, 做 walk-forward 寻优 + 因子IC检验。")
    elif stable >= len(results) / 2:
        lines.append("✅ **过半分组稳健**: 参数具备一定样本外泛化能力, 可推进多因子升级。")
        lines.append("   建议: 进入阶段1(因子IC)和阶段2(数据驱动权重)。")
    else:
        lines.append("⚠️ **分组分化明显**: 部分稳健,部分无Alpha,参数对不同行业泛化能力不均。")
        lines.append("   建议: 对无Alpha分组单独诊断, 检查是否样本不足或参数失效。")
    lines.append("")
    lines.append("### 不可忽视的样本量问题")
    lines.append("")
    lines.append("即使夏普扛住验证, 若两窗口交易合计 < 30 笔, 统计上仍不足以确认Alpha。")
    lines.append("建议后续补充蒙特卡洛重排检验(打乱交易顺序1000次, 看夏普分位数)。")

    return "\n".join(lines)


def regenerate_report():
    """从现有 walk_forward_result.json 用新判断逻辑重算 verdict 并重新生成报告

    用途: 修正判断逻辑后, 无需重跑 15 分钟回测, 直接基于已有结果重新判定。
    用法: python scripts/walk_forward_validation.py --regen
    """
    if not os.path.exists(RESULT_JSON):
        print(f"❌ 找不到 {RESULT_JSON}, 请先运行完整回测")
        return
    with open(RESULT_JSON, "r", encoding="utf-8") as f:
        data = json.load(f)
    run_time = data.get("run_time", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    results = data["results"]

    # 用新逻辑重算 verdict
    print("用新判断逻辑重算 verdict...")
    for g, r in results.items():
        t_sharpe = r["train"]["sharpe_ratio"]
        te_sharpe = r["test"]["sharpe_ratio"]
        verdict, reason = judge_robustness(t_sharpe, te_sharpe)
        r["verdict"] = verdict
        r["reason"] = reason
        print(f"  {g}: {verdict} — {reason}")

    # 保存更新后的 JSON
    with open(RESULT_JSON, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

    # 重新生成报告
    report = generate_report(results, run_time)
    with open(REPORT_MD, "w", encoding="utf-8") as f:
        f.write(report)
    print(f"\n✓ 已重新生成报告 → {REPORT_MD}")

    # 汇总
    print(f"\n{'=' * 60}")
    print("  重新判定后的汇总")
    print(f"{'=' * 60}")
    print(f"{'分组':<12} {'训练夏普':>8} {'测试夏普':>8}  结论")
    for g, r in results.items():
        print(f"{g:<12} {r['train']['sharpe_ratio']:>8} {r['test']['sharpe_ratio']:>8}  {r['verdict']}")


def main():
    # --regen 模式: 从已有结果重新生成报告, 不重跑回测
    if len(sys.argv) > 1 and sys.argv[1] == "--regen":
        regenerate_report()
        return

    run_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print("=" * 70)
    print("  Walk-Forward 验证 — 检验现有参数是否过拟合")
    print(f"  训练窗: {TRAIN_START} ~ {TRAIN_END}")
    print(f"  测试窗: {TEST_START} ~ {TEST_END}")
    print(f"  体制:   auto (ADX自动判断)")
    print("=" * 70)

    # ── 备份并清除用户偏好 (强制 auto, 不受 trending 偏好干扰) ──
    pref_existed = os.path.exists(_USER_PREF_FILE)
    if pref_existed:
        shutil.copy2(_USER_PREF_FILE, PREF_BACKUP)
        print(f"\n✓ 已备份用户偏好 → {PREF_BACKUP}")
    UserPreferences().clear_all()
    print("✓ 已清除用户偏好 (强制 auto 体制, 避免掩盖体制自适应)")

    try:
        watchlist = load_watchlist()
        dm = DataManager()

        # ── 拉取基准 ──
        print(f"\n拉取基准 {BENCHMARK} 数据 ({DATA_START} ~ {DATA_END})...")
        benchmark_df = dm.get_daily_kline(BENCHMARK, start_date=DATA_START, end_date=DATA_END)
        if benchmark_df is None or benchmark_df.empty:
            print(f"  ⚠️  基准数据拉取失败, 继续 (alpha 将为 0, 不影响夏普/回撤判断)")
            benchmark_df = None
        else:
            print(f"  ✓ {len(benchmark_df)} 条")

        results = {}

        # ── 逐分组回测 ──
        for group_name, stocks in watchlist.items():
            if group_name.startswith("_"):
                continue
            codes = [s["code"] for s in stocks]
            print(f"\n{'─' * 70}")
            print(f"分组: {group_name}  ({len(codes)}只: {', '.join(codes)})")
            print(f"{'─' * 70}")

            data_map = fetch_data(dm, codes)
            if len(data_map) < 2:
                print(f"  ⚠️  有效数据股票数 {len(data_map)} < 2, 跳过该分组")
                continue
            print(f"  ✓ 成功拉取 {len(data_map)}/{len(codes)} 只股票数据")

            # 训练窗
            print(f"  ▶ 训练窗 {TRAIN_START} ~ {TRAIN_END} 回测中...")
            m_train, _ = run_backtest(data_map, benchmark_df, TRAIN_START, TRAIN_END)
            print(f"    夏普={m_train.sharpe_ratio:.3f}  收益={m_train.total_return*100:.2f}%  "
                  f"回撤={m_train.max_drawdown*100:.2f}%  交易={m_train.trade_count}笔  "
                  f"胜率={m_train.win_rate*100:.1f}%")

            # 测试窗
            print(f"  ▶ 测试窗 {TEST_START} ~ {TEST_END} 回测中...")
            m_test, _ = run_backtest(data_map, benchmark_df, TEST_START, TEST_END)
            print(f"    夏普={m_test.sharpe_ratio:.3f}  收益={m_test.total_return*100:.2f}%  "
                  f"回撤={m_test.max_drawdown*100:.2f}%  交易={m_test.trade_count}笔  "
                  f"胜率={m_test.win_rate*100:.1f}%")

            # 衰减与判定
            decay = None
            if abs(m_train.sharpe_ratio) > 0.01:
                decay = round((1 - m_test.sharpe_ratio / m_train.sharpe_ratio) * 100, 1)
            verdict, reason = judge_robustness(m_train.sharpe_ratio, m_test.sharpe_ratio)

            print(f"  ➜ 夏普衰减: {decay}%  |  结论: {verdict} ({reason})")

            results[group_name] = {
                "stocks": codes,
                "train": metrics_to_dict(m_train),
                "test": metrics_to_dict(m_test),
                "sharpe_decay_pct": decay if decay is not None else "N/A",
                "verdict": verdict,
                "reason": reason,
            }

        if not results:
            print("\n❌ 没有任何分组成功完成回测")
            return

        # ── 保存 JSON ──
        os.makedirs(os.path.dirname(RESULT_JSON), exist_ok=True)
        with open(RESULT_JSON, "w", encoding="utf-8") as f:
            json.dump({
                "run_time": run_time,
                "config": {
                    "train_window": [TRAIN_START, TRAIN_END],
                    "test_window": [TEST_START, TEST_END],
                    "regime_mode": "auto",
                    "data_range": [DATA_START, DATA_END],
                },
                "results": results,
            }, f, ensure_ascii=False, indent=2)
        print(f"\n✓ 结构化结果 → {RESULT_JSON}")

        # ── 生成 Markdown 报告 ──
        report = generate_report(results, run_time)
        with open(REPORT_MD, "w", encoding="utf-8") as f:
            f.write(report)
        print(f"✓ 可读报告   → {REPORT_MD}")

        # ── 汇总 ──
        print(f"\n{'=' * 70}")
        print("  汇总")
        print(f"{'=' * 70}")
        print(f"{'分组':<12} {'训练夏普':>8} {'测试夏普':>8} {'衰减':>8}  结论")
        for g, r in results.items():
            d = r["sharpe_decay_pct"]
            d_str = f"{d}%" if isinstance(d, (int, float)) else d
            print(f"{g:<12} {r['train']['sharpe_ratio']:>8} {r['test']['sharpe_ratio']:>8} {d_str:>8}  {r['verdict']}")

    finally:
        # ── 恢复用户偏好 ──
        if os.path.exists(PREF_BACKUP):
            shutil.move(PREF_BACKUP, _USER_PREF_FILE)
            print("\n✓ 已恢复用户偏好")
        else:
            # 原本就没有偏好, 清除产生的空文件
            if os.path.exists(_USER_PREF_FILE):
                os.remove(_USER_PREF_FILE)
            print("\n✓ 用户偏好原本为空, 已清理")


if __name__ == "__main__":
    main()
