"""当前信号分析 — 基于最新收盘价的自选股操作策略"""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.data_fetcher import DataManager, Watchlist
from src.signal_engine import SignalEngine
from src.signal_engine.signals import SignalLevel
from src.config.group_config import GroupConfig
from src.signal_engine.scorer import Scorer
from src.indicators import IndicatorPipeline
import pandas as pd

dm = DataManager()
wl = Watchlist()
gc = GroupConfig()

# 加载数据
data_map = {}
for s in wl.get_all():
    code = s["code"]
    df = dm.get_daily_kline(code, start_date="2024-01-01")
    if df is not None and len(df) >= 120:
        data_map[code] = df

print("=" * 85)
print("  当前信号分析 (基于最新收盘价)")
print("=" * 85)

# 用信号引擎分析（不传 group_config 避免去重/冷却影响原始信号判断）
sig_engine = SignalEngine(dedup_days=5, group_config=gc)
scorer = Scorer()
pipeline = IndicatorPipeline()

# 先获取 MA60 方向
for code, df in data_map.items():
    ir = pipeline.run(df)
    ma60_dir = ir.get("MA60_DIRECTION")
    data_map[code] = (df, ir, ma60_dir)

results = []
for code, (df, ir, ma60_dir) in data_map.items():
    result = sig_engine.analyze(code, df)
    results.append(result)

# 按组分类输出
groups_order = ["科技成长型", "机械制造型", "周期资源型", "消费稳健型", "医药创新型"]
grouped = {}
for r in results:
    g = gc.get_group(r.symbol)
    grouped.setdefault(g, []).append(r)

for g in groups_order:
    if g not in grouped:
        continue
    print(f"\n{'─' * 85}")
    print(f"  【{g}】")
    print(f"{'─' * 85}")

    for r in sorted(grouped[g], key=lambda x: x.score, reverse=True):
        df, ir, _ = data_map[r.symbol]
        latest = df.iloc[-1]
        latest_date = str(latest["date"])[:10]
        close = latest["close"]

        # 获取分组参数
        params = gc.get_all_group_params(r.symbol)
        sc_threshold = params.get("score_threshold", 25)
        sc_ceiling = params.get("score_ceiling", 0)

        # 执行约束 (来自 SignalResult.execution, 由 classifier 接入后填充)
        execution = r.execution
        if execution is None:
            # 向后兼容: 无 execution 时构造默认值
            from src.signal_engine.classifier import ExecutionConstraint
            execution = ExecutionConstraint()

        # 执行状态标签 (独立于 7 级信号, 仅表示"能否操作")
        if r.hard_filter_blocked:
            exec_tag = "✗ 硬过滤"
        elif not r.level.is_actionable:
            exec_tag = "— 无需操作"
        elif execution.is_executable:
            exec_tag = "★ 可执行"
        else:
            exec_tag = f"○ {execution.blocking_reason[:12]}"

        # 获取 MA60 方向
        ma60_val = ir.get("MA60")
        ma60_dir_str = ""
        if ma60_val is not None:
            ma60_v = ma60_val.values.get("ma60")
            if ma60_v is not None:
                ma60_dir_str = f"MA60={ma60_v:.2f} | {'多头' if close > ma60_v else '空头'}"

        # 降级轨迹 (若有)
        demotion_info = ""
        if r.demotion_chain:
            demotion_info = f" | 降级: {' → '.join(r.demotion_chain)}"

        print(f"  {r.symbol} | {latest_date} | 收盘 {close:.2f} | {exec_tag}")
        print(f"    信号: {r.level.label} | 得分: {r.score:+.1f} | 置信度: {r.confidence:.0%}{demotion_info}")
        print(f"    阈值: [{sc_threshold}-{sc_ceiling if sc_ceiling else '无上限'}] | {ma60_dir_str}")
        if r.hard_filter_blocked:
            print(f"    拦截: {r.block_reason}")
        if r.block_detail and not r.hard_filter_blocked:
            print(f"    拦截: {r.block_detail}")
        print(f"    理由: {r.reason}")

        # 关键指标
        rsi = ir.get("RSI")
        macd = ir.get("MACD")
        vol_ratio = ir.get("VOL_RATIO")
        atr = ir.get("ATR")
        if rsi:
            print(f"    RSI={rsi.values.get('rsi', 'N/A'):.1f}" if isinstance(rsi.values.get('rsi'), (int, float)) else f"    RSI=N/A", end="")
        if macd:
            dif = macd.values.get("dif")
            dea = macd.values.get("dea")
            if dif is not None:
                print(f" | DIF={dif:.3f} DEA={dea:.3f}" if dea is not None else f" | DIF={dif:.3f}", end="")
        if vol_ratio:
            vr = vol_ratio.values.get("volume_ratio")
            if vr is not None:
                print(f" | 量比={vr:.2f}", end="")
        if atr:
            a = atr.values.get("atr")
            if a is not None:
                print(f" | ATR/Price={a/close*100:.2f}%", end="")
        print()

print(f"\n{'=' * 85}")
print("  可执行信号汇总")
print(f"{'=' * 85}")

actionable = []
for r in results:
    execution = r.execution
    if execution is None:
        from src.signal_engine.classifier import ExecutionConstraint
        execution = ExecutionConstraint()
    # 可执行: 信号可操作 + 无执行约束阻断
    if r.level.is_bullish and r.level.is_actionable and execution.is_executable:
        actionable.append(r)

if actionable:
    print("\n可买入信号:")
    for r in sorted(actionable, key=lambda x: x.score, reverse=True):
        g = gc.get_group(r.symbol)
        df, _, _ = data_map[r.symbol]
        close = df.iloc[-1]["close"]
        print(f"  {r.symbol} ({g}) | 收盘 {close:.2f} | {r.level.label} | 得分 {r.score:+.1f} | {r.reason}")
else:
    print("\n当前无符合条件的买入信号，建议观望等待。")

print()