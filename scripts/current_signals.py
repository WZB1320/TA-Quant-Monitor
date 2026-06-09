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
        cooldown = params.get("cooldown_days", 5)

        # 判断是否在冷却期
        in_cooldown = sig_engine.filter.is_in_cooldown(r.symbol, None, True, group_cooldown=cooldown)

        # 判断是否通过得分阈值
        score_pass = sc_threshold <= r.score <= (sc_ceiling if sc_ceiling > 0 else 100)

        # 综合判断
        if r.level.is_bullish and r.level.is_actionable and score_pass and not in_cooldown:
            action = "★ 可操作"
        elif r.level.is_bullish and r.level.is_actionable and not score_pass:
            action = "△ 得分不达标"
        elif r.level.is_bullish and r.level.is_actionable and in_cooldown:
            action = "○ 冷却期"
        elif r.level.is_bullish and not r.level.is_actionable:
            action = "  关注"
        elif r.level.is_bearish:
            action = "  偏空"
        else:
            action = "  观望"

        # 获取 MA60 方向
        ma60_val = ir.get("MA60")
        ma60_dir_str = ""
        if ma60_val is not None:
            ma60_v = ma60_val.values.get("ma60")
            if ma60_v is not None:
                ma60_dir_str = f"MA60={ma60_v:.2f} | {'多头' if close > ma60_v else '空头'}"

        print(f"  {r.symbol} | {latest_date} | 收盘 {close:.2f} | {action}")
        print(f"    信号: {r.level.label} | 得分: {r.score:+.1f} | 置信度: {r.confidence:.0%}")
        print(f"    阈值: [{sc_threshold}-{sc_ceiling if sc_ceiling else '无上限'}] | {ma60_dir_str}")
        if r.hard_filter_blocked:
            print(f"    拦截: {r.block_reason}")
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
print("  操作建议汇总")
print(f"{'=' * 85}")

actionable = []
for r in results:
    params = gc.get_all_group_params(r.symbol)
    sc_threshold = params.get("score_threshold", 25)
    sc_ceiling = params.get("score_ceiling", 0)
    cooldown = params.get("cooldown_days", 5)
    score_pass = sc_threshold <= r.score <= (sc_ceiling if sc_ceiling > 0 else 100)
    in_cooldown = sig_engine.filter.is_in_cooldown(r.symbol, None, True, group_cooldown=cooldown)

    if r.level.is_bullish and r.level.is_actionable and score_pass and not in_cooldown:
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