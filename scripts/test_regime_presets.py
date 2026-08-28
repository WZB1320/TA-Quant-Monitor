"""测试不同预设参数对科技成长组回测的影响

场景:
  - 原始: 不修改，用当前配置
  - 趋势上涨: 低门槛(35)、宽止损(2.5)、短冷却(5)
  - 震荡: 高门槛(50)、紧止损(1.5)、长冷却(12)
  
仅测试科技成长组的 7 只股票
"""
import os, sys, json, copy
from collections import defaultdict
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data_fetcher import DataManager, Watchlist
from src.backtest import BacktestEngine
from src.signal_engine.filter import SignalFilter
from src.backtest.position import PositionManager
from src.config.group_config import GroupConfig

# ── 预设参数 ──

TRENDING_PRESET = {
    "score_threshold": 35,
    "score_ceiling": 60,
    "cooldown_days": 5,
    "atr_stop_mult": 2.5,
    "max_consecutive_losses": 3,
    "consecutive_loss_suspend": 8,
    "atr_price_ratio_max": 0.10,
}

RANGING_PRESET = {
    "score_threshold": 48,       # 高门槛，只进强信号
    "score_ceiling": 55,         # ceiling必须 > threshold，窄窗口48-55
    "cooldown_days": 12,         # 长冷却，不频繁交易
    "atr_stop_mult": 1.5,        # 紧止损，快进快出
    "max_consecutive_losses": 2,
    "consecutive_loss_suspend": 15,
    "atr_price_ratio_max": 0.06,
}

PRESETS = {
    "原始参数": None,
    "趋势上涨(激进)": TRENDING_PRESET,
    "震荡(保守)": RANGING_PRESET,
}

# ── 加载数据 ──
dm = DataManager()
wl = Watchlist()
gc = GroupConfig()

GROUP = "科技成长型"
target_codes = [s["code"] for s in wl.get_all() if gc.get_group(s["code"]) == GROUP]

data_map = {}
for code in target_codes:
    df = dm.get_daily_kline(code, start_date="2024-01-01")
    if df is not None and len(df) >= 120:
        data_map[code] = df

print(f"科技成长组加载 {len(data_map)} 只股票: {list(data_map.keys())}")
print(f"数据范围: {data_map[list(data_map.keys())[0]]['date'].iloc[0]} ~ {data_map[list(data_map.keys())[0]]['date'].iloc[-1]}")
print()

# ── 跑三组测试 ──
results = {}

# 修复: 此前每个预设都删除 data/signal_history.json, 会摧毁 LIVE 模式实盘去重数据.
# 改为回测模式运行(内存去重, 不读写磁盘), 每次测试去重状态干净且互不污染.
from src.config.runtime_mode import set_mode, RuntimeMode
set_mode(RuntimeMode.BACKTEST)

for label, preset in PRESETS.items():
    # 创建独有的 GroupConfig 覆盖方法
    import src.config.group_config as gc_mod
    from src.config.group_config import GroupConfig
    
    # Monkey-patch get_all_group_params 和 get_atr_stop_mult
    orig_get_params = GroupConfig.get_all_group_params
    orig_get_atr = GroupConfig.get_atr_stop_mult
    orig_get_threshold = GroupConfig.get_score_threshold
    orig_get_ceiling = GroupConfig.get_score_ceiling
    orig_get_cooldown = GroupConfig.get_cooldown_days
    orig_get_csl = GroupConfig.get_consecutive_loss_suspend
    orig_get_mcl = GroupConfig.get_max_consecutive_losses
    orig_get_aprm = GroupConfig.get_atr_price_ratio_max
    
    if preset:
        def patched_get_params(self, code):
            p = orig_get_params(self, code)
            group = self.get_group(code)
            if group == GROUP:
                p.update(preset)
            return p
        
        def patched_get_atr(self, code):
            group = self.get_group(code)
            if group == GROUP and "atr_stop_mult" in preset:
                return preset["atr_stop_mult"]
            return orig_get_atr(self, code)
        
        def patched_get_threshold(self, code):
            group = self.get_group(code)
            if group == GROUP and "score_threshold" in preset:
                return preset["score_threshold"]
            return orig_get_threshold(self, code)
        
        def patched_get_ceiling(self, code):
            group = self.get_group(code)
            if group == GROUP and "score_ceiling" in preset:
                return preset["score_ceiling"]
            return orig_get_ceiling(self, code)
        
        def patched_get_cooldown(self, code):
            group = self.get_group(code)
            if group == GROUP and "cooldown_days" in preset:
                return preset["cooldown_days"]
            return orig_get_cooldown(self, code)
        
        def patched_get_csl(self, code):
            group = self.get_group(code)
            if group == GROUP and "consecutive_loss_suspend" in preset:
                return preset["consecutive_loss_suspend"]
            return orig_get_csl(self, code)
        
        def patched_get_mcl(self, code):
            group = self.get_group(code)
            if group == GROUP and "max_consecutive_losses" in preset:
                return preset["max_consecutive_losses"]
            return orig_get_mcl(self, code)
        
        def patched_get_aprm(self, code):
            group = self.get_group(code)
            if group == GROUP and "atr_price_ratio_max" in preset:
                return preset["atr_price_ratio_max"]
            return orig_get_aprm(self, code)

        GroupConfig.get_all_group_params = patched_get_params
        GroupConfig.get_atr_stop_mult = patched_get_atr
        GroupConfig.get_score_threshold = patched_get_threshold
        GroupConfig.get_score_ceiling = patched_get_ceiling
        GroupConfig.get_cooldown_days = patched_get_cooldown
        GroupConfig.get_consecutive_loss_suspend = patched_get_csl
        GroupConfig.get_max_consecutive_losses = patched_get_mcl
        GroupConfig.get_atr_price_ratio_max = patched_get_aprm

    # 运行回测
    engine = BacktestEngine(
        initial_capital=100_000,
        lookback_days=120,
        position_ratio=0.30,
        signal_dedup_days=5,
        risk_per_trade=0.02,
        atr_stop_mult=preset["atr_stop_mult"] if preset else 2.0,
    )
    metrics = engine.run(data_map)
    
    # 逐股盈亏
    stock_pnl = defaultdict(float)
    stock_trades = defaultdict(list)
    for t in engine.position_mgr.closed_trades:
        stock_pnl[t.symbol] += t.pnl
        stock_trades[t.symbol].append(t)
    
    results[label] = {
        "metrics": metrics,
        "stock_pnl": dict(stock_pnl),
        "stock_trades": dict(stock_trades),
        "closed_trades": engine.position_mgr.closed_trades,
    }
    
    # 恢复原始方法
    if preset:
        GroupConfig.get_all_group_params = orig_get_params
        GroupConfig.get_atr_stop_mult = orig_get_atr
        GroupConfig.get_score_threshold = orig_get_threshold
        GroupConfig.get_score_ceiling = orig_get_ceiling
        GroupConfig.get_cooldown_days = orig_get_cooldown
        GroupConfig.get_consecutive_loss_suspend = orig_get_csl
        GroupConfig.get_max_consecutive_losses = orig_get_mcl
        GroupConfig.get_atr_price_ratio_max = orig_get_aprm


# ── 汇总输出 ──

def fmt_pnl(v):
    return f"{v:+,.0f}" if v else "0"

def fmt_pnl_rate(v):
    return f"{v:+,.2f}%"

print()
print("=" * 120)
print("  科技成长组 三种预设参数 回测对比")
print("=" * 120)

# ── 全局指标对比 ──
print(f"\n{'指标':<25}", end="")
for label in PRESETS:
    print(f"{label:<22}", end="")
print()
print("-" * 120)

m_names = [
    ("总收益率", "total_return"),
    ("年化收益率", "annual_return"),
    ("最大回撤", "max_drawdown"),
    ("夏普比率", "sharpe_ratio"),
    ("胜率", "win_rate"),
    ("交易笔数", "trade_count"),
    ("总盈亏", "total_pnl"),
    ("盈亏比", "profit_factor"),
    ("平均持仓天数", "avg_holding_days"),
]

for display, key in m_names:
    print(f"{display:<25}", end="")
    for label in PRESETS:
        r = results[label]
        v = getattr(r["metrics"], key, None)
        if v is None:
            print(f"{'N/A':<22}", end="")
        elif key == "total_return":
            print(f"{v*100:+.2f}%{'':<16}", end="")
        elif key == "annual_return":
            print(f"{v*100:+.2f}%{'':<16}", end="")
        elif key == "max_drawdown":
            print(f"{v*100:+.2f}%{'':<16}", end="")
        elif key == "sharpe_ratio":
            print(f"{v:+.2f}{'':<17}", end="")
        elif key == "win_rate":
            print(f"{v*100:.1f}%{'':<17}", end="")
        elif key == "trade_count":
            print(f"{v:<22}", end="")
        elif key == "total_pnl":
            print(f"{v:+,.0f}{'':<17}", end="")
        elif key == "profit_factor":
            print(f"{v:.2f}{'':<18}", end="")
        elif key == "avg_holding_days":
            print(f"{v:.1f}天{'':<17}", end="")
    print()

# ── 逐股对比 ──
print(f"\n{'─' * 120}")
print(f"  逐股交易对比 (2026年)")
print(f"{'─' * 120}")

for code in target_codes:
    name = ""
    for s in wl.get_all():
        if s["code"] == code:
            name = s["name"]
            break
    
    print(f"\n  【{code} {name}】")
    print(f"  {'':>10}{'交易笔数':>8}{'总盈亏':>12}{'胜率':>8}{'平均盈':>10}", end="")
    
    # 列出每笔交易
    for label in PRESETS:
        labels_short = ["原始", "趋势上涨", "震荡"]
        idx = list(PRESETS.keys()).index(label)
        r = results[label]
        trades = r["stock_trades"].get(code, [])
        t2026 = [t for t in trades if str(t.entry_date) >= "2026"]
        pnl = sum(t.pnl for t in trades)
        wins = [t for t in trades if t.pnl > 0]
        wr = len(wins) / len(trades) * 100 if trades else 0
        avg_win = sum(t.pnl for t in wins) / len(wins) if wins else 0
        
        print(f"\n  {labels_short[idx]:>10} {len(trades):>8} {fmt_pnl(pnl):>12} {wr:>7.1f}% {avg_win:>9.0f}", end="")
        
        if trades:
            print(f"  ", end="")
            for t in trades:
                print(f"[{t.entry_date}→{t.exit_date} {t.pnl:+.0f}] ", end="")

print()
print(f"\n{'=' * 120}")
print(f"  结论:")
print(f"  - 趋势上涨预设: 低门槛+宽止损+短冷却 → 适合你判断会趋势上涨时使用，激进捕获")
print(f"  - 震荡预设:     高门槛+紧止损+长冷却 → 适合你判断是震荡市时使用，保守防守")
print(f"  - 自动(原始):   用当前配置，ADX体制自适应，不干预")
print(f"{'=' * 120}")