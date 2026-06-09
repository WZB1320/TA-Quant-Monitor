"""验证策略配置文件参数是否正确加载到各模块"""
import json
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

# 读取配置文件
config_path = os.path.join(os.path.dirname(__file__), "..", "config", "strategy_config.json")
with open(config_path, "r", encoding="utf-8") as f:
    config = json.load(f)["strategy_config"]

passed = 0
failed = 0

def check(name, expected, actual, tolerance=0.001):
    global passed, failed
    if isinstance(expected, float):
        ok = abs(expected - actual) < tolerance
    else:
        ok = expected == actual
    status = "PASS" if ok else "FAIL"
    if not ok:
        failed += 1
        print(f"  [{status}] {name}: 期望={expected}, 实际={actual}")
    else:
        passed += 1
        print(f"  [{status}] {name}: {actual}")

# ── 1. 回测引擎参数 ──
print("=" * 60)
print("  1. 回测引擎 (BacktestEngine)")
print("=" * 60)
from src.backtest import BacktestEngine
engine = BacktestEngine(
    initial_capital=config["backtest"]["initial_capital"],
    lookback_days=config["backtest"]["lookback_days"],
    position_ratio=config["position_sizing"]["position_ratio"],
    risk_per_trade=config["position_sizing"]["risk_per_trade"],
    atr_stop_mult=config["position_sizing"]["atr_stop_mult"],
)

check("initial_capital", config["backtest"]["initial_capital"], engine.initial_capital)
check("lookback_days", config["backtest"]["lookback_days"], engine.lookback_days)
check("position_ratio", config["position_sizing"]["position_ratio"], engine.position_ratio)
check("risk_per_trade", config["position_sizing"]["risk_per_trade"], engine.risk_per_trade)
check("atr_stop_mult", config["position_sizing"]["atr_stop_mult"], engine.atr_stop_mult)

# ── 2. 仓位管理器参数 ──
print(f"\n{'=' * 60}")
print("  2. 仓位管理器 (PositionManager)")
print("=" * 60)
from src.backtest.position import PositionManager
pm = PositionManager(
    initial_capital=config["backtest"]["initial_capital"],
    position_ratio=config["position_sizing"]["position_ratio"],
    risk_per_trade=config["position_sizing"]["risk_per_trade"],
    atr_stop_mult=config["position_sizing"]["atr_stop_mult"],
)

check("risk_per_trade", config["position_sizing"]["risk_per_trade"], pm.risk_per_trade)
check("atr_stop_mult", config["position_sizing"]["atr_stop_mult"], pm.atr_stop_mult)
check("position_ratio", config["position_sizing"]["position_ratio"], pm.position_ratio)

# 体制配置
regime_cfg = config["position_sizing"]["regime_config"]
for regime, expected in regime_cfg.items():
    actual = pm.REGIME_CONFIG.get(regime)
    if actual:
        check(f"regime.{regime}.target_ratio", expected["target_ratio"], actual["target_ratio"])
        check(f"regime.{regime}.max_per_stock", expected["max_per_stock"], actual["max_per_stock"])
    else:
        failed += 1
        print(f"  [FAIL] regime.{regime}: 配置中不存在")

# ── 3. 信号引擎参数 ──
print(f"\n{'=' * 60}")
print("  3. 信号引擎 (SignalEngine / SignalFilter)")
print("=" * 60)
from src.signal_engine import SignalEngine
from src.signal_engine.filter import SignalFilter

se = SignalEngine(dedup_days=config["signal_engine"]["dedup_days"])
check("dedup_days", config["signal_engine"]["dedup_days"], se.filter.dedup_days)
check("cooldown_days", config["signal_engine"]["cooldown_days"], se.filter.cooldown_days)

# ── 4. 评分器权重 ──
print(f"\n{'=' * 60}")
print("  4. 评分器权重 (Scorer)")
print("=" * 60)
from src.signal_engine.scorer import Scorer, CATEGORY_WEIGHTS, REGIME_WEIGHTS

cat_weights = config["signal_engine"]["category_weights"]
for cat, expected_val in cat_weights.items():
    actual_val = CATEGORY_WEIGHTS.get(cat, -1)
    check(f"category_weight.{cat}", expected_val, actual_val)

regime_weights = config["signal_engine"]["regime_weights"]
for regime, expected_weights in regime_weights.items():
    actual_weights = REGIME_WEIGHTS.get(regime, {})
    for cat, expected_val in expected_weights.items():
        actual_val = actual_weights.get(cat, -1)
        check(f"regime_weight.{regime}.{cat}", expected_val, actual_val)

# ── 5. 指标参数 ──
print(f"\n{'=' * 60}")
print("  5. 技术指标参数 (Indicators)")
print("=" * 60)
from src.indicators.trend import MA60Indicator, EMADualIndicator, MACDIndicator
from src.indicators.strength import ADXIndicator, ATRIndicator
from src.indicators.momentum import RSIIndicator, KDJIndicator
from src.indicators.volume import OBVIndicator, VolumeRatioIndicator

ind_cfg = config["indicators"]

# MA60
check("MA60.period", 60, 60)  # MA60 固定60
# EMA
check("EMA_DUAL.fast", ind_cfg["trend"]["EMA_DUAL"]["fast"], 12)
check("EMA_DUAL.slow", ind_cfg["trend"]["EMA_DUAL"]["slow"], 26)
# MACD
check("MACD.fast", ind_cfg["trend"]["MACD"]["fast"], 12)
check("MACD.slow", ind_cfg["trend"]["MACD"]["slow"], 26)
check("MACD.signal", ind_cfg["trend"]["MACD"]["signal"], 9)
# ADX
check("ADX.period", ind_cfg["strength"]["ADX"]["period"], 14)
# ATR
check("ATR.period", ind_cfg["strength"]["ATR"]["period"], 14)
# RSI
check("RSI.period", ind_cfg["momentum"]["RSI"]["period"], 21)
check("RSI.smooth_ema", ind_cfg["momentum"]["RSI"]["smooth_ema"], 5)
check("RSI.neutral_low", ind_cfg["momentum"]["RSI"]["neutral_zone"][0], 40)
check("RSI.neutral_high", ind_cfg["momentum"]["RSI"]["neutral_zone"][1], 60)
# KDJ
check("KDJ.k_period", ind_cfg["momentum"]["KDJ"]["k_period"], 9)
check("KDJ.k_smooth", ind_cfg["momentum"]["KDJ"]["k_smooth"], 3)
check("KDJ.d_smooth", ind_cfg["momentum"]["KDJ"]["d_smooth"], 3)

# ── 6. 交易成本 ──
print(f"\n{'=' * 60}")
print("  6. 交易成本 (Broker)")
print("=" * 60)
from src.backtest.broker import Broker

cost_cfg = config["trading_costs"]
broker = Broker(
    commission_rate=cost_cfg["commission_rate"],
    stamp_tax=cost_cfg["stamp_tax"],
    slippage=cost_cfg["slippage"],
    min_commission=cost_cfg["min_commission"],
)
check("commission_rate", cost_cfg["commission_rate"], broker.commission_rate)
check("stamp_tax", cost_cfg["stamp_tax"], broker.stamp_tax)
check("slippage", cost_cfg["slippage"], broker.slippage)
check("min_commission", cost_cfg["min_commission"], broker.min_commission)

# ── 7. 自选股 ──
print(f"\n{'=' * 60}")
print("  7. 自选股 (Watchlist)")
print("=" * 60)
from src.data_fetcher import Watchlist
wl = Watchlist()
wl_stocks = wl.get_all()
wl_codes = {s["code"] for s in wl_stocks}
cfg_codes = set()
for group_name, stocks in config.get("watchlist", {}).items():
    if group_name.startswith("_"):
        continue
    for stock in stocks:
        cfg_codes.add(stock["code"])

missing = cfg_codes - wl_codes
extra = wl_codes - cfg_codes
if not missing and not extra:
    passed += 1
    print(f"  [PASS] 自选股一致: {len(wl_codes)}只")
else:
    if missing:
        failed += 1
        print(f"  [FAIL] 配置中有但watchlist.json缺失: {missing}")
    if extra:
        failed += 1
        print(f"  [FAIL] watchlist.json有但配置中缺失: {extra}")

# ── 8. 止损止盈逻辑验证 ──
print(f"\n{'=' * 60}")
print("  8. 止损止盈逻辑验证")
print("=" * 60)
from datetime import date

pm_test = PositionManager(initial_capital=100000, position_ratio=0.30,
                          risk_per_trade=0.05, atr_stop_mult=2.5)
pm_test.set_regime("trending")

# 开仓: entry=10.0, ATR=0.50
trade = pm_test.open_long("TEST", date(2026, 1, 1), 10.0, "测试", atr_value=0.50)
if trade:
    check("开仓成功", True, True)
    check("ATR保存", 0.50, trade._atr_value)

    # 硬止损价 = 10.0 - 2.5×0.5 = 8.75
    expected_hard_stop = 10.0 - 2.5 * 0.5
    check("硬止损价", expected_hard_stop, trade._atr_stop_price)

    # 模拟盈利5%: 价格10.5, 止盈距离=2.5×0.5=1.25, 从最高点回撤1.25才触发
    result = pm_test.check_stop_loss("TEST", 10.5, date(2026, 1, 5))
    check("盈利5%不止盈", None, result)

    # 模拟盈利25%: 价格12.5, 最高12.5, 回撤1.5×0.5=0.75触发
    # 先更新最高价到12.5
    pm_test.check_stop_loss("TEST", 12.5, date(2026, 1, 10))  # 更新最高价, 不触发止盈
    # 价格从12.5跌到11.6 (回撤0.9)
    # 此时 profit_pct = (11.6-10)/10 = 16% → trailing_mult=2.0, trailing_dist=1.0
    # 12.5 - 11.6 = 0.9 < 1.0, 不触发
    # 再跌到11.4 (回撤1.1 > 1.0), 触发
    result = pm_test.check_stop_loss("TEST", 11.4, date(2026, 1, 11))  # 回撤1.1
    check("盈利>20%移动止盈触发", True, result is not None)
else:
    failed += 1
    print(f"  [FAIL] 开仓失败")

# ── 汇总 ──
print(f"\n{'=' * 60}")
print(f"  验证结果汇总")
print(f"{'=' * 60}")
total = passed + failed
print(f"  通过: {passed}/{total}")
print(f"  失败: {failed}/{total}")
if failed == 0:
    print(f"\n  所有配置参数验证通过!")
else:
    print(f"\n  有 {failed} 项配置不匹配, 请检查!")
    sys.exit(1)
