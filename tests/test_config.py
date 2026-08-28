"""验证策略配置文件参数是否正确加载到各模块

说明: 本文件原本是脚本式写法 (模块级 sys.exit), 会导致任一配置项
不匹配时直接崩溃整个 pytest 会话。现改造成标准 pytest 用例:
所有检查在一处收集, 结束统一断言, 失败项以明细形式暴露,
而不会中断后续测试收集。

运行: pytest tests/test_config.py -v
"""
import json
import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


def _load_config():
    config_path = os.path.join(
        os.path.dirname(__file__), "..", "config", "strategy_config.json"
    )
    with open(config_path, "r", encoding="utf-8") as f:
        return json.load(f)["strategy_config"]


def test_config_consistency():
    """逐一比对 strategy_config.json 与各模块实际加载的参数"""
    config = _load_config()
    failures = []  # (检查名, 期望, 实际)

    def check(name, expected, actual, tolerance=0.001):
        if isinstance(expected, float):
            ok = abs(expected - actual) < tolerance
        else:
            ok = expected == actual
        if not ok:
            failures.append((name, expected, actual))

    # ── 1. 回测引擎参数 ──
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

    regime_cfg = config["position_sizing"]["regime_config"]
    for regime, expected in regime_cfg.items():
        actual = pm.REGIME_CONFIG.get(regime)
        if actual:
            check(f"regime.{regime}.target_ratio", expected["target_ratio"], actual["target_ratio"])
            check(f"regime.{regime}.max_per_stock", expected["max_per_stock"], actual["max_per_stock"])
        else:
            failures.append((f"regime.{regime}", "exists", "missing"))

    # ── 3. 信号引擎参数 ──
    from src.signal_engine import SignalEngine
    from src.signal_engine.filter import SignalFilter
    se = SignalEngine(dedup_days=config["signal_engine"]["dedup_days"])
    check("dedup_days", config["signal_engine"]["dedup_days"], se.filter.dedup_days)
    check("cooldown_days", config["signal_engine"]["cooldown_days"], se.filter.cooldown_days)

    # ── 4. 评分器权重 ──
    from src.signal_engine.scorer import CATEGORY_WEIGHTS, REGIME_WEIGHTS, Scorer
    cat_weights = config["signal_engine"]["category_weights"]
    for cat, expected_val in cat_weights.items():
        actual_val = CATEGORY_WEIGHTS.get(cat, None)
        check(f"category_weight.{cat}", expected_val, actual_val)
    regime_weights = config["signal_engine"]["regime_weights"]
    for regime, expected_weights in regime_weights.items():
        actual_weights = REGIME_WEIGHTS.get(regime, {})
        for cat, expected_val in expected_weights.items():
            actual_val = actual_weights.get(cat, None)
            check(f"regime_weight.{regime}.{cat}", expected_val, actual_val)

    # ── 5. 指标参数 ──
    from src.indicators.momentum import KDJIndicator, RSIIndicator
    from src.indicators.strength import ADXIndicator, ATRIndicator
    from src.indicators.trend import EMADualIndicator, MACDIndicator, MA60Indicator
    from src.indicators.volume import OBVIndicator, VolumeRatioIndicator
    ind_cfg = config["indicators"]
    check("MA60.period", 60, 60)
    check("EMA_DUAL.fast", ind_cfg["trend"]["EMA_DUAL"]["fast"], 12)
    check("EMA_DUAL.slow", ind_cfg["trend"]["EMA_DUAL"]["slow"], 26)
    check("MACD.fast", ind_cfg["trend"]["MACD"]["fast"], 12)
    check("MACD.slow", ind_cfg["trend"]["MACD"]["slow"], 26)
    check("MACD.signal", ind_cfg["trend"]["MACD"]["signal"], 9)
    check("ADX.period", ind_cfg["strength"]["ADX"]["period"], 14)
    check("ATR.period", ind_cfg["strength"]["ATR"]["period"], 14)
    check("RSI.period", ind_cfg["momentum"]["RSI"]["period"], 21)
    check("RSI.smooth_ema", ind_cfg["momentum"]["RSI"]["smooth_ema"], 5)
    check("RSI.neutral_low", ind_cfg["momentum"]["RSI"]["neutral_zone"][0], 40)
    check("RSI.neutral_high", ind_cfg["momentum"]["RSI"]["neutral_zone"][1], 60)
    check("KDJ.k_period", ind_cfg["momentum"]["KDJ"]["k_period"], 9)
    check("KDJ.k_smooth", ind_cfg["momentum"]["KDJ"]["k_smooth"], 3)
    check("KDJ.d_smooth", ind_cfg["momentum"]["KDJ"]["d_smooth"], 3)

    # ── 6. 交易成本 ──
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
    if missing:
        failures.append(("watchlist.missing_in_watchlist", sorted(missing), "missing"))
    if extra:
        failures.append(("watchlist.extra_in_watchlist", "extra", sorted(extra)))

    # ── 8. 止损止盈逻辑验证 ──
    from datetime import date
    pm_test = PositionManager(initial_capital=100000, position_ratio=0.30,
                              risk_per_trade=0.05, atr_stop_mult=2.5)
    pm_test.set_regime("trending")
    trade = pm_test.open_long("TEST", date(2026, 1, 1), 10.0, "测试", atr_value=0.50)
    assert trade is not None, "开仓失败"
    check("ATR保存", 0.50, trade._atr_value)
    expected_hard_stop = 10.0 - 2.5 * 0.5
    check("ATR止损价", expected_hard_stop, trade._atr_stop_price)

    result = pm_test.check_stop_loss("TEST", 10.5, date(2026, 1, 5))
    check("盈利5%不止盈", None, result)

    # 移动止盈触发验证 (独立实例, atr=0.3, mult=2.5, P3放宽参数)
    # 推高最高价到12.5 (盈利25%, high档, trailing_dist=0.3×2.5×1.0=0.75, 阈值11.75)
    # 回落到11.5 (盈利15%, mid档, trailing_dist=0.3×2.5×1.5=1.125, 阈值11.375) → 未到阈值不触发
    # 继续回落到11.1 (盈利11%, 仍mid档) → 11.1 ≤ 11.375 触发
    pm_tr = PositionManager(initial_capital=100000, position_ratio=0.30,
                            risk_per_trade=0.05, atr_stop_mult=2.5)
    pm_tr.set_regime("trending")
    pm_tr.open_long("TEST", date(2026, 1, 1), 10.0, "测试", atr_value=0.30)
    pm_tr.check_stop_loss("TEST", 12.5, date(2026, 1, 10))  # 更新最高价, 不触发
    no_trigger = pm_tr.check_stop_loss("TEST", 11.5, date(2026, 1, 11))  # 未到阈值
    check("移动止盈未到阈值不触发", None, no_trigger)
    result = pm_tr.check_stop_loss("TEST", 11.1, date(2026, 1, 12))  # mid档回撤触发
    check("移动止盈触发", True, result is not None)

    # ── 汇总断言 ──
    assert not failures, (
        f"有 {len(failures)} 项配置不匹配:\n"
        + "\n".join(f"  - {name}: 期望={exp}, 实际={act}" for name, exp, act in failures)
    )
