"""
科技成长型分组回测 — 单次运行, 输出JSON结果
被 ab_test_runner.py 调用, 通过子进程隔离避免状态泄漏
"""
import sys
import os
import json
import numpy as np
import pandas as pd

project_root = os.path.join(os.path.dirname(__file__), "..")
sys.path.insert(0, project_root)

from src.data_fetcher import DataManager
from src.config.group_config import GroupConfig
from src.config.runtime_mode import set_mode, RuntimeMode
from src.backtest.engine import BacktestEngine

# 必须在创建任何 SignalEngine/Filter 之前设置为回测模式,
# 否则 Filter 会从磁盘加载实时信号历史, 导致误去重
set_mode(RuntimeMode.BACKTEST)

TECH_STOCKS = [
    "000725", "300450", "002138", "300433", "600522",
    "002272", "603002", "301188", "600552", "000100", "600487",
]


def main():
    label = sys.argv[1] if len(sys.argv) > 1 else "default"
    output_file = sys.argv[2] if len(sys.argv) > 2 else "result.json"

    print(f"Running backtest: {label}")

    dm = DataManager()
    data_map = {}
    for code in TECH_STOCKS:
        df = dm.get_daily_kline(code, start_date="2025-06-01", end_date="2026-07-13")
        if df is not None and not df.empty:
            data_map[code] = df

    benchmark_df = dm.get_daily_kline("000300", start_date="2025-06-01", end_date="2026-07-13")

    # Reset GroupConfig singleton (读取用户偏好, 保持 trending 模式)
    GroupConfig._instance = None
    GroupConfig._config = None

    # 验证配置
    gc = GroupConfig()
    params = gc.get_all_group_params("000725")
    print(f"  forced_regime: {params.get('forced_regime')}")
    print(f"  score_threshold: {params.get('score_threshold')}")

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
        forced_regime="trending",
    )

    metrics = engine.run(
        data_map=data_map,
        benchmark_df=None,
        start_date="2026-01-01",
        end_date="2026-07-13",
    )

    trades = engine.position_mgr.closed_trades
    open_positions = engine.position_mgr.open_positions

    # 获取回测最后一天的收盘价 (用于计算未平仓浮盈)
    from datetime import datetime as _dt
    last_date = _dt.strptime("2026-07-13", "%Y-%m-%d").date()
    last_prices = {}
    for symbol, df in data_map.items():
        for i, d in enumerate(df["date"]):
            d_date = pd.Timestamp(d).date() if isinstance(d, str) else d
            if d_date == last_date:
                last_prices[symbol] = float(df.iloc[i]["close"])
                break

    # 序列化已平仓交易明细
    trade_list = []
    for t in trades:
        trade_list.append({
            "symbol": t.symbol,
            "entry_date": str(t.entry_date),
            "entry_price": round(t.entry_price, 2),
            "exit_date": str(t.exit_date),
            "exit_price": round(t.exit_price, 2),
            "pnl_pct": round(float(t.pnl_pct) * 100, 1),
            "holding_days": t.holding_days,
            "exit_signal": t.exit_signal,
        })

    # 序列化未平仓持仓 (回测结束时仍持有)
    open_list = []
    for symbol, t in open_positions.items():
        current_price = last_prices.get(symbol, t.entry_price)
        unrealized_pct = (current_price - t.entry_price) / t.entry_price * 100
        open_list.append({
            "symbol": symbol,
            "entry_date": str(t.entry_date),
            "entry_price": round(t.entry_price, 2),
            "current_price": round(current_price, 2),
            "unrealized_pct": round(unrealized_pct, 1),
            "shares": t.shares,
            "entry_signal": t.entry_signal,
        })

    result = {
        "label": label,
        "metrics": {
            "total_return": round(metrics.total_return * 100, 2),
            "annual_return": round(metrics.annual_return * 100, 2),
            "max_drawdown": round(metrics.max_drawdown * 100, 2),
            "sharpe_ratio": round(metrics.sharpe_ratio, 3),
            "trade_count": metrics.trade_count,
            "win_count": metrics.win_count,
            "win_rate": round(metrics.win_rate * 100, 1),
            "avg_win_pct": round(metrics.avg_win_pct * 100, 2),
            "avg_loss_pct": round(metrics.avg_loss_pct * 100, 2),
            "profit_factor": round(metrics.profit_factor, 2),
            "avg_holding_days": round(metrics.avg_holding_days, 1),
        },
        "trades": trade_list,
        "open_positions": open_list,
    }

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    print(f"Done: {metrics.trade_count} closed trades, "
          f"{len(open_list)} open positions, return={metrics.total_return*100:.2f}%")
    print(f"Saved to {output_file}")


if __name__ == "__main__":
    main()
