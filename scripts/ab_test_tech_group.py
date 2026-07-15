"""
科技成长型分组 — T1档位倍率 A/B 回测对比

A组: T1=1.0× (默认)
B组: T1=2.0× (放宽T1回撤阈值)

回测区间: 2026-01-01 ~ 2026-07-13
基准: 沪深300 (不参与交易, 仅用于体制检测)
"""
import sys
import os
import json
import types
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

project_root = os.path.join(os.path.dirname(__file__), "..")
sys.path.insert(0, project_root)

from src.data_fetcher import DataManager
from src.config.group_config import GroupConfig
from src.backtest.engine import BacktestEngine
from src.backtest.position import PositionManager, Trade

# ── 科技成长型股票列表 ──
TECH_STOCKS = [
    "000725",  # 京东方A
    "300450",  # 先导智能
    "002138",  # 顺络电子
    "300433",  # 蓝思科技
    "600522",  # 中天科技
    "002272",  # 川润股份
    "603002",  # 宏昌电子
    "301188",  # 力诺药包
    "600552",  # 凯盛科技
    "000100",  # TCL科技
    "600487",  # 亨通光电
]


def make_patched_check_stop_loss(tier_mult_factors):
    """
    生成一个 monkey-patch 版的 check_stop_loss,
    使用自定义的 tier_mult_factors 替代硬编码的 0.6/0.8/1.0
    """
    def check_stop_loss(self, symbol: str, current_price: float,
                        current_date, signal_score=None):
        trade = self._open.get(symbol)
        if trade is None:
            return None

        # 更新最高价
        if current_price > trade.highest_price:
            trade.highest_price = current_price

        # 安全网: 10%硬止损
        if current_price <= trade.entry_price * 0.90:
            pnl_pct = (current_price - trade.entry_price) / trade.entry_price
            return self.close_position(symbol, current_date, current_price,
                                       f"安全网硬止损 ({pnl_pct*100:.1f}%, 保本底线)")

        atr_val = getattr(trade, '_atr_value', None)
        trade_stop_mult = getattr(trade, '_atr_stop_mult', self.atr_stop_mult)

        if atr_val is not None and atr_val > 0:
            profit_pct = (current_price - trade.entry_price) / trade.entry_price

            # ── 使用自定义倍率系数 ──
            if profit_pct > 0.20:
                trailing_mult = trade_stop_mult * tier_mult_factors["t3"]
                tier = f"T3(×{tier_mult_factors['t3']})"
            elif profit_pct > 0.10:
                trailing_mult = trade_stop_mult * tier_mult_factors["t2"]
                tier = f"T2(×{tier_mult_factors['t2']})"
            else:
                trailing_mult = trade_stop_mult * tier_mult_factors["t1"]
                tier = f"T1(×{tier_mult_factors['t1']})"

            stop_dist = atr_val * trade_stop_mult
            trailing_dist = atr_val * trailing_mult

            # ATR 硬止损
            if current_price <= trade.entry_price - stop_dist:
                pnl_pct = (current_price - trade.entry_price) / trade.entry_price
                return self.close_position(symbol, current_date, current_price,
                                           f"ATR硬止损 ({pnl_pct*100:.1f}%, ATR={atr_val:.2f})")

            # ATR 移动止盈
            if trade.highest_price > trade.entry_price:
                if current_price <= trade.highest_price - trailing_dist:
                    drawdown = (current_price - trade.highest_price) / trade.highest_price
                    return self.close_position(symbol, current_date, current_price,
                                               f"ATR移动止盈 ({tier} 最高{trade.highest_price:.2f}, "
                                               f"回撤{drawdown*100:.1f}%, 盈利{profit_pct*100:.1f}%)")
        else:
            # 无 ATR 时回退到固定百分比
            pnl_pct = (current_price - trade.entry_price) / trade.entry_price
            if pnl_pct <= -0.10:
                return self.close_position(symbol, current_date, current_price,
                                           f"硬止损 ({pnl_pct*100:.1f}%)")
            if trade.highest_price > trade.entry_price:
                drawdown = (current_price - trade.highest_price) / trade.highest_price
                if drawdown <= -0.05:
                    return self.close_position(symbol, current_date, current_price,
                                               f"移动止盈 (回撤{drawdown*100:.1f}%)")

        return None

    return check_stop_loss


def run_backtest(tier_mult_factors, label, data_map, benchmark_df,
                 start_date, end_date):
    """运行一轮回测并返回结果"""

    # ── monkey-patch PositionManager.check_stop_loss ──
    original_method = PositionManager.check_stop_loss
    patched = make_patched_check_stop_loss(tier_mult_factors)
    PositionManager.check_stop_loss = types.MethodType(
        lambda self, *args, **kwargs: patched(self, *args, **kwargs),
        PositionManager
    )
    # 实际上需要替换为实例方法, 用更简单的方式:
    PositionManager.check_stop_loss = patched

    try:
        # 重置 GroupConfig 单例 (避免状态残留)
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
            forced_regime=None,
        )

        metrics = engine.run(
            data_map=data_map,
            benchmark_df=benchmark_df,
            start_date=start_date,
            end_date=end_date,
        )

        # 收集交易明细
        trades = engine.position_mgr.closed_trades

        return {
            "label": label,
            "metrics": metrics,
            "trades": trades,
            "daily_values": engine.daily_values,
            "engine": engine,
        }

    finally:
        # 恢复原始方法
        PositionManager.check_stop_loss = original_method


def print_trade_details(trades, label):
    """打印交易明细"""
    print(f"\n  ── {label} 交易明细 ({len(trades)} 笔) ──")
    print(f"  {'股票':<8} {'入场日':<12} {'入场价':>8} {'退出日':<12} {'退出价':>8} {'盈利%':>8} {'持仓':>6} {'退出原因'}")
    print(f"  {'─'*8} {'─'*12} {'─'*8} {'─'*12} {'─'*8} {'─'*8} {'─'*6} {'─'*20}")

    for t in trades:
        pnl_pct = t.pnl_pct * 100 if hasattr(t.pnl_pct, '__float__') else float(t.pnl_pct) * 100
        print(f"  {t.symbol:<8} {str(t.entry_date):<12} {t.entry_price:>8.2f} {str(t.exit_date):<12} {t.exit_price:>8.2f} {pnl_pct:>+7.1f}% {t.holding_days:>5}天 {t.exit_signal[:30]}")


def main():
    print("=" * 100)
    print("  科技成长型分组 — T1档位倍率 A/B 回测对比")
    print("  回测区间: 2026-01-01 ~ 2026-07-13 | 初始资金: 10万 | risk_per_trade: 0.05")
    print("=" * 100)

    # 1. 获取K线数据
    print("\n[1] 获取K线数据...")
    dm = DataManager()

    start_fetch = "2025-06-01"  # 多取半年用于指标预热
    end_fetch = "2026-07-13"

    data_map = {}
    for code in TECH_STOCKS:
        df = dm.get_daily_kline(code, start_date=start_fetch, end_date=end_fetch)
        if df is not None and not df.empty:
            data_map[code] = df
            print(f"  ✓ {code}: {len(df)} 条 ({df.iloc[0]['date']} ~ {df.iloc[-1]['date']})")
        else:
            print(f"  ✗ {code}: 获取失败")

    print(f"\n  共获取 {len(data_map)} 只股票数据")

    # 2. 获取基准数据 (沪深300)
    print("\n[2] 获取基准数据...")
    benchmark_df = dm.get_daily_kline("000300", start_date=start_fetch, end_date=end_fetch)
    if benchmark_df is not None:
        print(f"  ✓ 沪深300: {len(benchmark_df)} 条")
    else:
        print("  ✗ 基准数据获取失败, 使用 None")
        benchmark_df = None

    # 3. A组: 默认参数
    print("\n[3] 运行A组回测 (T1=1.0×, T2=0.8×, T3=0.6×)...")
    default_factors = {"t1": 1.0, "t2": 0.8, "t3": 0.6}
    result_a = run_backtest(
        default_factors, "A组(默认T1=1.0×)",
        data_map, benchmark_df,
        "2026-01-01", "2026-07-13"
    )

    # 4. B组: T1=2.0
    print("\n[4] 运行B组回测 (T1=2.0×, T2=0.8×, T3=0.6×)...")
    adjusted_factors = {"t1": 2.0, "t2": 0.8, "t3": 0.6}
    result_b = run_backtest(
        adjusted_factors, "B组(T1=2.0×)",
        data_map, benchmark_df,
        "2026-01-01", "2026-07-13"
    )

    # 5. 打印交易明细
    print("\n" + "=" * 100)
    print("  交易明细对比")
    print("=" * 100)
    print_trade_details(result_a["trades"], "A组(默认T1=1.0×)")
    print_trade_details(result_b["trades"], "B组(T1=2.0×)")

    # 6. 绩效对比
    print("\n" + "=" * 100)
    print("  绩效对比")
    print("=" * 100)

    ma = result_a["metrics"]
    mb = result_b["metrics"]

    rows = [
        ("总收益率", f"{ma.total_return*100:+.2f}%", f"{mb.total_return*100:+.2f}%",
         f"{(mb.total_return - ma.total_return)*100:+.2f}%"),
        ("年化收益率", f"{ma.annual_return*100:+.2f}%", f"{mb.annual_return*100:+.2f}%",
         f"{(mb.annual_return - ma.annual_return)*100:+.2f}%"),
        ("最大回撤", f"{ma.max_drawdown*100:.2f}%", f"{mb.max_drawdown*100:.2f}%",
         f"{(mb.max_drawdown - ma.max_drawdown)*100:+.2f}%"),
        ("夏普比率", f"{ma.sharpe_ratio:.3f}", f"{mb.sharpe_ratio:.3f}",
         f"{mb.sharpe_ratio - ma.sharpe_ratio:+.3f}"),
        ("总交易次数", f"{ma.total_trades}", f"{mb.total_trades}",
         f"{mb.total_trades - ma.total_trades:+d}"),
        ("盈利交易", f"{ma.winning_trades}", f"{mb.winning_trades}",
         f"{mb.winning_trades - ma.winning_trades:+d}"),
        ("亏损交易", f"{ma.losing_trades}", f"{mb.losing_trades}",
         f"{mb.losing_trades - ma.losing_trades:+d}"),
        ("胜率", f"{ma.win_rate*100:.1f}%", f"{mb.win_rate*100:.1f}%",
         f"{(mb.win_rate - ma.win_rate)*100:+.1f}%"),
        ("平均盈利", f"{ma.avg_profit*100:+.2f}%", f"{mb.avg_profit*100:+.2f}%",
         f"{(mb.avg_profit - ma.avg_profit)*100:+.2f}%"),
        ("平均亏损", f"{ma.avg_loss*100:+.2f}%", f"{mb.avg_loss*100:+.2f}%",
         f"{(mb.avg_loss - ma.avg_loss)*100:+.2f}%"),
        ("盈亏比", f"{ma.profit_factor:.2f}", f"{mb.profit_factor:.2f}",
         f"{mb.profit_factor - ma.profit_factor:+.2f}"),
        ("平均持仓天数", f"{ma.avg_holding_days:.1f}", f"{mb.avg_holding_days:.1f}",
         f"{mb.avg_holding_days - ma.avg_holding_days:+.1f}"),
    ]

    print(f"\n  {'指标':<16} {'A组(T1=1.0×)':>16} {'B组(T1=2.0×)':>16} {'变化':>12}")
    print(f"  {'─'*16} {'─'*16} {'─'*16} {'─'*12}")
    for name, va, vb, diff in rows:
        print(f"  {name:<16} {va:>16} {vb:>16} {diff:>12}")

    # 7. 按退出原因分类统计
    print("\n" + "=" * 100)
    print("  按退出原因分类")
    print("=" * 100)

    for label, result in [("A组", result_a), ("B组", result_b)]:
        print(f"\n  {label}:")
        exit_reasons = {}
        for t in result["trades"]:
            reason = t.exit_signal.split("(")[0].strip()
            if reason not in exit_reasons:
                exit_reasons[reason] = {"count": 0, "pnl_sum": 0.0}
            exit_reasons[reason]["count"] += 1
            pnl_pct = t.pnl_pct * 100 if hasattr(t.pnl_pct, '__float__') else float(t.pnl_pct) * 100
            exit_reasons[reason]["pnl_sum"] += pnl_pct

        print(f"  {'退出原因':<20} {'次数':>6} {'平均盈利':>10} {'总盈利':>10}")
        print(f"  {'─'*20} {'─'*6} {'─'*10} {'─'*10}")
        for reason, stats in sorted(exit_reasons.items(), key=lambda x: -x[1]["count"]):
            avg = stats["pnl_sum"] / stats["count"] if stats["count"] > 0 else 0
            print(f"  {reason:<20} {stats['count']:>6} {avg:>+9.1f}% {stats['pnl_sum']:>+9.1f}%")

    # 8. 逐股票对比
    print("\n" + "=" * 100)
    print("  逐股票对比")
    print("=" * 100)

    stock_a = {}
    for t in result_a["trades"]:
        stock_a.setdefault(t.symbol, []).append(t)
    stock_b = {}
    for t in result_b["trades"]:
        stock_b.setdefault(t.symbol, []).append(t)

    print(f"\n  {'股票':<8} {'A组交易数':>8} {'A组总盈利':>10} │ {'B组交易数':>8} {'B组总盈利':>10} │ {'差异'}")
    print(f"  {'─'*8} {'─'*8} {'─'*10} │ {'─'*8} {'─'*10} │ {'─'*8}")

    all_stocks = sorted(set(list(stock_a.keys()) + list(stock_b.keys())))
    for code in all_stocks:
        ta = stock_a.get(code, [])
        tb = stock_b.get(code, [])
        pnl_a = sum(float(t.pnl_pct) * 100 for t in ta)
        pnl_b = sum(float(t.pnl_pct) * 100 for t in tb)
        diff = pnl_b - pnl_a
        print(f"  {code:<8} {len(ta):>8} {pnl_a:>+9.1f}% │ {len(tb):>8} {pnl_b:>+9.1f}% │ {diff:>+7.1f}%")

    print()


if __name__ == "__main__":
    main()
