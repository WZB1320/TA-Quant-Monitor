"""
虚拟券商 — 模拟订单执行

职责:
1. T+1 模拟: 信号当日产生, 次日以开盘价成交
2. 手续费: 买入/卖出各收佣金, A股最低5元
3. 滑点: 买入价上浮, 卖出价下浮
4. 涨跌停无法成交: 开盘即涨停/跌停视为无法交易

A股交易规则:
- T+1: 当日买入次日才能卖出
- 手续费: 佣金万2.5 + 印花税(卖出千1)
- 涨跌停: 主板10%, 科创/创业20%
"""
from datetime import date
from typing import Dict, Optional
import pandas as pd


class Broker:
    """虚拟券商"""

    def __init__(self, commission_rate: float = 0.00025,
                 stamp_tax: float = 0.001, slippage: float = 0.0001,
                 min_commission: float = 5.0):
        """
        Args:
            commission_rate: 佣金率 (默认万2.5)
            stamp_tax: 印花税率 (卖出千1, 仅卖出收取)
            slippage: 滑点率 (默认万一)
            min_commission: 最低佣金 (默认5元)
        """
        self.commission_rate = commission_rate
        self.stamp_tax = stamp_tax
        self.slippage = slippage
        self.min_commission = min_commission

    # ── 成交价计算 ──

    def buy_price(self, open_price: float) -> float:
        """计算买入成交价 = 开盘价 + 滑点"""
        return open_price * (1 + self.slippage)

    def sell_price(self, open_price: float) -> float:
        """计算卖出成交价 = 开盘价 - 滑点"""
        return open_price * (1 - self.slippage)

    # ── 手续费计算 ──

    def buy_commission(self, amount: float) -> float:
        """买入手续费 (仅佣金)"""
        return max(amount * self.commission_rate, self.min_commission)

    def sell_commission(self, amount: float) -> float:
        """卖出手续费 (佣金 + 印花税)"""
        comm = max(amount * self.commission_rate, self.min_commission)
        stamp = amount * self.stamp_tax
        return comm + stamp

    # ── 涨跌停检查 ──

    def can_trade(self, open_price: float, prev_close: float,
                  limit_pct: float = 0.10) -> bool:
        """
        检查是否能成交 (非一字涨跌停)

        Args:
            open_price: 当日开盘价
            prev_close: 前日收盘价
            limit_pct: 涨跌停幅度 (默认10%)

        Returns:
            True 可交易, False 一字板无法成交
        """
        limit_up = prev_close * (1 + limit_pct)
        limit_down = prev_close * (1 - limit_pct)
        # 一字板 (开盘即涨停或跌停)
        if open_price >= limit_up * 0.999 or open_price <= limit_down * 1.001:
            return False
        return True

    # ── 获取次日开盘价 ──

    def get_next_open(self, df: pd.DataFrame, signal_date_idx: int
                      ) -> Optional[dict]:
        """
        获取信号次日开盘价信息 (T+1 执行)

        Args:
            df: 完整K线 DataFrame (必须含 open, close 列, date 索引)
            signal_date_idx: 信号日期在 df 中的索引位置

        Returns:
            {"date": date, "open": float, "prev_close": float} 或 None
        """
        next_idx = signal_date_idx + 1
        if next_idx >= len(df):
            return None  # 已是最后一天, 无次日数据

        row = df.iloc[next_idx]
        prev_row = df.iloc[signal_date_idx]
        return {
            "date": row.name if hasattr(row.name, 'date') else row.get("date"),
            "open": float(row["open"]),
            "prev_close": float(prev_row["close"]),
        }