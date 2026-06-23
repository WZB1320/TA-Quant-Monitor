"""交易日历构建与日期定位

提取自 BacktestEngine, 提供独立的日历构建和 O(1) 日期定位。
原 BacktestEngine._locate_date 每次线性扫描 df, 回测 N 只股票 × M 天 = O(N×M×L)。
本模块预构建 {date: idx} 映射, 定位降为 O(1)。
"""
from datetime import date
from typing import Dict, List, Optional

import pandas as pd
import numpy as np


class TradingCalendar:
    """交易日历 + 日期索引缓存"""

    def __init__(self, data_map: Dict[str, pd.DataFrame]):
        self._all_dates: List[date] = self._build(data_map)
        # 为每只股票预构建 {date: idx} 映射, O(1) 定位
        self._index_maps: Dict[str, Dict[date, int]] = {
            symbol: self._build_index_map(df)
            for symbol, df in data_map.items()
        }

    @staticmethod
    def _build(data_map: Dict[str, pd.DataFrame]) -> List[date]:
        """构建交易日历 (多只股票的日期并集, 排序)"""
        all_dates = set()
        for df in data_map.values():
            dates = df["date"] if "date" in df.columns else df.index
            for d in dates:
                if isinstance(d, pd.Timestamp):
                    all_dates.add(d.date())
                elif isinstance(d, date):
                    all_dates.add(d)
                elif isinstance(d, str):
                    all_dates.add(pd.Timestamp(d).date())
        return sorted(all_dates)

    @staticmethod
    def _build_index_map(df: pd.DataFrame) -> Dict[date, int]:
        """为单只股票构建 {date: 行索引} 映射"""
        idx_map: Dict[date, int] = {}
        if df.index.name == "date" or isinstance(df.index, pd.DatetimeIndex):
            for i, d in enumerate(df.index):
                if isinstance(d, pd.Timestamp):
                    idx_map[d.date()] = i
                elif isinstance(d, date):
                    idx_map[d] = i
        elif "date" in df.columns:
            for i, d in enumerate(df["date"]):
                if isinstance(d, pd.Timestamp):
                    idx_map[d.date()] = i
                elif isinstance(d, date):
                    idx_map[d] = i
                elif isinstance(d, str):
                    idx_map[pd.Timestamp(d).date()] = i
        return idx_map

    @property
    def all_dates(self) -> List[date]:
        """所有交易日 (升序)"""
        return self._all_dates

    def locate(self, symbol: str, target) -> Optional[int]:
        """O(1) 定位 target 日期在 symbol 数据中的行索引

        Args:
            symbol: 股票代码
            target: date / pd.Timestamp / str

        Returns:
            行索引, 找不到返回 None
        """
        idx_map = self._index_maps.get(symbol)
        if idx_map is None:
            return None
        if isinstance(target, pd.Timestamp):
            target = target.date()
        elif isinstance(target, str):
            target = pd.Timestamp(target).date()
        return idx_map.get(target)

    def get_closing_prices(self, data_map: Dict[str, pd.DataFrame],
                           today) -> Dict[str, float]:
        """获取指定日期所有股票的收盘价"""
        prices = {}
        for symbol, df in data_map.items():
            idx = self.locate(symbol, today)
            if idx is not None:
                try:
                    prices[symbol] = float(df.iloc[idx]["close"])
                except (KeyError, IndexError):
                    pass
        return prices
