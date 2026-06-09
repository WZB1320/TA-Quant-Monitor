"""数据源抽象基类

所有数据源适配器必须实现此接口。
统一输入: 股票代码(str), 开始日期, 结束日期
统一输出: pd.DataFrame 标准列名(date/open/high/low/close/volume/amount)
"""
from abc import ABC, abstractmethod
from typing import Optional
import pandas as pd

# 标准输出列
STANDARD_COLUMNS = ["date", "open", "high", "low", "close", "volume", "amount"]


class DataSource(ABC):
    """数据源抽象基类"""

    name: str = "base"

    @abstractmethod
    def get_daily_kline(
        self,
        symbol: str,
        start_date: str,
        end_date: str,
        adjust: str = "qfq",
    ) -> Optional[pd.DataFrame]:
        """
        获取个股日线K线数据

        Args:
            symbol: 股票代码，各适配器自行解析格式
            start_date: 起始日期 "YYYYMMDD" 或 "YYYY-MM-DD"
            end_date: 结束日期
            adjust: 复权方式 "qfq"/"hfq"/""

        Returns:
            DataFrame with columns: date, open, high, low, close, volume, amount
            失败返回 None
        """
        ...

    def _standardize(self, df: pd.DataFrame) -> pd.DataFrame:
        """将各数据源的列名标准化为 STANDARD_COLUMNS"""
        col_map = {
            "日期": "date", "开盘": "open", "最高": "high",
            "最低": "low", "收盘": "close", "成交量": "volume", "成交额": "amount",
            "open": "open", "high": "high", "low": "low",
            "close": "close", "volume": "volume", "amount": "amount",
        }
        df = df.rename(columns=col_map)
        # 只保留标准列
        available = [c for c in STANDARD_COLUMNS if c in df.columns]
        df = df[available].copy()
        # 统一 dtype
        df["date"] = pd.to_datetime(df["date"]).dt.strftime("%Y-%m-%d")
        for col in ["open", "high", "low", "close", "volume", "amount"]:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")
        return df.sort_values("date").reset_index(drop=True)