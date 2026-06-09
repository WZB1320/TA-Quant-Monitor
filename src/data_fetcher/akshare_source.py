"""AKShare 数据源适配器

使用 AKShare 的 stock_zh_a_daily 接口（新浪数据源）
"""
from typing import Optional
import pandas as pd

from .base import DataSource


class AKShareSource(DataSource):
    """AKShare 数据源（新浪源）"""

    name = "akshare"

    def get_daily_kline(
        self,
        symbol: str,
        start_date: str,
        end_date: str,
        adjust: str = "qfq",
    ) -> Optional[pd.DataFrame]:
        import akshare as ak

        # stock_zh_a_daily 需要 "sh600519" 或 "sz000001" 格式
        code = self._format_symbol(symbol)

        try:
            df = ak.stock_zh_a_daily(
                symbol=code,
                start_date=start_date.replace("-", ""),
                end_date=end_date.replace("-", ""),
                adjust=adjust,
            )
        except Exception:
            return None

        if df is None or df.empty:
            return None

        df = self._standardize(df)
        return df

    @staticmethod
    def _format_symbol(symbol: str) -> str:
        """将纯数字代码转为 AKShare 需要的格式"""
        symbol = symbol.strip()
        if symbol.startswith(("sh", "sz", "bj")):
            return symbol
        code = symbol.zfill(6)
        if code.startswith(("6", "9")):
            return f"sh{code}"
        elif code.startswith(("4", "8")):
            return f"bj{code}"
        else:
            return f"sz{code}"