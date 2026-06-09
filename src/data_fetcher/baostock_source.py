"""BaoStock 数据源适配器"""
from typing import Optional
import pandas as pd

from .base import DataSource


class BaoStockSource(DataSource):
    """BaoStock 数据源"""

    name = "baostock"

    def get_daily_kline(
        self,
        symbol: str,
        start_date: str,
        end_date: str,
        adjust: str = "qfq",
    ) -> Optional[pd.DataFrame]:
        import baostock as bs

        code = self._format_symbol(symbol)

        # 复权参数: 1=前复权, 2=后复权, 3=不复权
        adjust_map = {"qfq": "1", "hfq": "2", "": "3", None: "3"}
        adj = adjust_map.get(adjust, "1")

        lg = bs.login()
        if lg.error_code != "0":
            return None

        try:
            rs = bs.query_history_k_data_plus(
                code,
                "date,open,high,low,close,volume,amount",
                start_date=start_date.replace("-", "-"),
                end_date=end_date.replace("-", "-"),
                frequency="d",
                adjustflag=adj,
            )
            if rs.error_code != "0":
                return None

            data = []
            while rs.next():
                data.append(rs.get_row_data())

            if not data:
                return None

            df = pd.DataFrame(data, columns=rs.fields)
        except Exception:
            return None
        finally:
            bs.logout()

        df = self._standardize(df)
        return df

    @staticmethod
    def _format_symbol(symbol: str) -> str:
        """将纯数字代码转为 BaoStock 需要的格式 sh.600519"""
        symbol = symbol.strip()
        if "." in symbol:
            return symbol
        if symbol.startswith(("sh.", "sz.", "bj.")):
            return symbol
        code = symbol.zfill(6)
        if code.startswith(("6", "9")):
            return f"sh.{code}"
        elif code.startswith(("4", "8")):
            return f"bj.{code}"
        else:
            return f"sz.{code}"