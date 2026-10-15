"""BaoStock 数据源适配器"""
from typing import Optional
import pandas as pd

from .base import DataSource


class BaoStockSource(DataSource):
    """BaoStock 数据源"""

    name = "baostock"

    # 复权参数 (baostock 官方口径, 2026-10-14 实测校准)
    #   "1" = 后复权, "2" = 前复权, "3" = 不复权
    # 实测证据 (sh.600887, 收盘价 2024-01-02 / 2026-09-28):
    #   "3" → 26.27    / 27.03       首末均为真实价              → 不复权
    #   "2" → 22.95    / 27.03       末日=真实价, 首日被压低      → 前复权
    #   "1" → 1961.73  / 2310.84     末日>真实价, 首日被放大      → 后复权
    #
    # 修正前此处为 {"qfq": "1", "hfq": "2"} (写反). 因 DATA_SOURCE_ORDER 中
    # akshare 优先而未被触发; 但 akshare 失败由 baostock 接管时, 会把后复权
    # 数据写入标记为 qfq 的缓存, 静默污染后续全部回测.
    ADJUST_MAP = {"qfq": "2", "hfq": "1", "": "3", None: "3"}
    DEFAULT_ADJUST = "2"  # 未知取值回退到前复权, 与 DataManager 默认值一致

    def get_daily_kline(
        self,
        symbol: str,
        start_date: str,
        end_date: str,
        adjust: str = "qfq",
    ) -> Optional[pd.DataFrame]:
        import baostock as bs

        code = self._format_symbol(symbol)

        adj = self.ADJUST_MAP.get(adjust, self.DEFAULT_ADJUST)

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