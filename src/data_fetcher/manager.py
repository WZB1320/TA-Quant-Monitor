"""
数据管理器 (DataManager)

核心职责:
1. 多数据源冗余: 按优先级尝试，主源失败自动切备用源
2. 本地缓存: 先查缓存，缓存未命中再请求远程，写入缓存
3. 统一接口: 对外提供 get_daily_kline()，屏蔽底层复杂度

数据流:
    get_daily_kline(symbol)
        → 检查缓存 → 命中? → 返回
        → 未命中 → 按优先级尝试远程数据源
                  → 成功 → 写入缓存 → 返回
                  → 全部失败 → 读缓存(即使过期) → 兜底返回
"""
from typing import Optional
from datetime import datetime, timedelta
import pandas as pd

from .base import DataSource
from .akshare_source import AKShareSource
from .baostock_source import BaoStockSource
from .cache import KLineCache
from src.config.settings import DATA_SOURCE_ORDER, DEFAULT_LOOKBACK_DAYS


class DataManager:
    """统一数据管理器"""

    def __init__(self):
        self.cache = KLineCache()
        self._sources: dict[str, DataSource] = {
            "akshare": AKShareSource(),
            "baostock": BaoStockSource(),
        }
        # 按配置的优先级排序
        self._source_order = [
            name for name in DATA_SOURCE_ORDER if name in self._sources
        ]

    @staticmethod
    def _expected_last_trade_date(end_dt: datetime, now: datetime = None) -> datetime:
        """返回 <= end_date 的最后一个预期交易日 (工作日近似).

        规则:
        - end_date 超过当前时间时截断到当前 (未来无数据)
        - 当天 16:00 前收盘数据尚未发布, 预期回退一天
        - 周末回退到周五

        注: 法定节假日按工作日近似 — 节假日当天缓存会被判定过期并多一次远程拉取
        (源端无新数据, 拉取后原样写回), 只影响请求量, 不影响正确性.
        """
        now = now or datetime.now()
        d = min(end_dt, now)
        # 当日收盘数据 ~16:00 后才可用
        if d.date() == now.date() and now.hour < 16:
            d -= timedelta(days=1)
        while d.weekday() >= 5:  # 5=周六, 6=周日
            d -= timedelta(days=1)
        # 归一化到零点: 与 strptime 解析的缓存日期 (零点) 做纯日期比较,
        # 避免 min() 带入 now 的时刻分量导致同日误判过期
        return d.replace(hour=0, minute=0, second=0, microsecond=0)

    def get_daily_kline(
        self,
        symbol: str,
        start_date: str = None,
        end_date: str = None,
        adjust: str = "qfq",
    ) -> Optional[pd.DataFrame]:
        """
        获取个股日线K线数据 (多源冗余 + 缓存)

        Args:
            symbol: 股票代码，如 "600519" / "000001"
            start_date: 起始日期 "YYYY-MM-DD"，默认一年前
            end_date: 结束日期 "YYYY-MM-DD"，默认今天
            adjust: 复权方式

        Returns:
            标准化的日线 DataFrame，失败返回 None
        """
        if end_date is None:
            end_date = datetime.now().strftime("%Y-%m-%d")
        if start_date is None:
            start_dt = datetime.now() - timedelta(days=DEFAULT_LOOKBACK_DAYS)
            start_date = start_dt.strftime("%Y-%m-%d")

        # Step 1: 查缓存
        cached = self.cache.load(symbol, start_date, end_date)
        if cached is not None and not cached.empty:
            # 缓存新鲜度: 末条日期须覆盖 <= end_date 的最后一个预期交易日.
            # 旧规则 (end - last).days <= 2 在收盘后当天分析时, 昨日缓存间隔仅1天
            # 即命中, 当日K线永远拉不下来, 实时信号系统性滞后一个交易日;
            # 回测场景下同样会漏掉窗口末尾 1~2 个交易日的数据.
            last_cached = cached["date"].max()
            end_dt = datetime.strptime(end_date, "%Y-%m-%d")
            last_dt = datetime.strptime(last_cached, "%Y-%m-%d")
            expected_last = self._expected_last_trade_date(end_dt)
            if last_dt >= expected_last:
                return cached
            # 缓存落后于预期最新交易日, 继续远程拉取

        # Step 2: 按优先级尝试远程数据源
        result = None
        errors = []
        for source_name in self._source_order:
            source = self._sources[source_name]
            try:
                result = source.get_daily_kline(symbol, start_date, end_date, adjust)
                if result is not None and not result.empty:
                    # 写入缓存
                    self.cache.save(symbol, result)
                    return result
                errors.append(f"{source_name}: returned empty")
            except Exception as e:
                errors.append(f"{source_name}: {type(e).__name__}({e})")

        # Step 3: 全部远程失败，用本地缓存兜底
        if cached is not None and not cached.empty:
            return cached

        return None