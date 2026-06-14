"""全局配置"""
import os

# 项目根目录
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA_DIR = os.path.join(ROOT_DIR, "data")

# 自选股列表文件
WATCHLIST_FILE = os.path.join(DATA_DIR, "watchlist.json")

# SQLite 缓存数据库
CACHE_DB = os.path.join(DATA_DIR, "kline_cache.db")

# SQLite 股票知识库
STOCK_KB_DB = os.path.join(DATA_DIR, "stock_knowledge.db")

# 默认获取日线数据的回溯天数
DEFAULT_LOOKBACK_DAYS = 365

# 数据源优先级: 越靠前越优先
DATA_SOURCE_ORDER = ["akshare", "baostock"]