"""自选股列表管理

JSON 文件存储，格式:
{
    "stocks": [
        {"code": "600519", "name": "贵州茅台", "market": "sh"},
        {"code": "000001", "name": "平安银行", "market": "sz"}
    ]
}
"""
import json
import os
from typing import List, Optional

from src.config.settings import WATCHLIST_FILE


# 默认自选股
_DEFAULT_WATCHLIST = [
    {"code": "600519", "name": "贵州茅台", "market": "sh"},
    {"code": "000001", "name": "平安银行", "market": "sz"},
    {"code": "000858", "name": "五粮液", "market": "sz"},
    {"code": "600036", "name": "招商银行", "market": "sh"},
    {"code": "300750", "name": "宁德时代", "market": "sz"},
    {"code": "601318", "name": "中国平安", "market": "sh"},
    {"code": "002475", "name": "立讯精密", "market": "sz"},
    {"code": "300059", "name": "东方财富", "market": "sz"},
    {"code": "688981", "name": "中芯国际", "market": "sh"},
]


class Watchlist:
    """自选股管理"""

    def __init__(self, file_path: str = WATCHLIST_FILE):
        self.file_path = file_path
        self._ensure_file()

    def _ensure_file(self):
        """确保文件存在，不存在则创建默认"""
        os.makedirs(os.path.dirname(self.file_path), exist_ok=True)
        if not os.path.exists(self.file_path):
            self._save(_DEFAULT_WATCHLIST)

    def _load(self) -> list:
        with open(self.file_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data.get("stocks", [])

    def _save(self, stocks: list):
        with open(self.file_path, "w", encoding="utf-8") as f:
            json.dump({"stocks": stocks}, f, ensure_ascii=False, indent=2)

    def get_all(self) -> List[dict]:
        """获取全部自选股"""
        return self._load()

    def get_codes(self) -> List[str]:
        """只获取代码列表"""
        return [s["code"] for s in self._load()]

    def add(self, code: str, name: str = "", market: str = ""):
        """添加自选股"""
        stocks = self._load()
        if any(s["code"] == code for s in stocks):
            return
        # 自动判断市场
        if not market:
            market = self._guess_market(code)
        stocks.append({"code": code, "name": name, "market": market})
        self._save(stocks)

    def remove(self, code: str):
        """移除自选股"""
        stocks = self._load()
        stocks = [s for s in stocks if s["code"] != code]
        self._save(stocks)

    @staticmethod
    def _guess_market(code: str) -> str:
        code = code.zfill(6)
        if code.startswith(("6", "9")):
            return "sh"
        elif code.startswith(("4", "8")):
            return "bj"
        else:
            return "sz"