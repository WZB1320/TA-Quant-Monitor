"""Pydantic 数据模型 — API 请求/响应结构定义"""
from typing import List, Optional
from pydantic import BaseModel, Field


# ── 自选股 ──

class StockItem(BaseModel):
    """单只股票"""
    code: str = Field(..., pattern=r"^\d{6}$", description="6位股票代码")
    name: str = Field("", description="股票名称")
    market: str = Field("", description="市场: sh/sz/bj")


class StockAddRequest(BaseModel):
    """添加股票请求"""
    code: str = Field(..., pattern=r"^\d{6}$", description="6位股票代码")
    name: str = Field("", description="股票名称")
    market: str = Field("", description="市场: sh/sz/bj, 留空自动判断")
    group: str = Field("", description="添加到指定分组, 留空则添加到 watchlist.json 但不分组")


class StockUpdateRequest(BaseModel):
    """更新股票请求 (换组/改名)"""
    name: Optional[str] = Field(None, description="新名称")
    market: Optional[str] = Field(None, description="新市场")
    group: Optional[str] = Field(None, description="移动到指定分组")


class WatchlistGroup(BaseModel):
    """自选股分组"""
    name: str = Field(..., description="分组名称")
    stocks: List[StockItem] = Field(default_factory=list, description="分组内股票列表")


class WatchlistResponse(BaseModel):
    """自选股完整响应"""
    groups: List[WatchlistGroup] = Field(default_factory=list, description="分组列表")
    ungrouped: List[StockItem] = Field(default_factory=list, description="未分组股票")


class StockSearchResult(BaseModel):
    """股票搜索结果"""
    code: str
    name: str
    market: str


class MessageResponse(BaseModel):
    """通用消息响应"""
    ok: bool = True
    message: str = ""


# ── 分组 ──

class GroupCreateRequest(BaseModel):
    """新建分组请求"""
    name: str = Field(..., min_length=1, max_length=20, description="分组名称")
    description: str = Field("", description="分组描述")
