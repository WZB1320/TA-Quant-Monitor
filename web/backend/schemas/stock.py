"""股票知识库相关 Pydantic 模型"""
from typing import List, Optional
from pydantic import BaseModel, Field


class StockSearchItem(BaseModel):
    """单条搜索结果"""
    code: str = Field(..., description="6位股票代码")
    name: str = Field(..., description="股票名称")
    market: str = Field(..., description="市场: sh/sz/bj")


class StockSearchResponse(BaseModel):
    """搜索响应"""
    results: List[StockSearchItem] = Field(default_factory=list, description="搜索结果列表")


class StockDetailResponse(BaseModel):
    """股票详情"""
    code: str
    name: str
    market: str
    py_initials: Optional[str] = None
    full_py: Optional[str] = None
    industry: Optional[str] = None


class SyncResponse(BaseModel):
    """同步响应"""
    ok: bool = True
    synced: int = Field(0, description="同步的股票数量")
    message: str = Field("", description="附加消息")