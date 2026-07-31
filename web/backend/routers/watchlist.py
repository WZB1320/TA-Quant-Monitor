"""自选股管理路由 — CRUD + 分组管理

数据存储:
  - data/watchlist.json: 简单股票列表 (现有)
  - config/strategy_config.json: 分组股票列表 + 分组策略参数
  - data/stock_knowledge.db: 股票知识库（添加时做匹配校验）

API 同时操作两个文件, 保持数据一致性:
  - watchlist.json 始终是所有股票的平铺列表 (兼容现有脚本)
  - strategy_config.json 的 watchlist 按分组组织 (前端展示用)
"""
import json
import logging
import os
import sys
from typing import List

from fastapi import APIRouter, HTTPException

# 确保项目根目录在 sys.path 中
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from src.config.settings import WATCHLIST_FILE, DATA_DIR
from services.knowledge_base import KnowledgeBase

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/watchlist", tags=["自选股管理"])

# strategy_config.json 路径
_CONFIG_FILE = os.path.join(_PROJECT_ROOT, "config", "strategy_config.json")


# ── 文件读写工具 ──

def _load_watchlist() -> list:
    """读取 watchlist.json 的 stocks 列表"""
    if not os.path.exists(WATCHLIST_FILE):
        return []
    with open(WATCHLIST_FILE, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data.get("stocks", [])


def _save_watchlist(stocks: list):
    """写入 watchlist.json"""
    os.makedirs(os.path.dirname(WATCHLIST_FILE), exist_ok=True)
    with open(WATCHLIST_FILE, "w", encoding="utf-8") as f:
        json.dump({"stocks": stocks}, f, ensure_ascii=False, indent=2)


def _load_config() -> dict:
    """读取 strategy_config.json"""
    if not os.path.exists(_CONFIG_FILE):
        return {"strategy_config": {}}
    with open(_CONFIG_FILE, "r", encoding="utf-8") as f:
        return json.load(f)


def _save_config(config: dict):
    """写入 strategy_config.json"""
    os.makedirs(os.path.dirname(_CONFIG_FILE), exist_ok=True)
    with open(_CONFIG_FILE, "w", encoding="utf-8") as f:
        json.dump(config, f, ensure_ascii=False, indent=2)


def _get_group_watchlist(config: dict) -> dict:
    """从 strategy_config 中获取分组自选股"""
    return config.get("strategy_config", {}).get("watchlist", {})


def _set_group_watchlist(config: dict, groups: dict):
    """更新 strategy_config 中的分组自选股"""
    config.setdefault("strategy_config", {})["watchlist"] = groups


def _guess_market(code: str) -> str:
    """根据代码判断市场"""
    code = code.zfill(6)
    if code.startswith(("6", "9")):
        return "sh"
    elif code.startswith(("4", "8")):
        return "bj"
    return "sz"


def _sync_watchlist_from_groups(config: dict):
    """从分组数据同步到 watchlist.json (保持兼容)

    将 strategy_config.watchlist 中所有分组的股票合并写入 watchlist.json
    """
    groups = _get_group_watchlist(config)
    all_stocks = []
    seen = set()
    for group_name, stocks in groups.items():
        if group_name.startswith("_"):
            continue
        for s in stocks:
            if s["code"] not in seen:
                all_stocks.append({"code": s["code"], "name": s["name"], "market": s["market"]})
                seen.add(s["code"])
    _save_watchlist(all_stocks)


# ── API 路由 ──

@router.get("", summary="获取自选股列表(含分组)")
def get_watchlist():
    """返回分组结构的自选股列表"""
    config = _load_config()
    groups_data = _get_group_watchlist(config)

    groups = []
    ungrouped = []

    for name, stocks in groups_data.items():
        if name.startswith("_"):
            continue
        groups.append({
            "name": name,
            "stocks": stocks,
        })

    # 检查 watchlist.json 中有但分组中没有的股票
    all_stocks = _load_watchlist()
    grouped_codes = set()
    for g in groups:
        for s in g["stocks"]:
            grouped_codes.add(s["code"])

    for s in all_stocks:
        if s["code"] not in grouped_codes:
            ungrouped.append(s)

    return {"groups": groups, "ungrouped": ungrouped}


@router.post("", summary="添加自选股")
def add_stock(req: dict):
    """添加股票到自选股, 可指定分组

    添加前校验股票是否存在于 A 股知识库中。

    Body: {"code": "000001", "name": "平安银行", "market": "sz", "group": "消费稳健型"}
    """
    code = req.get("code", "").strip()
    name = req.get("name", "").strip()
    market = req.get("market", "").strip()
    group = req.get("group", "").strip()

    if not code or len(code) != 6 or not code.isdigit():
        raise HTTPException(status_code=400, detail="股票代码必须为6位数字")

    if not market:
        market = _guess_market(code)

    # ── 知识库匹配校验 ──
    kb = KnowledgeBase()
    stock_info = kb.get_by_code(code)
    if not stock_info:
        raise HTTPException(
            status_code=400,
            detail=f"股票代码 {code} 不存在于 A 股市场，请检查输入"
        )
    # 名称以知识库为准自动纠正
    if name and name != stock_info["name"]:
        logger.info("名称自动纠正: %s → %s", name, stock_info["name"])
        name = stock_info["name"]

    stock = {"code": code, "name": name, "market": market}

    # 1. 添加到 watchlist.json
    stocks = _load_watchlist()
    if any(s["code"] == code for s in stocks):
        raise HTTPException(status_code=409, detail=f"股票 {code} 已在自选股列表中")
    stocks.append(stock)
    _save_watchlist(stocks)

    # 2. 添加到 strategy_config.json 的分组
    config = _load_config()
    groups = _get_group_watchlist(config)
    if group and not group.startswith("_"):
        if group not in groups:
            groups[group] = []
        if not any(s["code"] == code for s in groups[group]):
            groups[group].append(stock)
        _set_group_watchlist(config, groups)
        _save_config(config)

    return {"ok": True, "message": f"已添加 {name or code}"}


@router.delete("/{code}", summary="删除自选股")
def remove_stock(code: str):
    """从自选股列表中删除指定股票 (同时从所有分组中移除)"""
    code = code.strip()

    # 1. 从 watchlist.json 删除
    stocks = _load_watchlist()
    new_stocks = [s for s in stocks if s["code"] != code]
    if len(new_stocks) == len(stocks):
        raise HTTPException(status_code=404, detail=f"股票 {code} 不在自选股列表中")
    _save_watchlist(new_stocks)

    # 2. 从 strategy_config.json 的所有分组中删除
    config = _load_config()
    groups = _get_group_watchlist(config)
    for group_name, group_stocks in groups.items():
        if group_name.startswith("_"):
            continue
        groups[group_name] = [s for s in group_stocks if s["code"] != code]
    _set_group_watchlist(config, groups)
    _save_config(config)

    return {"ok": True, "message": f"已删除 {code}"}


@router.put("/{code}", summary="更新自选股(换组/改名)")
def update_stock(code: str, req: dict):
    """更新股票信息, 支持换组、改名

    Body: {"name": "新名称", "market": "sh", "group": "科技成长型"}
    """
    code = code.strip()
    new_name = req.get("name")
    new_market = req.get("market")
    new_group = req.get("group")

    # 1. 更新 watchlist.json
    stocks = _load_watchlist()
    found = False
    for s in stocks:
        if s["code"] == code:
            found = True
            if new_name is not None:
                s["name"] = new_name
            if new_market is not None:
                s["market"] = new_market
    if not found:
        raise HTTPException(status_code=404, detail=f"股票 {code} 不在自选股列表中")
    _save_watchlist(stocks)

    # 2. 处理换组
    config = _load_config()
    groups = _get_group_watchlist(config)

    if new_group is not None:
        # 从旧分组中移除
        updated_stock = None
        for group_name, group_stocks in groups.items():
            if group_name.startswith("_"):
                continue
            for s in group_stocks:
                if s["code"] == code:
                    updated_stock = s.copy()
                    break
            if updated_stock:
                groups[group_name] = [s for s in group_stocks if s["code"] != code]
                break

        # 添加到新分组
        if updated_stock:
            if new_name is not None:
                updated_stock["name"] = new_name
            if new_market is not None:
                updated_stock["market"] = new_market

            if new_group and not new_group.startswith("_"):
                if new_group not in groups:
                    groups[new_group] = []
                groups[new_group].append(updated_stock)
            # new_group 为空字符串表示移出分组(不加入任何组)

    else:
        # 仅更新名称/市场, 同步到分组数据
        for group_name, group_stocks in groups.items():
            if group_name.startswith("_"):
                continue
            for s in group_stocks:
                if s["code"] == code:
                    if new_name is not None:
                        s["name"] = new_name
                    if new_market is not None:
                        s["market"] = new_market

    _set_group_watchlist(config, groups)
    _save_config(config)

    return {"ok": True, "message": f"已更新 {code}"}


@router.post("/groups", summary="新建分组")
def create_group(req: dict):
    """新建自选股分组

    Body: {"name": "分组名", "description": "描述"}
    """
    name = req.get("name", "").strip()
    description = req.get("description", "").strip()

    if not name:
        raise HTTPException(status_code=400, detail="分组名称不能为空")
    if name.startswith("_"):
        raise HTTPException(status_code=400, detail="分组名称不能以下划线开头")

    config = _load_config()
    groups = _get_group_watchlist(config)

    if name in groups:
        raise HTTPException(status_code=409, detail=f"分组 '{name}' 已存在")

    groups[name] = []
    _set_group_watchlist(config, groups)

    # 同时在 group_config.groups 中创建对应配置
    group_config = config.get("strategy_config", {}).get("group_config", {})
    group_groups = group_config.get("groups", {})
    if name not in group_groups:
        group_groups[name] = {"description": description}
        group_config["groups"] = group_groups
        config["strategy_config"]["group_config"] = group_config

    _save_config(config)

    return {"ok": True, "message": f"已创建分组 '{name}'"}


@router.delete("/groups/{name}", summary="删除分组")
def delete_group(name: str):
    """删除分组 (分组内股票移至未分组, 不删除股票本身)"""
    config = _load_config()
    groups = _get_group_watchlist(config)

    if name not in groups:
        raise HTTPException(status_code=404, detail=f"分组 '{name}' 不存在")

    del groups[name]
    _set_group_watchlist(config, groups)

    # 同时删除 group_config 中的对应配置
    group_config = config.get("strategy_config", {}).get("group_config", {})
    group_groups = group_config.get("groups", {})
    if name in group_groups:
        del group_groups[name]
        group_config["groups"] = group_groups
        config["strategy_config"]["group_config"] = group_config

    _save_config(config)

    return {"ok": True, "message": f"已删除分组 '{name}'"}
