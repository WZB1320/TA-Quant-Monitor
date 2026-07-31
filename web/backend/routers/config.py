"""策略配置路由 — 按分组/模式读写 strategy_config.json

API:
  GET  /api/config              → 获取完整策略配置（兼容旧接口）
  PUT  /api/config              → 更新完整策略配置（兼容旧接口）
  GET  /api/config/groups       → 返回分组名列表
  GET  /api/config/groups/{name}?mode=base|trending|ranging
                                → 返回分组配置（mode 可选，用于 merge 预设回填）
  PUT  /api/config/groups/{name}
                                → 保存分组基础配置
  PUT  /api/config/groups/{name}/presets/{mode}
                                → 保存某个模式的预设覆盖
"""
import copy
import json
import os
from typing import Optional

from fastapi import APIRouter, Query

router = APIRouter(prefix="/api/config", tags=["配置"])

CONFIG_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))),
    "config", "strategy_config.json",
)

# preset 覆盖涉及的字段（与 manual_regime_presets 对齐）
PRESET_KEYS = [
    "score_threshold", "score_ceiling", "cooldown_days",
    "atr_stop_mult", "max_consecutive_losses", "consecutive_loss_suspend",
    "atr_price_ratio_max",
]


def _load_config() -> dict:
    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def _save_config(data: dict):
    with open(CONFIG_PATH, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def _get_group_config(cfg: dict, name: str) -> dict | None:
    """从完整配置中提取分组配置"""
    gc = cfg.get("strategy_config", {}).get("group_config", {})
    if name == "_default":
        return copy.deepcopy(gc.get("_default", {}))
    return copy.deepcopy(gc.get("groups", {}).get(name))


def _set_group_config(cfg: dict, name: str, data: dict):
    """将分组配置写回完整配置"""
    sc = cfg.setdefault("strategy_config", {})
    gc = sc.setdefault("group_config", {})
    if name == "_default":
        gc["_default"] = data
    else:
        gc.setdefault("groups", {})[name] = data


# ── 兼容旧接口 ──

@router.get("")
def get_full_config():
    return _load_config()


@router.put("")
def update_full_config(data: dict):
    _save_config(data)
    return {"status": "ok"}


# ── 分组列表 ──

@router.get("/groups")
def get_groups():
    cfg = _load_config()
    gc = cfg.get("strategy_config", {}).get("group_config", {})
    groups = list(gc.get("groups", {}).keys())
    return {"groups": groups}


# ── 获取分组配置（支持 mode 合并） ──

@router.get("/groups/{name}")
def get_group_config(name: str, mode: Optional[str] = Query(None)):
    cfg = _load_config()
    group_data = _get_group_config(cfg, name)

    if group_data is None:
        return {"error": f"分组 '{name}' 不存在"}

    presets = group_data.pop("manual_regime_presets", None) or {}

    # mode 不为空时：合并对应 preset 覆盖到基础配置（用于前端回填）
    if mode and mode in presets:
        preset = presets[mode]
        for key in preset:
            if key in group_data:
                group_data[key] = preset[key]

    return {
        "group_name": name,
        "config": group_data,
        "presets": presets,
    }


# ── 保存分组基础配置 ──

@router.put("/groups/{name}")
def save_group_config(name: str, data: dict):
    cfg = _load_config()
    existing = _get_group_config(cfg, name) or {}

    # 保留已有的 presets 不被覆盖
    data["manual_regime_presets"] = existing.get("manual_regime_presets")

    _set_group_config(cfg, name, data)
    _save_config(cfg)
    return {"status": "ok", "group_name": name}


# ── 保存模式预设 ──

@router.put("/groups/{name}/presets/{mode}")
def save_group_preset(name: str, mode: str, data: dict):
    """保存某个模式的预设覆盖（仅提取 PRESET_KEYS 中的字段）"""
    cfg = _load_config()
    group_data = _get_group_config(cfg, name)

    if group_data is None:
        return {"error": f"分组 '{name}' 不存在"}

    # 仅保留 preset 相关字段
    preset_data = {k: v for k, v in data.items() if k in PRESET_KEYS}

    group_data.setdefault("manual_regime_presets", {})[mode] = preset_data
    _set_group_config(cfg, name, group_data)
    _save_config(cfg)

    return {"status": "ok", "group_name": name, "mode": mode, "saved_keys": list(preset_data.keys())}