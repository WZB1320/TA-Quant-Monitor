"""分组配置加载器 — 从 strategy_config.json 读取分组参数

用法:
    from src.config.group_config import GroupConfig
    gc = GroupConfig()
    group = gc.get_group("000725")  # "科技成长型"
    weights = gc.get_regime_weights("000725")  # {"trending": {...}, ...}
    stop_mult = gc.get_atr_stop_mult("000725")  # 2.5
    boost = gc.get_max_per_stock_boost("000725")  # 1.2
"""
import json
import os
from typing import Dict, Optional


class GroupConfig:
    """分组配置 — 单例模式"""

    _instance = None
    _config = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._loaded = False
        return cls._instance

    def _load(self):
        if self._loaded:
            return
        config_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
            "config", "strategy_config.json"
        )
        try:
            with open(config_path, "r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception:
            self._loaded = True
            return

        sc = data.get("strategy_config", {})
        gc = sc.get("group_config", {})
        self._default = gc.get("_default", {})
        self._groups = gc.get("groups", {})

        # 构建 code → group_name 映射 (从 watchlist)
        wl = sc.get("watchlist", {})
        self._code_to_group: Dict[str, str] = {}
        for group_name, stocks in wl.items():
            if group_name.startswith("_"):
                continue
            for s in stocks:
                self._code_to_group[s["code"]] = group_name

        self._loaded = True

    def get_group(self, code: str) -> str:
        """获取股票所属分组名"""
        self._load()
        return self._code_to_group.get(code, "_default")

    def _get_group_config(self, code: str) -> dict:
        """获取股票所属分组的完整配置"""
        self._load()
        group_name = self._code_to_group.get(code)
        if group_name and group_name in self._groups:
            return self._groups[group_name]
        return self._default

    def get_regime_weights(self, code: str) -> dict:
        """获取分组专属体制权重"""
        gc = self._get_group_config(code)
        return gc.get("regime_weights", self._default.get("regime_weights", {}))

    def get_atr_stop_mult(self, code: str) -> float:
        """获取分组专属ATR止损倍率"""
        gc = self._get_group_config(code)
        return gc.get("atr_stop_mult", self._default.get("atr_stop_mult", 2.5))

    def get_max_per_stock_boost(self, code: str) -> float:
        """获取分组专属仓位加成"""
        gc = self._get_group_config(code)
        return gc.get("max_per_stock_boost", self._default.get("max_per_stock_boost", 1.0))

    def get_score_threshold(self, code: str) -> float:
        """获取分组专属最低得分阈值"""
        gc = self._get_group_config(code)
        return gc.get("score_threshold", self._default.get("score_threshold", 25))

    def get_score_ceiling(self, code: str) -> float:
        """获取分组专属得分上限 (超过此值视为过热信号, 0=不限制)"""
        gc = self._get_group_config(code)
        return gc.get("score_ceiling", self._default.get("score_ceiling", 0))

    def get_cooldown_days(self, code: str) -> int:
        """获取分组专属冷却期天数"""
        gc = self._get_group_config(code)
        return gc.get("cooldown_days", self._default.get("cooldown_days", 4))

    def get_consecutive_loss_suspend(self, code: str) -> int:
        """获取连亏暂停天数"""
        gc = self._get_group_config(code)
        return gc.get("consecutive_loss_suspend", self._default.get("consecutive_loss_suspend", 0))

    def get_max_consecutive_losses(self, code: str) -> int:
        """获取连亏触发暂停的阈值"""
        gc = self._get_group_config(code)
        return gc.get("max_consecutive_losses", self._default.get("max_consecutive_losses", 0))

    def get_vol_ratio_threshold(self, code: str) -> float:
        """获取分组专属量比阈值"""
        gc = self._get_group_config(code)
        return gc.get("vol_ratio_threshold", self._default.get("vol_ratio_threshold", 0.6))

    def get_atr_price_ratio_max(self, code: str) -> float:
        """获取ATR/Price最大波动率 (0=不限制)"""
        gc = self._get_group_config(code)
        return gc.get("atr_price_ratio_max", self._default.get("atr_price_ratio_max", 0))

    def get_require_macd_dif_above_zero(self, code: str) -> bool:
        """是否要求MACD DIF在零轴上方"""
        gc = self._get_group_config(code)
        return gc.get("require_macd_dif_above_zero", self._default.get("require_macd_dif_above_zero", False))

    def get_price_ma20_max_deviation(self, code: str) -> float:
        """获取价格偏离MA20最大比例 (0=不限制)"""
        gc = self._get_group_config(code)
        return gc.get("price_ma20_max_deviation", self._default.get("price_ma20_max_deviation", 0))

    def get_rsi_overbought(self, code: str) -> float:
        """获取RSI超买阈值 (0=不限制)"""
        gc = self._get_group_config(code)
        return gc.get("rsi_overbought", self._default.get("rsi_overbought", 0))

    def get_all_group_params(self, code: str) -> dict:
        """获取指定股票的所有分组参数 (一次性返回, 避免多次调用)"""
        gc = self._get_group_config(code)
        default = self._default
        return {
            "score_threshold": gc.get("score_threshold", default.get("score_threshold", 25)),
            "score_ceiling": gc.get("score_ceiling", default.get("score_ceiling", 0)),
            "cooldown_days": gc.get("cooldown_days", default.get("cooldown_days", 4)),
            "consecutive_loss_suspend": gc.get("consecutive_loss_suspend", default.get("consecutive_loss_suspend", 0)),
            "max_consecutive_losses": gc.get("max_consecutive_losses", default.get("max_consecutive_losses", 0)),
            "vol_ratio_threshold": gc.get("vol_ratio_threshold", default.get("vol_ratio_threshold", 0.6)),
            "atr_price_ratio_max": gc.get("atr_price_ratio_max", default.get("atr_price_ratio_max", 0)),
            "require_macd_dif_above_zero": gc.get("require_macd_dif_above_zero", default.get("require_macd_dif_above_zero", False)),
            "price_ma20_max_deviation": gc.get("price_ma20_max_deviation", default.get("price_ma20_max_deviation", 0)),
            "rsi_overbought": gc.get("rsi_overbought", default.get("rsi_overbought", 0)),
            "atr_stop_mult": gc.get("atr_stop_mult", default.get("atr_stop_mult", 2.5)),
            "max_per_stock_boost": gc.get("max_per_stock_boost", default.get("max_per_stock_boost", 1.0)),
            "regime_weights": gc.get("regime_weights", default.get("regime_weights", {})),
            "regime_filter_overrides": gc.get("regime_filter_overrides", {}),
            "breakout_ceiling_bonus": gc.get("breakout_ceiling_bonus", default.get("breakout_ceiling_bonus", 0)),
            "ma_alignment_ceiling_bonus": gc.get("ma_alignment_ceiling_bonus", default.get("ma_alignment_ceiling_bonus", 0)),
            "indicator_params": gc.get("indicator_params", {}),
            "indicator_weights": gc.get("indicator_weights", None),
            "strength_modifiers": gc.get("strength_modifiers", {}),
        }

    def get_regime_filter_overrides(self, code: str) -> dict:
        """获取分组专属的ADX体制过滤参数覆盖"""
        gc = self._get_group_config(code)
        return gc.get("regime_filter_overrides", {})