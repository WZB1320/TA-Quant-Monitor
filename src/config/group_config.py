"""分组配置加载器 — 从 strategy_config.json 读取分组参数

用法:
    from src.config.group_config import GroupConfig
    gc = GroupConfig()
    group = gc.get_group("000725")  # "科技成长型"
    weights = gc.get_regime_weights("000725")  # {"trending": {...}, ...}
    stop_mult = gc.get_atr_stop_mult("000725")  # 2.5
    boost = gc.get_max_per_stock_boost("000725")  # 1.2

用户手动体制选择持久化到 data/user_preferences.json,
与策略配置分离, 避免互相污染。服务重启后偏好保留。
"""
import json
import os
from typing import Dict, Optional

from src.config.user_preferences import UserPreferences


class GroupConfig:
    """分组配置 — 单例模式 (策略参数只读, 用户偏好持久化)"""

    _instance = None
    _config = None

    def __new__(cls, user_prefs: UserPreferences = None):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._loaded = False
            cls._instance._user_prefs = user_prefs  # 首次创建时注入
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

        # 手动模式预设 (仅科技成长型)
        self._manual_presets: Dict[str, dict] = {}
        for group_name, cfg in self._groups.items():
            presets = cfg.get("manual_regime_presets")
            if presets:
                self._manual_presets[group_name] = presets

        # 用户手动选择的体制 — 持久化到 UserPreferences (懒初始化)
        if self._user_prefs is None:
            self._user_prefs = UserPreferences()

        self._loaded = True

    def get_group(self, code: str) -> str:
        """获取股票所属分组名"""
        self._load()
        return self._code_to_group.get(code, "_default")

    # ── 手动模式（用户判断方向） ──

    def set_user_regime(self, group_name: str, regime: str):
        """设置用户手动选择的体制模式 (持久化到磁盘)

        Args:
            group_name: 分组名称
            regime: "trending" (趋势上涨) / "ranging" (震荡) / "auto" (自动判断)
        """
        self._load()
        self._user_prefs.set_regime(group_name, regime)

    def set_user_regime_all(self, regime: str):
        """将体制选择应用到所有分组 (未选择分组时使用)

        Args:
            regime: "trending" / "ranging" / "auto"
        """
        self._load()
        group_names = set(self._code_to_group.values())
        for group_name in group_names:
            self._user_prefs.set_regime(group_name, regime)

    def get_user_regime(self, group_name: str) -> str:
        """获取用户手动选择的体制模式 (从磁盘读取)"""
        self._load()
        return self._user_prefs.get_regime(group_name)

    def clear_user_regime(self):
        """清除所有用户手动选择（恢复自动判断）"""
        self._load()
        self._user_prefs.clear_all()

    def _merge_preset(self, code: str, group_config: dict) -> dict:
        """将手动模式预设合并到分组配置中"""
        self._load()
        group_name = self._code_to_group.get(code)
        if not group_name:
            return group_config
        regime = self._user_prefs.get_regime(group_name)
        if not regime or regime == "auto":
            return group_config
        presets = self._manual_presets.get(group_name, {})
        preset = presets.get(regime)
        if not preset:
            return group_config
        # 合并: preset 覆盖 group_config 中对应的键
        merged = dict(group_config)
        merged.update(preset)
        return merged

    # ── 分组配置读取 ──

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
        gc = self._merge_preset(code, gc)
        return gc.get("atr_stop_mult", self._default.get("atr_stop_mult", 2.5))

    def get_max_per_stock_boost(self, code: str) -> float:
        """获取分组专属仓位加成"""
        gc = self._get_group_config(code)
        return gc.get("max_per_stock_boost", self._default.get("max_per_stock_boost", 1.0))

    def get_score_threshold(self, code: str) -> float:
        """获取分组专属最低得分阈值"""
        gc = self._get_group_config(code)
        gc = self._merge_preset(code, gc)
        return gc.get("score_threshold", self._default.get("score_threshold", 25))

    def get_score_ceiling(self, code: str) -> float:
        """获取分组专属得分上限 (超过此值视为过热信号, 0=不限制)"""
        gc = self._get_group_config(code)
        gc = self._merge_preset(code, gc)
        return gc.get("score_ceiling", self._default.get("score_ceiling", 0))

    def get_cooldown_days(self, code: str) -> int:
        """获取分组专属冷却期天数"""
        gc = self._get_group_config(code)
        gc = self._merge_preset(code, gc)
        return gc.get("cooldown_days", self._default.get("cooldown_days", 4))

    def get_consecutive_loss_suspend(self, code: str) -> int:
        """获取连亏暂停天数"""
        gc = self._get_group_config(code)
        gc = self._merge_preset(code, gc)
        return gc.get("consecutive_loss_suspend", self._default.get("consecutive_loss_suspend", 0))

    def get_max_consecutive_losses(self, code: str) -> int:
        """获取连亏触发暂停的阈值"""
        gc = self._get_group_config(code)
        gc = self._merge_preset(code, gc)
        return gc.get("max_consecutive_losses", self._default.get("max_consecutive_losses", 0))

    def get_vol_ratio_threshold(self, code: str) -> float:
        """获取分组专属量比阈值"""
        gc = self._get_group_config(code)
        return gc.get("vol_ratio_threshold", self._default.get("vol_ratio_threshold", 0.6))

    def get_atr_price_ratio_max(self, code: str) -> float:
        """获取ATR/Price最大波动率 (0=不限制)"""
        gc = self._get_group_config(code)
        gc = self._merge_preset(code, gc)
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
        gc = self._merge_preset(code, gc)
        default = self._default
        # 手动模式: 获取 forced_regime (从持久化偏好读取)
        group_name = self._code_to_group.get(code)
        forced_regime = self._user_prefs.get_regime(group_name) if group_name else "auto"
        if forced_regime == "auto":
            forced_regime = None
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
            "forced_regime": forced_regime,  # 手动模式: 强制体制, 覆盖ADX自动检测
        }

    def get_regime_filter_overrides(self, code: str) -> dict:
        """获取分组专属的ADX体制过滤参数覆盖"""
        gc = self._get_group_config(code)
        return gc.get("regime_filter_overrides", {})