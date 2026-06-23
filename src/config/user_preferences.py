"""用户运行时偏好持久化

存储用户在运行时设置的非策略参数 (如手动体制选择),
与 strategy_config.json (策略参数) 分离, 避免互相污染。

存储文件: data/user_preferences.json
"""
import json
import logging
import os
from typing import Dict

from src.config.settings import DATA_DIR

logger = logging.getLogger(__name__)

_USER_PREF_FILE = os.path.join(DATA_DIR, "user_preferences.json")


class UserPreferences:
    """用户运行时偏好 (非策略参数)"""

    def __init__(self, file_path: str = None):
        self._file_path = file_path or _USER_PREF_FILE
        self._prefs: Dict[str, str] = self._load()

    def _load(self) -> Dict[str, str]:
        """从磁盘加载偏好"""
        try:
            if os.path.exists(self._file_path):
                with open(self._file_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    return data.get("regime_overrides", {})
        except Exception as e:
            logger.warning("加载用户偏好失败: %s", e)
        return {}

    def _save(self) -> None:
        """持久化到磁盘"""
        try:
            os.makedirs(os.path.dirname(self._file_path), exist_ok=True)
            with open(self._file_path, "w", encoding="utf-8") as f:
                json.dump({"regime_overrides": self._prefs}, f,
                          ensure_ascii=False, indent=2)
        except Exception as e:
            logger.warning("保存用户偏好失败: %s", e)

    def set_regime(self, group_name: str, regime: str) -> None:
        """设置分组的手动体制选择

        Args:
            group_name: 分组名称
            regime: "trending" / "ranging" / "auto" (auto 表示清除)
        """
        if regime in ("trending", "ranging"):
            self._prefs[group_name] = regime
        else:
            # "auto" 或无效值 → 清除该分组的覆盖
            self._prefs.pop(group_name, None)
        self._save()

    def get_regime(self, group_name: str) -> str:
        """获取分组的手动体制选择, 默认 "auto" """
        return self._prefs.get(group_name, "auto")

    def clear_all(self) -> None:
        """清除所有用户偏好"""
        self._prefs.clear()
        self._save()

    def as_dict(self) -> Dict[str, str]:
        """返回当前偏好的副本"""
        return dict(self._prefs)
