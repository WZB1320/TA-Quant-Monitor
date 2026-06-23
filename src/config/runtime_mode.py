"""运行时模式 — 区分回测与实时分析

回测模式 (BACKTEST):
  - SignalFilter 不读写磁盘, 仅用内存历史
  - 用户偏好不持久化
  - 保证回测可复现, 不污染实时信号历史

实时模式 (LIVE):
  - SignalFilter 读写 signal_history.json
  - 用户偏好持久化到 user_preferences.json
"""
from enum import Enum


class RuntimeMode(Enum):
    BACKTEST = "backtest"
    LIVE = "live"


# 全局模式, 默认实时 (向后兼容)
_current_mode: RuntimeMode = RuntimeMode.LIVE


def set_mode(mode: RuntimeMode) -> None:
    """设置全局运行时模式"""
    global _current_mode
    _current_mode = mode


def get_mode() -> RuntimeMode:
    """获取当前运行时模式"""
    return _current_mode


def is_backtest() -> bool:
    """当前是否为回测模式"""
    return _current_mode == RuntimeMode.BACKTEST


def is_live() -> bool:
    """当前是否为实时模式"""
    return _current_mode == RuntimeMode.LIVE
