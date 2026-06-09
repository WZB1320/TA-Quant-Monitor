"""
指标基类 + 统一输出结构

每个指标都是独立的插件，遵循统一接口:
  输入: pd.DataFrame (标准化日线数据)
  输出: IndicatorResult
"""
from dataclasses import dataclass, field
from typing import Optional
import pandas as pd


@dataclass
class IndicatorResult:
    """指标计算结果"""
    name: str                              # 指标名称
    category: str                          # 类别: trend/strength/momentum/volume
    direction: int                         # +1(看多) / 0(中性) / -1(看空)
    signal: str                            # 文字信号: buy/sell/neutral
    strength: float                        # 信号强度 0~1
    description: str                       # 可读描述
    values: dict = field(default_factory=dict)  # 原始计算值，供调试/展示


class BaseIndicator:
    """指标基类 — 插件式接口"""

    name: str = "base"
    category: str = "base"

    def calculate(self, df: pd.DataFrame) -> IndicatorResult:
        raise NotImplementedError

    def __call__(self, df: pd.DataFrame) -> IndicatorResult:
        return self.calculate(df)

    def _make_result(self, direction: int, signal: str, strength: float,
                     description: str, **values) -> IndicatorResult:
        return IndicatorResult(
            name=self.name,
            category=self.category,
            direction=direction,
            signal=signal,
            strength=min(max(strength, 0.0), 1.0),
            description=description,
            values=values,
        )