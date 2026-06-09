"""
信号结果数据结构 + 信号级别定义

信号级别 (由强到弱):
  STRONG_BUY  → 三重共振：趋势+动量+量价 同时看多
  BUY         → 至少两类指标看多
  WEAK_BUY    → 单类指标看多，可关注
  NEUTRAL     → 指标矛盾或无信号
  WEAK_SELL   → 单类指标看空
  SELL        → 至少两类指标看空
  STRONG_SELL → 三重共振看空
"""
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List
from src.indicators.base import IndicatorResult


class SignalLevel(Enum):
    STRONG_BUY = 3
    BUY = 2
    WEAK_BUY = 1
    NEUTRAL = 0
    WEAK_SELL = -1
    SELL = -2
    STRONG_SELL = -3

    @property
    def label(self) -> str:
        labels = {
            SignalLevel.STRONG_BUY: "强买入",
            SignalLevel.BUY: "买入",
            SignalLevel.WEAK_BUY: "关注(偏多)",
            SignalLevel.NEUTRAL: "观望",
            SignalLevel.WEAK_SELL: "注意(偏空)",
            SignalLevel.SELL: "卖出",
            SignalLevel.STRONG_SELL: "强卖出",
        }
        return labels[self]

    @property
    def is_bullish(self) -> bool:
        return self.value > 0

    @property
    def is_bearish(self) -> bool:
        return self.value < 0

    @property
    def is_actionable(self) -> bool:
        """是否需要操作 (买入/卖出级别以上)"""
        return abs(self.value) >= 2


@dataclass
class CategorySummary:
    """单个类别汇总"""
    category: str                         # trend/strength/momentum/volume
    direction: int                        # 类别整体方向
    consensus: int                        # 看多指标数
    dissensus: int                        # 看空指标数
    indicators: Dict[str, IndicatorResult] = field(default_factory=dict)


@dataclass
class SignalResult:
    """最终信号"""
    symbol: str                           # 股票代码
    level: SignalLevel                    # 信号级别
    score: float                          # 综合得分 (-100 ~ +100)
    confidence: float                     # 置信度 0~1
    reason: str                           # 一句话总结
    details: str                          # 详细分析
    category_summary: Dict[str, CategorySummary] = field(default_factory=dict)
    hard_filter_blocked: bool = False     # 是否被硬过滤拦截
    block_reason: str = ""                # 拦截原因

    def __repr__(self):
        return (f"<{self.symbol} {self.level.label} "
                f"score={self.score:+.1f} conf={self.confidence:.1%} | {self.reason}>")