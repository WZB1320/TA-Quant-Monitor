"""pytest 共享 fixture"""
import os
import sys

import pytest

# 确保项目根目录在 sys.path
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

# 测试默认在回测模式运行, 避免污染实时数据
from src.config.runtime_mode import set_mode, RuntimeMode
set_mode(RuntimeMode.BACKTEST)


@pytest.fixture
def make_indicator():
    """构造 IndicatorResult 的工厂 fixture"""
    from src.indicators.base import IndicatorResult

    def _make(name, category, direction, strength=0.8, signal="buy",
              description="", **values):
        return IndicatorResult(
            name=name,
            category=category,
            direction=direction,
            signal=signal,
            strength=strength,
            description=description,
            values=values,
        )
    return _make
