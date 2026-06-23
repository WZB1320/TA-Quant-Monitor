"""
回测模块 — 验证信号引擎在历史数据上的实际收益表现
"""
from .engine import BacktestEngine
from .position import PositionManager, Trade, Side
from .broker import Broker
from .metrics import BacktestMetrics, compute_metrics
from .report import generate_report, generate_summary
from .calendar import TradingCalendar
from .regime_detector import RegimeDetector
from .market_filter import MarketFilter
from .signal_executor import SignalExecutor

__all__ = [
    "BacktestEngine",
    "PositionManager", "Trade", "Side",
    "Broker",
    "BacktestMetrics", "compute_metrics",
    "generate_report", "generate_summary",
    "TradingCalendar",
    "RegimeDetector",
    "MarketFilter",
    "SignalExecutor",
]