"""策略自进化层 — 教训驱动的参数变异与选择 (P8)"""
from .auto_applier import EvolutionAB, build_candidate, compare_runs

__all__ = ["EvolutionAB", "build_candidate", "compare_runs"]
