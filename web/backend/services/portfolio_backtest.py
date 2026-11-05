"""组合回测编排服务 — P5 分组独立回测 + 权重合并

从 routers/backtest.py 抽取而来（2026-09-29，**纯移动，零逻辑改动**）。
抽取目的：让 HTTP 路由变薄壳，回测编排可被其他调用方复用（收盘复盘大屏 builder）。

调用契约：
    run_portfolio_backtest(req, on_progress=None) -> dict
    - req: BacktestRequest（分组展示、mode、起止日期、本金、基准）
    - on_progress(progress:int, text:str) 可选进度回调，仅影响 UI 状态，不影响回测结果
    - 返回: {"status": "done"|"error", ...}，与重构前路由返回体完全一致

运行时模式：函数内部自行切换 BACKTEST 模式并在 finally 恢复，
           SignalFilter 走纯内存历史，不读写 signal_history.json。

回归闸口：与重构前同参数逐日比对 daily_values 必须完全一致。
"""
import copy
import logging
import os
import sys
from datetime import datetime
from typing import Callable, Optional

from pydantic import BaseModel

# 确保项目根目录在 sys.path
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from src.data_fetcher import DataManager, Watchlist
from src.config.group_config import GroupConfig
from src.config.runtime_mode import set_mode, RuntimeMode, get_mode
from src.backtest import BacktestEngine
from src.backtest.metrics import compute_metrics
from src.backtest.rotation_strategy import RotationStrategy
from src.memory.failure_case import (
    FailureCaseRecorder, compute_stock_pool_snapshot, make_backtest_source_ref,
)
from src.memory.positive_case import PositiveCaseRecorder
from src.memory.strategy_memory import compute_strategy_version

logger = logging.getLogger(__name__)


# ── P5 组合配置 (2026-08-07 固化) ──
# 分组独立回测: 每组独立engine, 按权重分配资金, 合并daily_values计算组合级指标.
# 权威来源: config/strategy_config.json 的 strategy_config.portfolio_config_p5
# (单源: 配置页修改权重后回测立即跟随; 旧实现硬编码在本文件, 配置页改动对回测不生效).
# 下方 _FALLBACK_PORTFOLIO 仅在配置文件缺失/字段不全时兜底, 与配置文件当前值一致.
_FALLBACK_PORTFOLIO = {
    "weights": {
        "科技成长型": 0.40,    # 核心引擎 (2026YTD +41%)
        "消费稳健型": 0.05,    # 全组启用 (P2均值回归, Alpha为负但参与组合分散)
        "周期资源型": 0.375,   # 核心引擎 (2026YTD +38%, 降5%让位消费/医药)
        "医药创新型": 0.05,    # 全组启用 (P3均值回归, Alpha为负但参与组合分散)
        "机械制造型": 0.10,    # P4c启用 (动量轮动策略, 测试窗Alpha-9.97%)
    },
    # 周期组仅趋势市交易 (ADX>25), 避免震荡市频繁止损
    "regimes_filter": {"周期资源型": ["trending"]},
    # 科技组ATR收紧至1.8 (默认2.0), 控制回撤
    "atr_override": {"科技成长型": 1.8},
    # 组级12%回撤保护 (引擎真实降仓, 非事后净值调整)
    "dd_protection_threshold": -0.12,
    "dd_protection_recovery": -0.06,
    "dd_protection_reduced_ratio": 0.5,
}


def _load_portfolio_config() -> dict:
    """从 strategy_config.json 读取组合配置 (每次回测时读取, 配置页改动即时生效).

    返回: {"weights": {...}, "regimes_cfg": {...}, "atr_override": {...}, "dd_config": {...}}
    """
    import json
    cfg = {
        "weights": dict(_FALLBACK_PORTFOLIO["weights"]),
        "regimes_cfg": {k: set(v) for k, v in _FALLBACK_PORTFOLIO["regimes_filter"].items()},
        "atr_override": dict(_FALLBACK_PORTFOLIO["atr_override"]),
        "dd_config": {
            "threshold": _FALLBACK_PORTFOLIO["dd_protection_threshold"],
            "recovery": _FALLBACK_PORTFOLIO["dd_protection_recovery"],
            "reduced_ratio": _FALLBACK_PORTFOLIO["dd_protection_reduced_ratio"],
        },
    }
    try:
        path = os.path.join(_PROJECT_ROOT, "config", "strategy_config.json")
        with open(path, "r", encoding="utf-8") as f:
            pc = json.load(f).get("strategy_config", {}).get("portfolio_config_p5", {})
        if pc.get("weights"):
            cfg["weights"] = dict(pc["weights"])
        if pc.get("regimes_filter"):
            cfg["regimes_cfg"] = {k: set(v) for k, v in pc["regimes_filter"].items()}
        if pc.get("atr_override"):
            cfg["atr_override"] = dict(pc["atr_override"])
        if pc.get("dd_protection_threshold") is not None:
            cfg["dd_config"] = {
                "threshold": pc["dd_protection_threshold"],
                "recovery": pc.get("dd_protection_recovery", -0.06),
                "reduced_ratio": pc.get("dd_protection_reduced_ratio", 0.5),
            }
    except Exception as e:
        logger.warning("读取组合配置失败, 使用内置默认值: %s", e)
    return cfg


def _build_group_params_snapshot(*, group_name: str, group_cfg: dict, strategy_mode: str,
                                 group_capital: float, atr_stop_mult: Optional[float],
                                 mean_reversion_config: Optional[dict],
                                 forced_regime: str, trade_regimes, dd_config: dict,
                                 weight: float, mode: str) -> dict:
    """构建该组本次回测的完整参数快照 (L1 记录的 strategy_version 来源)"""
    return {
        "group_name": group_name,
        "group_config": group_cfg or {},
        "engine": {
            "initial_capital": group_capital,
            "position_ratio": 0.3,
            "risk_per_trade": 0.05,
            # 最小手数(100股)约束诊断: position_ratio=0.3 时单笔目标仓位
            # ≈ capital*0.3*0.95, 可买最高价 = 该值/100. A 股常见价格中位数
            # 约 ¥15~20, 若可买最高价 <20 元则多数股票凑不齐 1 手, 信号被系统性跳过.
            "max_affordable_price": round((group_capital * 0.3 * 0.95) / 100, 2),
            "capital_constrained": (group_capital * 0.3 * 0.95) / 100 < 20.0,
            "atr_stop_mult": atr_stop_mult,
            "signal_dedup_days": 5,
            "forced_regime": forced_regime,
            "trade_regimes": sorted(trade_regimes) if trade_regimes else None,
            "dd_protection": dd_config,
            "mean_reversion": mean_reversion_config,
        },
        "portfolio": {"weight": weight, "mode": mode},
        "strategy_mode": strategy_mode,
    }


def _record_group_cases(recorder: FailureCaseRecorder, *, run_id: Optional[str],
                        group_name: str, group_codes: list, data_map: dict,
                        bench_df, daily_values, trades: list, group_capital: float,
                        strategy_mode: str, params_snapshot: dict,
                        regime_dist: Optional[dict], mode: str,
                        regime_slices: Optional[dict] = None,
                        positive_recorder: Optional[PositiveCaseRecorder] = None) -> None:
    """记录单组回测的 L1 失效案例 (未命中 entry_flags 时由 recorder 自行跳过)
    与 P5 正样本轨迹 (未命中正样本标签时由 positive_recorder 自行跳过)
    """
    if daily_values is None or len(daily_values) == 0:
        return

    start = str(daily_values.index[0])[:10]
    end = str(daily_values.index[-1])[:10]

    bench_series = None
    if bench_df is not None:
        bench_series = BacktestEngine._align_benchmark(bench_df, daily_values.index)
    metrics = compute_metrics(
        daily_values=daily_values,
        trades=trades,
        initial_capital=group_capital,
        benchmark_values=bench_series,
    )

    sub_map = {c: data_map[c] for c in group_codes if c in data_map}
    stock_pool_snapshot = compute_stock_pool_snapshot(
        sub_map, bench_df, start, end, stock_codes=group_codes)

    recorder.record_backtest(
        source_ref={"run_id": run_id, "group": group_name, "mode": mode},
        group_name=group_name,
        stock_codes=group_codes,
        window={"start": start, "end": end},
        strategy_mode=strategy_mode,
        params=params_snapshot,
        metrics=metrics,
        trades=trades,
        benchmark_return=metrics.benchmark_return,
        regime_dist=regime_dist,
        regime_slices=regime_slices,
        stock_pool_snapshot=stock_pool_snapshot,
    )

    if positive_recorder is not None:
        positive_recorder.record_backtest(
            source_ref={"run_id": run_id, "group": group_name, "mode": mode},
            group_name=group_name,
            stock_codes=group_codes,
            window={"start": start, "end": end},
            strategy_mode=strategy_mode,
            params=params_snapshot,
            metrics=metrics,
            trades=trades,
            benchmark_return=metrics.benchmark_return,
            regime_dist=regime_dist,
            regime_slices=regime_slices,
        )


def _group_metrics_snapshot(daily_values, trades, group_capital, bench_df) -> dict:
    """单组精简指标快照 (供 P8 组级 A/B 对比)。无数据时返回 None 值占位。"""
    if daily_values is None or len(daily_values) < 2:
        return {"alpha": None, "total_return": None, "trade_count": len(trades or [])}
    bench_series = None
    if bench_df is not None:
        bench_series = BacktestEngine._align_benchmark(bench_df, daily_values.index)
    m = compute_metrics(
        daily_values=daily_values, trades=trades or [],
        initial_capital=group_capital, benchmark_values=bench_series,
    )
    return {
        "alpha": round(m.alpha, 6),
        "total_return": round(m.total_return, 6),
        "trade_count": m.trade_count,
    }


class BacktestRequest(BaseModel):
    groups: list[str] = []          # 展示分组(仅过滤前端交易明细展示, 不影响回测参与)
    mode: str = "base"              # base / trending / ranging
    start_date: Optional[str] = None  # YYYY-MM-DD
    end_date: Optional[str] = None    # YYYY-MM-DD
    initial_capital: float = 100000
    benchmark: str = "sh.000300"       # 基准指数代码 (沪深300, 与回测脚本一致)


def run_portfolio_backtest(
    req: BacktestRequest,
    on_progress: Optional[Callable[[int, str], None]] = None,
    config_override: Optional[dict] = None,
    record_memory: bool = True,
) -> dict:
    """同步执行 P5 组合回测（阻塞直到完成）。

    纯移动自 routers/backtest.py::run_backtest 主体，逻辑与返回体一字未改。
    进度上报改为可选回调，不传则静默。

    P8 扩展参数 (仅进化 A/B 使用, 正常调用方无需关心):
        config_override: 候选参数覆盖
            {"groups": {组名: 完整组配置dict},       # 替换 gc._groups[组名]
             "atr_override": {组名: atr值},           # 合并进组合级 atr_override
             "weights": {组名: 权重}}                 # 合并进组合级权重
        record_memory: False 时不写 L1/正样本/运行基线/谱系/建议验证
                       (候选参数跑出的结果不属于生产轨迹, 不应污染记忆层)
    """
    def _progress(p: int, text: Optional[str] = None) -> None:
        if on_progress is not None:
            on_progress(p, text or "")

    # 切换到 BACKTEST 模式: SignalFilter 使用纯内存历史, 不读写 signal_history.json
    # 避免污染实时信号数据, 且避免上次回测/分析的去重记录影响本次回测
    prev_mode = get_mode()
    set_mode(RuntimeMode.BACKTEST)

    # P8: config_override 需要临时替换 gc._groups 中的组配置, 记录原始值供 finally 恢复
    override_groups = (config_override or {}).get("groups") or {}
    _gc_backup: dict = {}
    gc: Optional[GroupConfig] = None

    try:
        _progress(0, "加载自选股...")

        # 1. 读取自选股
        wl = Watchlist()
        all_stocks = wl.get_all()
        if not all_stocks:
            return {"status": "error", "message": "自选股列表为空"}

        # 注意: 不删除 signal_history.json — BACKTEST 模式下 SignalFilter 为纯内存历史,
        # 不读写该文件; 删除只会摧毁 LIVE 模式积累的跨会话去重数据, 导致实时信号重复触发.

        gc = GroupConfig()
        if override_groups:
            import copy as _copy
            for g_name, g_cfg in override_groups.items():
                _gc_backup[g_name] = _copy.deepcopy(gc._groups.get(g_name))

        # 筛选展示语义: req.groups 仅用于前端交易明细展示过滤, 不影响回测参与.
        # 回测始终跑全部活跃组(权重>0), 确保收益与P5基线一致, 避免选子集导致现金稀释.

        # regime 作为请求级参数传入 BacktestEngine, 不再写盘 user_preferences.json
        # 避免污染其他页面 (信号看板/个股回测) 的 regime 配置
        # base → auto (ADX 自动判断); trending/ranging → 该模式
        forced_regime = req.mode if req.mode in ("trending", "ranging") else "auto"

        total = len(all_stocks)
        _progress(10, f"加载日线数据 (共{total}只)...")

        # 2. 拉取日线数据
        # 数据获取起点要比回测起点早至少 120 个交易日 (lookback), 否则引擎无法计算指标
        dm = DataManager()
        if req.start_date:
            from datetime import timedelta
            start_dt = datetime.strptime(req.start_date, "%Y-%m-%d")
            if start_dt >= datetime(2025, 12, 1):
                # P5 验证场景 (2026-01-01 起): 硬编码与脚本一致, 确保结果可复现.
                # 对应 verify_frontend_p5.py / ytd_2026_group_comparison.py 的 DATA_START.
                fetch_start = "2025-06-01"
            else:
                # 其他场景 (如训练窗 2024-07-01): 动态往前推 240 天, 确保有足够 lookback
                fetch_start = (start_dt - timedelta(days=240)).strftime("%Y-%m-%d")
        else:
            fetch_start = "2024-01-01"
        end = req.end_date or datetime.now().strftime("%Y-%m-%d")

        data_map = {}
        stock_info = {}  # code → {name, group}
        for i, stock in enumerate(all_stocks):
            code = stock["code"]
            try:
                df = dm.get_daily_kline(code, start_date=fetch_start, end_date=end)
                if df is not None and len(df) >= 120:
                    data_map[code] = df
                    stock_info[code] = {
                        "name": stock.get("name", code),
                        "group": gc.get_group(code),
                    }
            except Exception as e:
                logger.warning("获取 %s 数据失败: %s", code, e)
            _progress(10 + int((i + 1) / len(all_stocks) * 50))

        if not data_map:
            return {"status": "error", "message": "所有股票数据获取失败"}

        _progress(65, "执行回测引擎...")

        # 3. 获取基准数据
        # 基准获取失败必须直接报错终止, 不能 bench_df=None 继续跑:
        #   - 机械组轮动策略 rotation.run 内部 bench_df.copy() 对 None 抛 AttributeError,
        #     会让整个组合回测崩溃;
        #   - RegimeDetector 失去基准后体制判断失真, 周期组 trade_regimes={trending}
        #     过滤会错误拦截/放行信号, 回测结果不可信.
        try:
            bench_df = dm.get_daily_kline(req.benchmark, start_date=fetch_start, end_date=end)
        except Exception as e:
            logger.warning("获取基准数据失败: %s", e)
            bench_df = None
        if bench_df is None or bench_df.empty:
            return {
                "status": "error",
                "message": f"基准数据获取失败 ({req.benchmark}), 回测已终止: "
                           "体制判断与轮动策略均依赖基准数据, 请检查网络/数据源后重试",
            }

        # 4. 运行回测引擎 — P5 分组独立回测 + 权重合并
        # 每组独立创建 BacktestEngine, 按配置权重分配资金, 应用差异化配置:
        #   - 周期组: trade_regimes={trending} 仅趋势市交易
        #   - 科技组: atr_stop_mult=1.8 收紧止损
        #   - 所有组: dd_protection_config 12%引擎真实降仓
        # 权重0%的组(已暂停)跳过不交易, 对应资金留作现金缓冲.
        # benchmark_df 传入 engine.run: RegimeDetector 需要它判断 trending/ranging,
        # 周期组的 trade_regimes={trending} 过滤依赖此判断.

        # 组合配置: 从 strategy_config.json 读取 (单源, 配置页改动即时生效)
        pc = _load_portfolio_config()

        # P8: 候选参数覆盖 (仅进化 A/B 使用) — 合并组合级 weights / atr_override / regimes_cfg
        if config_override:
            for key in ("weights", "atr_override"):
                ov = config_override.get(key)
                if isinstance(ov, dict):
                    pc[key].update(ov)
            # regimes_cfg: 候选覆盖 (value 是 list, 引擎内部转 set)
            ov_regimes = config_override.get("regimes_cfg")
            if isinstance(ov_regimes, dict):
                for g, regimes in ov_regimes.items():
                    if regimes is None:
                        pc["regimes_cfg"][g] = None
                    else:
                        pc["regimes_cfg"][g] = set(regimes)
        PORTFOLIO_WEIGHTS = pc["weights"]
        REGIMES_CFG = pc["regimes_cfg"]
        ATR_OVERRIDE = pc["atr_override"]
        DD_CONFIG = pc["dd_config"]

        # 确定参与的组: 始终用全部活跃组(权重>0), 不受 req.groups 影响.
        # req.groups 仅用于前端交易明细展示过滤 (筛选展示语义).
        active_groups = [g for g, w in PORTFOLIO_WEIGHTS.items() if w > 0]

        if not active_groups:
            return {"status": "error", "message": "全部分组权重均为0(已暂停), 无可回测的组"}

        # 权重: 始终用原始配置权重, 不归一化.
        # P5 策略设计了现金缓冲作为风控组成部分, 归一化会抹掉现金缓冲,
        # 导致满仓运行 + MarketFilter 减仓 → 收益大幅下降 (33% → 7%).
        # 与命令行脚本(ytd_2026_group_comparison.py)和验证脚本(verify_frontend_p5.py)保持一致.
        weights_normalized = {g: PORTFOLIO_WEIGHTS[g] for g in active_groups}

        portfolio_daily_values = None
        all_closed_trades = []
        all_open_positions = {}
        total_invested = 0
        failure_recorder = FailureCaseRecorder()
        positive_recorder = PositiveCaseRecorder()
        group_metrics_map: dict = {}

        for idx, group_name in enumerate(active_groups):
            weight = weights_normalized[group_name]
            group_capital = req.initial_capital * weight
            total_invested += group_capital

            # 该组的股票
            group_codes = [c for c, info in stock_info.items() if info["group"] == group_name]
            group_codes = [c for c in group_codes if c in data_map]
            if not group_codes:
                continue

            _progress(65, f"执行回测引擎: {group_name} ({idx+1}/{len(active_groups)})...")

            # 重新从磁盘读取分组配置(热更新): 配置经 API 修改后无需重启后端即可生效.
            # 注: 此前用 `GroupConfig._instance = None` 重置是死代码(gc 仍是旧实例引用),
            # 改用 reload() 原地刷新同一单例, 使下方读取的分组参数真实生效.
            gc.reload()

            # P8: 候选组配置覆盖 — reload 从磁盘重读会覆盖预置值, 必须在 reload 之后替换;
            # 引擎运行时实时读取 gc._groups, 就地替换才能让候选参数真实生效
            if group_name in override_groups:
                gc._groups[group_name] = copy.deepcopy(override_groups[group_name])

            # 读取该组的策略配置
            group_cfg = gc._groups.get(group_name, {})
            strategy_mode = group_cfg.get("strategy_mode", "trend_following")

            # 失效案例 L1 记录的临时载体 (两个策略分支各自填充)
            group_daily = None
            group_trades = []
            group_regime_dist = None
            group_run_id = None
            group_atr_mult = None
            group_mr_config = None

            if strategy_mode == "rotation":
                # ── 动量轮动策略 (机械组专用) ──
                # 使用独立RotationStrategy, 不走BacktestEngine
                rotation = RotationStrategy(initial_capital=group_capital)
                sub_map = {c: data_map[c] for c in group_codes}
                rot_result = rotation.run(
                    sub_map,
                    bench_df=bench_df,
                    start_date=req.start_date or "2024-07-01",
                    end_date=end,
                )

                # 合并 daily_values
                if rot_result["daily_values"] is not None:
                    if portfolio_daily_values is None:
                        portfolio_daily_values = rot_result["daily_values"].copy()
                    else:
                        portfolio_daily_values = portfolio_daily_values.add(
                            rot_result["daily_values"], fill_value=0)

                # 收集交易记录 (RotationTrade对象与Trade对象格式兼容)
                all_closed_trades.extend(rot_result["closed_trades"])
                all_open_positions.update(rot_result["open_positions"])

                # 失效案例: 轮动组没有 BacktestEngine, 使用独立 run_id
                group_daily = rot_result["daily_values"]
                group_trades = list(rot_result["closed_trades"])
                group_run_id = None  # 由 recorder 侧 source_ref 生成

            else:
                # ── 标准趋势跟踪/均值回归策略 ──
                # 桥接: strategy_config.mean_reversion_exit → BacktestEngine.mean_reversion_config
                mean_reversion_config = group_cfg.get("mean_reversion_exit") or None

                group_engine = BacktestEngine(
                    initial_capital=group_capital,
                    lookback_days=120,
                    position_ratio=0.3,
                    commission_rate=0.00025,
                    stamp_tax=0.0005,
                    slippage=0.0001,
                    signal_dedup_days=5,
                    risk_per_trade=0.05,
                    atr_stop_mult=ATR_OVERRIDE.get(group_name, 2.0),
                    group_config=gc,
                    forced_regime=forced_regime,
                    benchmark_df_for_memory=bench_df,
                    trade_regimes=REGIMES_CFG.get(group_name),
                    dd_protection_config=DD_CONFIG,
                    mean_reversion_config=mean_reversion_config,
                )

                sub_map = {c: data_map[c] for c in group_codes}
                group_engine.run(
                    sub_map,
                    benchmark_df=bench_df,  # 传基准数据: RegimeDetector 需要它判断 trending/ranging
                                            # (周期组 trade_regimes={trending} 过滤依赖 regime 判断)
                    start_date=req.start_date,
                    end_date=end,
                )

                # 失效案例: 采集该组的净值/交易/体制分布
                group_daily = group_engine.daily_values
                group_trades = list(group_engine.position_mgr.closed_trades)
                group_regime_dist = getattr(group_engine, "regime_dist", None)
                group_regime_slices = getattr(group_engine, "regime_slices", None)
                group_run_id = getattr(group_engine.memory, "run_id", None)
                group_atr_mult = ATR_OVERRIDE.get(group_name, 2.0)
                group_mr_config = mean_reversion_config

                # 合并 daily_values (各组净值按日期对齐相加)
                if group_engine.daily_values is not None:
                    if portfolio_daily_values is None:
                        portfolio_daily_values = group_engine.daily_values.copy()
                    else:
                        portfolio_daily_values = portfolio_daily_values.add(
                            group_engine.daily_values, fill_value=0)

                # 收集交易记录 (含部分平仓的回撤保护降仓记录)
                all_closed_trades.extend(group_engine.position_mgr.closed_trades)
                all_open_positions.update(group_engine.position_mgr.open_positions)

            # 单组指标快照 (无论是否 record_memory 都计算, 供 P8 组级 A/B)
            group_metrics_map[group_name] = _group_metrics_snapshot(
                daily_values=group_daily, trades=group_trades,
                group_capital=group_capital, bench_df=bench_df,
            )

            # 失效案例 L1 + 正样本: 记录该组回测结果 (record_memory=False 时跳过;
            # 记录失败绝不影响主流程)
            if record_memory:
                try:
                    params_snapshot = _build_group_params_snapshot(
                        group_name=group_name,
                        group_cfg=group_cfg,
                        strategy_mode=strategy_mode,
                        group_capital=group_capital,
                        atr_stop_mult=group_atr_mult,
                        mean_reversion_config=group_mr_config,
                        forced_regime=forced_regime,
                        trade_regimes=REGIMES_CFG.get(group_name),
                        dd_config=DD_CONFIG,
                        weight=weight,
                        mode=req.mode,
                    )
                    if group_run_id is None:
                        group_run_id = make_backtest_source_ref()["run_id"]
                    _record_group_cases(
                        failure_recorder,
                        run_id=group_run_id,
                        group_name=group_name,
                        group_codes=group_codes,
                        data_map=data_map,
                        bench_df=bench_df,
                        daily_values=group_daily,
                        trades=group_trades,
                        group_capital=group_capital,
                        strategy_mode=strategy_mode,
                        params_snapshot=params_snapshot,
                        regime_dist=group_regime_dist,
                        regime_slices=group_regime_slices,
                        mode=req.mode,
                        positive_recorder=positive_recorder,
                    )
                except Exception:
                    logger.warning("失效案例/正样本记录失败 (group=%s)", group_name, exc_info=True)

            _progress(65 + int((idx + 1) / len(active_groups) * 20))

        # 加上现金部分 (权重0%的组的资金 + 归一化后的剩余现金)
        cash = req.initial_capital - total_invested
        if portfolio_daily_values is not None:
            portfolio_daily_values = portfolio_daily_values + cash

        # 用合并后的 daily_values 计算组合级 metrics
        if portfolio_daily_values is None or len(portfolio_daily_values) == 0:
            return {"status": "error", "message": "回测未产生净值数据"}

        # 基准对齐: 只对齐一次, metrics 计算与前端曲线输出共用同一序列
        # (旧实现对齐两次: 第二次结果受 benchmark_return != 0 门控,
        #  基准收益恰好为 0 时前端拿不到真实序列, 只能回退线性近似)
        bench_series = None
        if bench_df is not None:
            bench_series = BacktestEngine._align_benchmark(
                bench_df, portfolio_daily_values.index)

        if bench_series is not None and len(bench_series) > 0:
            metrics = compute_metrics(
                daily_values=portfolio_daily_values,
                trades=all_closed_trades,
                initial_capital=req.initial_capital,
                benchmark_values=bench_series,
            )
        else:
            metrics = compute_metrics(
                daily_values=portfolio_daily_values,
                trades=all_closed_trades,
                initial_capital=req.initial_capital,
            )

        _progress(90, "格式化结果...")

        # 5. 格式化交易记录 (含已平仓 + 未平仓)
        # 合并各组的 closed_trades, 按 entry_date 排序
        trades_out = []
        for t in all_closed_trades:
            info = stock_info.get(t.symbol, {})
            trades_out.append({
                "symbol": t.symbol,
                "name": info.get("name", t.symbol),
                "group": info.get("group", ""),
                "mode": req.mode,
                "entry_date": str(t.entry_date)[:10],
                "entry_price": round(t.entry_price, 3),
                "exit_date": str(t.exit_date)[:10] if t.exit_date else "",
                "exit_price": round(t.exit_price, 3) if t.exit_price else None,
                "shares": t.shares,
                "cost": round(t.entry_price * t.shares, 2),
                "pnl": round(t.pnl, 2),
                "pnl_pct": round(t.pnl_pct * 100, 2),
                "holding_days": t.holding_days,
                "signal": t.entry_signal,
                "exit_signal": t.exit_signal,
                "commission": round(t.commission, 2),
                "status": "closed",
            })

        # 加入未平仓持仓 (回测结束时仍持有的仓位)
        # 使用 daily_values 的最后一个日期作为最后日期
        last_date = None
        if metrics.daily_values is not None and len(metrics.daily_values) > 0:
            last_date = metrics.daily_values.index[-1]
            # 获取最后一天的收盘价
            from src.backtest.calendar import TradingCalendar
            calendar = TradingCalendar(data_map)
            prices_last = calendar.get_closing_prices(data_map, last_date)
            for symbol, t in all_open_positions.items():
                info = stock_info.get(symbol, {})
                current_price = prices_last.get(symbol)
                # 计算浮动盈亏 (未扣除卖出手续费)
                unrealized_pnl = (current_price - t.entry_price) * t.shares - t.commission if current_price else 0
                unrealized_pnl_pct = (unrealized_pnl / (t.entry_price * t.shares) * 100) if t.shares > 0 else 0
                trades_out.append({
                    "symbol": symbol,
                    "name": info.get("name", symbol),
                    "group": info.get("group", ""),
                    "mode": req.mode,
                    "entry_date": str(t.entry_date)[:10],
                    "entry_price": round(t.entry_price, 3),
                    "exit_date": "",  # 未平仓
                    "exit_price": round(current_price, 3) if current_price else None,
                    "shares": t.shares,
                    "cost": round(t.entry_price * t.shares, 2),
                    "pnl": round(unrealized_pnl, 2),
                    "pnl_pct": round(unrealized_pnl_pct, 2),
                    "holding_days": (last_date - t.entry_date).days if hasattr(last_date, '__sub__') else 0,
                    "signal": t.entry_signal,
                    "exit_signal": "未平仓",
                    "commission": round(t.commission, 2),
                    "status": "open",
                })

        # 按买入日期排序
        trades_out.sort(key=lambda x: x["entry_date"])

        # 6. 格式化日净值序列 (供前端绘图)
        daily_values = metrics.daily_values
        dates = []
        values = []
        if daily_values is not None and len(daily_values) > 0:
            for d, v in daily_values.items():
                dates.append(str(d)[:10])
                values.append(round(float(v), 2))

        # 基准净值: 直接复用上方对齐好的 bench_series (归一化净值, 起点=1.0),
        # 不再二次对齐. 保留 6 位小数 — 前端需乘以 initial_capital 还原量纲,
        # 只留 2 位会让 100000 本金下的曲线误差达数百元.
        bench_dates = []
        bench_values = []
        if bench_series is not None:
            for d, v in bench_series.items():
                bench_dates.append(str(d)[:10])
                bench_values.append(round(float(v), 6))

        # P4/P7/P8: 计算组合级运行标识 (无论是否 record_memory 都计算,
        # 进化 A/B 需要从返回值拿 run_id 与 strategy_version)
        outcome_metrics = {
            "alpha": round(metrics.alpha, 6),
            "total_return": round(metrics.total_return, 6),
            "annual_return": round(metrics.annual_return, 6),
            "max_drawdown": round(metrics.max_drawdown, 6),
            "sharpe_ratio": round(metrics.sharpe_ratio, 4),
            "win_rate": round(metrics.win_rate, 4),
            "trade_count": metrics.trade_count,
            "benchmark_return": round(metrics.benchmark_return, 6),
        }
        outcome_window = {
            "start": str(daily_values.index[0])[:10],
            "end": str(daily_values.index[-1])[:10],
        }

        # P7 版本谱系: 组合级版本 = mode + 组合配置 + 各组 strategy_mode
        portfolio_version_params = {
            "mode": req.mode,
            "portfolio_config": {
                "weights": dict(PORTFOLIO_WEIGHTS),
                "regimes_filter": {k: sorted(v) for k, v in REGIMES_CFG.items()},
                "atr_override": dict(ATR_OVERRIDE),
                "dd_protection": dict(DD_CONFIG),
            },
            "group_modes": {
                g: gc._groups.get(g, {}).get("strategy_mode", "trend_following")
                for g in active_groups
            },
        }
        # 候选只改组级参数时 group_modes 不变, 会把候选与基线哈希成同一版本.
        # 无覆盖时省略该键, 以保持基线版本号与既有谱系记录一致.
        if override_groups:
            portfolio_version_params["group_config_overrides"] = {
                g: copy.deepcopy(override_groups[g])
                for g in active_groups if g in override_groups
            }
        portfolio_version = compute_strategy_version(portfolio_version_params)
        outcome_run_id = make_backtest_source_ref()["run_id"]

        # 记忆层写入 (record_memory=False 时跳过: 候选参数结果不属于生产轨迹)
        if record_memory:
            try:
                from src.memory.run_outcome import RunOutcomeStore
                from src.memory.version_lineage import VersionLineage
                from src.ai.suggestion_validator import SuggestionValidator

                outcome_store = RunOutcomeStore()
                outcome_store.record(
                    run_id=outcome_run_id,
                    window=outcome_window,
                    metrics=outcome_metrics,
                    mode=req.mode,
                    groups=active_groups,
                    strategy_version=portfolio_version,
                )

                lineage = VersionLineage()
                lineage.record_version(
                    version=portfolio_version,
                    params=portfolio_version_params,
                    run_id=outcome_run_id,
                    window=outcome_window,
                    mode=req.mode,
                    groups=active_groups,
                    metrics=outcome_metrics,
                )

                # 自动验证 (观察性: 同窗口 before/after 对比), 结论回写 L2 + 版本谱系
                SuggestionValidator(
                    outcome_store=outcome_store,
                    version_lineage=lineage,
                ).validate(dry_run=False)
            except Exception:
                logger.warning("P4 运行基线/自动验证/P7 谱系记录失败", exc_info=True)

        _progress(100, "完成")

        return {
            "status": "done",
            "run_id": outcome_run_id,
            "strategy_version": portfolio_version,
            "metrics": {
                "total_return": round(metrics.total_return, 6),
                "annual_return": round(metrics.annual_return, 6),
                "max_drawdown": round(metrics.max_drawdown, 6),
                "sharpe_ratio": round(metrics.sharpe_ratio, 4),
                "trade_count": metrics.trade_count,
                "win_rate": round(metrics.win_rate, 4),
                "profit_factor": round(metrics.profit_factor, 4) if metrics.profit_factor != float('inf') else 999.0,
                "total_pnl": round(metrics.total_pnl, 2),
                "initial_capital": metrics.initial_capital,
                "final_value": round(metrics.final_value, 2),
                "win_count": metrics.win_count,
                "avg_holding_days": round(metrics.avg_holding_days, 1),
                "volatility": round(metrics.volatility, 4),
                "benchmark_return": round(metrics.benchmark_return, 6),
                "alpha": round(metrics.alpha, 6),
            },
            "trades": trades_out,
            "daily_values": {"dates": dates, "values": values},
            "benchmark_values": {"dates": bench_dates, "values": bench_values},
            "mode": req.mode,
            "groups": active_groups,
            "group_metrics": group_metrics_map,
            "display_groups": req.groups,
            "portfolio_config": {
                "weights": {g: w for g, w in PORTFOLIO_WEIGHTS.items() if w > 0},
                "cash_ratio": round(cash / req.initial_capital, 4),
            },
        }

    except Exception as e:
        logger.exception("回测执行异常")
        return {"status": "error", "message": str(e)}
    finally:
        set_mode(prev_mode)  # 恢复原运行时模式
        # P8: 恢复被 config_override 替换的组配置 (单例是全局的, 不能留候选参数)
        if _gc_backup and gc is not None:
            for g_name, g_cfg in _gc_backup.items():
                if g_cfg is None:
                    gc._groups.pop(g_name, None)
                else:
                    gc._groups[g_name] = g_cfg
