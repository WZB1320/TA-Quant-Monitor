"""策略记忆层 — 结构化记录信号、参数、结果与上下文

Phase 1: 仅记录, 不分析. 为后续 AI 优化积累数据.

存储:
  - 实盘: data/strategy_memory_YYYY-MM.jsonl (按月切分, 避免单文件无限增长)
         旧格式 data/strategy_memory.jsonl 仍可读取 (向后兼容)
  - 回测: data/backtest_memory/{run_id}.jsonl (每次 run 一个文件, 便于横向对比)

记录类型:
  - SignalRecord: 信号产出时写入 (SignalEngine.analyze)
      记录: 标的/时间/regime/level/score/参数快照/指标快照/执行状态
  - OutcomeRecord: 交易平仓时写入 (BacktestEngine.run)
      记录: 入场/出场/盈亏/退出归因/市场上下文
      通过 (symbol, analysis_date, run_id) 关联到 SignalRecord

设计说明:
  - params_snapshot 是策略调整追踪的核心字段, 其 hash 即 strategy_version
  - 信号与结果分两类记录, 因为产出时无法知道结果, 强行合并会丢失信号上下文
  - 回测单独成文件, 避免多次回测污染实盘记忆
  - live 按月切分: 进程跨月运行时自动切换到新月文件
  - 信号记录幂等: 去重键为 (标的, 交易日, 级别, 参数版本), 同一事实重复分析只落一条
    (见 signal_dedupe_key). 没有幂等之前, 每次重新分析/后端热重载都会追加一份
    全新 run 的完整快照, 实测 7 月冗余 70%、8 月 82%、9 月 77%.
"""
import os
import glob
import json
import uuid
import hashlib
import logging
import threading
from datetime import datetime
from typing import Optional

logger = logging.getLogger(__name__)


# 项目根目录 (src/memory/ → src/ → 项目根)
_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
_DATA_DIR = os.path.join(_ROOT, "data")


def compute_strategy_version(params: dict) -> str:
    """计算参数快照的版本 hash (MD5 前 8 位)

    当 config 页修改任何参数时, 此 hash 变化.
    AI 后续可按 strategy_version 分组, 对比不同参数集的收益表现.
    """
    raw = json.dumps(params, sort_keys=True, default=str, ensure_ascii=False)
    return hashlib.md5(raw.encode("utf-8")).hexdigest()[:8]


def generate_run_id(source: str = "live") -> str:
    """生成 run_id

    Args:
        source: "backtest" | "live"
    """
    prefix = "bt" if source == "backtest" else "live"
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    short_uuid = uuid.uuid4().hex[:8]
    return f"{prefix}_{ts}_{short_uuid}"


# 信号记录去重键字段 (顺序即键顺序)
_SIGNAL_KEY_FIELDS = ("symbol", "analysis_date", "level", "strategy_version")


def signal_dedupe_key(record: dict) -> tuple:
    """信号记录去重键 = (标的, 交易日, 级别, 参数版本)

    语义: 同一只股票、同一个交易日、同一套参数下产出的同一个级别的信号, 是"同一个事实",
    重复分析(重新生成大屏 / 后端热重载 / 手工重跑)不应追加第二条。

    交易日缺失时退化为写入日期 (recorded_at 前 10 字符):
      - 同一写入日内的重复生成仍能判重
      - 不会把不同交易日的信号误判成同一条 (避免了 null 全部撞成一个键)
    """
    analysis_date = record.get("analysis_date")
    if analysis_date:
        date_key = str(analysis_date)[:10]
    else:
        date_key = f"rec:{(record.get('recorded_at') or '')[:10]}"
    return (
        str(record.get("symbol")),
        date_key,
        str(record.get("level")),
        str(record.get("strategy_version")),
    )


def find_live_memory_files(data_dir: Optional[str] = None) -> list:
    """查找所有 live memory 文件 (按月切分的新格式 + 旧格式)

    返回排序后的文件路径列表. 旧格式文件排在最后.

    Args:
        data_dir: 数据目录, None 则用默认 _DATA_DIR

    Returns:
        [path, ...] 按月份升序排列
    """
    d = data_dir or _DATA_DIR
    files = []
    # 新格式: strategy_memory_YYYY-MM.jsonl
    for f in glob.glob(os.path.join(d, "strategy_memory_*.jsonl")):
        files.append(f)
    # 旧格式: strategy_memory.jsonl (向后兼容)
    old = os.path.join(d, "strategy_memory.jsonl")
    if os.path.exists(old):
        files.append(old)
    return sorted(files)


class StrategyMemory:
    """策略记忆层 — append-only JSONL 记录器

    用法:
      # 回测 (BacktestEngine 内部创建)
      memory = StrategyMemory(source="backtest")

      # 实盘 (路由层创建, 传给 SignalEngine)
      memory = StrategyMemory(source="live")

      # 记录信号 (SignalEngine.analyze 内部调用)
      memory.record_signal({...})

      # 记录结果 (BacktestEngine 平仓时调用)
      memory.record_outcome({...})

    线程安全:
      Phase 1 假设单线程写入. 多并发实盘请求写同一文件时,
      依赖 OS 对 < 4KB 单行 write 的 append 原子性.
      开启 dedupe 时, 判重索引的读改写与落盘在同一把锁内完成.

    幂等 (dedupe=True, 默认开启):
      信号记录按 signal_dedupe_key 判重, 索引在首次写入时从当前文件整体重建,
      因此跨进程重启(uvicorn --reload 热重载)依然有效 —— 索引不是进程内记忆,
      而是"磁盘上已存在的事实"的投影.
    """

    def __init__(self, source: str = "live", run_id: Optional[str] = None,
                 enabled: bool = True, dedupe: bool = True,
                 data_dir: Optional[str] = None):
        """
        Args:
            source: "backtest" | "live"
            run_id: 运行 ID, None 则自动生成
            enabled: False 则完全不写盘 (用于禁用记忆)
            dedupe: True 则对信号记录按 signal_dedupe_key 幂等写入
            data_dir: 数据目录, None 则用默认 _DATA_DIR (测试可注入临时目录)
        """
        self.source = source
        self.run_id = run_id or generate_run_id(source)
        self.enabled = enabled
        self.dedupe = dedupe
        self._data_dir = data_dir or _DATA_DIR
        self._file_path = self._resolve_file_path()
        # 判重索引 (惰性加载): None = 尚未加载; _seen_file 用于识别月份切换/文件变化
        self._seen: Optional[set] = None
        self._seen_file: Optional[str] = None
        self._index_lock = threading.Lock()

    @property
    def strategy_version(self) -> str:
        """占位: 实际 version 由调用方按 params_snapshot 计算"""
        return ""

    def _resolve_file_path(self) -> str:
        """根据 source 决定存储路径

        live source 按月切分: strategy_memory_YYYY-MM.jsonl
        """
        if self.source == "backtest":
            backtest_dir = os.path.join(self._data_dir, "backtest_memory")
            os.makedirs(backtest_dir, exist_ok=True)
            return os.path.join(backtest_dir, f"{self.run_id}.jsonl")
        else:
            os.makedirs(self._data_dir, exist_ok=True)
            return self._get_monthly_file_path(datetime.now())

    def _get_monthly_file_path(self, dt: datetime) -> str:
        """按月切分的 live memory 文件路径: strategy_memory_YYYY-MM.jsonl"""
        month_str = dt.strftime("%Y-%m")
        return os.path.join(self._data_dir, f"strategy_memory_{month_str}.jsonl")

    # ── 判重索引 ──

    def _seen_keys(self) -> set:
        """获取已落盘信号键集合 (惰性加载, 文件变化时重建)

        索引从文件整体重建而非进程内累加: 后端每次热重载都是新进程,
        进程内累加的索引会随之丢失, 判重也就失效了。
        """
        if self._seen is None or self._seen_file != self._file_path:
            self._seen = self._load_keys(self._file_path)
            self._seen_file = self._file_path
        return self._seen

    @staticmethod
    def _load_keys(file_path: str) -> set:
        """读取 JSONL, 收集所有 signal 记录的去重键; 文件不存在/损坏行直接跳过"""
        keys = set()
        if not os.path.exists(file_path):
            return keys
        with open(file_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError:
                    logger.warning("记忆文件存在无法解析的行, 已跳过: %s", file_path)
                    continue
                if record.get("record_type") == "signal":
                    keys.add(signal_dedupe_key(record))
        return keys

    def record_signal(self, data: dict) -> bool:
        """记录一条信号记录

        data 应包含 (由 SignalEngine 构建):
          symbol, analysis_date, group, regime, level, score, confidence,
          executable, block_reason, strategy_version, params_snapshot,
          indicators, category_consensus, cat_scores, price_at_signal

        Returns:
            True = 已写入; False = 被去重或记忆层已禁用
        """
        if not self.enabled:
            return False
        record = {
            "record_type": "signal",
            "record_id": f"sig_{uuid.uuid4().hex[:12]}",
            "source": self.source,
            "run_id": self.run_id,
            "recorded_at": datetime.now().isoformat(timespec="seconds"),
            **data,
        }
        if self.dedupe:
            key = signal_dedupe_key(record)
            if not self._write_unique(record, key):
                logger.info("记忆层去重: 跳过重复信号 %s (run_id=%s)", key, self.run_id)
                return False
            return True
        self._append(record)
        return True

    def _write_unique(self, record: dict, key: tuple) -> bool:
        """键不存在则落盘并登记; 已存在返回 False

        先占位后写入: 写入抛异常时回滚占位, 避免"没写成功却被当成已存在"而永久丢数据。
        """
        with self._index_lock:
            seen = self._seen_keys()
            if key in seen:
                return False
            seen.add(key)
            try:
                self._append(record)
            except Exception:
                seen.discard(key)
                raise
            return True

    def record_outcome(self, data: dict) -> None:
        """记录一条结果记录

        data 应包含 (由 BacktestEngine 构建):
          signal_ref, signal_level_at_entry, signal_score_at_entry,
          symbol, entry_date, entry_price, exit_date, exit_price,
          shares, pnl, pnl_pct, holding_days, commission_total,
          exit_reason, exit_reason_detail, market_context_exit
        """
        if not self.enabled:
            return
        record = {
            "record_type": "outcome",
            "record_id": f"out_{uuid.uuid4().hex[:12]}",
            "source": self.source,
            "run_id": self.run_id,
            "recorded_at": datetime.now().isoformat(timespec="seconds"),
            **data,
        }
        self._append(record)

    def _append(self, record: dict) -> None:
        """追加写入一行 JSON (JSONL 格式)

        live source 按月切分: 进程跨月运行时自动切换到新月文件.
        """
        if self.source == "live":
            now = datetime.now()
            current_month_file = self._get_monthly_file_path(now)
            if current_month_file != self._file_path:
                logger.info("Live memory 按月切分: %s → %s",
                            os.path.basename(self._file_path),
                            os.path.basename(current_month_file))
                self._file_path = current_month_file

        line = json.dumps(record, ensure_ascii=False, default=str)
        with open(self._file_path, "a", encoding="utf-8") as f:
            f.write(line + "\n")

    @property
    def file_path(self) -> str:
        """当前记忆文件路径 (用于日志/展示)"""
        return self._file_path
