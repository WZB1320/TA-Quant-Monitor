"""周报生成器 — 从策略记忆层聚合数据, 调用 LLM 生成策略分析周报

数据流:
  记忆层 JSONL → 加载/关联 → 统计聚合 → 构建 Prompt → LLM → 周报 Markdown

用法:
  from src.ai import WeeklyReportGenerator
  gen = WeeklyReportGenerator()
  report = gen.generate("data/backtest_memory/bt_xxx.jsonl")
  print(report)
"""
import os
import re
import json
import glob
import logging
import threading
from collections import defaultdict
from datetime import datetime
from typing import Optional

from .llm_client import LLMClient
from .suggestion_tracker import SuggestionTracker

logger = logging.getLogger(__name__)

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
_BACKTEST_MEM_DIR = os.path.join(_PROJECT_ROOT, "data", "backtest_memory")
_REPORT_DIR = os.path.join(_PROJECT_ROOT, "data", "reports")
_REPORT_INDEX_FILE = os.path.join(_REPORT_DIR, "index.json")

# 保护周报索引文件并发写 (read-modify-write 操作需要互斥)
_report_index_lock = threading.Lock()


# ── System Prompt ──

SYSTEM_PROMPT = """你是一位专业的量化策略分析师，擅长解读交易数据、分析策略表现、提供改进建议。

你的任务：基于结构化的策略记忆数据，生成一份专业的策略周报。

报告结构（使用 Markdown）：

## 一、本周概览
- 数据区间、信号/交易总数、整体收益情况
- 一句话总结策略本周表现
- 若上下文中有"最近周报趋势对比", 简述本周相较前几周是改善、恶化还是震荡

## 二、盈亏归因分析
- 按信号等级分解：哪个级别的信号贡献了主要盈亏
- 按退出原因分解：止盈/止损/信号平仓各自的占比和效果
- 按 Regime 分解：不同市场体制下的表现差异

## 三、策略有效性分析
- 哪些参数组合/regime 下表现较好或较差
- 信号得分与实际收益的相关性
- 执行约束（冷却/去重）是否过度限制了交易机会
- 若有多个 strategy_version, 对比不同参数组合的表现差异, 指出哪个版本更优及可能原因

## 四、风险提示
- 连续亏损情况、胜率是否偏低
- 持仓时间是否异常
- 最大单笔亏损是否超出预期

## 五、改进建议
- 具体可执行的参数微调方向（如"建议将 score_threshold 从 25 调整至 30"）
- 每条建议需说明理由和预期效果
- 不涉及个股推荐
- 若上下文中有"上周建议回顾", 需评价上周建议的执行情况和效果
- 若上下文中有"最近周报趋势对比", 需结合趋势方向给出建议 (如连续恶化时应更保守)

要求：
1. 基于数据说话，不要编造数字
2. 分析要有深度，指出数字背后的可能原因
3. 建议要具体且可执行，但用词谨慎（用"建议考虑"而非"必须"）
4. 使用中文，Markdown 格式
5. 报告结尾声明："本报告由 AI 基于历史数据生成，仅供参考，不构成投资建议。"

结构化建议输出（供系统追踪, 不展示给用户）：
在报告正文之后, 用以下格式输出结构化建议. 若无建议则输出空数组.
每条建议必须包含 category/target/current_value/suggested_value/rationale 五个字段.

<<SUGGESTIONS>>
[
  {
    "category": "param_adjust",
    "target": "score_threshold",
    "current_value": 35,
    "suggested_value": 25,
    "rationale": "当前阈值过滤掉80%信号, 建议放宽以提高交易频次"
  }
]
<<END>>
"""


class WeeklyReportGenerator:
    """策略周报生成器

    读取记忆层 JSONL, 聚合统计, 调用 LLM 生成周报.
    """

    def __init__(self, llm: Optional[LLMClient] = None):
        self.llm = llm or LLMClient()

    def generate(self, memory_file: Optional[str] = None,
                 save: bool = True,
                 report_meta: Optional[dict] = None) -> str:
        """生成周报

        Args:
            memory_file: 记忆文件路径. None 则自动查找最新的回测记忆文件.
            save: 是否保存到 data/reports/ 目录
            report_meta: 周报元信息, 含 week_start/week_end/weeks_ago.
                         传入则启用反馈闭环 (加载历史建议 + 保存新建议).

        Returns:
            周报 Markdown 文本 (已移除结构化建议块)
        """
        self._last_suggestions_saved = 0

        # 1. 确定文件
        file_path = memory_file or self._find_latest_memory_file()
        if not file_path or not os.path.exists(file_path):
            raise FileNotFoundError(
                f"未找到记忆文件. 请先运行回测生成数据.\n"
                f"查找路径: {_BACKTEST_MEM_DIR}"
            )

        logger.info("加载记忆文件: %s", file_path)

        # 2. 加载并关联数据
        signals, outcomes = self._load_records(file_path)
        joined = self._join_signals_outcomes(signals, outcomes)

        # 3. 聚合统计
        stats = self._aggregate(file_path, signals, outcomes, joined)

        # 4. 反馈闭环: 检查历史建议是否已被应用 + 加载上下文
        suggestion_ctx = ""
        tracker = SuggestionTracker()
        all_signals_flat = []
        for run_id, sigs in signals.items():
            all_signals_flat.extend(sigs.values())

        if all_signals_flat:
            current_params = all_signals_flat[0].get("params_snapshot", {})
            current_version = all_signals_flat[0].get("strategy_version", "")
            # 检查 pending 建议是否已被应用 (params 已变更)
            newly_applied = tracker.check_applied(current_params, current_version)
            if newly_applied:
                logger.info("检测到 %d 条建议已被应用", len(newly_applied))

        # 加载历史建议上下文 (pending + applied)
        suggestion_ctx = self._build_suggestion_context(tracker)

        # 5. 构建数据摘要 (+ 历史建议上下文 + 历史趋势上下文)
        data_summary = self._build_data_summary(stats)
        if suggestion_ctx:
            data_summary += "\n" + suggestion_ctx

        # 注入最近 4 周趋势对比 (让 LLM 看到历史表现方向)
        historical_ctx = self._build_historical_context(max_weeks=4)
        if historical_ctx:
            data_summary += "\n" + historical_ctx

        # 6. 调用 LLM 生成周报
        if self.llm.is_available:
            logger.info("调用 LLM 生成周报 (model=%s)...", self.llm.model)
            try:
                raw_report = self.llm.chat(SYSTEM_PROMPT, data_summary)
            except Exception as e:
                logger.error("LLM 调用失败, 降级输出统计摘要: %s", e)
                raw_report = self._fallback_report(stats, str(e))
        else:
            logger.warning("LLM 未配置, 输出统计摘要 (不含 AI 分析)")
            raw_report = self._fallback_report(stats, "LLM 未配置, 请参考 .env.example 设置 DEEPSEEK_API_KEY")

        # 7. 解析结构化建议块 + 移除
        suggestions = self._extract_suggestions(raw_report)
        report = self._strip_suggestion_block(raw_report)

        # 8. 保存建议到追踪器
        if suggestions and report_meta:
            report_id = f"weekly_{report_meta.get('week_start', datetime.now().strftime('%Y%m%d'))}"
            saved_count = tracker.save_suggestions(
                suggestions, report_id,
                report_meta.get("week_start", ""),
                report_meta.get("week_end", ""),
            )
            self._last_suggestions_saved = saved_count
            logger.info("已保存 %d 条结构化建议", saved_count)

        # 9. 保存周报
        if save:
            saved_path = self._save_report(report, stats, report_meta)
            logger.info("周报已保存: %s", saved_path)

        return report

    def generate_stream(self, memory_file: Optional[str] = None,
                        save: bool = True,
                        report_meta: Optional[dict] = None):
        """流式生成周报, yield SSE 事件 dict

        与 generate() 共享前置处理 (加载/聚合/反馈闭环/历史趋势),
        但 LLM 调用改为流式输出, 逐块 yield 给前端.

        <<SUGGESTIONS>> 块实时检测: 发现标记后停止发送 delta,
        后续内容 (结构化建议) 仅内部累积用于解析, 不发给前端.

        Yields:
            {"type": "delta", "content": "..."}     — 文本块 (前端拼接显示)
            {"type": "done", "report_id": "...",    — 完成
             "suggestions_saved": N,
             "saved_to": "path"}
            {"type": "error", "message": "..."}     — 错误
        """
        self._last_suggestions_saved = 0

        try:
            # 1-3. 加载并聚合数据 (同 generate)
            file_path = memory_file or self._find_latest_memory_file()
            if not file_path or not os.path.exists(file_path):
                raise FileNotFoundError(
                    f"未找到记忆文件. 查找路径: {_BACKTEST_MEM_DIR}"
                )
            signals, outcomes = self._load_records(file_path)
            joined = self._join_signals_outcomes(signals, outcomes)
            stats = self._aggregate(file_path, signals, outcomes, joined)

            # 4. 反馈闭环 (同 generate)
            tracker = SuggestionTracker()
            all_signals_flat = []
            for run_id, sigs in signals.items():
                all_signals_flat.extend(sigs.values())
            if all_signals_flat:
                current_params = all_signals_flat[0].get("params_snapshot", {})
                current_version = all_signals_flat[0].get("strategy_version", "")
                newly_applied = tracker.check_applied(current_params, current_version)
                if newly_applied:
                    logger.info("检测到 %d 条建议已被应用", len(newly_applied))

            suggestion_ctx = self._build_suggestion_context(tracker)

            # 5. 构建数据摘要 (同 generate)
            data_summary = self._build_data_summary(stats)
            if suggestion_ctx:
                data_summary += "\n" + suggestion_ctx
            historical_ctx = self._build_historical_context(max_weeks=4)
            if historical_ctx:
                data_summary += "\n" + historical_ctx

            # 6. 流式调用 LLM, 实时检测 <<SUGGESTIONS>> 标记
            if self.llm.is_available:
                logger.info("流式调用 LLM (model=%s)...", self.llm.model)
                MARKER = "<<SUGGESTIONS>>"
                marker_len = len(MARKER)
                buffer = ""
                sent_end = 0  # buffer 中已发送的位置

                try:
                    for chunk in self.llm.stream_chat(SYSTEM_PROMPT, data_summary):
                        buffer += chunk
                        idx = buffer.find(MARKER)
                        if idx >= 0:
                            # 标记出现: 发送标记前的内容, 停止后续 delta
                            if idx > sent_end:
                                yield {"type": "delta", "content": buffer[sent_end:idx]}
                                sent_end = idx
                            # 不再 yield delta, 但内部继续累积
                        else:
                            # 标记未出现: 发送新内容, 保留安全余量 (防跨块)
                            safe_end = len(buffer) - marker_len
                            if safe_end > sent_end:
                                yield {"type": "delta", "content": buffer[sent_end:safe_end]}
                                sent_end = safe_end
                except RuntimeError as e:
                    yield {"type": "error", "message": str(e)}
                    return

                # 流结束: 发送标记前剩余内容 (或无标记时的全部剩余)
                idx = buffer.find(MARKER)
                if idx >= 0:
                    if idx > sent_end:
                        yield {"type": "delta", "content": buffer[sent_end:idx]}
                else:
                    if len(buffer) > sent_end:
                        yield {"type": "delta", "content": buffer[sent_end:]}

                raw_report = buffer
            else:
                # LLM 未配置: 降级输出统计摘要
                logger.warning("LLM 未配置, 输出统计摘要 (不含 AI 分析)")
                raw_report = self._fallback_report(
                    stats, "LLM 未配置, 请参考 .env.example 设置 DEEPSEEK_API_KEY")
                yield {"type": "delta", "content": raw_report}

            # 7-8. 解析建议 + 保存 (同 generate)
            suggestions = self._extract_suggestions(raw_report)
            report = self._strip_suggestion_block(raw_report)

            if suggestions and report_meta:
                report_id = f"weekly_{report_meta.get('week_start', datetime.now().strftime('%Y%m%d'))}"
                saved_count = tracker.save_suggestions(
                    suggestions, report_id,
                    report_meta.get("week_start", ""),
                    report_meta.get("week_end", ""),
                )
                self._last_suggestions_saved = saved_count
                logger.info("已保存 %d 条结构化建议", saved_count)

            # 9. 保存周报
            saved_to = None
            if save:
                saved_to = self._save_report(report, stats, report_meta)
                logger.info("周报已保存: %s", saved_to)

            yield {
                "type": "done",
                "suggestions_saved": self._last_suggestions_saved,
                "suggestions_count": len(suggestions),
                "saved_to": saved_to,
                "llm_available": self.llm.is_available,
            }

        except FileNotFoundError as e:
            yield {"type": "error", "message": str(e)}
        except Exception as e:
            logger.exception("流式生成周报失败")
            yield {"type": "error", "message": f"生成失败: {e}"}

    # ── 数据加载 ──

    def _find_latest_memory_file(self) -> Optional[str]:
        """查找最新的记忆文件

        优先查找回测记忆文件; 若无则查找 live memory 文件
        (含按月切分的 strategy_memory_YYYY-MM.jsonl + 旧格式).
        """
        files = glob.glob(os.path.join(_BACKTEST_MEM_DIR, "*.jsonl"))
        if not files:
            # 检查 live memory 文件 (按月切分 + 旧格式)
            live_files = glob.glob(os.path.join(
                _PROJECT_ROOT, "data", "strategy_memory_*.jsonl"))
            old_live = os.path.join(_PROJECT_ROOT, "data", "strategy_memory.jsonl")
            if os.path.exists(old_live):
                live_files.append(old_live)
            if not live_files:
                return None
            files = live_files
        return max(files, key=os.path.getmtime)

    def _load_records(self, file_path: str) -> tuple:
        """加载 JSONL, 返回 (signals_dict, outcomes_list)

        signals_dict: {run_id: {symbol_date_key: record}}
        outcomes_list: [record, ...]
        """
        signals = defaultdict(dict)
        outcomes = []

        with open(file_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                record = json.loads(line)
                if record["record_type"] == "signal":
                    key = f"{record['symbol']}_{record['analysis_date']}"
                    signals[record["run_id"]][key] = record
                elif record["record_type"] == "outcome":
                    outcomes.append(record)

        return signals, outcomes

    def _join_signals_outcomes(self, signals: dict, outcomes: list) -> list:
        """关联 outcome → signal, 返回合并记录列表"""
        joined = []
        for out in outcomes:
            ref = out.get("signal_ref", {})
            run_id = ref.get("run_id")
            key = f"{ref.get('symbol')}_{ref.get('analysis_date')}"
            sig = signals.get(run_id, {}).get(key)

            joined.append({
                "symbol": out["symbol"],
                "level": out.get("signal_level_at_entry") or (sig["level"] if sig else "unknown"),
                "score": out.get("signal_score_at_entry"),
                "regime": sig["regime"] if sig else "unknown",
                "group": sig["group"] if sig else None,
                "pnl": out["pnl"],
                "pnl_pct": out["pnl_pct"],
                "holding_days": out["holding_days"],
                "exit_reason": out["exit_reason"],
                "executable": sig["executable"] if sig else None,
                "strategy_version": sig["strategy_version"] if sig else "unknown",
            })
        return joined

    # ── 统计聚合 ──

    def _aggregate(self, file_path: str, signals: dict,
                   outcomes: list, joined: list) -> dict:
        """聚合所有统计数据"""
        all_signals = []
        for run_id, sigs in signals.items():
            all_signals.extend(sigs.values())

        # 数据概览
        analysis_dates = [s.get("analysis_date") for s in all_signals if s.get("analysis_date")]
        entry_dates = [o.get("entry_date") for o in outcomes if o.get("entry_date")]
        exit_dates = [o.get("exit_date") for o in outcomes if o.get("exit_date")]
        all_dates = analysis_dates + entry_dates + exit_dates
        all_dates_sorted = sorted(d for d in all_dates if d)

        stats = {
            "file_path": os.path.basename(file_path),
            "source": all_signals[0].get("source", "unknown") if all_signals else "unknown",
            "run_id": list(signals.keys())[0] if signals else "unknown",
            "date_range": (
                f"{all_dates_sorted[0]} ~ {all_dates_sorted[-1]}"
                if all_dates_sorted else "N/A"
            ),
            "signal_count": len(all_signals),
            "outcome_count": len(outcomes),
            "joined_count": len(joined),
            "raw_outcomes": outcomes,
        }

        # 信号统计
        stats["signal_by_level"] = self._group_count(all_signals, "level")
        stats["signal_by_regime"] = self._group_count(all_signals, "regime")
        stats["signal_by_group"] = self._group_count(all_signals, "group")

        # 执行状态
        exec_stats = {"executable": 0, "blocked": 0}
        block_reasons = defaultdict(int)
        for s in all_signals:
            if s.get("executable"):
                exec_stats["executable"] += 1
            else:
                exec_stats["blocked"] += 1
                reason = s.get("block_reason", "unknown")
                block_reasons[reason] += 1
        stats["exec_stats"] = exec_stats
        stats["block_reasons"] = dict(block_reasons)

        # 交易统计
        if joined:
            pnls = [t["pnl"] for t in joined]
            pnl_pcts = [t["pnl_pct"] for t in joined]
            holdings = [t["holding_days"] for t in joined if t["holding_days"] is not None]
            wins = sum(1 for p in pnls if p > 0)

            stats["trade_stats"] = {
                "total": len(joined),
                "wins": wins,
                "win_rate": wins / len(joined) * 100 if joined else 0,
                "avg_return_pct": sum(pnl_pcts) / len(pnl_pcts) * 100 if pnl_pcts else 0,
                "total_pnl": sum(pnls),
                "max_win": max(pnls) if pnls else 0,
                "max_loss": min(pnls) if pnls else 0,
                "avg_holding": sum(holdings) / len(holdings) if holdings else 0,
            }
        else:
            stats["trade_stats"] = None

        # 交叉分析
        stats["by_level"] = self._trade_stats_by_group(joined, "level")
        stats["by_exit_reason"] = self._trade_stats_by_group(joined, "exit_reason")
        stats["by_regime"] = self._trade_stats_by_group(joined, "regime")

        # 策略版本
        versions = defaultdict(int)
        for s in all_signals:
            v = s.get("strategy_version", "unknown")
            versions[v] += 1
        stats["strategy_versions"] = dict(versions)

        # 版本对比: 按 strategy_version 分组统计交易绩效, 用于回答"调参后效果如何"
        stats["version_performance"] = self._stats_by_version(joined)

        # 参数摘要 (取第一条信号的 params_snapshot)
        if all_signals:
            params = all_signals[0].get("params_snapshot", {})
            exec_params = params.get("execution_params", {})
            ind_params = params.get("indicator_params", {})
            stats["params_summary"] = {
                "ma_short": ind_params.get("ma_short"),
                "ma_long": ind_params.get("ma_long"),
                "score_threshold": exec_params.get("score_threshold"),
                "cooldown_days": exec_params.get("cooldown_days"),
                "max_consecutive_losses": exec_params.get("max_consecutive_losses"),
                "forced_regime": params.get("forced_regime"),
            }
        else:
            stats["params_summary"] = {}

        return stats

    @staticmethod
    def _group_count(records: list, key: str) -> dict:
        """按字段分组计数"""
        groups = defaultdict(int)
        for r in records:
            val = r.get(key) or "unknown"
            groups[val] += 1
        return dict(sorted(groups.items(), key=lambda x: -x[1]))

    @staticmethod
    def _trade_stats_by_group(records: list, key: str) -> list:
        """按指定字段分组统计交易"""
        groups = defaultdict(list)
        for r in records:
            groups[r.get(key, "unknown")].append(r)

        result = []
        for name, trades in sorted(groups.items()):
            n = len(trades)
            wins = sum(1 for t in trades if t["pnl"] > 0)
            avg_pct = sum(t["pnl_pct"] for t in trades) / n * 100
            avg_hold = sum(t["holding_days"] for t in trades if t["holding_days"]) / n
            result.append({
                "name": name,
                "count": n,
                "win_rate": wins / n * 100 if n else 0,
                "avg_return_pct": avg_pct,
                "avg_holding": avg_hold,
            })
        return result

    @staticmethod
    def _stats_by_version(joined: list) -> list:
        """按 strategy_version 分组统计交易绩效, 用于版本间对比

        当一次回测/实盘期间参数变更时, 会出现多个 strategy_version.
        此方法按版本切分, 输出每个版本的交易数/胜率/收益, 供 AI 诊断"调参效果".
        """
        groups = defaultdict(list)
        for r in joined:
            v = r.get("strategy_version", "unknown")
            groups[v].append(r)

        result = []
        for version, trades in groups.items():
            n = len(trades)
            wins = sum(1 for t in trades if t["pnl"] > 0)
            pnl_pcts = [t["pnl_pct"] for t in trades]
            pnls = [t["pnl"] for t in trades]
            result.append({
                "version": version,
                "count": n,
                "win_rate": wins / n * 100 if n else 0,
                "avg_return_pct": sum(pnl_pcts) / n * 100 if n else 0,
                "total_pnl": sum(pnls),
            })
        # 按交易数降序
        result.sort(key=lambda x: -x["count"])
        return result

    # ── Prompt 构建 ──

    def _build_data_summary(self, stats: dict) -> str:
        """构建喂给 LLM 的数据摘要文本"""
        lines = ["【策略记忆数据汇总】", ""]

        # 概览
        lines.append(f"数据来源: {stats['source']}")
        lines.append(f"数据文件: {stats['file_path']}")
        lines.append(f"Run ID: {stats['run_id']}")
        lines.append(f"数据区间: {stats['date_range']}")
        lines.append(f"信号记录: {stats['signal_count']} 条")
        lines.append(f"结果记录: {stats['outcome_count']} 条")
        lines.append(f"交易关联: {stats['joined_count']} 笔")
        lines.append("")

        # 信号统计
        lines.append("【信号统计】")
        lines.append(f"- 按等级: {self._fmt_dict(stats['signal_by_level'])}")
        lines.append(f"- 按 Regime: {self._fmt_dict(stats['signal_by_regime'])}")
        lines.append(f"- 按分组: {self._fmt_dict(stats['signal_by_group'])}")
        lines.append(f"- 可执行: {stats['exec_stats']['executable']}, "
                      f"被拦截: {stats['exec_stats']['blocked']}")
        if stats["block_reasons"]:
            lines.append(f"- 拦截原因: {self._fmt_dict(stats['block_reasons'])}")
        lines.append("")

        # 交易统计
        ts = stats.get("trade_stats")
        if ts:
            lines.append("【交易统计】")
            lines.append(f"- 总交易: {ts['total']} 笔")
            lines.append(f"- 胜率: {ts['win_rate']:.1f}%")
            lines.append(f"- 平均收益: {ts['avg_return_pct']:+.2f}%")
            lines.append(f"- 总盈亏: {ts['total_pnl']:+,.2f}")
            lines.append(f"- 最大盈利: {ts['max_win']:+,.2f}")
            lines.append(f"- 最大亏损: {ts['max_loss']:+,.2f}")
            lines.append(f"- 平均持仓: {ts['avg_holding']:.0f} 天")
            lines.append("")
        else:
            lines.append("【交易统计】无交易记录\n")

        # 交叉分析表
        lines.append("【按信号等级分解】")
        lines.append(self._fmt_trade_table(stats["by_level"]))
        lines.append("")

        lines.append("【按退出原因分解】")
        lines.append(self._fmt_trade_table(stats["by_exit_reason"]))
        lines.append("")

        lines.append("【按 Regime 分解】")
        lines.append(self._fmt_trade_table(stats["by_regime"]))
        lines.append("")

        # 策略版本
        lines.append("【策略版本】")
        lines.append(f"- 版本数: {len(stats['strategy_versions'])}")
        for ver, cnt in stats["strategy_versions"].items():
            lines.append(f"  {ver}: {cnt} 条信号")
        lines.append("")

        # 版本对比表 (仅当存在多个版本时输出, 单版本时省略)
        version_perf = stats.get("version_performance", [])
        if len(version_perf) > 1:
            lines.append("【版本对比 (不同参数组合的交易表现)】")
            lines.append(self._fmt_version_table(version_perf))
            lines.append("")

        # 超额收益提示 (benchmark_5d_return 非空时)
        bench_count = sum(
            1 for o in stats.get("raw_outcomes", [])
            if o.get("market_context_exit", {}).get("benchmark_5d_return") is not None
        )
        if bench_count > 0:
            lines.append(f"【超额收益数据】{bench_count} 笔交易有 benchmark 5日收益数据, 可做超额分析")
            lines.append("")

        # 参数摘要
        ps = stats.get("params_summary", {})
        if ps:
            lines.append("【参数快照摘要】")
            lines.append(f"- MA周期: {ps.get('ma_short')}/{ps.get('ma_long')}")
            lines.append(f"- 得分阈值: {ps.get('score_threshold')}")
            lines.append(f"- 冷却天数: {ps.get('cooldown_days')}")
            lines.append(f"- 连亏暂停: {ps.get('max_consecutive_losses')}次")
            lines.append(f"- 强制Regime: {ps.get('forced_regime') or 'auto'}")
            lines.append("")

        lines.append("请基于以上数据生成策略周报。")

        return "\n".join(lines)

    @staticmethod
    def _fmt_dict(d: dict) -> str:
        """格式化字典为 "key: val, key: val" 字符串"""
        return ", ".join(f"{k}: {v}" for k, v in d.items())

    @staticmethod
    def _fmt_trade_table(rows: list) -> str:
        """格式化交易统计为 Markdown 表格"""
        if not rows:
            return "(无数据)"
        lines = ["| 名称 | 笔数 | 胜率 | 平均收益 | 均持仓 |",
                 "|------|------|------|----------|--------|"]
        for r in rows:
            lines.append(
                f"| {r['name']} | {r['count']} | {r['win_rate']:.0f}% | "
                f"{r['avg_return_pct']:+.2f}% | {r['avg_holding']:.0f}d |"
            )
        return "\n".join(lines)

    @staticmethod
    def _fmt_version_table(rows: list) -> str:
        """格式化版本对比为 Markdown 表格"""
        if not rows:
            return "(无版本数据)"
        lines = ["| 版本 | 交易数 | 胜率 | 平均收益 | 总盈亏 |",
                 "|------|--------|------|----------|--------|"]
        for r in rows:
            lines.append(
                f"| {r['version']} | {r['count']} | {r['win_rate']:.0f}% | "
                f"{r['avg_return_pct']:+.2f}% | {r['total_pnl']:+,.2f} |"
            )
        return "\n".join(lines)

    # ── 反馈闭环: 建议解析与上下文构建 ──

    def _build_suggestion_context(self, tracker: SuggestionTracker) -> str:
        """构建历史建议上下文, 注入 LLM prompt

        让 LLM 能看到:
        - pending 建议: 上次建议了什么, 是否仍未应用
        - applied 建议: 已应用的建议, 应用前后的效果对比 (若有)
        """
        pending = tracker.load_pending_suggestions()
        applied = tracker.load_applied_suggestions()

        if not pending and not applied:
            return ""

        lines = ["【上周建议回顾】"]

        if pending:
            lines.append(f"\n未应用的建议 ({len(pending)} 条):")
            for i, s in enumerate(pending[-5:], 1):  # 最近 5 条
                target = s.get("target", "?")
                curr = s.get("current_value", "?")
                sug = s.get("suggested_value", "?")
                rationale = s.get("rationale", "")[:80]
                lines.append(f"  {i}. [{target}] {curr} → {sug}")
                lines.append(f"     理由: {rationale}")
                lines.append(f"     状态: 未应用 (建议可能仍有效)")

        if applied:
            lines.append(f"\n已应用的建议 ({len(applied)} 条):")
            for i, s in enumerate(applied[-5:], 1):
                target = s.get("target", "?")
                curr = s.get("current_value", "?")
                sug = s.get("suggested_value", "?")
                applied_at = s.get("applied_at", "?")[:10]
                new_ver = s.get("applied_version", "?")
                validation = s.get("validation")
                lines.append(f"  {i}. [{target}] {curr} → {sug}")
                lines.append(f"     应用时间: {applied_at}, 新版本: {new_ver}")
                if validation:
                    verdict = validation.get("verdict", "?")
                    lines.append(f"     验证结果: {verdict}")
                else:
                    lines.append(f"     状态: 已应用, 待验证效果")

        lines.append('\n请在「改进建议」部分评价上述建议的执行情况和效果。')
        return "\n".join(lines)

    @staticmethod
    def _extract_suggestions(raw_report: str) -> list:
        """从 LLM 输出中解析 <<SUGGESTIONS>>...<<END>> 块

        Returns:
            建议字典列表, 解析失败则返回空列表
        """
        pattern = r'<<SUGGESTIONS>>\s*(.*?)\s*<<END>>'
        match = re.search(pattern, raw_report, re.DOTALL)
        if not match:
            logger.debug("未找到 SUGGESTIONS 块")
            return []

        json_str = match.group(1).strip()
        try:
            suggestions = json.loads(json_str)
            if isinstance(suggestions, list):
                logger.info("解析到 %d 条结构化建议", len(suggestions))
                return suggestions
        except json.JSONDecodeError as e:
            logger.warning("SUGGESTIONS 块 JSON 解析失败: %s", e)

        return []

    @staticmethod
    def _strip_suggestion_block(raw_report: str) -> str:
        """从报告中移除 <<SUGGESTIONS>>...<<END>> 块 (不展示给用户)"""
        pattern = r'<<SUGGESTIONS>>\s*.*?\s*<<END>>'
        cleaned = re.sub(pattern, '', raw_report, flags=re.DOTALL)
        # 清理尾部多余空行
        return cleaned.rstrip() + '\n'

    # ── 历史趋势上下文 ──

    @staticmethod
    def _load_report_index() -> list:
        """加载周报索引文件, 不存在或损坏则返回空列表"""
        if not os.path.exists(_REPORT_INDEX_FILE):
            return []
        try:
            with open(_REPORT_INDEX_FILE, "r", encoding="utf-8") as f:
                index = json.load(f)
                return index if isinstance(index, list) else []
        except (json.JSONDecodeError, OSError):
            return []

    def _build_historical_context(self, max_weeks: int = 4) -> str:
        """构建最近 N 周的趋势对比上下文, 注入 LLM prompt

        从周报索引中读取最近 max_weeks 篇报告的关键指标,
        让 LLM 能看到策略的历史表现趋势 (改善/恶化/震荡),
        从而给出更有深度的分析.

        Returns:
            历史趋势文本, 无历史数据时返回空字符串
        """
        index = self._load_report_index()
        if not index:
            return ""

        # 按 generated_at 降序, 取最近 max_weeks 条
        index.sort(key=lambda r: r.get("generated_at", ""), reverse=True)
        recent = index[:max_weeks]

        # 按时间正序排列 (最早的在前, 最近的在后, 便于看趋势)
        recent.reverse()

        lines = ["【最近周报趋势对比】"]
        lines.append("")
        lines.append("| 周区间 | 信号数 | 交易数 | 胜率 | 平均收益 | 总盈亏 | 建议数 |")
        lines.append("|--------|--------|--------|------|----------|--------|--------|")

        for r in recent:
            # 周区间: 优先用 week_start~week_end, 否则用 data_range
            ws = r.get("week_start")
            we = r.get("week_end")
            if ws and we:
                week_label = f"{ws[5:]}~{we[5:]}"  # 取 MM-DD 部分
            else:
                dr = r.get("data_range", "N/A")
                week_label = dr[:20] if dr else "N/A"

            sig = r.get("signal_count", 0)
            trades = r.get("trade_count", 0)
            wr = r.get("win_rate")
            wr_str = f"{wr:.0f}%" if wr is not None else "N/A"
            ar = r.get("avg_return_pct")
            ar_str = f"{ar:+.2f}%" if ar is not None else "N/A"
            pnl = r.get("total_pnl")
            pnl_str = f"{pnl:+,.0f}" if pnl is not None else "N/A"
            sug = r.get("suggestions_count", 0)

            lines.append(
                f"| {week_label} | {sig} | {trades} | {wr_str} | "
                f"{ar_str} | {pnl_str} | {sug} |"
            )

        lines.append("")
        lines.append("请结合上述历史趋势, 分析本周表现是改善、恶化还是震荡, "
                     "并在「改进建议」中考虑趋势方向.")

        return "\n".join(lines)

    # ── 降级输出 (LLM 不可用时) ──

    def _fallback_report(self, stats: dict, reason: str) -> str:
        """LLM 不可用时, 输出纯统计摘要"""
        lines = ["# 策略周报 (统计摘要)", ""]
        lines.append(f"> ⚠️ AI 分析不可用: {reason}")
        lines.append("> 以下为纯数据统计, 不含 AI 分析与建议。\n")

        lines.append(f"**数据文件**: {stats['file_path']}")
        lines.append(f"**数据区间**: {stats['date_range']}")
        lines.append(f"**信号/交易**: {stats['signal_count']} / {stats['joined_count']}")
        lines.append("")

        ts = stats.get("trade_stats")
        if ts:
            lines.append("## 交易统计\n")
            lines.append(f"- 胜率: {ts['win_rate']:.1f}%")
            lines.append(f"- 平均收益: {ts['avg_return_pct']:+.2f}%")
            lines.append(f"- 总盈亏: {ts['total_pnl']:+,.2f}")
            lines.append("")

        lines.append("## 按信号等级分解\n")
        lines.append(self._fmt_trade_table(stats["by_level"]))
        lines.append("")
        lines.append("## 按退出原因分解\n")
        lines.append(self._fmt_trade_table(stats["by_exit_reason"]))

        # 版本对比 (仅当存在多个版本时输出)
        version_perf = stats.get("version_performance", [])
        if len(version_perf) > 1:
            lines.append("\n## 版本对比\n")
            lines.append(self._fmt_version_table(version_perf))

        return "\n".join(lines)

    # ── 保存 ──

    def _save_report(self, report: str, stats: dict,
                     report_meta: Optional[dict] = None) -> str:
        """保存周报到 data/reports/ 目录, 并更新索引文件"""
        os.makedirs(_REPORT_DIR, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"weekly_{timestamp}.md"
        filepath = os.path.join(_REPORT_DIR, filename)

        header = f"<!-- 生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} -->\n"
        header += f"<!-- 数据源: {stats['file_path']} -->\n"
        header += f"<!-- 区间: {stats['date_range']} -->\n"
        if report_meta:
            header += f"<!-- 周区间: {report_meta.get('week_start', '?')} ~ {report_meta.get('week_end', '?')} -->\n"
            header += f"<!-- weeks_ago: {report_meta.get('weeks_ago', 0)} -->\n"
        header += "\n"

        with open(filepath, "w", encoding="utf-8") as f:
            f.write(header + report)

        # 更新索引
        self._update_report_index(filename, timestamp, stats, report_meta)

        return filepath

    def _update_report_index(self, filename: str, timestamp: str,
                            stats: dict, report_meta: Optional[dict]) -> None:
        """追加一条周报元信息到索引文件 data/reports/index.json

        使用模块级 threading.Lock 保护 read-modify-write 操作,
        避免多线程并发写索引时丢失条目.
        """
        # 构建索引条目 (锁外构建, 减少锁持有时间)
        filepath = os.path.join(_REPORT_DIR, filename)
        file_size = os.path.getsize(filepath) if os.path.exists(filepath) else 0

        ts = stats.get("trade_stats")
        entry = {
            "report_id": f"weekly_{timestamp}",
            "filename": filename,
            "generated_at": datetime.now().isoformat(timespec="seconds"),
            "week_start": report_meta.get("week_start") if report_meta else None,
            "week_end": report_meta.get("week_end") if report_meta else None,
            "data_range": stats.get("date_range"),
            "source": stats.get("source"),
            "run_id": stats.get("run_id"),
            "signal_count": stats.get("signal_count", 0),
            "outcome_count": stats.get("outcome_count", 0),
            "trade_count": stats.get("joined_count", 0),
            "win_rate": ts["win_rate"] if ts else None,
            "avg_return_pct": ts["avg_return_pct"] if ts else None,
            "total_pnl": ts["total_pnl"] if ts else None,
            "suggestions_count": getattr(self, "_last_suggestions_saved", 0),
            "llm_available": self.llm.is_available,
            "file_size": file_size,
        }

        # 锁内执行 read-modify-write
        with _report_index_lock:
            index = []
            if os.path.exists(_REPORT_INDEX_FILE):
                try:
                    with open(_REPORT_INDEX_FILE, "r", encoding="utf-8") as f:
                        index = json.load(f)
                        if not isinstance(index, list):
                            index = []
                except (json.JSONDecodeError, OSError):
                    index = []

            index.append(entry)

            with open(_REPORT_INDEX_FILE, "w", encoding="utf-8") as f:
                json.dump(index, f, ensure_ascii=False, indent=2)

        logger.info("周报索引已更新: %s (共 %d 条)", filename, len(index))
