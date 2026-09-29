"""验证实盘记忆层幂等 — 真实信号分析跑两遍, 第二次不应追加任何记忆记录

为什么需要这个脚本:
  单元测试证明的是 StrategyMemory 自己的行为; 而历史缺陷出在"实时分析路径反复触发"
  这条真实链路上 (SignalAnalyzer → SignalEngine.analyze_batch → record_signal)。
  本脚本直接跑真实分析, 用"文件行数是否增长"作为判据。

判据:
  1. 两次分析的信号结果完全一致 (否则中途数据变了, 判据不成立)
  2. 第二次分析新增记录数 == 0
  第一次分析的增量是"确有新增的事实", 属正常写入, 不作为判据。

注意:
  - 以 BACKTEST 模式运行, 信号过滤器走纯内存历史, 不写 data/signal_history.json
    (与 dashboard_builder 生成大屏时的做法一致)

用法:
  python scripts/verify_memory_idempotency.py
"""
import json
import os
import sys
import warnings

warnings.filterwarnings("ignore")

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(ROOT)
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "web", "backend"))

from src.config.runtime_mode import set_mode, RuntimeMode  # noqa: E402
set_mode(RuntimeMode.BACKTEST)

from src.memory import StrategyMemory  # noqa: E402
from services.signal_analyzer import SignalAnalyzer  # noqa: E402


def _count_lines(path: str) -> int:
    if not os.path.exists(path):
        return 0
    with open(path, "r", encoding="utf-8") as f:
        return sum(1 for line in f if line.strip())


def _last_record(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        lines = [ln for ln in f if ln.strip()]
    return json.loads(lines[-1])


def _signals(res: dict) -> dict:
    return {r["code"]: (r["level"], r["score"]) for r in res["results"]}


def main() -> int:
    analyzer = SignalAnalyzer()
    mem = analyzer._memory
    path = mem.file_path

    print(f"记忆文件: {path}")
    print(f"本次 run_id: {mem.run_id}\n")

    n0 = _count_lines(path)
    print(f"[起点] 记录数 {n0}")

    res1 = analyzer.run_analysis(group="", user_regime="auto")
    assert res1.get("status") == "done", f"第一次分析失败: {res1.get('message')}"
    n1 = _count_lines(path)
    print(f"[第 1 次分析] {res1['summary']} → 记录数 {n1} (新增 {n1 - n0}, 新增即确有新增的事实)")

    res2 = analyzer.run_analysis(group="", user_regime="auto")
    assert res2.get("status") == "done", f"第二次分析失败: {res2.get('message')}"
    n2 = _count_lines(path)
    print(f"[第 2 次分析] {res2['summary']} → 记录数 {n2} (新增 {n2 - n1})")

    same_signals = _signals(res1) == _signals(res2)
    print(f"[一致性] 两次信号结果{'完全一致' if same_signals else '不一致 —— 数据中途变化, 判据不成立'}"
          f" (标的数 {len(res1['results'])}/{len(res2['results'])})")

    # 新实例模拟"后端重启"(uvicorn 热重载 / 手工重跑): 索引必须从磁盘重建才判得出来
    fresh = StrategyMemory(source="live", run_id="verify_probe")
    probe = _last_record(path)
    wrote = fresh.record_signal({
        k: probe.get(k) for k in ("symbol", "analysis_date", "level", "strategy_version")
    } | {"score": 0.0})
    n3 = _count_lines(fresh.file_path)
    print(f"[重启模拟] 新实例重复写入最后一条记录 {probe.get('symbol')}/{probe.get('analysis_date')}"
          f"/{probe.get('level')} → {'仍被写入(判重失效!)' if wrote else '已被判重跳过'}"
          f", 记录数 {n3}")

    print()
    ok = same_signals and (n2 - n1) == 0 and not wrote and n3 == n2
    print("结论:", "幂等成立 —— 同一交易日的重复分析不再追加记忆记录"
          if ok else "未通过, 修复不生效")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
