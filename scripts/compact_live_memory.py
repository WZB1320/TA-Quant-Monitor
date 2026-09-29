"""压缩实盘记忆文件 — 回填交易日 + 去除重复

背景 (实测数据 2026-07 ~ 2026-09):
  live 记忆层此前没有幂等写入, 每次重新分析 / 后端热重载 (uvicorn --reload) 都会追加
  一份全新 run 的完整信号快照; 且实时分析路径不传 analysis_date, 记录里该字段恒为 null。
  实测: 7 月 392 条来自 10 个写入日 (其中 3 天多次生成), 8 月 383/18 天, 9 月 250/8 天。
  冗余全部来自"同一写入日的多次生成", 单次生成的写入日零冗余。

本脚本做两件事:

  1. 回填 analysis_date —— price_at_signal 就是"分析数据最后一根的收盘价", 是交易日的指纹
     取该标的 [写入日 - 7天, 写入日] 区间内 且收盘价相等的最大日期 (容差 0.005 ≈ A股最小
     价差的一半, 等价于"同价位")。语义上正是"那天能取到的最新一根"。
     窗口是必需的: 缓存覆盖 2021 至今, 不加窗口时一个 2026-09 的记录会因为同价位而匹配到
     2025-01 的历史交易日 (实测 9 月文件出现 6 条 2024/2025 年的错误日期)。
     窗口外匹配不到、或同窗口内无匹配 → 保留 null, 不猜。

  2. 按"整条记录指纹"去重 —— 除 record_id/run_id/recorded_at/analysis_date 外全部字段
     (含完整指标快照) 逐字段一致, 才判定为同一条事实的副本, 保留首次出现的记录。
     不用"价格+级别+版本"做分组键: 实测 8 月有 5 条 / 9 月有 1 条记录属于
     "不同交易日恰好同价位", 只按价格分组会把这些真事实当副本删掉。

用法:
  python scripts/compact_live_memory.py                       # 预演, 只打印统计 (默认)
  python scripts/compact_live_memory.py --apply               # 落盘, 原文件备份为 <file>.bak
  python scripts/compact_live_memory.py --no-backfill-date    # 只去重, 不推断日期
  python scripts/compact_live_memory.py data/strategy_memory_2026-08.jsonl --apply
"""
import argparse
import collections
import json
import os
import shutil
import sqlite3
import sys
from datetime import datetime, timedelta
from typing import Optional

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from src.memory import find_live_memory_files, signal_dedupe_key  # noqa: E402

# A 股最小价差 0.01, 取半价差作容差 = 同一个价位
_PRICE_TOL = 0.005
# 分析日与写入日的最大允许间隔: 正常是 0~1 个交易日 (隔夜/节假日最多几天);
# 超过这个窗口的"同价位"只是历史巧合, 不能当成交交易日
_MAX_BACKFILL_LAG_DAYS = 7
_KLINE_DB = os.path.join(_ROOT, "data", "kline_cache.db")

# 指纹忽略的字段: 这四项是"这一次写入"的元信息, 不构成事实本身
_VOLATILE_FIELDS = ("record_id", "run_id", "recorded_at", "analysis_date", "source")


def load_close_index(db_path: str = _KLINE_DB) -> dict:
    """读取 K 线缓存 → {symbol: [(date, close), ...] 按日期升序}"""
    if not os.path.exists(db_path):
        raise FileNotFoundError(f"K 线缓存不存在, 无法回填日期: {db_path}")
    con = sqlite3.connect(db_path)
    try:
        rows = con.execute("SELECT symbol, date, close FROM kline_cache").fetchall()
    finally:
        con.close()

    index = collections.defaultdict(list)
    for symbol, date, close in rows:
        index[symbol].append((str(date)[:10], float(close)))
    for series in index.values():
        series.sort()
    return index


def content_fingerprint(record: dict) -> str:
    """整条记录指纹 (忽略写入元字段) — 同指纹 = 同一条事实的副本"""
    payload = {k: v for k, v in record.items() if k not in _VOLATILE_FIELDS}
    return json.dumps(payload, sort_keys=True, ensure_ascii=False, default=str)


def infer_trade_date(record: dict, close_index: dict,
                     not_after: Optional[str] = None,
                     max_lag_days: int = _MAX_BACKFILL_LAG_DAYS):
    """用 price_at_signal 反查交易日

    只在 [not_after - max_lag_days, not_after] 窗口内匹配: 分析日不可能比写入日早太多,
    窗口外的同价位是历史巧合。

    Returns:
        (交易日 or None, 是否命中窗口内最后一根 —— 即数据管线当时是否最新)
    """
    price = record.get("price_at_signal")
    symbol = record.get("symbol")
    if price is None or not symbol or not not_after:
        return None, False
    try:
        lower = (datetime.strptime(not_after, "%Y-%m-%d")
                 - timedelta(days=max_lag_days)).strftime("%Y-%m-%d")
    except ValueError:
        return None, False

    series = [(d, c) for d, c in close_index.get(symbol, [])
              if lower <= d <= not_after]
    if not series:
        return None, False

    matched = [d for d, c in series if abs(c - float(price)) <= _PRICE_TOL]
    if not matched:
        return None, False
    trade_date = matched[-1]
    return trade_date, trade_date == series[-1][0]


def compact_records(records: list, close_index: dict, backfill: bool = True) -> tuple:
    """返回 (压缩后记录列表, 统计字典)"""
    # 1. 按整条记录指纹分组 (保持首次出现顺序)
    groups = collections.OrderedDict()
    passthrough = []
    for rec in records:
        if rec.get("record_type") != "signal":
            # 结果记录不参与压缩 (同一标的可多次进出场), 原样保留
            passthrough.append(rec)
            continue
        groups.setdefault(content_fingerprint(rec), []).append(rec)

    stat = collections.Counter()
    stat["records_in"] = len(records)
    stat["facts"] = len(groups)

    kept = []
    for members in groups.values():
        stat["duplicates_removed"] += len(members) - 1
        rec = dict(members[0])  # 保留首次出现的记录 (最接近信号当日)

        if rec.get("analysis_date"):
            stat["date_already_present"] += 1
        elif not backfill:
            stat["date_unresolved"] += 1
        else:
            # 上界取组内"最早一次写入时间": 与是哪一次重跑无关, 结果稳定
            earliest_at = min((m.get("recorded_at") or "") for m in members)[:10]
            trade_date, was_current = infer_trade_date(rec, close_index, earliest_at)
            if trade_date:
                rec["analysis_date"] = trade_date
                stat["date_backfilled"] += 1
                stat["date_hit_latest_bar"] += 1 if was_current else 0
            else:
                stat["date_unresolved"] += 1

        kept.append(rec)

    # 2. 日期回填可能把两条真事实推成同一个去重键 (不同交易日恰好同价位):
    #    保留最早一次写入的那条日期, 其余置空 —— 宁可缺日期, 不可错日期
    #    置空会改变键本身, 因此迭代到不动点 (带纪元无日期的记录键为 rec:写入日, 不会误撞)
    for _ in range(5):
        by_key = collections.defaultdict(list)
        for rec in kept:
            by_key[signal_dedupe_key(rec)].append(rec)
        nulled = 0
        for rows in by_key.values():
            if len(rows) < 2:
                continue
            rows.sort(key=lambda r: r.get("recorded_at") or "")
            for extra in rows[1:]:
                if extra.get("analysis_date"):
                    extra["analysis_date"] = None
                    nulled += 1
        stat["date_nulled_on_conflict"] += nulled
        if not nulled:
            break

    kept.extend(passthrough)
    stat["records_out"] = len(kept)

    # 3. 自证: 压缩后去重键必须唯一
    keys = [signal_dedupe_key(r) for r in kept if r.get("record_type") == "signal"]
    stat["residual_duplicate_keys"] = len(keys) - len(set(keys))
    return kept, dict(stat)


def process_file(path: str, close_index: dict, apply: bool = False,
                 backfill: bool = True) -> dict:
    """处理单个文件; apply=False 时不写盘"""
    records = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))

    kept, stat = compact_records(records, close_index, backfill=backfill)
    stat["file"] = os.path.basename(path)

    if apply:
        shutil.copy2(path, path + ".bak")
        tmp = path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            for rec in kept:
                f.write(json.dumps(rec, ensure_ascii=False, default=str) + "\n")
        os.replace(tmp, path)
        stat["backup"] = os.path.basename(path) + ".bak"

    return stat


def main() -> int:
    parser = argparse.ArgumentParser(description="压缩实盘记忆文件 (回填交易日 + 去重)")
    parser.add_argument("files", nargs="*", help="目标文件; 缺省 = data/ 下所有 live 记忆文件")
    parser.add_argument("--apply", action="store_true", help="落盘 (默认只预演)")
    parser.add_argument("--no-backfill-date", action="store_true", help="只去重, 不推断日期")
    args = parser.parse_args()

    files = args.files or find_live_memory_files()
    if not files:
        print("未找到任何 live 记忆文件")
        return 0

    close_index = load_close_index() if not args.no_backfill_date else {}
    print(f"{'[预演] ' if not args.apply else '[落盘] '}待处理 {len(files)} 个文件"
          f"{' (含日期回填)' if close_index else ''}\n")

    total = collections.Counter()
    for path in files:
        stat = process_file(path, close_index, apply=args.apply,
                            backfill=not args.no_backfill_date)
        total.update({k: v for k, v in stat.items() if isinstance(v, int)})
        print(f"{stat['file']}:")
        print(f"  记录 {stat['records_in']} → {stat['records_out']}"
              f"  (唯一事实 {stat['facts']} 条, 去除副本 {stat['duplicates_removed']} 条)")
        print(f"  日期: 回填 {stat.get('date_backfilled', 0)}"
              f" / 已有 {stat.get('date_already_present', 0)}"
              f" / 推断不出 {stat.get('date_unresolved', 0)}"
              f" / 歧义置空 {stat.get('date_nulled_on_conflict', 0)}")
        print(f"  压缩后残留重复键: {stat['residual_duplicate_keys']}")
        if stat.get("backup"):
            print(f"  备份: {stat['backup']}")
        print()

    print("合计:")
    print(f"  记录 {total['records_in']} → {total['records_out']}"
          f"  去除副本 {total['duplicates_removed']} 条"
          f" ({total['duplicates_removed'] / max(total['records_in'], 1) * 100:.1f}%)")
    print(f"  日期回填 {total['date_backfilled']} 条"
          f" (其中命中当日最新一根 {total['date_hit_latest_bar']} 条),"
          f" 推断不出 {total['date_unresolved']} 条, 歧义置空 {total['date_nulled_on_conflict']} 条")
    print(f"  残留重复键 {total['residual_duplicate_keys']}")
    if not args.apply:
        print("\n以上为预演结果, 未改动任何文件; 加 --apply 落盘 (自动备份 .bak)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
