"""
导出消费/医药/机械三个暂停组的详细交易记录
回测区间: 2026-01-01 ~ 2026-08-06
输出: 每组每笔交易明细 (开仓/平仓日期、信号、盈亏) + 止损触发统计

用法: python scripts/export_paused_groups_trades.py
"""
import sys
import os
import json
import shutil
import warnings
import pandas as pd
from datetime import datetime

warnings.filterwarnings("ignore")
project_root = os.path.join(os.path.dirname(__file__), "..")
sys.path.insert(0, project_root)

from src.config.runtime_mode import set_mode, RuntimeMode
set_mode(RuntimeMode.BACKTEST)

from src.backtest.engine import BacktestEngine
from src.data_fetcher import DataManager
from src.config.group_config import GroupConfig
from src.config.user_preferences import UserPreferences, _USER_PREF_FILE

BT_START, BT_END = "2026-01-01", "2026-08-06"
DATA_START, DATA_END = "2025-06-01", "2026-08-06"
BENCHMARK = "sh.000300"
TOTAL_CAPITAL = 1000000

REGIMES_CFG = {"周期资源型": {"trending"}}
ATR_OVERRIDE = {"科技成长型": 1.8}
DD_CONFIG = BacktestEngine.DEFAULT_DD_PROTECTION_CONFIG

PAUSED_GROUPS = ["消费稳健型", "医药创新型", "机械制造型"]

REPORT_MD = os.path.join(project_root, "data", "paused_groups_trade_details.md")
TRADES_CSV = os.path.join(project_root, "data", "paused_groups_trades.csv")


def load_watchlist_and_names():
    cfg_path = os.path.join(project_root, "config", "strategy_config.json")
    with open(cfg_path, "r", encoding="utf-8") as f:
        cfg = json.load(f)
    wl = {}
    names = {}
    for g, stocks in cfg["strategy_config"]["watchlist"].items():
        if g.startswith("_") or not isinstance(stocks, list):
            continue
        wl[g] = [s["code"] for s in stocks]
        for s in stocks:
            names[s["code"]] = s["name"]
    return wl, names


def run_group_and_get_trades(data_map, benchmark_df, group_codes, group_name,
                             start, end, dd_config):
    """跑单组回测, 返回交易记录列表"""
    GroupConfig._instance = None
    GroupConfig._config = None
    engine = BacktestEngine(
        initial_capital=TOTAL_CAPITAL, lookback_days=120, position_ratio=0.3,
        commission_rate=0.00025, stamp_tax=0.001, slippage=0.0001,
        signal_dedup_days=5, risk_per_trade=0.05, atr_stop_mult=2.0,
        forced_regime=None, trade_regimes=REGIMES_CFG.get(group_name),
        dd_protection_config=dd_config,
    )
    sub_map = {c: data_map[c] for c in group_codes if c in data_map}
    m = engine.run(sub_map, benchmark_df=benchmark_df, start_date=start, end_date=end)

    trades = []
    for t in engine.position_mgr.closed_trades:
        trades.append({
            "group": group_name,
            "symbol": t.symbol,
            "entry_date": str(t.entry_date),
            "entry_price": round(t.entry_price, 3),
            "exit_date": str(t.exit_date) if t.exit_date else "",
            "exit_price": round(t.exit_price, 3) if t.exit_price else 0,
            "exit_signal": t.exit_signal or "",
            "shares": t.shares,
            "pnl": round(t.pnl, 2),
            "pnl_pct": round(t.pnl_pct * 100, 2),
            "holding_days": t.holding_days,
        })
    return trades, m


def categorize_exit(signal):
    """分类平仓信号"""
    s = signal or ""
    if "回撤保护" in s:
        return "回撤保护降仓"
    if "止损" in s or "硬止损" in s or "ATR" in s:
        return "止损"
    if "止盈" in s or "移动" in s or "trailing" in s.lower():
        return "止盈"
    if "信号" in s or "卖出" in s or "看空" in s:
        return "信号退出"
    return "其他"


def main():
    run_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print("=" * 95)
    print(f"  导出暂停组(消费/医药/机械)详细交易记录")
    print(f"  回测区间: {BT_START} ~ {BT_END}")
    print("=" * 95)

    pref_existed = os.path.exists(_USER_PREF_FILE)
    pref_backup = _USER_PREF_FILE + ".trades_bak"
    if pref_existed:
        shutil.copy2(_USER_PREF_FILE, pref_backup)
    UserPreferences().clear_all()

    try:
        watchlist, names = load_watchlist_and_names()
        dm = DataManager()
        print("\n拉取数据...")
        all_codes = [c for codes in watchlist.values() for c in codes]
        data_map = {}
        for code in all_codes:
            df = dm.get_daily_kline(code, start_date=DATA_START, end_date=DATA_END)
            if df is not None and len(df) > 80:
                data_map[code] = df
        benchmark_df = dm.get_daily_kline(BENCHMARK, start_date=DATA_START, end_date=DATA_END)
        print(f"  股票 {len(data_map)}/{len(all_codes)}, 基准 {len(benchmark_df)}条")

        all_trades = []
        group_summaries = {}

        for g in PAUSED_GROUPS:
            g_codes = [c for c in watchlist.get(g, []) if c in data_map]
            if len(g_codes) < 2:
                print(f"  {g}: 股票不足, 跳过")
                continue
            print(f"\n跑 {g} 回测...")
            trades, m = run_group_and_get_trades(data_map, benchmark_df, g_codes, g,
                                                 BT_START, BT_END, DD_CONFIG)
            # 补充股票名称
            for t in trades:
                t["name"] = names.get(t["symbol"], "")
                t["exit_category"] = categorize_exit(t["exit_signal"])
            all_trades.extend(trades)
            group_summaries[g] = {
                "trade_count": len(trades),
                "total_pnl": round(sum(t["pnl"] for t in trades), 2),
                "win_count": sum(1 for t in trades if t["pnl"] > 0),
                "loss_count": sum(1 for t in trades if t["pnl"] <= 0),
                "trades": trades,
            }
            print(f"  {g}: {len(trades)}笔交易, 总盈亏{sum(t['pnl'] for t in trades):+.2f}, "
                  f"盈{sum(1 for t in trades if t['pnl'] > 0)}笔/亏{sum(1 for t in trades if t['pnl'] <= 0)}笔")

        # 导出 CSV
        if all_trades:
            df = pd.DataFrame(all_trades)
            df = df[["group", "symbol", "name", "entry_date", "entry_price",
                     "exit_date", "exit_price", "exit_signal", "exit_category",
                     "shares", "pnl", "pnl_pct", "holding_days"]]
            df = df.sort_values(["group", "entry_date"])
            df.to_csv(TRADES_CSV, index=False, encoding="utf-8-sig")
            print(f"\n✓ CSV → {TRADES_CSV}")

        # 生成 Markdown 报告
        report = generate_report(run_time, group_summaries, names)
        with open(REPORT_MD, "w", encoding="utf-8") as f:
            f.write(report)
        print(f"✓ 报告 → {REPORT_MD}")

        # 打印汇总
        print(f"\n{'='*95}")
        print(f"  暂停组交易记录汇总")
        print(f"{'='*95}")
        for g, s in group_summaries.items():
            print(f"\n  [{g}] {s['trade_count']}笔, 总盈亏{s['total_pnl']:+.2f}, "
                  f"盈{s['win_count']}/亏{s['loss_count']}")
            # 按平仓类别统计
            cat_stats = {}
            for t in s["trades"]:
                cat = t["exit_category"]
                if cat not in cat_stats:
                    cat_stats[cat] = {"count": 0, "pnl": 0}
                cat_stats[cat]["count"] += 1
                cat_stats[cat]["pnl"] += t["pnl"]
            print(f"    平仓类别统计:")
            for cat, cs in sorted(cat_stats.items(), key=lambda x: x[1]["count"], reverse=True):
                print(f"      {cat:10s}: {cs['count']}笔, 盈亏{cs['pnl']:+.2f}")

            # 列出所有交易
            print(f"\n    {'股票':<10} {'开仓日':>12} {'平仓日':>12} {'平仓信号':<22} {'盈亏%':>7} {'天数':>4}")
            print(f"    {'-'*80}")
            for t in sorted(s["trades"], key=lambda x: x["entry_date"]):
                print(f"    {t['name'][:4]:<10} {t['entry_date']:>12} {t['exit_date']:>12} "
                      f"{t['exit_signal'][:20]:<22} {t['pnl_pct']:>+7.2f} {t['holding_days']:>4}")

    finally:
        if pref_existed and os.path.exists(pref_backup):
            shutil.move(pref_backup, _USER_PREF_FILE)


def generate_report(run_time, group_summaries, names):
    L = []
    L.append(f"# 暂停组(消费/医药/机械)详细交易记录\n")
    L.append(f"**运行时间**: {run_time}")
    L.append(f"**回测区间**: {BT_START} ~ {BT_END}")
    L.append(f"**配置**: P5 (各组单独满仓回测, 资金{TOTAL_CAPITAL})\n")

    for g, s in group_summaries.items():
        L.append(f"## {g}\n")
        L.append(f"**交易笔数**: {s['trade_count']}笔 | **总盈亏**: {s['total_pnl']:+.2f} | "
                 f"**盈**: {s['win_count']}笔 / **亏**: {s['loss_count']}笔\n")

        # 平仓类别统计
        cat_stats = {}
        for t in s["trades"]:
            cat = t["exit_category"]
            if cat not in cat_stats:
                cat_stats[cat] = {"count": 0, "pnl": 0}
            cat_stats[cat]["count"] += 1
            cat_stats[cat]["pnl"] += t["pnl"]

        L.append("### 平仓类别统计\n")
        L.append("| 平仓类别 | 笔数 | 总盈亏 |")
        L.append("|---------|------|--------|")
        for cat, cs in sorted(cat_stats.items(), key=lambda x: x[1]["count"], reverse=True):
            L.append(f"| {cat} | {cs['count']} | {cs['pnl']:+.2f} |")
        L.append("")

        # 交易明细
        L.append("### 交易明细\n")
        L.append("| 股票 | 开仓日 | 开仓价 | 平仓日 | 平仓价 | 平仓信号 | 类别 | 盈亏% | 天数 |")
        L.append("|------|--------|--------|--------|--------|---------|------|-------|------|")
        for t in sorted(s["trades"], key=lambda x: x["entry_date"]):
            L.append(f"| {t['name']} | {t['entry_date']} | {t['entry_price']} | "
                     f"{t['exit_date']} | {t['exit_price']} | {t['exit_signal']} | "
                     f"{t['exit_category']} | {t['pnl_pct']:+.2f} | {t['holding_days']} |")
        L.append("")

    return "\n".join(L)


if __name__ == "__main__":
    main()
