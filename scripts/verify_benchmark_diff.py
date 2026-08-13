"""快速验证: benchmark 代码差异对收益的影响"""
import sys, os, shutil, warnings
import pandas as pd, numpy as np

warnings.filterwarnings("ignore")
project_root = os.path.join(os.path.dirname(__file__), "..")
sys.path.insert(0, project_root)

from src.config.runtime_mode import set_mode, RuntimeMode
set_mode(RuntimeMode.BACKTEST)

from src.backtest.engine import BacktestEngine
from src.data_fetcher import DataManager
from src.config.group_config import GroupConfig
from src.config.user_preferences import UserPreferences, _USER_PREF_FILE

PORTFOLIO_WEIGHTS = {"科技成长型": 0.40, "周期资源型": 0.425}
REGIMES_CFG = {"周期资源型": {"trending"}}
ATR_OVERRIDE = {"科技成长型": 1.8}
DD_CONFIG = BacktestEngine.DEFAULT_DD_PROTECTION_CONFIG

# 测试3种benchmark
BENCHMARKS = ["000001", "sh.000001", "sh.000300"]

def test_benchmark(bench_code, data_map, dm):
    """用指定benchmark跑P5回测"""
    bench_df = dm.get_daily_kline(bench_code, start_date="2025-06-01", end_date="2026-08-06")
    if bench_df is None or len(bench_df) == 0:
        return {"error": f"无法拉取benchmark {bench_code}", "bench_len": 0}

    portfolio_nav = None
    total_trades = 0
    group_info = {}

    for g, weight in PORTFOLIO_WEIGHTS.items():
        gc = GroupConfig()
        # 获取该组股票
        cfg_path = os.path.join(project_root, "config", "strategy_config.json")
        import json
        with open(cfg_path, "r", encoding="utf-8") as f:
            cfg = json.load(f)
        group_codes = [s["code"] for s in cfg["strategy_config"]["watchlist"].get(g, [])
                       if isinstance(s, dict) and s["code"] in data_map]

        if not group_codes:
            continue

        GroupConfig._instance = None
        GroupConfig._config = None

        capital = 100000 * weight
        engine = BacktestEngine(
            initial_capital=capital, lookback_days=120, position_ratio=0.3,
            commission_rate=0.00025, stamp_tax=0.001, slippage=0.0001,
            signal_dedup_days=5, risk_per_trade=0.05,
            atr_stop_mult=ATR_OVERRIDE.get(g, 2.0),
            group_config=gc, forced_regime="auto",
            benchmark_df_for_memory=bench_df,
            trade_regimes=REGIMES_CFG.get(g),
            dd_protection_config=DD_CONFIG,
        )
        sub_map = {c: data_map[c] for c in group_codes}
        m = engine.run(sub_map, benchmark_df=bench_df, start_date="2026-01-01", end_date="2026-08-06")

        if engine.daily_values is not None:
            portfolio_nav = engine.daily_values.copy() if portfolio_nav is None else portfolio_nav.add(engine.daily_values, fill_value=0)
        total_trades += m.trade_count
        group_info[g] = {"return": m.total_return, "trades": m.trade_count}

    cash = 100000 * 0.175
    if portfolio_nav is not None:
        portfolio_nav = portfolio_nav + cash

    total_return = (portfolio_nav.iloc[-1] / 100000) - 1 if portfolio_nav is not None else 0
    return {
        "total_return_pct": round(total_return * 100, 2),
        "trades": total_trades,
        "bench_len": len(bench_df),
        "groups": group_info,
    }


def main():
    print("=" * 90)
    print("  Benchmark代码差异验证 — 000001 vs sh.000001 vs sh.000300")
    print("=" * 90)

    pref_existed = os.path.exists(_USER_PREF_FILE)
    pref_backup = _USER_PREF_FILE + ".bench_bak"
    if pref_existed:
        shutil.copy2(_USER_PREF_FILE, pref_backup)
    UserPreferences().clear_all()

    try:
        dm = DataManager()

        # 先测试每个benchmark能拉到多少数据
        print("\nBenchmark数据拉取测试:")
        for code in BENCHMARKS:
            df = dm.get_daily_kline(code, start_date="2025-06-01", end_date="2026-08-06")
            if df is not None and len(df) > 0:
                first_date = df["date"].iloc[0] if "date" in df.columns else "?"
                last_close = df["close"].iloc[-1] if "close" in df.columns else "?"
                first_close = df["close"].iloc[0] if "close" in df.columns else "?"
                ret = (float(last_close) / float(first_close) - 1) * 100 if first_close != "?" else 0
                print(f"  {code:12s}: {len(df)}条, 首日{first_date}, 收益{ret:+.2f}%, 末收盘{last_close}")
            else:
                print(f"  {code:12s}: 拉取失败 (None)")

        # 拉取股票数据(只拉科技+周期)
        import json
        cfg_path = os.path.join(project_root, "config", "strategy_config.json")
        with open(cfg_path, "r", encoding="utf-8") as f:
            cfg = json.load(f)
        all_codes = []
        for g in ["科技成长型", "周期资源型"]:
            all_codes.extend([s["code"] for s in cfg["strategy_config"]["watchlist"].get(g, [])
                            if isinstance(s, dict)])
        data_map = {}
        for code in all_codes:
            df = dm.get_daily_kline(code, start_date="2025-06-01", end_date="2026-08-06")
            if df is not None and len(df) >= 120:
                data_map[code] = df
        print(f"\n股票数据: {len(data_map)}/{len(all_codes)}")

        # 对比3种benchmark
        print(f"\n{'='*90}")
        print(f"{'Benchmark':<15} {'收益%':>8} {'交易数':>6} {'科技组':>20} {'周期组':>20}")
        print(f"{'-'*75}")
        for bench_code in BENCHMARKS:
            r = test_benchmark(bench_code, data_map, dm)
            if "error" in r:
                print(f"{bench_code:<15} 错误: {r['error']}")
            else:
                g_info = r.get("groups", {})
                tech = g_info.get("科技成长型", {})
                cyc = g_info.get("周期资源型", {})
                print(f"{bench_code:<15} {r['total_return_pct']:>+8.2f} {r['trades']:>6} "
                      f"科技{tech.get('return',0)*100:+.1f}%/{tech.get('trades',0)}笔  "
                      f"周期{cyc.get('return',0)*100:+.1f}%/{cyc.get('trades',0)}笔")

    finally:
        if pref_existed and os.path.exists(pref_backup):
            shutil.move(pref_backup, _USER_PREF_FILE)


if __name__ == "__main__":
    main()
