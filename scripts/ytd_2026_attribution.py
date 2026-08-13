"""
收益差异归因对比 — 定位 P5(34.22%) vs "原来(~47%)" 的收益差异来源

4种配置对比 2026-01-01 ~ 2026-08-06:
  A. P3退出 + 消费10% + 无降仓保护     (≈ "原来" P3配置)
  B. P3退出 + 消费0%  + 无降仓保护     (分离: 暂停消费的影响)
  C. P3退出 + 消费0%  + 12%降仓(P5)    (分离: 降仓保护的影响)
  D. P2退出 + 消费10% + 无降仓保护     (P2退出参数, 分离: 退出参数的影响)

A vs B = 暂停消费组的影响
B vs C = 12%降仓保护的影响
A vs C = 总优化影响(P3→P5)
A vs D = 退出参数(P2→P3)的影响

用法: python scripts/ytd_2026_attribution.py
"""
import sys
import os
import json
import shutil
import warnings
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
project_root = os.path.join(os.path.dirname(__file__), "..")
sys.path.insert(0, project_root)

from src.config.runtime_mode import set_mode, RuntimeMode
set_mode(RuntimeMode.BACKTEST)

from src.backtest.engine import BacktestEngine
from src.data_fetcher import DataManager
from src.config.group_config import GroupConfig
from src.config.user_preferences import UserPreferences, _USER_PREF_FILE
from src.backtest.position import PositionManager

BT_START, BT_END = "2026-01-01", "2026-08-06"
DATA_START, DATA_END = "2025-06-01", "2026-08-06"
BENCHMARK = "sh.000300"
TOTAL_CAPITAL = 1000000

REGIMES_CFG = {"周期资源型": {"trending"}}
ATR_OVERRIDE = {"科技成长型": 1.8}

# P2 退出参数 (trailing 紧, 震荡市不禁用)
P2_STOP_PARAMS = {
    'hard_stop_pct': 0.10,
    'trail_tier1_threshold': 0.10, 'trail_tier2_threshold': 0.20,
    'trail_mult_low': 1.0, 'trail_mult_mid': 0.8, 'trail_mult_high': 0.6,
    'no_atr_hard_stop_pct': 0.10, 'no_atr_trail_drawdown': 0.05,
}
P2_REGIME_EXIT = {}  # 空dict=禁用分体制退出(不禁用trailing)

# P5 权重 (消费暂停)
WEIGHTS_P5 = {"科技成长型": 0.40, "消费稳健型": 0.0, "周期资源型": 0.425,
              "医药创新型": 0.0, "机械制造型": 0.0}
# P3 权重 (消费10%)
WEIGHTS_P3 = {"科技成长型": 0.40, "消费稳健型": 0.10, "周期资源型": 0.425,
              "医药创新型": 0.0, "机械制造型": 0.0}

DD_CONFIG = BacktestEngine.DEFAULT_DD_PROTECTION_CONFIG


def load_watchlist():
    cfg_path = os.path.join(project_root, "config", "strategy_config.json")
    with open(cfg_path, "r", encoding="utf-8") as f:
        cfg = json.load(f)
    return {g: [s["code"] for s in stocks]
            for g, stocks in cfg["strategy_config"]["watchlist"].items()
            if not g.startswith("_") and isinstance(stocks, list)}


def run_config(data_map, benchmark_df, watchlist, weights, stop_params, regime_exit, dd_config, label):
    """跑一组配置的组合回测"""
    GroupConfig._instance = None
    GroupConfig._config = None

    portfolio_nav = None
    group_details = {}
    total_trades = 0
    total_dd_triggers = 0
    total_dd_reduce_trades = 0

    for g, codes in watchlist.items():
        if g not in weights or weights[g] == 0:
            group_details[g] = {"skipped": True}
            continue
        g_codes = [c for c in codes if c in data_map]
        if len(g_codes) < 2:
            continue
        capital = TOTAL_CAPITAL * weights[g]
        regimes = REGIMES_CFG.get(g)
        atr_mult = ATR_OVERRIDE.get(g, 2.0)

        GroupConfig._instance = None
        GroupConfig._config = None
        engine = BacktestEngine(
            initial_capital=capital, lookback_days=120, position_ratio=0.3,
            commission_rate=0.00025, stamp_tax=0.001, slippage=0.0001,
            signal_dedup_days=5, risk_per_trade=0.05, atr_stop_mult=atr_mult,
            forced_regime=None, trade_regimes=regimes,
            stop_loss_params=stop_params,
            regime_exit_config=regime_exit,
            dd_protection_config=dd_config,
        )
        sub_map = {c: data_map[c] for c in g_codes if c in data_map}
        m = engine.run(sub_map, benchmark_df=benchmark_df, start_date=BT_START, end_date=BT_END)
        dd_trades = sum(1 for t in engine.position_mgr.closed_trades if "回撤保护" in (t.exit_signal or ""))
        group_details[g] = {
            "return": getattr(m, "total_return", 0) or 0,
            "sharpe": getattr(m, "sharpe_ratio", 0) or 0,
            "max_dd": getattr(m, "max_drawdown", 0) or 0,
            "trades": getattr(m, "trade_count", 0) or 0,
            "dd_triggers": engine.dd_protection_stats.get("triggers", 0),
            "dd_reduce_trades": dd_trades,
            "daily_values": engine.daily_values.copy() if engine.daily_values is not None else None,
        }
        total_trades += group_details[g]["trades"]
        total_dd_triggers += group_details[g]["dd_triggers"]
        total_dd_reduce_trades += group_details[g]["dd_reduce_trades"]

        nav = group_details[g]["daily_values"]
        if nav is not None:
            portfolio_nav = nav if portfolio_nav is None else portfolio_nav.add(nav, fill_value=0)

    if portfolio_nav is None:
        return {"label": label, "error": "无净值"}

    invested = sum(w * TOTAL_CAPITAL for w in weights.values() if w > 0)
    cash = TOTAL_CAPITAL - invested
    portfolio_nav = portfolio_nav + cash
    nav_idx = pd.to_datetime(portfolio_nav.index)
    portfolio_nav = pd.Series(portfolio_nav.values, index=nav_idx)

    daily_ret = portfolio_nav.pct_change().dropna()
    total_return = (portfolio_nav.iloc[-1] / TOTAL_CAPITAL) - 1
    sharpe = (daily_ret.mean() / daily_ret.std() * np.sqrt(252)
              if daily_ret.std() > 0 else 0.0)
    cummax = portfolio_nav.cummax()
    max_dd = ((portfolio_nav - cummax) / cummax).min()

    # 沪深300收益
    bench = benchmark_df.copy()
    if "date" not in bench.columns:
        bench = bench.reset_index()
    bench["date"] = pd.to_datetime(bench["date"])
    bench_s = bench.set_index("date")["close"].astype(float)
    bt_start_ts = pd.Timestamp(BT_START)
    bt_end_ts = pd.Timestamp(BT_END)
    bench_s = bench_s[(bench_s.index >= bt_start_ts) & (bench_s.index <= bt_end_ts)]
    bench_return = (bench_s.iloc[-1] / bench_s.iloc[0]) - 1 if len(bench_s) > 0 else 0

    return {
        "label": label,
        "total_return_pct": round(total_return * 100, 2),
        "alpha_pct": round((total_return - bench_return) * 100, 2),
        "sharpe": round(sharpe, 3),
        "max_dd_pct": round(max_dd * 100, 2),
        "trades": total_trades,
        "dd_triggers": total_dd_triggers,
        "dd_reduce_trades": total_dd_reduce_trades,
        "bench_return_pct": round(bench_return * 100, 2),
        "group_details": group_details,
    }


def main():
    print("=" * 100)
    print(f"  收益差异归因对比 — 4种配置 × {BT_START}~{BT_END}")
    print(f"  目标: 定位 P5(34.22%) vs '原来(~47%)' 的收益差异来源")
    print("=" * 100)

    pref_existed = os.path.exists(_USER_PREF_FILE)
    pref_backup = _USER_PREF_FILE + ".attr_bak"
    if pref_existed:
        shutil.copy2(_USER_PREF_FILE, pref_backup)
    UserPreferences().clear_all()

    try:
        watchlist = load_watchlist()
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

        # P3 退出参数 = None (用默认 DEFAULT_STOP_PARAMS, 已固化P3值)
        # P2 退出参数 = P2_STOP_PARAMS + P2_REGIME_EXIT

        configs = [
            ("A. P3退出+消费10%+无降仓",   WEIGHTS_P3, None,            None,         None),
            ("B. P3退出+消费0%+无降仓",    WEIGHTS_P5, None,            None,         None),
            ("C. P3退出+消费0%+12%降仓(P5)",WEIGHTS_P5, None,            None,         DD_CONFIG),
            ("D. P2退出+消费10%+无降仓",   WEIGHTS_P3, P2_STOP_PARAMS,  P2_REGIME_EXIT, None),
        ]

        results = []
        for label, weights, sp, re_cfg, dd_cfg in configs:
            print(f"\n跑 [{label}] ...", end="", flush=True)
            r = run_config(data_map, benchmark_df, watchlist, weights, sp, re_cfg, dd_cfg, label)
            results.append(r)
            if "error" not in r:
                print(f" 收益{r['total_return_pct']:+.2f}% Alpha{r['alpha_pct']:+.2f}% "
                      f"夏普{r['sharpe']:.3f} 回撤{r['max_dd_pct']:.1f}% "
                      f"降仓触发{r['dd_triggers']}次/{r['dd_reduce_trades']}笔")
            else:
                print(f" 错误: {r['error']}")

        # 汇总
        print(f"\n{'='*100}")
        print(f"  收益差异归因汇总 ({BT_START}~{BT_END})")
        print(f"{'='*100}")
        print(f"\n  {'配置':<32} {'收益%':>8} {'Alpha%':>8} {'夏普':>7} {'回撤%':>7} {'降仓触发':>8}")
        print(f"  {'-'*75}")
        for r in results:
            if "error" not in r:
                print(f"  {r['label']:<32} {r['total_return_pct']:>+8.2f} {r['alpha_pct']:>+8.2f} "
                      f"{r['sharpe']:>7.3f} {r['max_dd_pct']:>7.1f} {r['dd_triggers']:>5}次/{r['dd_reduce_trades']}笔")
        print(f"  {'沪深300':<32} {results[0]['bench_return_pct']:>+8.2f}")

        # 归因分析
        print(f"\n{'='*100}")
        print(f"  收益差异归因分析")
        print(f"{'='*100}")
        if len(results) >= 4 and "error" not in results[0] and "error" not in results[3]:
            a = results[0]["total_return_pct"]  # P3+消费10%+无降仓
            b = results[1]["total_return_pct"]  # P3+消费0%+无降仓
            c = results[2]["total_return_pct"]  # P5(P3+消费0%+12%降仓)
            d = results[3]["total_return_pct"]  # P2退出+消费10%+无降仓

            print(f"\n  A vs B (暂停消费组的影响):     {a:+.2f}% → {b:+.2f}% = {b-a:+.2f}%")
            print(f"  B vs C (12%降仓保护的影响):    {b:+.2f}% → {c:+.2f}% = {c-b:+.2f}%")
            print(f"  A vs D (退出参数P2→P3的影响):  {d:+.2f}% → {a:+.2f}% = {a-d:+.2f}%")
            print(f"  A vs C (总优化 P3→P5):         {a:+.2f}% → {c:+.2f}% = {c-a:+.2f}%")
            print(f"  D vs C (原始P2 → 当前P5):      {d:+.2f}% → {c:+.2f}% = {c-d:+.2f}%")

        # 各组明细
        print(f"\n{'='*100}")
        print(f"  各组明细 (按配置)")
        print(f"{'='*100}")
        for r in results:
            if "error" in r:
                continue
            print(f"\n  [{r['label']}]")
            for g, gd in r["group_details"].items():
                if gd.get("skipped"):
                    continue
                print(f"    {g:12s}: 收益{gd['return']*100:+.2f}% 夏普{gd['sharpe']:.3f} "
                      f"回撤{gd['max_dd']*100:.1f}% 交易{gd['trades']}笔 "
                      f"降仓{gd['dd_triggers']}次/{gd['dd_reduce_trades']}笔")

    finally:
        if pref_existed and os.path.exists(pref_backup):
            shutil.move(pref_backup, _USER_PREF_FILE)


if __name__ == "__main__":
    main()
