"""
A/B 测试编排器

1. 运行A组 (默认 T1=1.0)
2. 编辑 position.py 将 T1 改为 2.0
3. 运行B组 (T1=2.0)
4. 恢复 position.py
5. 对比结果
"""
import subprocess
import sys
import os
import json
import shutil

PROJECT_ROOT = os.path.join(os.path.dirname(__file__), "..")
POSITION_PY = os.path.join(PROJECT_ROOT, "src", "backtest", "position.py")
BACKUP = POSITION_PY + ".bak"
RESULT_A = os.path.join(PROJECT_ROOT, "result_a.json")
RESULT_B = os.path.join(PROJECT_ROOT, "result_b.json")


def edit_t1_multiplier(new_val_str):
    """编辑 position.py 中 T1 档位的倍率

    原始: trailing_mult = trade_stop_mult          # 2.5 (盈利<10%, 给足空间)
    修改: trailing_mult = trade_stop_mult * {new_val}   # ...
    """
    with open(POSITION_PY, "r", encoding="utf-8") as f:
        content = f.read()

    # 原始行 (T1 档位, 无乘数)
    old_line = "trailing_mult = trade_stop_mult          # 2.5 (盈利<10%, 给足空间)"
    new_line = f"trailing_mult = trade_stop_mult * {new_val_str}   # T1 modified ({new_val_str})"

    if old_line not in content:
        # 可能已经是修改后的状态
        import re
        # 匹配已有的 T1 modified 行
        pattern = r'trailing_mult = trade_stop_mult \* [\d.]+\s+# T1 modified.*'
        if re.search(pattern, content):
            content = re.sub(pattern, new_line, content)
        else:
            print("ERROR: Cannot find T1 line in position.py")
            return False
    else:
        content = content.replace(old_line, new_line)

    with open(POSITION_PY, "w", encoding="utf-8") as f:
        f.write(content)
    return True


def restore_position_py():
    """恢复原始 position.py"""
    if os.path.exists(BACKUP):
        shutil.copy2(BACKUP, POSITION_PY)
        os.remove(BACKUP)
        print("Restored position.py from backup")
    else:
        # 如果没有备份, 把 T1 改回 1.0
        edit_t1_multiplier("1.0")
        # 清理注释
        with open(POSITION_PY, "r", encoding="utf-8") as f:
            content = f.read()
        content = content.replace(
            "trailing_mult = trade_stop_mult * 1.0   # T1 modified (1.0)",
            "trailing_mult = trade_stop_mult          # 2.5 (盈利<10%, 给足空间)"
        )
        with open(POSITION_PY, "w", encoding="utf-8") as f:
            f.write(content)
        print("Restored T1 to 1.0")


def run_backtest(label, output_file):
    """运行单次回测"""
    print(f"\n{'='*60}")
    print(f"  Running: {label}")
    print(f"{'='*60}")

    result = subprocess.run(
        [sys.executable, "scripts/run_group_backtest.py", label, output_file],
        cwd=PROJECT_ROOT,
        capture_output=False,
    )
    return result.returncode == 0


def compare_results():
    """对比A/B结果"""
    with open(RESULT_A, "r", encoding="utf-8") as f:
        ra = json.load(f)
    with open(RESULT_B, "r", encoding="utf-8") as f:
        rb = json.load(f)

    ma = ra["metrics"]
    mb = rb["metrics"]

    print("\n" + "=" * 100)
    print("  科技成长型分组 — T1档位倍率 A/B 回测对比")
    print("  回测区间: 2026-01-01 ~ 2026-07-13 | 初始资金: 10万 | risk_per_trade: 0.05")
    print("=" * 100)

    # 绩效对比
    print(f"\n  {'指标':<16} {'A组(T1=1.0×)':>16} {'B组(T1=2.0×)':>16} {'变化':>12}")
    print(f"  {'─'*16} {'─'*16} {'─'*16} {'─'*12}")

    rows = [
        ("总收益率", f"{ma['total_return']:+.2f}%", f"{mb['total_return']:+.2f}%",
         f"{mb['total_return'] - ma['total_return']:+.2f}%"),
        ("年化收益率", f"{ma['annual_return']:+.2f}%", f"{mb['annual_return']:+.2f}%",
         f"{mb['annual_return'] - ma['annual_return']:+.2f}%"),
        ("最大回撤", f"{ma['max_drawdown']:.2f}%", f"{mb['max_drawdown']:.2f}%",
         f"{mb['max_drawdown'] - ma['max_drawdown']:+.2f}%"),
        ("夏普比率", f"{ma['sharpe_ratio']:.3f}", f"{mb['sharpe_ratio']:.3f}",
         f"{mb['sharpe_ratio'] - ma['sharpe_ratio']:+.3f}"),
        ("总交易次数", f"{ma['trade_count']}", f"{mb['trade_count']}",
         f"{mb['trade_count'] - ma['trade_count']:+d}"),
        ("盈利交易", f"{ma['win_count']}", f"{mb['win_count']}",
         f"{mb['win_count'] - ma['win_count']:+d}"),
        ("胜率", f"{ma['win_rate']:.1f}%", f"{mb['win_rate']:.1f}%",
         f"{mb['win_rate'] - ma['win_rate']:+.1f}%"),
        ("平均盈利", f"{ma['avg_win_pct']:+.2f}%", f"{mb['avg_win_pct']:+.2f}%",
         f"{mb['avg_win_pct'] - ma['avg_win_pct']:+.2f}%"),
        ("平均亏损", f"{ma['avg_loss_pct']:+.2f}%", f"{mb['avg_loss_pct']:+.2f}%",
         f"{mb['avg_loss_pct'] - ma['avg_loss_pct']:+.2f}%"),
        ("盈亏比", f"{ma['profit_factor']:.2f}", f"{mb['profit_factor']:.2f}",
         f"{mb['profit_factor'] - ma['profit_factor']:+.2f}"),
        ("平均持仓天数", f"{ma['avg_holding_days']:.1f}", f"{mb['avg_holding_days']:.1f}",
         f"{mb['avg_holding_days'] - ma['avg_holding_days']:+.1f}"),
    ]
    for name, va, vb, diff in rows:
        print(f"  {name:<16} {va:>16} {vb:>16} {diff:>12}")

    # 交易明细对比
    print("\n" + "=" * 100)
    print("  交易明细对比")
    print("=" * 100)

    for label, result in [("A组(默认T1=1.0×)", ra), ("B组(T1=2.0×)", rb)]:
        trades = result["trades"]
        open_positions = result.get("open_positions", [])
        print(f"\n  ── {label} (已平仓 {len(trades)} 笔 + 未平仓 {len(open_positions)} 笔) ──")
        print(f"  {'股票':<8} {'入场日':<12} {'入场价':>8} {'退出日':<12} {'退出价':>8} {'盈利%':>8} {'持仓':>6} {'退出原因'}")
        print(f"  {'─'*8} {'─'*12} {'─'*8} {'─'*12} {'─'*8} {'─'*8} {'─'*6} {'─'*24}")

        for t in trades:
            reason = t["exit_signal"][:24]
            print(f"  {t['symbol']:<8} {t['entry_date']:<12} {t['entry_price']:>8.2f} "
                  f"{t['exit_date']:<12} {t['exit_price']:>8.2f} {t['pnl_pct']:>+7.1f}% "
                  f"{t['holding_days']:>5}天 {reason}")

        # 未平仓持仓
        if open_positions:
            print(f"\n  ── {label} 未平仓持仓 (回测结束时仍持有) ──")
            print(f"  {'股票':<8} {'入场日':<12} {'入场价':>8} {'当前价':>8} {'浮盈%':>8} {'持仓':>6} {'入场信号'}")
            print(f"  {'─'*8} {'─'*12} {'─'*8} {'─'*8} {'─'*8} {'─'*6} {'─'*24}")
            for t in open_positions:
                signal = t.get("entry_signal", "")[:24]
                print(f"  {t['symbol']:<8} {t['entry_date']:<12} {t['entry_price']:>8.2f} "
                      f"{t['current_price']:>8.2f} {t['unrealized_pct']:>+7.1f}% "
                      f"{t['shares']:>5}股 {signal}")

    # 按退出原因分类
    print("\n" + "=" * 100)
    print("  按退出原因分类")
    print("=" * 100)

    for label, result in [("A组", ra), ("B组", rb)]:
        trades = result["trades"]
        print(f"\n  {label}:")
        exit_reasons = {}
        for t in trades:
            reason = t["exit_signal"].split("(")[0].strip()
            if reason not in exit_reasons:
                exit_reasons[reason] = {"count": 0, "pnl_sum": 0.0}
            exit_reasons[reason]["count"] += 1
            exit_reasons[reason]["pnl_sum"] += t["pnl_pct"]

        print(f"  {'退出原因':<20} {'次数':>6} {'平均盈利':>10} {'总盈利':>10}")
        print(f"  {'─'*20} {'─'*6} {'─'*10} {'─'*10}")
        for reason, stats in sorted(exit_reasons.items(), key=lambda x: -x[1]["count"]):
            avg = stats["pnl_sum"] / stats["count"] if stats["count"] > 0 else 0
            print(f"  {reason:<20} {stats['count']:>6} {avg:>+9.1f}% {stats['pnl_sum']:>+9.1f}%")

    # 逐股票对比
    print("\n" + "=" * 100)
    print("  逐股票对比")
    print("=" * 100)

    stock_a = {}
    for t in ra["trades"]:
        stock_a.setdefault(t["symbol"], []).append(t)
    stock_b = {}
    for t in rb["trades"]:
        stock_b.setdefault(t["symbol"], []).append(t)

    print(f"\n  {'股票':<8} {'A组交易数':>8} {'A组总盈利':>10} │ {'B组交易数':>8} {'B组总盈利':>10} │ {'差异'}")
    print(f"  {'─'*8} {'─'*8} {'─'*10} │ {'─'*8} {'─'*10} │ {'─'*8}")

    all_stocks = sorted(set(list(stock_a.keys()) + list(stock_b.keys())))
    for code in all_stocks:
        ta = stock_a.get(code, [])
        tb = stock_b.get(code, [])
        pnl_a = sum(t["pnl_pct"] for t in ta)
        pnl_b = sum(t["pnl_pct"] for t in tb)
        diff = pnl_b - pnl_a
        print(f"  {code:<8} {len(ta):>8} {pnl_a:>+9.1f}% │ {len(tb):>8} {pnl_b:>+9.1f}% │ {diff:>+7.1f}%")

    print()


def main():
    # 1. 备份 position.py
    print("Backing up position.py...")
    shutil.copy2(POSITION_PY, BACKUP)

    try:
        # 2. 运行A组 (默认 T1=1.0)
        success_a = run_backtest("A组(默认T1=1.0×)", RESULT_A)

        if not success_a:
            print("A组回测失败!")
            return

        # 3. 编辑 position.py: T1 = 2.0
        print("\nEditing position.py: T1 multiplier 1.0 → 2.0")
        if not edit_t1_multiplier("2.0"):
            print("编辑失败!")
            return

        # 4. 运行B组 (T1=2.0)
        success_b = run_backtest("B组(T1=2.0×)", RESULT_B)

    finally:
        # 5. 恢复 position.py
        print("\nRestoring position.py...")
        restore_position_py()

    if not success_b:
        print("B组回测失败!")
        return

    # 6. 对比结果
    compare_results()


if __name__ == "__main__":
    main()
