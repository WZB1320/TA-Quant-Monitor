"""
验证机械组启用 — 确认配置正确 + 前端回测集成正常
"""
import sys, os, json, requests
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

os.environ.setdefault("PYTHONIOENCODING", "utf-8")

# 1. 验证配置加载
print("=" * 80)
print("  [1] 配置加载验证")
print("=" * 80)

cfg_path = os.path.join(os.path.dirname(__file__), "..", "config", "strategy_config.json")
with open(cfg_path, "r", encoding="utf-8") as f:
    cfg = json.load(f)

groups = cfg["strategy_config"]["group_config"]["groups"]
mech = groups["机械制造型"]
print(f"  机械组 strategy_mode: {mech.get('strategy_mode', 'N/A')}")
print(f"  机械组 description: {mech.get('description', 'N/A')[:60]}")

portfolio = cfg["strategy_config"]["portfolio_config_p5"]
print(f"\n  组合权重:")
for g, w in portfolio["weights"].items():
    status = "活跃" if w > 0 else "暂停"
    print(f"    {g}: {w*100:.1f}% ({status})")
print(f"  现金缓冲: {portfolio['cash_ratio']*100:.1f}%")
total = sum(portfolio["weights"].values()) + portfolio["cash_ratio"]
print(f"  总计: {total*100:.1f}% {'✅' if abs(total - 1.0) < 0.001 else '❌'}")

# 2. 验证GroupConfig读取
print(f"\n{'='*80}")
print("  [2] GroupConfig读取验证")
print("=" * 80)

from src.config.group_config import GroupConfig
gc = GroupConfig()
gc._load()
mech_cfg = gc._groups.get("机械制造型", {})
mode = mech_cfg.get("strategy_mode", "trend_following")
print(f"  GroupConfig机械组 strategy_mode: {mode}")
print(f"  rotation模式识别: {'✅ 正确' if mode == 'rotation' else '❌ 未识别'}")

# 3. 验证RotationStrategy导入
print(f"\n{'='*80}")
print("  [3] RotationStrategy导入验证")
print("=" * 80)

from src.backtest.rotation_strategy import RotationStrategy
rs = RotationStrategy(initial_capital=10000)
print(f"  RotationStrategy实例化: ✅")
print(f"  初始资金: {rs.capital}")
print(f"  初始现金: {rs.cash}")

# 4. 运行前端回测API
print(f"\n{'='*80}")
print("  [4] 前端回测API验证")
print("=" * 80)

try:
    resp = requests.post(
        "http://localhost:8000/api/backtest/run",
        json={
            "groups": [],
            "mode": "base",
            "start_date": "2025-07-01",
            "end_date": "2026-06-30",
            "initial_capital": 1000000,
            "benchmark": "sh.000300"
        },
        timeout=120
    )
    result = resp.json()

    if result.get("status") == "done":
        m = result["metrics"]
        print(f"  回测状态: ✅ 完成")
        print(f"  组合收益: {m['total_return']*100:+.2f}%")
        print(f"  Alpha: {m['alpha']*100:+.2f}%")
        print(f"  夏普: {m['sharpe_ratio']:.3f}")
        print(f"  回撤: {m['max_drawdown']*100:.2f}%")
        print(f"  交易数: {m['trade_count']}")

        # 检查活跃组
        active = result.get("groups", [])
        print(f"\n  活跃组: {active}")
        print(f"  机械组参与: {'✅ 是' if '机械制造型' in active else '❌ 否'}")

        # 检查交易记录中是否有机械组
        trades = result.get("trades", [])
        mech_trades = [t for t in trades if t.get("group") == "机械制造型"]
        print(f"  机械组交易数: {len(mech_trades)}")
        if mech_trades:
            print(f"  机械组首笔交易: {mech_trades[0]['name']} {mech_trades[0]['pnl_pct']:+.2f}%")

        # 组合配置
        pc = result.get("portfolio_config", {})
        print(f"\n  组合配置:")
        for g, w in pc.get("weights", {}).items():
            print(f"    {g}: {w*100:.1f}%")
        print(f"    现金: {pc.get('cash_ratio', 0)*100:.1f}%")

        # 验证机械组退出原因
        if mech_trades:
            exit_reasons = {}
            for t in mech_trades:
                r = t.get("exit_signal", "")
                if "调仓" in r: key = "月度调仓换出"
                elif "固定止损" in r: key = "固定止损-12%"
                elif "MA60" in r: key = "MA60破位退出"
                elif "trailing" in r: key = "ATR trailing止盈"
                elif "回测结束" in r: key = "回测结束平仓"
                else: key = r
                exit_reasons[key] = exit_reasons.get(key, 0) + 1
            print(f"\n  机械组退出原因: {exit_reasons}")

    elif result.get("status") == "error":
        print(f"  回测状态: ❌ 错误")
        print(f"  错误信息: {result.get('message', 'N/A')}")
    else:
        print(f"  回测状态: {result.get('status', 'unknown')}")
        print(f"  响应: {str(result)[:200]}")

except requests.exceptions.ConnectionError:
    print("  ⚠️ 后端服务未启动, 跳过API验证")
    print("  请先启动后端: cd web/backend && python main.py")
except Exception as e:
    print(f"  ❌ 异常: {e}")

print(f"\n{'='*80}")
print("  验证完成")
print("=" * 80)
