"""验证 fetch_start 硬编码后前端收益与脚本完全一致"""
import requests, time

body = {
    "groups": ["科技成长型", "周期资源型"],
    "mode": "base",
    "start_date": "2026-01-01",
    "end_date": "2026-08-06",
    "initial_capital": 100000,
    "benchmark": "sh.000300",
}

print("发送回测请求 (fetch_start 已硬编码为 2025-06-01)...")
t0 = time.time()
resp = requests.post("http://127.0.0.1:8000/api/backtest/run", json=body, timeout=300)
data = resp.json()
elapsed = time.time() - t0

if data.get("status") == "error":
    print(f"❌ 错误: {data.get('message')}")
    raise SystemExit(1)

m = data["metrics"]
pc = data.get("portfolio_config", {})
print(f"耗时: {elapsed:.1f}s\n")
print("=" * 55)
print("  前端回测结果 (硬编码 fetch_start=2025-06-01)")
print("=" * 55)
print(f"  组合收益:   {m['total_return']*100:+.2f}%")
print(f"  Alpha:     {m['alpha']*100:+.2f}%")
print(f"  夏普:      {m['sharpe_ratio']:.3f}")
print(f"  回撤:      {m['max_drawdown']*100:.2f}%")
print(f"  交易笔数:   {m['trade_count']}")
print(f"  胜率:      {m['win_rate']*100:.1f}%")
print(f"  基准收益:   {m['benchmark_return']*100:+.2f}%")
print(f"  现金比例:   {pc.get('cash_ratio', 0)*100:.1f}%")
print()
print(f"  对比验证脚本(verify_frontend_p5.py): +33.25%")
diff = abs(m['total_return'] - 0.3325)
print(f"  差异: {diff*100:.2f}%")
if diff < 0.005:
    print("  ✅ 完全一致! (差异<0.5%, 因资金规模四舍五入)")
elif diff < 0.02:
    print("  ✅ 基本一致 (差异<2%)")
else:
    print(f"  ⚠️ 仍有差异 {diff*100:.2f}%, 需进一步排查")
