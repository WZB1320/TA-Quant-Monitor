"""直接测试 SignalAnalyzer 完整流程（含调试输出）"""
import sys
import os
import logging

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

logging.basicConfig(level=logging.DEBUG, format="%(levelname)s [%(name)s] %(message)s")

# 模拟 web/backend 的路径环境
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
WEB_BACKEND = os.path.join(PROJECT_ROOT, "web", "backend")
sys.path.insert(0, WEB_BACKEND)

from services.signal_analyzer import SignalAnalyzer

print("=== 测试 1: trending 模式 ===")
analyzer = SignalAnalyzer()
result = analyzer.run_analysis(group="科技成长型", user_regime="trending")
print(f"返回状态: {result.get('status')}")
print(f"分析时间: {result.get('analyzed_at')}")
print(f"汇总: {result.get('summary')}")
if result.get('results'):
    for r in result['results'][:3]:
        print(f"  {r['code']} {r['name']:6s} score={r['score']} level={r['level']} confidence={r['confidence']} action={r['action']}")

print("\n=== 测试 2: ranging 模式 ===")
analyzer2 = SignalAnalyzer()  # 同一个单例
result2 = analyzer2.run_analysis(group="科技成长型", user_regime="ranging")
print(f"返回状态: {result2.get('status')}")
print(f"汇总: {result2.get('summary')}")
if result2.get('results'):
    for r in result2['results'][:3]:
        print(f"  {r['code']} {r['name']:6s} score={r['score']} level={r['level']} confidence={r['confidence']} action={r['action']}")

print("\n=== 对比: 两次分析的 actionable 数量差异 ===")
print(f"trending actionable: {result.get('summary', {}).get('actionable', 'N/A')}")
print(f"ranging actionable:  {result2.get('summary', {}).get('actionable', 'N/A')}")