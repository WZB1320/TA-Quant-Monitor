"""手动模式覆盖功能 — 单元测试

验证内容：
1. 选择 trending 预设后，科技成长型股票参数切换到趋势参数
2. 选择 ranging 预设后，参数切换到震荡参数
3. 选择 auto 后恢复原组默认参数
4. 其他分组（消费稳健型）不受手动模式影响
5. clear_user_regime 清除所有设置
"""

import sys
import os

# 确保项目根在 sys.path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.config.group_config import GroupConfig
from src.data_fetcher.watchlist import Watchlist


def sep(title):
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}")


def test():
    gc = GroupConfig()
    wl = Watchlist()

    # 获取科技成长型的一只股票
    tech_stocks = [s for s in wl.get_all() if gc.get_group(s["code"]) == "科技成长型"]
    other_stocks = [s for s in wl.get_all() if gc.get_group(s["code"]) == "消费稳健型"]

    if not tech_stocks:
        print("ERROR: 自选股中没有科技成长型的股票，测试无法进行")
        print("请先在 watchlist 中添加一只科技成长型股票")
        return

    tech_code = tech_stocks[0]["code"]
    tech_name = tech_stocks[0]["name"]
    other_code = other_stocks[0]["code"] if other_stocks else tech_code
    other_name = other_stocks[0]["name"] if other_stocks else tech_name

    print(f"测试股票 (科技成长型): {tech_code} {tech_name}")
    if other_stocks:
        print(f"测试股票 (消费稳健型): {other_code} {other_name}")

    # ── 1. 默认参数（auto） ──
    sep("1. 默认模式 (auto)")
    params = gc.get_all_group_params(tech_code)
    print(f"  科技成长型 score_threshold:     {params['score_threshold']}")
    print(f"  科技成长型 score_ceiling:       {params['score_ceiling']}")
    print(f"  科技成长型 cooldown_days:       {params['cooldown_days']}")
    print(f"  科技成长型 atr_stop_mult:       {params['atr_stop_mult']}")
    print(f"  科技成长型 max_consecutive_losses: {params['max_consecutive_losses']}")
    print(f"  科技成长型 consecutive_loss_suspend: {params['consecutive_loss_suspend']}")
    print(f"  科技成长型 atr_price_ratio_max:  {params['atr_price_ratio_max']}")

    # ── 2. trending 预设 ──
    sep("2. 切换到 trending (趋势上涨)")
    gc.set_user_regime("科技成长型", "trending")
    print(f"  当前 regime: {gc.get_user_regime('科技成长型')}")

    params_t = gc.get_all_group_params(tech_code)
    print(f"  score_threshold:     {params_t['score_threshold']}  (预期: 35)")
    print(f"  score_ceiling:       {params_t['score_ceiling']}    (预期: 60)")
    print(f"  cooldown_days:       {params_t['cooldown_days']}    (预期: 5)")
    print(f"  atr_stop_mult:       {params_t['atr_stop_mult']}    (预期: 2.5)")
    print(f"  max_consecutive_losses: {params_t['max_consecutive_losses']}  (预期: 3)")
    print(f"  consecutive_loss_suspend: {params_t['consecutive_loss_suspend']}  (预期: 8)")
    print(f"  atr_price_ratio_max:  {params_t['atr_price_ratio_max']}  (预期: 0.10)")

    assert params_t["score_threshold"] == 35, f"FAIL: score_threshold={params_t['score_threshold']}, expected 35"
    assert params_t["score_ceiling"] == 60, f"FAIL: score_ceiling={params_t['score_ceiling']}, expected 60"
    assert params_t["cooldown_days"] == 5, f"FAIL: cooldown_days={params_t['cooldown_days']}, expected 5"
    assert params_t["atr_stop_mult"] == 2.5, f"FAIL: atr_stop_mult={params_t['atr_stop_mult']}, expected 2.5"
    assert params_t["max_consecutive_losses"] == 3, f"FAIL"
    assert params_t["consecutive_loss_suspend"] == 8, f"FAIL"
    assert params_t["atr_price_ratio_max"] == 0.10, f"FAIL"
    print("  >>> trending 预设参数全部正确!")

    # ── 3. ranging 预设 ──
    sep("3. 切换到 ranging (震荡)")
    gc.set_user_regime("科技成长型", "ranging")
    print(f"  当前 regime: {gc.get_user_regime('科技成长型')}")

    params_r = gc.get_all_group_params(tech_code)
    print(f"  score_threshold:     {params_r['score_threshold']}  (预期: 48)")
    print(f"  score_ceiling:       {params_r['score_ceiling']}    (预期: 55)")
    print(f"  cooldown_days:       {params_r['cooldown_days']}    (预期: 12)")
    print(f"  atr_stop_mult:       {params_r['atr_stop_mult']}    (预期: 1.5)")
    print(f"  max_consecutive_losses: {params_r['max_consecutive_losses']}  (预期: 2)")
    print(f"  consecutive_loss_suspend: {params_r['consecutive_loss_suspend']}  (预期: 15)")
    print(f"  atr_price_ratio_max:  {params_r['atr_price_ratio_max']}  (预期: 0.06)")

    assert params_r["score_threshold"] == 48, f"FAIL: score_threshold={params_r['score_threshold']}, expected 48"
    assert params_r["score_ceiling"] == 55, f"FAIL: score_ceiling={params_r['score_ceiling']}, expected 55"
    assert params_r["cooldown_days"] == 12, f"FAIL: cooldown_days={params_r['cooldown_days']}, expected 12"
    assert params_r["atr_stop_mult"] == 1.5, f"FAIL: atr_stop_mult={params_r['atr_stop_mult']}, expected 1.5"
    assert params_r["max_consecutive_losses"] == 2, f"FAIL"
    assert params_r["consecutive_loss_suspend"] == 15, f"FAIL"
    assert params_r["atr_price_ratio_max"] == 0.06, f"FAIL"
    print("  >>> ranging 预设参数全部正确!")

    # ── 4. 恢复 auto ──
    sep("4. 恢复 auto (自动判断)")
    gc.set_user_regime("科技成长型", "auto")
    print(f"  当前 regime: {gc.get_user_regime('科技成长型')}")

    params_auto = gc.get_all_group_params(tech_code)
    assert params_auto["score_threshold"] == params["score_threshold"], "恢复后 score_threshold 不一致"
    assert params_auto["cooldown_days"] == params["cooldown_days"], "恢复后 cooldown_days 不一致"
    print(f"  score_threshold: {params_auto['score_threshold']} (已恢复默认)")
    print(f"  cooldown_days: {params_auto['cooldown_days']} (已恢复默认)")
    print("  >>> auto 恢复到默认参数正确!")

    # ── 5. 其他分组不受影响 ──
    sep("5. 其他分组不受影响")
    gc.set_user_regime("科技成长型", "trending")

    params_other = gc.get_all_group_params(other_code)
    # 消费稳健型的默认值: score_threshold=30, cooldown_days=5
    print(f"  消费稳健型 score_threshold: {params_other['score_threshold']} (预期: 30, 不受科技成长型影响)")
    print(f"  消费稳健型 cooldown_days: {params_other['cooldown_days']} (预期: 5, 不受科技成长型影响)")
    assert params_other["score_threshold"] == 30, f"FAIL: 其他分组被意外修改，score_threshold={params_other['score_threshold']}"
    assert params_other["cooldown_days"] == 5, f"FAIL: 其他分组被意外修改"
    print("  >>> 其他分组确认未受影响!")

    # ── 6. clear_user_regime ──
    sep("6. clear_user_regime 清除所有设置")
    gc.clear_user_regime()
    params_cleared = gc.get_all_group_params(tech_code)
    assert params_cleared["score_threshold"] == params["score_threshold"], "clear 后未恢复默认"
    print(f"  clear 后 score_threshold: {params_cleared['score_threshold']} (已恢复默认)")
    print("  >>> clear_user_regime 正常工作!")

    # ── 完成 ──
    sep("结果")
    print("  所有 6 项测试全部通过! 手动模式覆盖功能正常工作。")


if __name__ == "__main__":
    test()