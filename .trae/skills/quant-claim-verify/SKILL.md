---
name: "quant-claim-verify"
description: "量化组级业绩结论验证器 — 在TA-Quant-Monitor项目中, 任何关于组(科技/消费/周期/医药/机械)的业绩、Alpha、收益、胜率、转正、改善、恶化等结论之前必须调用. Invoke BEFORE making any claim about group-level performance in TA-Quant-Monitor project (path e:\\AI\\TA-Quant-Monitor). Triggers: user asks 'XX组业绩/Alpha/为什么变好变差'; assistant wants to say '转正/改善/恶化/意外/突然'."
---

# 量化结论验证器 (Quant Claim Verifier)

**适用项目**: 仅 `e:\AI\TA-Quant-Monitor`。其他项目跳过本skill。

## 强制调用场景 (MANDATORY Triggers)

在下述任何结论**之前**必须先执行本skill的验证流程：

1. 用户询问某组的 Alpha / 收益 / 业绩 / 表现 / 为什么变好变差
2. 用户询问"均启用时业绩如何" / "暂停组启用后效果"
3. 助手要陈述"转正 / 改善 / 恶化 / 意外 / 突然 / 比之前好/差"
4. 助手要给出"医药组/消费组/机械组/科技组/周期组的XX指标是Y%"
5. 任何"单组业绩结论"的陈述

## 验证流程 (Verification Procedure)

### 步骤1: 指标定义自检
确认要陈述的指标是哪一个，严禁混用：
- **Alpha** = 组合(组)收益 − 同期同口径基准收益。**不是**单笔 pnl_pct 的平均值
- **收益(total_return)** = 期末净值 / 期初资金 − 1
- **胜率(win_rate)** = 盈利交易数 / 总交易数
- **平均盈亏(avg pnl_pct)** = 单笔交易 pnl_pct 的平均（与Alpha无关）

### 步骤2: 历史基线锚定
读取 `e:\AI\TA-Quant-Monitor\data\group_alpha_baselines.json` 获取该组历史Alpha。
若新结论与历史基线差距 > 历史绝对值的50%，或符号反转，**必须**独立验证后再下结论。

### 步骤3: 独立单组回测验证
**严禁**从组合聚合数据反推单组结论。必须独立跑单组回测：

```bash
"C:\Users\zongb\AppData\Local\Programs\Python\Python313\python.exe" -u e:\AI\TA-Quant-Monitor\scripts\verify_group_alpha.py --group <组名> --start <YYYY-MM-DD> --end <YYYY-MM-DD>
```

脚本会输出该组独立Alpha并与基线对比，差距过大时输出告警。

### 步骤4: 置信度标注
下结论时必须标注来源：
- `[已验证]` — 已运行步骤3脚本，数值来自独立回测
- `[推断未验证]` — 从聚合数据反推，未独立验证（**禁止用于单组结论**）
- `[记忆引用]` — 引自历史记录，可能过时，需重新验证

## 5条核心规则

1. **指标严格区分**: Alpha ≠ 平均pnl_pct ≠ 胜率。报告业绩必须六件套(收益/基准/Alpha/Sharpe/回撤/交易数)
2. **历史基线锚定**: 声明"改善/转正/恶化"前，必须读取基线库对比。符号反转或差距>50%必须独立验证
3. **验证先于结论**: 单组结论必须用独立脚本验证，不能从组合层聚合反推
4. **反乐观偏差**: 先报告负面/中性事实，再说正面。禁用"意外转正/突然变好"等措辞，除非已验证
5. **错误自检**: 结果出来后自检——与历史差距大吗？指标定义对吗？是独立验证还是反推？

## 3类补充场景

- **源码落地验证**: 声明"已修改/已实现"前，必须grep源码确认（避免P4"声称改5处实际0处"）
- **因果解释约束**: 用户问"为什么X"时，不能编造原因。先验证事实再解释，解释不了就说"不确定"
- **置信度标注**: 任何结论必须标注来源（已验证/推断未验证/记忆引用）

## 常见错误模式（禁止）

| 错误 | 正确做法 |
|------|---------|
| 把平均pnl_pct当Alpha说 | 跑独立回测取真实Alpha |
| 从组合聚合数据反推单组Alpha | 跑verify_group_alpha.py |
| "意外转正"未验证就下结论 | 先验证，标注[已验证] |
| 声明"已修改"未grep源码 | grep确认后再说 |
| 历史基线差距>50%仍直接确认 | 必须独立验证 |

## 基线库维护

每次正式回测后，更新 `e:\AI\TA-Quant-Monitor\data\group_alpha_baselines.json`：
```json
{
  "科技成长型": {"train": {"alpha": 0.05, "date": "2026-08-12"}, "test": {...}, "full": {...}},
  "消费稳健型": {...},
  "周期资源型": {...},
  "医药创新型": {...},
  "机械制造型": {...}
}
```
