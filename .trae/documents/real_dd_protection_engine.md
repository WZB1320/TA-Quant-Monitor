# 引擎真实降仓回撤保护 — 实现规划

## Context（为什么做这个改动）

当前回撤保护是**「事后净值降仓模型」**：在脚本里对组合净值序列做事后数学调整（保护期日收益 × 0.5），不是引擎层面的真实交易。这导致两个问题：

1. **不真实**：触发时并不真的卖出股票，没有交易成本、没有真实减仓动作，只是报告里的数字游戏。
2. **8%阈值形同虚设**：P3退出参数优化后，组合回撤已被压到 7.5~7.6%（够不到8%），保护在标准两窗口从不触发；且即使触发，也只是事后调数字，不是真实风控。

用户决定：**维持8%阈值不降，但把回撤保护从「事后模型」升级为「引擎层面真实降仓」**——触发时引擎实际卖出部分持仓（单向，恢复时不自动买回，让新信号自然重建仓位）。这样8%才是一把真实的风控闸门，在极端行情下真正减损。

**核心约束**：
- 单向降仓（只卖不买）：触发时真实卖出，恢复时仅切换状态标志、不买回。
- 不破坏现有行为：8%不触发时，结果必须与P3基线完全一致（向后兼容）。
- 8%作为"最后防线"定位：温和市场不触发是对的，极端行情才出手。

---

## 改动文件与具体方案

### 1. `src/backtest/position.py` — 新增 `reduce_position()` 方法

在 `close_position()` 之后新增部分平仓方法。**不改动**现有 `close_position`/`open_long`/`check_stop_loss`/`Trade` 数据类。

```python
def reduce_position(self, symbol, exit_date, exit_price, reduce_ratio, signal="回撤保护降仓"):
    """部分平仓 — 按比例卖出部分股数, 剩余继续持有 (用于组合级回撤保护真实降仓).

    机制:
      - 计算 reduce_ratio 比例的卖出股数 (向下取整到100股)
      - 创建一个"部分平仓" Trade (复制entry信息, shares=卖出股数), close它, 加入closed_trades
      - 减少open Trade的shares (entry_price/highest_price/ATR止损参数保持不变)
      - 回笼现金 (扣佣金+印花税)
    """
    trade = self._open.get(symbol)
    if trade is None: return None
    sell_shares = int(trade.shares * reduce_ratio / 100) * 100
    if sell_shares < 100: return None
    # 部分平仓Trade
    partial = Trade(symbol, trade.side, trade.entry_date, trade.entry_price,
                    shares=sell_shares, entry_signal=trade.entry_signal)
    revenue = sell_shares * exit_price
    comm = max(revenue * self.commission_rate, 5)
    stamp = revenue * 0.001
    total_comm = comm + stamp
    partial.close(exit_date, exit_price, signal, commission=total_comm)
    self.cash += (revenue - total_comm)
    self.closed_trades.append(partial)
    # 减少open Trade股数 (entry_price/highest_price/_atr_*不变, 止损逻辑对剩余持仓延续)
    trade.shares -= sell_shares
    return partial
```

**关键点**：
- 部分平仓 Trade 的 `exit_signal="回撤保护降仓"`，便于 metrics 里识别和过滤。
- open Trade 的 `entry_price`/`highest_price`/`_atr_stop_price` 等保持不变 → 止损逻辑对剩余持仓无缝延续。
- 不需要 `increase_position`（单向，不买回）。

### 2. `src/backtest/engine.py` — 新增构造参数 + 每日循环插入回撤检查

**构造函数新增参数**（L39-L61 签名 + L97-L105 赋值）：
```python
dd_protection_config: dict = None,  # None=不启用(向后兼容), {"threshold":-0.08,"recovery":-0.04,"reduced_ratio":0.5}
```
在 `__init__` 里保存：
```python
self.dd_protection_config = dd_protection_config
```

**`run()` 方法初始化保护状态**（在 L152 `position_mgr` 创建后）：
```python
# 组合级回撤保护状态 (真实降仓)
self._dd_enabled = dd_protection_config is not None
self._dd_threshold = (dd_protection_config or {}).get("threshold", -0.08)
self._dd_recovery = (dd_protection_config or {}).get("recovery", -0.04)
self._dd_reduced_ratio = (dd_protection_config or {}).get("reduced_ratio", 0.5)
self._nav_peak = self.initial_capital
self._in_protection = False
self._dd_triggers = 0  # 触发次数统计
self._dd_reduce_days = 0  # 降仓天数统计
```

**每日循环插入回撤检查**（在 L271 信号暂存之后、L273 记录净值之前）：
```python
# 4.5 组合级回撤保护 (真实降仓, 单向)
if self._dd_enabled and in_backtest_range:
    current_nav = self.position_mgr.total_value(prices_today)
    if current_nav > self._nav_peak:
        self._nav_peak = current_nav
    dd = (current_nav - self._nav_peak) / self._nav_peak if self._nav_peak > 0 else 0
    if not self._in_protection and dd < self._dd_threshold:
        # 触发: 对每个持仓真实部分平仓 (卖出 reduced_ratio 比例)
        self._in_protection = True
        self._dd_triggers += 1
        for sym in list(self.position_mgr.open_positions.keys()):
            if sym in prices_today:
                self.position_mgr.reduce_position(
                    sym, today, prices_today[sym],
                    reduce_ratio=self._dd_reduced_ratio)
    elif self._in_protection and dd > self._dd_recovery:
        # 恢复: 仅切换状态, 不买回 (单向, 让新信号自然重建仓位)
        self._in_protection = False
    if self._in_protection:
        self._dd_reduce_days += 1
```

**统计输出**：在 `run()` 返回前，把 `_dd_triggers`/`_dd_reduce_days` 挂到 `self.metrics` 或单独属性，供脚本读取。

### 3. `scripts/real_dd_validation.py` — 新建验证脚本

三层验证：

**(A) 标准两窗口 + 8%（不破坏验证）**：训练窗 2024-07~2025-06、测试窗 2025-07~2026-06。
- 预期：8%不触发（回撤7.5~7.6%），结果与P3基线完全一致（Alpha/夏普/回撤/收益）。
- 确认：引擎真实降仓逻辑在不触发时不影响任何行为。

**(B) 标准两窗口 + 4%阈值（强制触发验证）**：用4%阈值强制触发降仓，对比「引擎真实降仓」vs「事后模型」。
- 预期：真实降仓会执行部分卖出（trade_count增加、closed_trades含"回撤保护降仓"记录、现金增加）。
- 确认：真实降仓逻辑正确执行，NAV受真实卖出影响（与事后模型数值接近但不完全相同，因有交易成本且peak基于真实NAV）。

**(C) 2022熊市极端窗口 + 8%（真实减损验证）**：窗口 2022-01-01~2022-12-31（沪深300全年跌21%）。
- 预期：8%真实触发，引擎实际卖出部分持仓，组合回撤显著小于无保护对照。
- 确认：8%在极端行情下真正起作用、真实减损。
- 数据：已验证2022年基准+个股数据可拉取（242条）。

脚本复用 `p3_full_validation.py` 的数据拉取/分组回测结构，但**移除事后 `apply_drawdown_protection`**，改为给 `BacktestEngine` 传 `dd_protection_config`。对比三档：无保护 / 8%引擎真实降仓 / 4%引擎真实降仓。

---

## 不改动的部分（明确边界）

- `Trade` 数据类不改（部分平仓通过新建Trade实现，不动数据类）。
- `close_position`/`open_long`/`check_stop_loss` 不改（`reduce_position` 是新增独立方法）。
- `strategy_config.json` 不改（dd保护参数走引擎构造参数，不放配置文件；脚本里显式传入）。
- 现有事后模型脚本（p2/p3/ytd等）保留不动，作为历史对照；新验证用新脚本+引擎参数。

---

## 验证方案（端到端）

1. **语法检查**：`python -m py_compile src/backtest/position.py src/backtest/engine.py scripts/real_dd_validation.py`
2. **跑 (A) 标准两窗口8%**：
   ```
   python scripts/real_dd_validation.py
   ```
   预期输出：训练窗 Alpha+1.59%/夏普1.437/回撤-7.6%、测试窗 Alpha+47.13%/夏普3.362/回撤-7.5%，触发0次。与P3基线一致 → 确认不破坏。
3. **跑 (B) 4%强制触发**：脚本内含4%档，输出真实降仓的trade_count增加、含"回撤保护降仓"exit_signal、触发次数>0。
4. **跑 (C) 2022熊市8%**：输出8%真实触发、组合回撤 < 无保护对照、减损幅度量化。
5. **回归**：现有 `p3_full_validation.py` 仍能跑（它不用新参数，向后兼容），结果不变。

通过标准：(A)与P3基线一致、(B)真实卖出生效、(C)8%在熊市真实减损。

---

## 关键设计决策（已与用户确认）

| 决策 | 选择 | 理由 |
|------|------|------|
| 阈值 | 维持8% | 用户选择不降阈值，8%作"最后防线" |
| 方向 | 单向（只卖不买） | 用户选择；更保守、更真实，不盲目买回崩盘市场 |
| 触发依据 | 真实NAV（含降仓效果）的peak | 自洽、真实 |
| 恢复行为 | 仅切换状态标志，不买回 | 单向；让新信号自然重建仓位 |
| 参数传递 | 引擎构造参数 `dd_protection_config` | 向后兼容（None=不启用），不污染配置文件 |
