# 复权口径改造 — 项目实施计划 v1.0

| 项 | 内容 |
|---|---|
| 版本 | v1.0 ｜ 2026-10-22 |
| 基线 | `pre-adjust-20261016`（+123.26%，229 笔，快照 `data/baseline_snapshot_20261016.db`） |
| 对照入口 | `scripts/run_baseline_backtest.py`（唯一合法对照，禁止用 `run_backtest.py` 取基线） |
| 状态 | 设计已定，待开工 |
| 代码改动范围 | `src/backtest/*`（4 个文件）+ `src/data_fetcher/*`（1 处）+ 新增 2 个模块 |

---

## 0. 一页纸摘要

### 目标
1. **成交层真实化**：成交价、股数、现金、市值、NAV 全部改用不复权（raw）行情，使每笔交易可手算对账。
2. **事件层补齐**：送股/转增调整股数，现金分红进现金账，按持股期限计红利税。
3. **可复现**：消除 qfq 随新除权漂移导致的历史不可重放。

### 非目标（明确不做）
- **不动信号层**。信号继续用 qfq，一行不改。
- **不追求"更精确"的复权因子**。已实测精度被"一分钱"封顶（max 偏差 0.0066 元）。
- **不做多源数值融合**。多源只用于交叉校验，输出单一值。
- **不引入东财/腾讯行情源**（单位地雷：东财成交量单位是手、腾讯 `amount` 列实为成交量）。
- **不建模派息到账延迟**（明细无派息日字段，影响 <1bp）。

### 里程碑

| ID | 里程碑 | 关键产出 | 可独立交付 |
|---|---|---|---|
| **M1** | 数据基座 | `corporate_action` 表 + raw 双轨 + 三方校验报告 | ✅ 不动交易逻辑 |
| **M2** | 成交层真实账本 | 成交/市值/NAV 切 raw + 事件日处理 + 红利税 | ✅ |
| **M3** | 比率接口跨界 | ATR 以比率传递，消除"忘记换算"类 bug | ✅ |
| **M4** | 对账工具 | `scripts/verify_trade_ledger.py` 常驻回归 | ✅ |
| **M5** | 数据完整性 + 新基线 | 单批次保证 + 过期哨兵 + 归因报告 | ✅ |

**每个里程碑独立验收、可随时停。** M1 完成即可交付审计能力，不依赖后续。

---

## 1. 核心设计

### 1.1 口径分层（唯一契约）

```
                  ┌── 信号层: qfq  ── 只产出比率，不出绝对金额
行情双轨 ─────────┤
 raw + qfq        │   ATR/close、止损距离/close、距MA20百分比
                  │
事件表 (D, s) ─┐  │
               ▼  ▼
             成交层: raw + 事件日处理
             - T+1 成交价 = raw open
             - 除权日: shares ×= (1+s)；ref_price = (ref_price − D)/(1+s)
             - 现金: cash += D_pre × shares；平仓时扣红利税
             - 涨跌停基准 = 除权参考价 (prev_close − D)/(1+s)
             - NAV / 持仓市值 = raw close
```

**跨界唯一约定：信号层只传无量纲比率，成交层用 raw 价格自行还原成元。**
不做"逐处除以换算系数 K_t"——比率接口从结构上消灭"新增代码忘记换算"的可能，不靠纪律。

### 1.2 为什么信号层用 qfq 而不是 hfq（已决策，勿再改）

切 raw 之后，**qfq 漂移问题自动失效**：

- qfq 漂移 = 每次除权后整段历史乘一个常数（实测 600519 −2.31%、300308 −0.15%）
- 信号层全是比例型指标，实测整序列 ×3.7 / ×91.3 信号 **100% 一致**；滚动 vs 静态基准差 2.91e-14
- 漂移唯一能伤害的是绝对金额，而绝对金额已全部搬去 raw

选 qfq 而非 hfq 是为了**保住与基线的可比性**——否则"复权改造"与"精度变更"两个变量混进同一次归因，结果无法解释。

### 1.3 红利税：按持股期限三档（已决策）

| 持股期限 | 税率 | 依据 |
|---|---|---|
| ≤ 1 个月（含） | **20%** | 财税〔2015〕101号，全额计入应纳税所得额 |
| 1 个月 ~ 1 年（含） | **10%** | 暂减按 50% 计入 |
| > 1 年 | **0%** | 免征 |

**实测支撑（229 笔基线）**：持仓 mean 29 / p50 24 / **max 103 天** → **0% 档永不触发**；
资本加权综合税率 **13.49%**（≤1月档占 35.0%，1月~1年档占 65.0%），不是 10%。

**关键机制**：持股 ≤1 年的**派息时不扣缴**，待卖出时由中登按先进先出算持股期限、券商从资金账户扣收。
即**税率在卖出时才确定**——回测里 `holding_days` 已算好，实现只需一行：

```python
rate = 0.20 if holding_days <= 30 else 0.10   # 0% 档本策略不触发
```

### 1.4 派息基数口径（易错点，必须遵守）

税率既按 `holding_days` 自算，基数就必须是**含税 `dps_pre`**——
公告的"扣税后"值恒定按 10% 算（窗口内实测 39/39 事件隐含税率 = 10.00%），与实际 13.49% 不符。

| 场景 | 基数 | 税率来源 |
|---|---|---|
| `share_ratio == 0`（**窗口内 39/39 全是这种**） | `dps_pre` | `holding_days` 三档自算 |
| `share_ratio > 0` | `dps_after` | 不再叠加（送股按面值 1 元/股代扣的税已内扣） |

### 1.5 除权处理顺序（决定正确性）

**除权调整必须在当日止损检查之前执行。** 顺序对了，"除权日误止损"根本不会发生，无需特判：

```
除权日 prev_close = P，派息 D → 除权参考价 (P − D)/(1+s) ≈ 当日 raw 价格
同时 ref_price 同步调整为 (ref_price − D)/(1+s)
故 硬止损判据 current_price <= ref_price × (1 − 12%) 天然不成立
```

反之若 `ref_price` 不调整，价格跳空 D 会被当成暴跌 → 误触发止损。这是本次改造**最关键的一处正确性**。

---

## 2. 数据模型

### 2.1 `corporate_action` 表（落 `data/kline_cache.db`，不新建库文件）

现有库已含 `kline_cache`（40130 行）与迁移备份表，`_init_db` 已有成熟的迁移范式可复用。

```sql
CREATE TABLE IF NOT EXISTS corporate_action (
    symbol          TEXT NOT NULL,
    ex_date         TEXT NOT NULL,      -- 除权除息日
    record_date     TEXT,               -- 股权登记日：entry_date <= record_date 才有权领取
    dps_pre         REAL,               -- 每股派息（含税，元）  <- 计算基数
    dps_after       REAL,               -- 每股派息（税后，元，公告值） <- 仅校验/送股兜底
    share_ratio     REAL DEFAULT 0,     -- 送股比例 s（每1股送几股）
    transfer_ratio  REAL DEFAULT 0,     -- 转增比例
    multiplier      REAL,               -- 复权乘数：行情反推值（裁判）
    mult_baostock   REAL,               -- baostock backAdjustFactor 逐行比值
    mult_em         REAL,               -- 东财明细重构值 prev/((prev−D)/(1+s))
    verified        INTEGER DEFAULT 0,  -- 1=三方闭合(<0.01元) 0=待裁决
    reason          TEXT,               -- 未闭合原因，如"因子有明细无→疑似配股"
    checked_at      TEXT,               -- 本条最后校验时间 —— 过期哨兵用
    PRIMARY KEY (symbol, ex_date)
);
```

### 2.2 新增模块

| 模块 | 职责 |
|---|---|
| `src/data_fetcher/event_book.py` | `EventBook` 类：事件表读写、除权调整计算、过期哨兵 |
| `src/data_fetcher/event_validator.py`（脚本化） | 三方交叉校验，产出 JSON 报告 |

`EventBook` 对外接口：

```python
class EventBook:
    def events_between(symbol, start, end) -> List[CorporateAction]
    def apply_to_position(symbol, ex_date, shares, ref_price, highest_price)
        -> (shares, ref_price, highest_price, cash_dividend, s)
    def adjusted_prev_close(symbol, date, raw_prev_close) -> float   # 涨跌停基准
    def staleness(symbols) -> (max_age_days, missing_symbols)       # 哨兵
```

### 2.3 raw 行情双轨

`kline_cache` 主键已是 `(symbol, date, adjust)`——**零迁移**，直接以 `adjust=''` 写入 raw 行即可。

---

## 3. 任务分解（WBS）

### M1 数据基座 〔依赖：无〕

| ID | 任务 | 改动文件 | 验收判据 |
|---|---|---|---|
| T1.1 | 建 `corporate_action` 表 | `src/data_fetcher/cache.py` `_init_db` | 表建成功；重复初始化幂等 |
| T1.2 | `EventBook` 模块 | 新增 `src/data_fetcher/event_book.py` | 单测覆盖 4 个接口 |
| T1.3 | raw 双轨落库 | `manager.py`（新增 `get_daily_kline_raw`） | `adjust=''` 行可读写，与 qfq 行数一致 |
| T1.4 | 三方交叉校验脚本 | 新增 `scripts/validate_corporate_actions.py` | 见下 |
| T1.5 | 全 46 只跑通 + 出报告 | — | 闭合率 + 未闭合事件清单（含配股/增发） |

**T1.4 校验规则**：

| 来源 | 取数 | 角色 |
|---|---|---|
| 新浪 `hfq/raw` 比值跳变 | 已有链路 | **裁判**（市场实际发生的乘数） |
| baostock `backAdjustFactor` 逐行比值 | `query_adjust_factor` | 厂商因子 |
| 东财 `stock_fhps_detail_em` | 公告 D、s 重构 | 拆分记账 |

**判据（绝对金额，不是相对误差）**：`|Δ乘数| × 前收 < 0.01 元`
容差必须随价格自适应——4 元股的一分钱相对分辨率是 2.5e-3，1500 元股是 6.7e-4，统一 `1e-5` 会把舍入噪声全判成事件。

**已验证基线**：10 只标的 50 个事件，双向查全 **0 漏 0 多**，绝对偏差 p50 0.00062 元 / max 0.00661 元，超过 0.01 元的 **0/50**。

**禁止事项**：baostock 的 `foreAdjustFactor` **不能用**（不是逐事件乘数，实测 2021-06-11 给 0.8218 而实际 1.0212）。只能用 `backAdjustFactor` 的逐行比值。

### M2 成交层真实账本 〔依赖：M1〕

| ID | 任务 | 改动位置 | 验收判据 |
|---|---|---|---|
| T2.1 | 日循环插除权处理（**必须在止损检查之前**） | `engine.py:259` 循环开头 | 除权日先调整持仓再检查止损 |
| T2.2 | 持仓除权调整：shares / ref_price / highest_price | `position.py` 新增 `apply_corporate_action()` | 送转 shares ×= (1+s)；ref_price/highest_price = (x−D)/(1+s) |
| T2.3 | 分红进现金 + 累计 `dividend_gross` | `position.py` + `Trade` dataclass | `cash += dps_pre × shares` |
| T2.4 | 平仓结算红利税 | `position.py:close_position` / `reduce_position` | `tax = dividend_gross × rate`；按比例拆分（减仓场景） |
| T2.5 | PnL 含净分红 | `position.py:48-51` `Trade.close` | `pnl = (exit−entry)×shares + dividend_net − commission` |
| T2.6 | 涨跌停基准改除权参考价 | `broker.py:76-97` + `signal_executor.py:81` | 除权日用 `(prev_close−D)/(1+s)` 作基准 |
| T2.7 | 成交价/市值/NAV 切 raw | `broker.py:121`、`calendar.py:92`、`position.py:363` | 全部读 raw 序列 |
| T2.8 | 硬止损改判 `ref_price` | `position.py:448` | `current_price <= ref_price × (1 − hard_stop_pct)` |

**回归要求**：M2 完成后**必须**跑 `scripts/run_baseline_backtest.py` 确认信号层未受影响（229 笔、+123.26% 应完全一致——因为信号层未动，且 M3 之前 ATR 仍是 qfq 元）。
若此处出现差异，说明误改了信号层或数据，**停下来查，不要继续**。

### M3 比率接口跨界 〔依赖：M2〕

| ID | 任务 | 改动位置 | 验收判据 |
|---|---|---|---|
| T3.1 | ATR 以比率传递 | `signal_executor.py:104-116` → `atr_ratio = atr / close` | 传入 `atr_ratio` 而非 `atr_value` |
| T3.2 | 开仓用比率算仓位与止损价 | `position.py:225-253`、`283-285` | `stop_dist = atr_ratio × mult × entry_price`；`stop_price = entry_price × (1 − atr_ratio × mult)` |
| T3.3 | 止损/移动止盈用比率 + ref_price | `position.py:479-494` | `stop_dist = atr_ratio × mult × ref_price` |

**验收**：构造除权日附近用例——除权日不误触发止损、除权后止损价同步下移。

### M4 对账工具 〔依赖：M2〕

| ID | 任务 | 改动文件 | 验收判据 |
|---|---|---|---|
| T4.1 | `scripts/verify_trade_ledger.py` | 新增 | 独立重算任意一笔，输出逐项对账表 |
| T4.2 | 选 10 笔手算对账（5 笔跨除权日 + 5 笔普通） | — | **一致到分** |
| T4.3 | 纳入单测 | `tests/unit/test_verify_trade_ledger.py` | 常驻回归 |

**对账表输出格式**（这就是将来对接券商流水的雏形）：

```
标的/期间: 600887  2025-06-03 → 2025-08-12  持股 70 天
  买入  1100 股 @ 25.97 (raw open)              -28,567.00
  除权日 2025-06-06  每股派 1.22（含税）
        分红现金  1100 × 1.22                   +1,342.00
        红利税    持股70天 → 10%                  -134.20
  卖出  1100 股 @ 27.14                         +29,854.00
  佣金/印花税/过户费                                  -xx.xx
  净盈亏                                       +2,xxx.xx
  === 与回测记录差异: 0.00 ===
```

### M5 数据完整性 + 新基线 〔依赖：M3、M4〕

| ID | 任务 | 改动位置 | 验收判据 |
|---|---|---|---|
| T5.1 | 回测取数单批次保证 | `manager.py` / 基线脚本 | 禁止跨批次切片（缓存 38/46 只是 9 批次拼接的） |
| T5.2 | 过期哨兵 | `EventBook.staleness` | `MIN(checked_at)` 距今 >90 天 或 标的表内 0 行 → **warn 不阻断** |
| T5.3 | 重跑基线 | `scripts/run_baseline_backtest.py` | tag `raw-ledger-YYYYMMDD` |
| T5.4 | 逐笔差异归因 | 新增报告 | 归因到 4 来源，**零"不明原因"** |

**T5.4 归因四来源**：①一手取整路径分叉 ②红利税 ③佣金最低阈值 ④除权日涨跌停判定变化。

**预期变化（先说清，免得吵"为什么结果全变了"）**：
- 总收益下修 **约 1pp 量级**（红利税 + 整手粒度），不是十几个点
- 交易路径会分叉：仓位市值中位数 8503 元、一手 2065 元（**一手 ≈ 仓位 24%**），
  qfq→raw 尺度变化 5~9% 会让相当比例交易跳一手
- **验收不要求逐笔对得上旧基线**，只要求差异可归因

---

## 4. 代码坐标速查表

| 环节 | 位置 | 现状 | 归属 |
|---|---|---|---|
| 取数默认 adjust | `manager.py:69` | `adjust="qfq"` | M1 |
| 缓存读写 | `cache.py:117/145` | 已支持 adjust 维度 | 无需改 |
| 信号切片 | `engine.py:317-321` | `df_slice` → `analyze` | **不动** |
| T+1 成交价 | `broker.py:121` | `row["open"]` | M2 → raw |
| T+1 前收基准 | `broker.py:122` | `prev_row["close"]` | M2 → raw + 除权参考价 |
| 涨跌停判定 | `broker.py:92-93` | `prev_close × (1±pct)` | M2 → 除权参考价 |
| 板块涨跌停幅度 | `broker.py:63-74` | **已实现** 科创/创业20%、北交所30% | 不动 |
| 开仓股数 | `position.py:225-253` | ATR(元) × price | M3 → 比率 |
| 开仓止损价 | `position.py:283-285` | `entry − ATR×mult` | M3 → 比率 |
| 持仓市值 | `position.py:358-363` | `shares × close` | M2 → raw |
| 止损检查入口 | `engine.py:269-276` | `prices_today` | M1 插除权处理在前 |
| 硬止损 | `position.py:448` | `entry × (1−12%)` | M2 → ref_price |
| ATR 止损/移动止盈 | `position.py:479-494` | `atr_val`（元） | M3 → 比率 + ref_price |
| 最高价更新 | `position.py:438` | `highest_price` | M2 → 除权调整 |
| 回撤降仓 | `engine.py:337-355` / `position.py:317` | 部分平仓 | M2 → 分红按比例拆分 |
| 日净值 NAV | `engine.py:359` / `calendar.py:92` | `close` | M2 → raw |
| 红利税 | 无 | — | M2 新增 |
| 交易 PnL | `position.py:48-51` | `(exit−entry)×shares−comm` | M2 → 加净分红 |

---

## 5. 测试计划

| 层级 | 内容 | 位置 |
|---|---|---|
| 单元 | `EventBook` 除权调整、税率三档、比率换算 | `tests/unit/test_event_book.py`（新增） |
| 单元 | `Broker` 除权参考价涨跌停判定 | `tests/unit/test_broker.py`（扩充） |
| 单元 | `PositionManager` 除权后 shares/ref_price/highest_price | `tests/unit/test_position_manager.py`（扩充） |
| 单元 | ATR 比率接口仓位与止损价 | `tests/unit/test_position_manager.py`（扩充） |
| 集成 | 对账脚本 10 笔手算 | `tests/integration/`（新增） |
| 回归 | **全量单测**（当前 250 个 test 函数） | `python -m pytest tests/ -q` |

**纪律**：每个 M 结束必须全量单测通过 + 基线可复现。任一 M 失败不得进入下一 M。

---

## 6. 风险登记

| # | 风险 | 影响 | 应对 |
|---|---|---|---|
| R1 | 除权处理插在止损检查之后 → 除权日误止损 | 高（静默错误） | T2.1 强制顺序；构造除权用例单测 |
| R2 | `highest_price` 未除权调整 → 移动止盈误触发 | 高 | T2.2 一并调整；单测覆盖 |
| R3 | 用公告"扣税后"值当基数 → 税率被锁死在 10% | 中 | §1.4 明确用 `dps_pre`；代码注释写死规则 |
| R4 | 减仓场景分红未按比例拆分 | 中（二阶） | T2.4 按比例拆分；显式标注已知简化 |
| R5 | 误传 baostock `foreAdjustFactor` | 高（因子全错） | 校验脚本只取 `backAdjustFactor`；注释警告 |
| R6 | 跨批次拼接台阶污染 | 中 | T5.1 单批次保证 |
| R7 | 新基线"结果全变"引发争议 | 中（沟通） | 提前声明预期下修 1pp；验收只要求可归因 |
| R8 | 事件表过期未更新 → 静默漏分红 | 中 | T5.2 哨兵 warn |

---

## 7. 回滚方案

| 层级 | 手段 |
|---|---|
| 代码 | 每个 M 一个 commit，可逐 M revert |
| 数据 | `corporate_action` 是**新增表**，删除即回到 M1 前状态，`kline_cache` 不受影响 |
| 行情 | raw 以 `adjust=''` 独立存储，与 qfq 行共存，删 raw 行不影响现有 qfq |
| 基线 | `data/baseline_snapshot_20261016.db` 冻结快照永久保留，随时可复现原基线 |

**关键**：整个改造**不修改任何现有 qfq 数据行**，因此任何时刻都能安全回滚。

---

## 8. 完成定义（DoD）

- [ ] M1：46 只标的事件表闭合率报告 + 未闭合清单；配股/增发标的已识别
- [ ] M2：`run_baseline_backtest.py` 结果与基线一致（证明信号层未受影响）
- [ ] M3：除权日用例单测通过
- [ ] M4：10 笔手算对账**一致到分**
- [ ] M5：新基线 tag `raw-ledger-YYYYMMDD`；逐笔差异归因 4 来源，**零"不明原因"**
- [ ] 全量单测通过（≥250 + 新增）
- [ ] 红利税口径已按 §1.4 落地（`dps_pre` 为基数）

---

## 附录 A 术语

| 术语 | 含义 |
|---|---|
| **raw** | 不复权，真实成交价。除权日有跳空坑。**唯一物理事实** |
| **qfq** | 前复权，把除权日之前的历史价格压低抹平跳空，最新一天 = 真实价 |
| **hfq** | 后复权，把除权日之后的价格抬高，最老一天 = 真实价，绝对价位无经济含义 |
| **D** | 每股派息（元）。`dps_pre` 含税 / `dps_after` 税后 |
| **s** | 送转比例（每 1 股送几股） |
| **除权参考价** | `(前收 − D) / (1 + s)`，除权日涨跌停判定基准 |
| **ref_price** | 除权调整后的成本基准，用于止损/止盈百分比计算（区别于实际成交价 `entry_price`） |

## 附录 B 已验证结论（可直接引用，无需重测）

| # | 结论 | 证据 |
|---|---|---|
| 1 | raw 跨源位级一致 | 新浪 vs 腾讯 4 只 × 906 天 × OHLC，最大差 0.0000 元 |
| 2 | 复权因子准确到物理极限 | 50 事件双向查全 0 漏 0 多，max 偏差 0.0066 元 < 一个最小报价单位 |
| 3 | 复权不调整 volume | 新浪/baostock raw vs qfq volume 逐日 0 差异 |
| 4 | 腾讯 `hfq` 是坏数据 | 906 天中 849 天检出"跳变"；corr(ratio变化, 当日收益) = −0.86 |
| 5 | 东财成交量单位是**手** | 新浪volume/东财成交量 = 100.00003 |
| 6 | 腾讯 `amount` 列实为成交量（手） | 新浪volume/腾讯amount = 100.0000 |
| 7 | 公告比例 ≠ 实际执行比例 | 300308 "10转4" 名义 s=0.4，行情反推实际 0.3918（回购股不参与分配） |
| 8 | 窗口内事件全为纯派息 | 12 只 39 事件，零送股零转增；公告隐含税率 39/39 = 10.00% |
| 9 | 无加仓，单一 lot | 229 笔按 symbol 区间重叠 0 对，`holding_days` 可信 |
| 10 | qfq 跨时间漂移 | 600031 −0.944% / 600519 −2.312% / 300308 −0.154%（单交易日） |
