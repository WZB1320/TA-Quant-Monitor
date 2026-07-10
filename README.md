# QuantMonitor — 量化交易监控系统

一套面向量化投资交易员的 **A 股多因子策略回测与 AI 分析平台**，支持自选股分组管理、分组级策略参数配置、历史回测、绩效可视化，以及基于 LLM 的策略周报自动生成与反馈闭环。

## 功能概览

| 模块 | 说明 |
|------|------|
| **自选股管理** | 分组管理自选股（科技成长型 / 消费稳健型 / 周期资源型 …），支持搜索添加、换组、删除 |
| **策略参数配置** | 按分组独立配置指标参数、指标权重、体制权重、信号引擎参数，互不干扰 |
| **实时信号分析** | 实时分析自选股信号，按等级/方向分组展示，支持体制切换对比 |
| **组合回测分析** | 选定时间区间与分组运行组合回测，展示净值曲线、回撤曲线、6 项核心绩效指标、交易明细 |
| **个股回测分析** | 单只股票深度回测，支持多体制对比、信号分布、交易明细 |
| **AI 策略周报** | 基于策略记忆数据自动生成 AI 分析周报，含盈亏归因、策略有效性、风险提示、改进建议 |
| **建议反馈闭环** | AI 建议自动追踪（pending → applied → validated），参数变更自动检测，历史建议效果回顾 |
| **流式报告生成** | SSE 流式输出周报，实时展示生成过程，结构化建议块实时过滤不外露 |

## 技术架构

```
┌──────────────────────────────────────────────────────────────────────┐
│  前端  (React + Vite + TypeScript + Ant Design)                   │
│  ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐          │
│  │ 自选股   │ │ 策略配置 │ │ 信号分析 │ │ 回测分析 │          │
│  └────┬─────┘ └────┬─────┘ └────┬─────┘ └────┬─────┘          │
│       └───────────────┼─────────────┼──────────────┘                   │
│                     └───────────────┬──────────────────────────────┘
│                              Axios / REST + SSE 接口                 │
└──────────────────────────────────────┬─────────────────────────────────┘
                                 │
┌────────────────────────────────┼───────────────────────────────────┐
│  后端  (FastAPI + Uvicorn)                                      │
│  ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐       │
│  │ 自选股   │ │ 策略配置 │ │ 信号分析 │ │ 回测分析 │       │
│  │ 增删改查 │ │  接口    │ │  接口    │ │  接口    │       │
│  └────┬─────┘ └────┬─────┘ └────┬─────┘ └────┬─────┘       │
│  ┌──────────┐ ┌──────────┐ ┌──────────┐                     │
│  │ AI 知识库 │ │建议追踪器│ │周报生成器│                     │
│  │ 周报生成 │ │  服务   │ │ SSE 流  │                     │
│  └──────────┘ └──────────┘ └──────────┘                     │
│       └───────────────────┼──────────────────────┘                │
│                       JSON 文件存储                               │
└───────────────────────────────┬──────────────────────────────────┘
                                │
┌───────────────────────────────┼──────────────────────────────────┐
│  核心引擎  (Python)                                             │
│  ┌──────────┐ ┌──────────┐ ┌──────────┐                    │
│  │ 数据获取 │ │ 指标计算 │ │ 信号引擎 │                    │
│  │ AkShare  │ │ 计算管线 │ │ 评分器   │                    │
│  └──────────┘ └──────────┘ └──────────┘                    │
│  ┌──────────┐ ┌──────────┐ ┌──────────┐                    │
│  │ 回测引擎 │ │ 仓位管理 │ │ 市场过滤 │                    │
│  └──────────┘ └──────────┘ └──────────┘                    │
│  ┌──────────┐ ┌──────────┐                                    │
│  │ 体制检测 │ │ 记忆层   │                                    │
│  └──────────┘ └──────────┘                                    │
│  ┌──────────────────────────────────────────┐                  │
│  │         AI 分析层 (LLM 客户端 + 周报生成器) │                  │
│  └──────────────────────────────────────────┘                  │
└───────────────────────────────────────────────────────────────────┘
```

## 目录结构

```
.
├── config/
│   └── strategy_config.json          # 策略参数配置（分组级）
├── data/
│   ├── watchlist.json               # 自选股列表
│   ├── kline_cache.db             # K线数据缓存 (SQLite)
│   ├── stock_knowledge.db            # 股票知识库 (SQLite)
│   ├── signal_history.json        # 信号去重历史
│   ├── user_preferences.json      # 用户偏好（体制选择等）
│   ├── strategy_memory.jsonl      # 策略记忆（旧格式，向后兼容）
│   ├── strategy_memory_YYYY-MM.jsonl # 策略记忆（按月切分新格式）
│   ├── report_suggestions.jsonl    # AI 建议追踪记录
│   ├── backtest_memory/          # 回测记忆文件（每次 run 一个文件）
│   │   └── bt_{timestamp}_{id}.jsonl
│   └── reports/                   # 生成的周报
│       ├── index.json             # 周报索引
│       └── weekly_{timestamp}.md
├── src/                           # 核心引擎 (Python)
│   ├── ai/                        # AI 分析层
│   │   ├── llm_client.py         #   LLM 客户端（支持多模型适配）
│   │   ├── report_generator.py  #   周报生成器（统计聚合 + LLM 调用）
│   │   ├── scheduler.py         #   周报调度器（按自然周切分）
│   │   └── suggestion_tracker.py #   建议追踪器（反馈闭环）
│   ├── backtest/                  # 回测引擎
│   │   ├── engine.py             #   回测主循环
│   │   ├── broker.py             #   模拟撮合
│   │   ├── position.py           #   仓位管理
│   │   ├── metrics.py            #   绩效指标
│   │   ├── report.py             #   回测报告
│   │   ├── market_filter.py      #   市场过滤器（大盘风控）
│   │   ├── regime_detector.py    #   体制检测器
│   │   ├── signal_executor.py    #   信号执行器
│   │   └── calendar.py           #   交易日历
│   ├── config/                    # 配置管理
│   │   ├── settings.py           #   全局设置
│   │   ├── group_config.py      #   分组配置加载
│   │   ├── user_preferences.py   #   用户偏好管理
│   │   └── runtime_mode.py       #   运行模式（LIVE/BACKTEST）
│   ├── data_fetcher/              # 数据获取
│   │   ├── akshare_source.py     #   AkShare 数据源
│   │   ├── baostock_source.py   #   BaoStock 数据源
│   │   ├── cache.py             #   K线缓存
│   │   ├── manager.py           #   数据管理器
│   │   └── watchlist.py         #   自选股加载
│   ├── indicators/                # 技术指标
│   │   ├── trend.py             #   趋势指标 (MA60, EMA, MACD)
│   │   ├── strength.py          #   强度指标 (ADX, ATR)
│   │   ├── momentum.py         #   动量指标 (RSI, KDJ)
│   │   ├── volume.py            #   量价指标 (OBV, VOL_RATIO)
│   │   └── pipeline.py          #   指标计算管线
│   ├── signal_engine/             # 信号引擎
│   │   ├── engine.py            #   信号生成主逻辑
│   │   ├── scorer.py           #   多因子评分
│   │   ├── filter.py            #   信号过滤
│   │   ├── validator.py        #   信号校验
│   │   ├── classifier.py       #   信号分类器
│   │   └── signals.py          #   信号定义与等级
│   └── memory/                    # 策略记忆层
│       └── strategy_memory.py  #   记忆记录器（信号+交易）
├── web/
│   ├── backend/                 # Web 后端
│   │   ├── app.py              #   FastAPI 入口
│   │   ├── schemas.py          #   Pydantic 模型
│   │   ├── routers/            #   API 路由
│   │   │   ├── watchlist.py   #     自选股 CRUD
│   │   │   ├── stocks.py      #     股票搜索
│   │   │   ├── signals.py     #     信号分析
│   │   │   ├── config.py      #     策略配置
│   │   │   ├── backtest.py    #     组合回测
│   │   │   ├── stock_backtest.py #   个股回测
│   │   │   └── ai_report.py   #     AI 周报与建议
│   │   ├── services/            #   业务服务
│   │   │   ├── knowledge_base.py   # 股票知识库
│   │   │   └── signal_analyzer.py # 信号分析服务
│   │   └── schemas/
│   │       └── stock.py         #   数据模型
│   └── frontend/               # Web 前端
│       └── src/
│           ├── api/index.ts    #   Axios 接口封装
│           ├── components/     #   公共组件
│           ├── pages/          #   页面组件
│           │   ├── Watchlist/  #     自选股管理
│           │   ├── Config/       #     策略配置
│           │   ├── Signals/    #     信号分析
│           │   ├── Backtest/   #     组合回测
│           │   ├── StockBacktest/ #  个股回测
│           │   └── AIReport/   #     AI 周报
│           └── stores/         #   Zustand 状态管理
├── scripts/                     # 辅助脚本
│   ├── current_signals.py     #   查看当前信号
│   ├── analyze_trades.py     #   交易分析
│   ├── plot_equity.py        #   净值曲线绘制
│   ├── analyze_positions.py  #   持仓分析
│   ├── compare_returns.py   #   收益对比
│   ├── review_trades.py     #   交易复盘
│   ├── sync_stock_kb.py      #   同步股票知识库
│   └── test_regime_presets.py #   测试体制预设
├── tests/                      # 单元测试
│   ├── unit/                   #   单元测试
│   └── integration/            #   集成测试
├── run_backtest.py             # 一键回测入口
├── generate_weekly_report.py # 生成周报入口
├── start.bat                   # Windows 启动脚本
└── .env.example              # 环境变量示例
```

## 快速开始

### 环境要求

- Python 3.10+
- Node.js 18+
- npm 9+

### 1. 安装 Python 依赖

```bash
pip install fastapi uvicorn akshare baostock pandas numpy
```

### 2. 配置 LLM API（可选，用于 AI 周报）

复制 `.env.example` 为 `.env` 并配置：

```env
DEEPSEEK_API_KEY=sk-your-api-key-here
DEEPSEEK_BASE_URL=https://api.deepseek.com
DEEPSEEK_MODEL=deepseek-chat
```

支持的模型提供商：DeepSeek、豆包、通义千问（可在 `llm_client.py` 中配置。

### 3. 启动后端

```bash
cd web/backend
uvicorn app:app --reload --port 8000
```

后端接口文档: http://localhost:8000/docs

### 4. 安装前端依赖并启动

```bash
cd web/frontend
npm install
npm run dev
```

前端页面: http://localhost:5173

### 5. 命令行回测 (无需前端)

```bash
python run_backtest.py
```

### 6. 命令行生成 AI 周报

```bash
python generate_weekly_report.py
```

## API 接口总览

### 自选股管理

| 方法 | 路径 | 说明 |
|------|------|------|
| GET | `/api/watchlist/groups` | 获取所有分组及股票 |
| POST | `/api/watchlist/groups/{group}` | 获取指定分组股票 |
| POST | `/api/watchlist/stocks` | 添加股票到分组 |
| DELETE | `/api/watchlist/stocks/{code}` | 从分组删除股票 |
| PUT | `/api/watchlist/stocks/{code}/group` | 股票换组 |
| GET | `/api/stocks/search?q=` | 搜索股票 |

### 策略配置

| 方法 | 路径 | 说明 |
|------|------|------|
| GET | `/api/config/groups` | 获取所有分组配置 |
| GET | `/api/config/groups/{group}` | 获取指定分组配置 |
| PUT | `/api/config/groups/{group}` | 更新分组配置 |
| GET | `/api/config/groups/{group}/reset` | 重置分组参数为默认 |
| GET | `/api/config/user/regime` | 获取用户选择的体制 |
| PUT | `/api/config/user/regime` | 设置用户体制偏好 |

### 信号分析

| 方法 | 路径 | 说明 |
|------|------|------|
| GET | `/api/signals/analyze` | 分析自选股信号 |
| GET | `/api/signals/analyze/{group}` | 分析指定分组信号 |

### 回测分析

| 方法 | 路径 | 说明 |
|------|------|------|
| POST | `/api/backtest/run` | 运行组合回测 |
| POST | `/api/stock-backtest/run` | 运行个股回测 |

### AI 周报

| 方法 | 路径 | 说明 |
|------|------|------|
| POST | `/api/ai/weekly-report` | 生成周报（指定记忆文件） |
| POST | `/api/ai/weekly-report/auto` | 按自然周生成周报（反馈闭环） |
| POST | `/api/ai/weekly-report/stream` | 流式生成周报（SSE） |
| GET | `/api/ai/llm-status` | 检查 LLM 配置状态 |
| GET | `/api/ai/suggestions` | 列出所有 AI 建议 |
| GET | `/api/ai/suggestions/pending` | 列出待应用建议 |
| GET | `/api/ai/reports` | 列出所有周报（索引） |
| GET | `/api/ai/reports/{report_id}` | 检索单篇周报内容 |

## 策略体系

### 多因子评分模型

信号引擎对每只股票计算综合得分，由 **4 大类 8 个指标** 加权汇总：

| 类别 | 指标 | 作用 |
|------|------|------|
| 趋势 | MA60, EMA 双线, MACD | 判断多空方向与趋势强度 |
| 强度 | ADX, ATR | 识别市场体制（趋势/震荡）与波动率 |
| 动量 | RSI, KDJ | 捕捉超买超卖与拐点信号 |
| 量价 | OBV, 量比 | 验证量价共振与资金流向 |

### 体制自适应权重

系统根据 ADX 自动识别当前市场体制，动态切换指标权重：

- **趋势市** (ADX > 25): 趋势类权重提升至 40%
- **震荡市** (ADX < 20): 动量类权重提升至 50%
- **过渡期** (20 < ADX < 25): 均衡配置

用户也可通过前端手动强制切换体制，对比不同体制下的信号表现。

### 分组差异化策略

每个自选股分组拥有独立的策略参数：

| 分组 | 特点 | 得分阈值 | ATR止损 | RSI周期 | 核心差异 |
|------|------|----------|---------|---------|----------|
| 科技成长型 | 高Beta、题材驱动 | 45 | 2.0× | 9 | RSI短周期+OBV高权重 |
| 消费稳健型 | 慢牛低波动 | 30 | 1.8× | 21 | MA60高权重+MACD标准 |
| 周期资源型 | 大宗驱动、周期强 | 25 | 2.5× | 21 | ADX高权重+量价并重 |
| 医药创新型 | 事件驱动、均值回归 | 40 | 2.5× | 21 | RSI/KDJ高权重+MACD降权 |
| 机械制造型 | 趋势与量价并重 | 40 | 2.5× | 21 | ADX+VOL_RATIO强化 |

### 仓位与风控

- **ATR 动态止损**: 入场价 - N×ATR，N 按分组配置（1.8 ~ 2.5）
- **移动止盈**: 盈利 10% 以下 2.5×ATR → 10~20% 2.0×ATR → 20% 以上 1.5×ATR
- **连亏保护**: 连续亏损 N 笔后暂停交易 M 天（按分组配置）
- **冷却期**: 卖出后 N 天内不再买入同一标的
- **市场过滤**: 大盘趋势向下时限制开仓（可配置）

## AI 周报与反馈闭环

### 工作原理

系统基于策略记忆层数据，调用 LLM 自动生成专业的策略分析周报，并形成完整的建议反馈闭环：

```
策略记忆 → 统计聚合 → LLM 生成 → 结构化建议 → 保存追踪
    ↑                                            ↓
    └──── 效果验证 ← 参数变更检测 ← 应用建议 ←──────┘
```

### 周报内容结构

1. **本周概览** — 数据区间、信号/交易总数、整体收益
2. **盈亏归因分析** — 按信号等级/退出原因/体制分解
3. **策略有效性分析** — 参数组合表现、信号得分相关性、执行约束影响
4. **风险提示** — 连续亏损、胜率、持仓时间、最大单笔亏损
5. **改进建议** — 具体可执行的参数微调方向

### 建议状态流转

```
pending（待应用）
  ├──→ applied（已应用）─→ validated（已验证）
  └──→ superseded（已过期/被取代）
```

- **pending**: AI 生成建议，等待用户应用
- **applied**: 检测到参数已变更为建议值，自动标记为已应用
- **validated**: 应用后有足够交易数据验证效果（待实现）
- **superseded**: 超过 4 周未应用，被新建议取代

### 流式生成（SSE）

周报生成支持 Server-Sent Events 流式输出：

- LLM 输出逐块推送到前端，用户无需等待完整报告
- 结构化建议块（`<<SUGGESTIONS>>...<<END>>）实时过滤，不外露给用户
- 安全余量机制处理跨块标记边界

## 策略记忆层

### 记录类型

记忆层记录两类数据：

| 类型 | 触发时机 | 关键字段 |
|------|----------|----------|
| **SignalRecord** | 信号产出时 | 标的/时间/regime/等级/得分/参数快照/指标快照/执行状态 |
| **OutcomeRecord** | 交易平仓时 | 入场/出场/盈亏/退出归因/市场上下文 |

两类记录通过 `(symbol, analysis_date, run_id)` 关联。

### 存储方式

- **实盘记忆**: `data/strategy_memory_YYYY-MM.jsonl`（按月切分）
- **回测记忆**: `data/backtest_memory/{run_id}.jsonl`（每次 run 独立文件）

### 版本追踪

`strategy_version` 是参数快照的 MD5 哈希（前 8 位），用于：

- 检测策略参数变更检测
- 不同参数版本间的绩效对比
- AI 建议应用效果验证

## 运行模式

系统支持两种运行模式，通过 `RuntimeMode` 控制：

| 模式 | 说明 | 信号历史 | 用户偏好 |
|------|------|----------|----------|
| **LIVE** | 实盘模式 | 读写 `signal_history.json` | 持久化到 `user_preferences.json` |
| **BACKTEST** | 回测模式 | 不写盘（内存去重） | 不持久化 |

回测模式确保回测不会污染实盘信号数据。

## 数据源

- **AkShare**: A 股实时行情、股票搜索、财务数据
- **BaoStock**: 历史日 K 线数据
- **本地缓存**: SQLite 缓存 K 线数据，减少重复请求
- **股票知识库**: SQLite 存储股票基础信息，支持本地搜索

## 自定义配置

### 添加自选股

编辑 `data/watchlist.json`，或通过前端页面操作：

```json
{
  "科技成长型": [
    { "code": "000725", "name": "京东方A", "market": "sz" }
  ]
}
```

### 调整策略参数

在 `config/strategy_config.json` 的 `groups` 中修改对应分组的参数，或通过前端策略配置页面调整。

### 新增分组

1. 在 `watchlist.json` 中添加新分组及股票
2. 在 `strategy_config.json` 的 `groups` 中添加对应策略参数
3. 未配置的分组自动使用 `_default` 参数

## 开源协议

MIT

## 作者声明

本项目为个人学习研究作品，**个人用户免费使用**。禁止任何形式的商业销售、转售或用于盈利性服务。欢迎 Star 与 Fork，转载请注明出处。

**风险提示**：本系统生成的所有信号、报告、建议仅供学习研究参考使用，不构成任何投资建议。股市有风险，投资需谨慎。
