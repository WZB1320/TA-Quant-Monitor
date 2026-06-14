# QuantMonitor AI 助手实现方案

## 一、需求概述

在回测分析完成后，允许用户针对分析结果与 AI 助手进行对话提问。AI 助手需支持：
- 理解回测结果数据（净值曲线、交易明细、绩效指标）
- 回答用户关于策略表现、交易逻辑、改进建议等问题
- **支持联网查询**（实时新闻、公告、行业数据等）

## 二、前端交互方案

### 2.1 UI 形态：悬浮对话面板

采用右下角悬浮按钮 + 侧边抽屉/弹层面板的形式，不干扰现有页面布局：

```
┌─────────────────────────────────────────────┐
│  BacktestPage                                │
│  ┌──────────────────────────────────────┐   │
│  │  回测结果图表...                      │   │
│  │                                      │   │
│  └──────────────────────────────────────┘   │
│                                    ┌─────┐  │
│                                    │ 🤖  │  │  ← 悬浮按钮 (Fab)
│                                    └─────┘  │
└─────────────────────────────────────────────┘

点击后展开:
┌─────────────────────────────────────────────┐
│  BacktestPage                    ┌────────┐ │
│  ┌────────────────────────────┐  │ AI助手  │ │
│  │                            │  │ ─────── │ │
│  │                            │  │ [消息流] │ │
│  │                            │  │ 用户:.. │ │
│  │                            │  │ AI: ... │ │
│  │                            │  │         │ │
│  │                            │  │ [输入框] │ │
│  └────────────────────────────┘  └────────┘ │
└─────────────────────────────────────────────┘
```

### 2.2 组件规划

| 组件 | 路径 | 职责 |
|------|------|------|
| `AIChatWidget` | `components/AIChat/AIChatWidget.tsx` | 悬浮按钮 + 面板容器 |
| `AIChatPanel` | `components/AIChat/AIChatPanel.tsx` | 对话面板（消息列表 + 输入区） |
| `ChatMessage` | `components/AIChat/ChatMessage.tsx` | 单条消息渲染（支持 Markdown） |
| `useAIChat` | `components/AIChat/useAIChat.ts` | 对话状态管理、SSE 连接 |

### 2.3 状态管理（Zustand Store）

```typescript
// stores/aiChat.ts
interface AIChatState {
  isOpen: boolean
  messages: ChatMessage[]
  isLoading: boolean
  context: BacktestContext | null  // 当前回测上下文
  toggle: () => void
  sendMessage: (content: string) => Promise<void>
  setContext: (ctx: BacktestContext) => void
  clearMessages: () => void
}

interface ChatMessage {
  id: string
  role: 'user' | 'assistant' | 'system'
  content: string
  timestamp: number
  sources?: SearchResult[]  // 联网查询来源
}

interface BacktestContext {
  result: BacktestResult      // 绩效指标
  trades: Trade[]             // 交易明细（前20条摘要）
  equityCurve: EquityPoint[]  // 净值曲线（采样）
  groupName: string           // 当前分组
  dateRange: [string, string] // 回测区间
}
```

### 2.4 接入点

在 `BacktestPage` 中引入 `AIChatWidget`，当回测完成（`result` 有值）时自动将结果数据注入 AI Chat Context：

```tsx
// pages/Backtest/index.tsx
import { AIChatWidget } from '../../components/AIChat'

export default function BacktestPage() {
  const [result, setResult] = useState<any>(null)
  // ...

  return (
    <div>
      {/* 现有回测结果展示 */}
      {/* ... */}
      <AIChatWidget backtestResult={result} />
    </div>
  )
}
```

## 三、后端 AI 服务架构

### 3.1 整体架构

```
┌─────────────────────────────────────────────────────────────┐
│  前端 (React)                                                │
│  ┌─────────────┐  SSE/HTTP  ┌───────────────────────────┐  │
│  │ AIChatPanel │◄──────────►│  /api/ai/chat (Streaming) │  │
│  └─────────────┘            └───────────────────────────┘  │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│  后端 (FastAPI)                                              │
│  ┌───────────────────────────────────────────────────────┐  │
│  │  AI Router                                            │  │
│  │  POST /api/ai/chat                                    │  │
│  │    ├── 接收: messages[], context, enable_web_search   │  │
│  │    ├── 构建 System Prompt + Context                   │  │
│  │    ├── 若 enable_web_search: 调用 Search Tool         │  │
│  │    ├── 调用 LLM API (Streaming)                       │  │
│  │    └── SSE 流式返回                                   │  │
│  └───────────────────────────────────────────────────────┘  │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────────────┐  │
│  │ LLM Client  │  │ Search Tool │  │ Context Builder     │  │
│  │ (OpenAI SDK)│  │ (联网查询)   │  │ (回测数据格式化)     │  │
│  └─────────────┘  └─────────────┘  └─────────────────────┘  │
└─────────────────────────────────────────────────────────────┘
```

### 3.2 新增后端模块

```
web/backend/
├── routers/
│   └── ai.py              # AI 对话路由 (/api/ai/*)
├── services/
│   ├── llm_client.py      # LLM 调用封装 (OpenAI SDK)
│   ├── search_tool.py     # 联网查询工具
│   └── context_builder.py # 回测数据 → Prompt 上下文
├── schemas/
│   └── ai.py              # Pydantic 模型
```

### 3.3 API 设计

#### POST `/api/ai/chat` — 流式对话

**请求体：**
```json
{
  "messages": [
    {"role": "user", "content": "为什么我的策略在3月份回撤这么大？"}
  ],
  "context": {
    "result": {"total_return": 0.24, "max_drawdown": -0.05, ...},
    "trades": [...],
    "equity_curve": [...],
    "group_name": "科技成长型",
    "date_range": ["2024-01-01", "2024-12-31"]
  },
  "enable_web_search": true,
  "model": "deepseek-chat"
}
```

**响应：** SSE Stream
```
data: {"type": "text", "content": "根据"}
data: {"type": "text", "content": "您的回测结果"}
data: {"type": "search_result", "results": [{"title": "...", "url": "..."}]}
data: {"type": "done"}
```

### 3.4 LLM Client 设计

支持多模型切换，配置文件管理：

```python
# services/llm_client.py
from openai import AsyncOpenAI

class LLMClient:
    def __init__(self, config: dict):
        self.client = AsyncOpenAI(
            api_key=config['api_key'],
            base_url=config['base_url']
        )
        self.model = config['model']

    async def chat_stream(self, messages: list, tools: list = None):
        """流式调用，支持 Function Calling"""
        stream = await self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            tools=tools,
            stream=True,
        )
        async for chunk in stream:
            yield chunk
```

**支持的模型（配置化）：**
- DeepSeek (deepseek-chat / deepseek-reasoner)
- 豆包 (doubao-pro-32k)
- 通义千问 (qwen-max)
- OpenAI GPT-4 (可选)

### 3.5 联网查询实现

两种方案，推荐 **方案A**（简单直接）：

#### 方案A：LLM 自带联网（推荐）

部分模型 API 已内置联网能力（如 DeepSeek 的 `search` 工具），只需在请求中开启：

```python
# 调用时传入 tools 参数
response = await client.chat.completions.create(
    model="deepseek-chat",
    messages=messages,
    tools=[{"type": "builtin_function", "function": {"name": "web_search"}}],
)
```

**优点：** 零额外开发，模型自己决定何时搜索、如何总结。
**缺点：** 依赖模型服务商，部分模型不支持。

#### 方案B：自建 Search Tool（Function Calling）

当模型不自带联网时，通过 Function Calling 让模型主动调用搜索：

```python
# services/search_tool.py
import aiohttp

async def web_search(query: str, num: int = 5) -> list:
    """调用搜索引擎 API（如 Serper.dev / Bing API）"""
    async with aiohttp.ClientSession() as session:
        async with session.post(
            "https://google.serper.dev/search",
            headers={"X-API-KEY": SERPER_API_KEY},
            json={"q": query, "num": num}
        ) as resp:
            data = await resp.json()
            return [
                {"title": r["title"], "url": r["link"], "snippet": r["snippet"]}
                for r in data.get("organic", [])
            ]

# 注册为 Tool
SEARCH_TOOL = {
    "type": "function",
    "function": {
        "name": "web_search",
        "description": "搜索互联网获取实时信息",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "搜索关键词"}
            },
            "required": ["query"]
        }
    }
}
```

**流程：**
1. 用户提问 → LLM 判断需要联网 → 返回 `function_call: web_search`
2. 后端执行搜索 → 将结果拼接进消息历史 → 再次调用 LLM
3. LLM 基于搜索结果生成最终回答 → SSE 流式返回

### 3.6 Context Builder — 回测数据格式化

将回测结果转换为 LLM 可理解的结构化文本：

```python
# services/context_builder.py

def build_backtest_context(ctx: dict) -> str:
    result = ctx['result']
    trades = ctx['trades'][:20]  # 取前20条

    return f"""【当前回测结果上下文】
分组: {ctx['group_name']}
回测区间: {ctx['date_range'][0]} 至 {ctx['date_range'][1]}

【绩效指标】
- 总收益率: {result['total_return']:.2%}
- 年化收益率: {result['annual_return']:.2%}
- 最大回撤: {result['max_drawdown']:.2%}
- 夏普比率: {result['sharpe_ratio']:.2f}
- 交易次数: {result['trade_count']}
- 胜率: {result['win_rate']:.2%}
- 盈亏比: {result['profit_factor']:.2f}

【交易明细摘要】(前20笔)
{"\n".join(format_trade(t) for t in trades)}

请基于以上数据回答用户问题。若用户询问涉及最新市场信息，请使用联网搜索工具获取实时数据后再作答。
"""
```

### 3.7 System Prompt 设计

```
你是一位专业的量化策略分析师，擅长解读回测结果、分析交易逻辑、提供策略优化建议。

当前用户正在查看其量化策略的回测报告，你需要：
1. 基于提供的回测数据回答用户问题
2. 分析绩效指标的含义与改进空间
3. 若用户询问最新市场动态、个股新闻、行业政策等，主动使用联网搜索工具获取信息
4. 回答应专业、简洁，使用中文

注意事项：
- 不要编造数据，所有分析必须基于提供的回测上下文
- 涉及预测或建议时，需声明"仅供参考，不构成投资建议"
- 联网搜索结果需标注来源
```

## 四、数据流设计

```
用户完成回测
    │
    ▼
┌─────────────────┐
│ BacktestPage    │ ──setContext()──► ┌─────────────┐
│ 渲染结果图表     │                   │ aiChat Store │
└─────────────────┘                   └──────┬──────┘
                                             │
用户点击 AI 助手，输入问题                      │
    │                                        │
    ▼                                        ▼
┌─────────────────┐    HTTP POST     ┌──────────────┐
│ AIChatPanel     │ ───────────────► │ /api/ai/chat │
│ sendMessage()   │   + context      │              │
└─────────────────┘                  └──────┬───────┘
                                            │
                                            ▼
                                    ┌───────────────┐
                                    │ ContextBuilder │
                                    │ 拼接 Prompt   │
                                    └───────┬───────┘
                                            │
                              ┌─────────────┴─────────────┐
                              ▼                           ▼
                        ┌──────────┐              ┌─────────────┐
                        │ LLM 判断  │──需要联网?──►│ Search Tool │
                        │          │    Yes       │  执行搜索    │
                        └────┬─────┘              └──────┬──────┘
                             │ No                        │
                             ▼                           ▼
                        ┌──────────┐              ┌──────────┐
                        │ 直接生成  │              │ 搜索结果  │
                        │ 回答      │              │ 注入上下文│
                        └────┬─────┘              └────┬─────┘
                             │                         │
                             └──────────┬──────────────┘
                                        ▼
                                  ┌────────────┐
                                  │ SSE Stream │
                                  │ 流式返回    │
                                  └─────┬──────┘
                                        │
                                        ▼
                                  ┌────────────┐
                                  │ AIChatPanel │
                                  │ 逐字渲染    │
                                  └────────────┘
```

## 五、实现步骤（推荐优先级）

### Phase 1: 基础对话（MVP）
1. 前端：`AIChatWidget` + `AIChatPanel` UI 组件
2. 后端：`/api/ai/chat` 路由，支持基础对话（不联网）
3. 接入 LLM API（DeepSeek 或豆包）
4. Context Builder 将回测数据注入 Prompt

### Phase 2: 联网查询
5. 增加 `enable_web_search` 开关
6. 实现 Search Tool（方案A优先，方案B兜底）
7. 前端展示搜索来源卡片

### Phase 3: 体验优化
8. 消息历史持久化（localStorage / 后端存储）
9. 支持 Markdown 渲染（表格、代码块）
10. 支持复制消息、重新生成
11. 对话上下文压缩（token 超限处理）

## 六、技术选型总结

| 层级 | 技术 | 说明 |
|------|------|------|
| 前端 UI | Ant Design + 自定义 CSS | 悬浮面板、消息气泡 |
| 前端状态 | Zustand | 对话状态、上下文管理 |
| 通信协议 | SSE (Server-Sent Events) | 流式输出，比 WebSocket 轻量 |
| 后端框架 | FastAPI | 原生支持 SSE |
| LLM SDK | OpenAI Python SDK | 兼容 DeepSeek / 豆包 / 千问 |
| 搜索工具 | 模型内置 / Serper.dev | 优先使用模型自带联网 |
| 配置管理 | JSON / Pydantic Settings | API Key、模型参数可配置 |

## 七、风险评估

| 风险 | 影响 | 缓解措施 |
|------|------|----------|
| LLM API 费用 | 中 | 限制单用户对话轮次、上下文压缩 |
| Token 超限 | 中 | 交易明细截断、净值曲线采样 |
| 搜索 API 费用 | 低 | 用户手动开启联网开关 |
| 响应延迟 | 中 | SSE 流式输出 + loading 状态 |
| 数据隐私 | 低 | 回测数据本地处理，不上传股票持仓 |
