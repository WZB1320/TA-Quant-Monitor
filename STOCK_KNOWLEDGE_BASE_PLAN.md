# A 股自选股知识库 + 联想搜索实现方案

## 一、需求理解

**现状**：用户添加自选股时，输入股票代码/名称后点击「搜索」，后端调用 `akshare.stock_zh_a_spot_em()` 实时拉取全市场数据并模糊匹配。每次搜索都请求一次 akshare，延迟高、无缓存、不支持输入联想。

**目标**：
1. 建立本地 A 股知识库（全量股票列表）
2. 输入时实时联想（无需点击搜索）
3. 添加的股票必须与知识库中的名称匹配（防止输错代码/名称）

## 二、知识库数据源方案

### 2.1 数据来源

| 来源 | 接口 | 说明 |
|------|------|------|
| AKShare | `ak.stock_zh_a_spot_em()` | A 股实时行情列表（含代码、名称、市场） |
| AKShare | `ak.stock_info_a_code_name()` | A 股基础信息（更轻量，适合初始化） |

### 2.2 存储方案：SQLite 本地知识库

在现有 `data/kline_cache.db` 旁新增 `data/stock_knowledge.db`：

```sql
-- 股票主表
CREATE TABLE stocks (
    code        TEXT PRIMARY KEY,  -- 6位数字代码
    name        TEXT NOT NULL,      -- 股票名称
    market      TEXT NOT NULL,      -- sh / sz / bj
    py_initials TEXT,               -- 拼音首字母 (如: ZGYH)
    full_py     TEXT,               -- 全拼 (如: zhongguoyinhang)
    industry    TEXT,               -- 所属行业 (可选)
    updated_at  TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- 全文搜索索引 (SQLite FTS5)
CREATE VIRTUAL TABLE stock_search USING fts5(
    code, name, py_initials, full_py,
    content='stocks', content_rowid='rowid'
);
```

**为什么用 SQLite**：
- 项目已有 SQLite 使用经验（kline_cache.db）
- 无需额外数据库服务，零部署成本
- 支持 FTS5 全文搜索，联想查询性能足够（A 股约 5000+ 只股票）
- 单机场景完全够用

**关于 FTS5 中文搜索的说明**：

FTS5 默认 tokenizer（unicode61）按空格/标点分词，对中文无效——`MATCH '平安'` 无法匹配「平安银行」。因此采用**混合搜索策略**：

| 输入类型 | 判断方式 | 搜索方式 |
|----------|----------|----------|
| 纯数字 | `query.isdigit()` | `code LIKE 'xxx%'` 前缀匹配 |
| 纯英文（拼音） | `query.isascii()` | FTS5 MATCH 搜索 `py_initials` 和 `full_py` 字段 |
| 中文 | 其余情况 | `name LIKE '%xxx%'` 模糊匹配 |

这样既利用了 FTS5 对拼音搜索的高性能，又避免了中文分词问题。A 股仅 5000+ 条记录，中文 `LIKE` 模糊匹配的性能完全可接受。

### 2.3 数据同步机制

```
首次启动 / 每日定时
    │
    ▼
┌─────────────────┐
│ 调用 akshare    │  stock_info_a_code_name()
│ 获取全量股票列表 │  或 stock_zh_a_spot_em()
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ 生成拼音字段     │  中文名称 → 拼音首字母 + 全拼
│ (pypinyin库)    │  如: 平安银行 → PA / pinganyinhang
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ 写入 SQLite     │  REPLACE INTO stocks (...)
│ 更新 FTS 索引   │  INSERT INTO stock_search (...)
└─────────────────┘
```

**同步触发时机**：
1. **首次启动**：后端启动时检查表是否为空，空则自动同步
2. **定时刷新**：每日开盘前自动更新（新上市/退市/更名）
3. **手动刷新**：提供 `/api/stocks/sync` 管理接口

## 三、联想搜索前端交互方案

### 3.1 交互改造

将现有的 `Input.Search` + 点击搜索 改为 `AutoComplete` 组件，实现输入即联想：

```
用户输入 "pingan"
    │
    ▼
┌─────────────────────────────────────────┐
│  输入框                                  │
│  ┌───────────────────────────────────┐  │
│  │ pingan                            │  │
│  └───────────────────────────────────┘  │
│  ┌───────────────────────────────────┐  │
│  │ 🔍 000001  平安银行        [深]   │  │
│  │ 🔍 000002  万科A           [深]   │  │  ← 联想下拉
│  │ 🔍 600000  浦发银行        [沪]   │  │
│  └───────────────────────────────────┘  │
└─────────────────────────────────────────┘
```

**支持多种输入方式**：
- 代码匹配：`000001` → 平安银行
- 名称匹配：`平安` → 平安银行
- 拼音首字母：`PA` → 平安银行
- 全拼匹配：`pingan` → 平安银行

### 3.2 前端组件变更

```tsx
// 改造前: Input.Search + onSearch + 手动点击
<Input.Search
  placeholder="输入股票代码或名称搜索"
  onSearch={handleSearch}      // 点击搜索按钮才触发
/>

// 改造后: AutoComplete + onChange 防抖
<AutoComplete
  options={suggestions}        // 联想结果
  onSearch={debouncedSearch}   // 输入即触发（300ms 防抖）
  onSelect={handleSelect}      // 选中即添加
  placeholder="输入代码 / 名称 / 拼音"
>
  <Input />
</AutoComplete>
```

## 四、后端搜索 API 改造

### 4.1 现有搜索接口（将被废弃）

```python
# web/backend/routers/watchlist.py —— 原有实现，改造后移除
@router.get("/search")
def search_stock(q: str = "", limit: int = 10):
    # 每次调用 akshare.stock_zh_a_spot_em() —— 实时但慢
```

### 4.2 改造后搜索接口（统一入口）

原有 `/api/watchlist/search` 端点**废弃移除**，搜索功能统一由新的 `/api/stocks/search` 提供：

```python
# web/backend/routers/stocks.py  (新增路由)
from services.knowledge_base import KnowledgeBase

kb = KnowledgeBase()

@router.get("/stocks/search", summary="联想搜索股票")
def search_stocks(q: str = "", limit: int = 10):
    """从本地知识库联想搜索，支持代码/名称/拼音"""
    if not q or len(q.strip()) < 1:
        return {"results": []}
    results = kb.search(q.strip(), limit)
    return {"results": results}

@router.post("/stocks/sync", summary="同步股票知识库")
def sync_stocks():
    """手动触发全量同步"""
    count = kb.sync_from_akshare()

    return {"ok": True, "synced": count}
```

### 4.3 搜索逻辑（混合搜索策略）

```python
# services/knowledge_base.py
import sqlite3
from pypinyin import lazy_pinyin

class KnowledgeBase:
    def search(self, query: str, limit: int = 10) -> list:
        """多字段匹配搜索（混合策略）

        搜索策略：
        - 纯数字 → code LIKE 前缀匹配
        - 纯英文 → FTS5 MATCH（拼音首字母 + 全拼）
        - 含中文 → name LIKE 模糊匹配
        """
        conn = sqlite3.connect(self.db_path)

        if query.isdigit():
            # 纯数字 → 按代码前缀匹配
            sql = "SELECT code, name, market FROM stocks WHERE code LIKE ? LIMIT ?"
            params = (f"{query}%", limit)
            rows = conn.execute(sql, params).fetchall()

        elif query.isascii():
            # 纯英文（拼音）→ FTS5 全文搜索
            # 注意：FTS5 MATCH 不支持前缀通配符直接拼在末尾，
            # 需要对输入做通配处理，如 "ping" 转为 "ping*"
            fts_query = f'"{query}"*' if " " not in query else query
            sql = """
                SELECT s.code, s.name, s.market
                FROM stock_search ss
                JOIN stocks s ON ss.rowid = s.rowid
                WHERE stock_search MATCH ?
                ORDER BY rank
                LIMIT ?
            """
            try:
                rows = conn.execute(sql, (fts_query, limit)).fetchall()
            except sqlite3.OperationalError:
                # FTS5 查询语法错误时回退到 LIKE
                rows = []

            # FTS5 没结果时回退到 LIKE（处理输入为部分拼音的情况）
            if not rows:
                sql = """SELECT code, name, market FROM stocks
                         WHERE py_initials LIKE ? OR full_py LIKE ?
                         LIMIT ?"""
                rows = conn.execute(sql, (f"{query}%", f"{query}%", limit)).fetchall()

        else:
            # 含中文 → name LIKE 模糊匹配
            sql = "SELECT code, name, market FROM stocks WHERE name LIKE ? LIMIT ?"
            rows = conn.execute(sql, (f"%{query}%", limit)).fetchall()

        conn.close()
        return [{"code": r[0], "name": r[1], "market": r[2]} for r in rows]
```

## 五、匹配校验机制

用户添加股票时，必须与知识库匹配：

```python
# routers/watchlist.py —— add_stock 改造
@router.post("")
def add_stock(req: dict):
    code = req.get("code", "").strip()
    name = req.get("name", "").strip()

    # 1. 从知识库校验
    stock_info = kb.get_by_code(code)
    if not stock_info:
        raise HTTPException(status_code=400, detail=f"股票代码 {code} 不存在于 A 股市场")

    # 2. 名称不匹配时，以知识库为准（或报错）
    if name and name != stock_info["name"]:
        # 方案A: 自动纠正
        name = stock_info["name"]
        # 方案B: 报错提示
        # raise HTTPException(status_code=400, detail=f"名称不匹配，应为: {stock_info['name']}")

    # 3. 继续原有添加逻辑...
```

## 六、完整改动点清单

### 6.1 新增文件

| 文件 | 说明 |
|------|------|
| `web/backend/services/__init__.py` | services 包初始化（新目录） |
| `web/backend/services/knowledge_base.py` | 知识库核心类（SQLite 操作 + 搜索 + 同步） |
| `web/backend/routers/stocks.py` | 股票搜索/同步路由（新增 `/api/stocks/*` 端点） |
| `web/backend/schemas/stock.py` | Pydantic 模型（StockSearchResult 等） |
| `scripts/sync_stock_kb.py` | 独立脚本：手动同步知识库 |

### 6.2 修改文件

| 文件 | 改动内容 |
|------|----------|
| `web/backend/app.py` | 注册 `stocks` 路由；启动时检查知识库初始化 |
| `web/backend/routers/watchlist.py` | **移除** `search_stock` 端点（搜索统一到 `/api/stocks/search`）；`add_stock` 增加知识库匹配校验 |
| `web/frontend/src/pages/Watchlist/index.tsx` | `Input.Search` → `AutoComplete`；`handleSearch` 改为防抖联想 |
| `web/frontend/src/api/index.ts` | 新增 `stockApi`，搜索调用 `/stocks/search`；`watchlistApi.search` 废弃 |
| `src/config/settings.py` | 新增 `STOCK_KB_DB` 路径常量 |
| `requirements.txt` | 新增依赖：`pypinyin` |

### 6.3 配置/依赖变更

| 项 | 变更 |
|----|------|
| Python 依赖 | `pip install pypinyin`（中文转拼音） |
| 数据文件 | 新增 `data/stock_knowledge.db`（SQLite，约 1-2MB） |
| 配置常量 | `src/config/settings.py` 新增 `STOCK_KB_DB` 路径 |
| 环境变量 | 可选：`STOCK_KB_AUTO_SYNC=1` 控制启动时是否自动同步 |

## 七、实现步骤（推荐优先级）

### Phase 1: 知识库搭建
1. 创建 `knowledge_base.py`：SQLite 建表、FTS5 索引
2. 实现 `sync_from_akshare()`：从 akshare 拉取全量数据
3. 实现 `search()`：多字段匹配查询
4. 添加 `scripts/sync_stock_kb.py` 手动同步脚本

### Phase 2: 后端 API 改造
5. 新建 `routers/stocks.py`：搜索接口 + 同步接口
6. 在 `watchlist.py` 中**移除** `search_stock` 端点（约 20 行），`add_stock` 增加知识库匹配校验
7. `app.py` 注册 `stocks_router` + 启动时自动初始化知识库
8. `settings.py` 新增 `STOCK_KB_DB` 路径常量

### Phase 3: 前端交互改造
9. `Watchlist/index.tsx`：`Input.Search` → `AutoComplete`
10. 添加 `lodash.debounce` 或手写防抖逻辑
11. 支持选中即添加（减少一步点击）

### Phase 4: 体验优化
12. 拼音搜索支持（`pypinyin` 生成拼音字段）
13. 搜索高亮匹配部分
14. 无结果时提示"未找到，请检查输入"

## 八、数据流图

```
用户打开「添加股票」弹窗
    │
    ▼
┌─────────────────────────────┐
│ 前端: AutoComplete 输入框    │
│ 输入 "pingan" (300ms 防抖)  │
└─────────────┬───────────────┘
              │ GET /api/stocks/search?q=pingan
              ▼
┌─────────────────────────────┐
│ 后端: /stocks/search         │
│ KnowledgeBase.search()       │
│   ├── 纯数字 → code LIKE     │
│   ├── 纯英文 → FTS5 MATCH    │
│   │   (拼音首字母 + 全拼)    │
│   └── 含中文 → name LIKE     │
└─────────────┬───────────────┘
              │ 返回 [{code,name,market}]
              ▼
┌─────────────────────────────┐
│ 前端: 渲染联想下拉列表        │
│ 000001 平安银行 [深]         │
└─────────────┬───────────────┘
              │ 用户选中
              ▼
┌─────────────────────────────┐
│ 前端: 调用 addStock()        │
│ POST /api/watchlist          │
│ {code:"000001", name:"平安银行"}
└─────────────┬───────────────┘
              ▼
┌─────────────────────────────┐
│ 后端: add_stock 校验         │
│ kb.get_by_code("000001")     │
│   → 存在，名称匹配 ✓         │
│   → 继续添加到 watchlist     │
└─────────────────────────────┘
```

## 九、风险评估

| 风险 | 影响 | 缓解措施 |
|------|------|----------|
| akshare 接口变更 | 中 | 封装数据源适配器，接口变更时只改一处 |
| 新上市/退市/更名股票 | 低 | 每日定时同步 + 手动同步接口 |
| SQLite 并发写入 | 低 | 单机使用，写入量极小 |
| pypinyin 多音字（如"行"→hang/xing） | 低 | 知识库同时存储拼音首字母和全拼；覆盖常用股票名即可，极端情况用户可用代码搜索 |
| 中文搜索性能（LIKE '%xxx%'） | 低 | A 股仅 5000+ 条，全表扫描也在毫秒级；热门搜索可加 LRU 内存缓存 |
| 首次同步耗时（akshare 接口约 3-5 秒） | 低 | 后台异步同步，前端显示"知识库初始化中"；仅启动时执行一次 |
| FTS5 MATCH 语法异常（用户输入特殊字符） | 低 | 代码中 try/except 捕获 `sqlite3.OperationalError`，回退到 LIKE 查询 |
