# FinSight · 中文投研助理 Agent

> **Equity Research Copilot** —— 面向中文 A 股 / 港美股市场的投研助理 Agent。
>
> 基于 **LangGraph ReAct** 工具编排 · **多路混合 RAG**（本地研报库 + DuckDuckGo Web）· **Qwen3-Rerank** 精排 · **三策略上下文压缩** · **东方财富免 Key 行情/基本面** · **滚动摘要长记忆** · **Streamlit 投研主题工作台**。

[![Python](https://img.shields.io/badge/Python-3.10+-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![LangGraph](https://img.shields.io/badge/LangGraph-ReAct-1C3C5A)](https://langchain-ai.github.io/langgraph/)
[![Streamlit](https://img.shields.io/badge/UI-Streamlit-FF4B4B?logo=streamlit&logoColor=white)](https://streamlit.io/)
[![Tongyi Qwen](https://img.shields.io/badge/LLM-Tongyi%20Qwen-FF6A00)](https://dashscope.aliyun.com/)
![Tools](https://img.shields.io/badge/Tools-10%20%E4%B8%AA-blue)
![Recall@5](https://img.shields.io/badge/Recall%405-100%25-brightgreen)
![Token Saved](https://img.shields.io/badge/Token-%E2%88%9254.7%25-success)

---

## 关键指标（30 题黄金集 · 三档对照实测）

| 指标 | Vector-only | **+Rerank** | **+Rerank+Compression** |
|---|---|---|---|
| Recall@5 | 100% | **100%** | **100%** |
| MRR | 1.000 | **1.000** | **1.000** |
| Precision@actual | 90.7% | **88.1%** | **88.1%** |
| 平均输入 tokens | 947 | **429（−54.7%）** | **429** |
| 平均返回文档数 | 4.93 | **2.30** | **2.30** |
| 延迟 P50 / P95 (ms) | 2163 / 9293 | 4521 / 15370 | 4521 / 15370 |

> **核心结论**：当前语料下检索召回已饱和（Recall=100%），Rerank 的真正价值在于**精度收紧 + 输入瘦身**——平均输入 token 从 947 降到 429，预期 LLM 推理成本同比下降 ~50%。
> 完整报告见 [`tests/eval_results.md`](tests/eval_results.md)，自动生成。

---

## 快速开始

   ```bash
# 1. 安装依赖
   pip install -r requirements.txt

# 2. 配置 API Key（DashScope 提供 LLM / Embedding / Rerank）
export DASHSCOPE_API_KEY=sk-xxxxxxxx        # Linux / macOS
# 或 PowerShell: $env:DASHSCOPE_API_KEY="sk-xxxxxxxx"

# 3. 一键重建本地索引（首次或数据有变更时）
python scripts/rebuild_index.py --yes

# 4. 启动 Streamlit 工作台
   streamlit run app.py
   ```

打开浏览器至 `http://localhost:8501` 即可使用。建议体验：

- **对话模式**：「贵州茅台（600519）现在的股价 + 基本面」 / 「什么是 ROE？请用杜邦分析拆解一下」
- **报告模式**：「帮我写一份英伟达（NVDA）的个股速评，带上最新行情和估值」

---

## 能力速览

### 三类核心能力

| 维度 | 能力 |
|---|---|
| **本地投研问答** | 研报 / 年报 / 公告 / 政策 / 行业百科 / 财经术语，结构化分块 + 向量检索 + Rerank + 按需压缩 |
| **实时市场数据** | A 股 / 港股 / 美股的行情快照、基本面快照（PE-TTM / PB / 市值 / 换手率），多种 ticker 写法兼容、NASDAQ 拿不到自动回退 NYSE |
| **结构化报告** | 个股速评 / 行业速评 / 晨会纪要 三种模板，按用户意图自动选用 |

### 10 个 @tool（注册到 LangGraph ReAct Agent）

| # | 工具 | 作用 |
|---|---|---|
| 1 | `rag_summarize` | 本地投研语料库：检索 → Rerank → LLM 总结 |
| 2 | `rag_retrieve` | 仅检索原文（带出处），不调 LLM 生成 |
| 3 | `hybrid_summarize` | 本地 + DuckDuckGo Web → 统一 Rerank → LLM 总结 |
| 4 | `hybrid_search` | 同上，仅返回精排后的参考文本 |
| 5 | `web_search` | DuckDuckGo HTML，用于实时新闻 / 公告 / 行情评论 |
| 6 | `get_stock_quote` | 行情快照（最新价 / 涨跌 / 成交），东财 push2，免 Key |
| 7 | `get_stock_basics` | 基本面快照（市值 / PE-TTM / PE-LYR / PB / 换手率），同上 |
| 8 | `convert_currency` | 汇率换算（USD / CNY / HKD / EUR / JPY），open.er-api.com |
| 9 | `compute_financial_metric` | 财务指标 / 估值 / 同环比纯算术 |
| 10 | `get_market_datetime` | A 股 / 港股 / 美股市场当地时间 |

> 工具调用由模型按问题自主决定（非硬编码路由），description 里描述了**何时调用 / 何时不要调用**，配合 `prompts/main_prompt.txt` 投研 Copilot 角色卡引导。
> 实现见 `tools/agent_tool.py` / `tools/finance_tool.py`。

### 数据语料（`data/`）

| 子目录 | 内容 | 状态 |
|---|---|---|
| `glossary/` | 财经术语词典（80+ 术语 / 6 大估值方法 / 三大报表速查） | ✅ 已内置 |
| `industry_kb/` | 行业百科（新能源车 / 半导体 / AI 算力） | ✅ 已内置 |
| `_demo/` | 明标虚构演示样本（虚构公司 999001 + 虚构 AI 算力研报） | ✅ 已内置 |
| `research_reports/` | 真实券商研报 | ⏳ 待用户按 `data/README.md` 接入 |
| `company_filings/` | 上市公司年报 / 季报 / 公告 | ⏳ 待用户按 `data/README.md` 接入 |
| `policy/` | 央行 / 证监会 / 部委政策原文 | ⏳ 待用户按 `data/README.md` 接入 |

---

## 工作流：从用户输入到结构化回答

每次提问都会进入 **ReAct 循环**：模型根据系统提示词决定是否调用工具 → 执行工具 → 把结果作为 ToolMessage 写回对话 → 再生成回复，直到结束。流式输出仅展示 **最终助手文本的增量**，工具调用详情可在展开器中查看。

### 端到端总览（合并对话 + 报告模式）

```
┌──────────────────────────┐
│      用户输入             │
│  Streamlit 输入框         │
└─────────────┬────────────┘
              │
              ▼
┌──────────────────────────┐         ┌─────────────────────────┐
│      app.py              │ ◀──────▶│ 长记忆管理器               │
│  对话历史 + 报告模式开关    │         │ (recent window +         │
└─────────────┬────────────┘         │  rolling summary)        │
              │                      └─────────────────────────┘
              ▼
┌──────────────────────────┐
│     ReactAgent           │
│ 历史 → Human/AI Message  │
│ + 历史摘要 + 本轮问题      │
└─────────────┬────────────┘
              │
              ▼
┌──────────────────────────┐         ┌─────────────────────────┐
│  LangGraph create_agent  │ ◀──────▶│ middleware              │
│  + TOOLS（10 个）         │         │ runtime.context["report"]│
│                          │         │ 切换 main / report prompt│
└─────────────┬────────────┘         └─────────────────────────┘
              │
              ▼
┌──────────────────────────┐
│   ChatTongyi 推理         │
│   是否调用工具？           │
└─────────────┬────────────┘
              │
        ┌─────┴────────────────────┐
        │                          │
        ▼                          ▼
┌──────────────────┐     ┌─────────────────────────────┐
│  直接文本回答     │     │ 模型发出 tool_calls          │
│  （未走工具）     │     │  rag / hybrid / web /       │
│                  │     │  stock / metric / fx ...     │
└────────┬─────────┘     └────────────┬─────────────────┘
         │                            │
         │                            ▼
         │              ┌─────────────────────────────┐
         │              │ 执行工具                     │
         │              │ → ToolMessage               │
         │              │ → 主模型再推理 (ReAct 多步)  │
         │              │ (rag/hybrid：本地侧可开      │
         │              │  Query 扩展，见下节)         │
         │              └────────────┬─────────────────┘
         │                           │
         └────────────┬──────────────┘
                      ▼
            ┌──────────────────────┐
            │ 最终 AIMessage        │
            │ Streamlit 流式展示    │
            └──────────────────────┘
```

当工具为 `rag_summarize` / `rag_retrieve` / `hybrid_search` / `hybrid_summarize` 时，**本地 Chroma 检索**可在 `config/rag.yml` 中开启 Query 扩展（多检索用语合并粗排）；Agent 层 ReAct 编排**不变**，仍是「选工具 → 执行 → 再推理」。

### Hybrid RAG 数据流（本地 + Web 合并精排）

`hybrid_summarize` / `hybrid_search` 内部把**两路召回合并到统一候选池**，**单次** Rerank 精排，避免双源加权偏置。

```
┌─────────────────────┐     ┌──────────────────────────────────┐
│   Web 召回           │     │  本地向量检索                     │
│  始终用**同一用户    │     │  可选 Query 扩展（rag.yml）：      │
│  query**             │     │  LLM 生多条检索用语 → 并行各      │
│  DuckDuckGo HTML     │     │  top-K → 合并 / 去重 / cap        │
│  → Document(web)     │     │  → Document(local)               │
└──────────┬──────────┘     └──────────┬───────────────────────┘
           │   Web 一条 · 本地可多 query │
           └──────────────┬─────────────┘
           ▼
┌─────────────────────┐
                │   合并候选池          │
                │  source_channel 标记  │
└──────────┬──────────┘
                         ▼
     ┌─────────────────────┐
                │  Qwen3-Rerank 精排   │
                │  query = 用户原问    │
                │  • 阈值过滤（≥0.25）│
                │  • 去重（≥0.80）    │
                │  • Instruct 引导     │
     └──────────┬──────────┘
              ▼
     ┌─────────────────────┐
                │  上下文压缩（按需）  │
                │  AUTO 策略路由       │
                │  short → none        │
                │  code/fact → extract │
                │  long → hybrid       │
                └──────────┬──────────┘
                           ▼
            ┌──────────────────────────┐
            │  Top-N 参考资料            │
            │  → ToolMessage             │
            │  → Agent 主模型组织回答     │
            └──────────────────────────┘
```

> 单路本地 RAG（`rag_summarize` / `rag_retrieve`）流程一致，只是合并池里只有 `Document(local)`，无 Web 一支。**本地向量检索**侧可配置多条 query 扩展（见下文「RAG 链路细节 → Query 扩展流程」）；Web 侧仍用原始用户 query。

### Demo 三连击（已可在内置数据集上跑通）

| Demo | 输入示例 | 走线 | 考点 |
|---|---|---|---|
| **1. 纯本地 RAG** | "什么是 ROE？请用杜邦分析拆解一下" | `rag_summarize` → 命中 `glossary/financial_terms.md` → Rerank Top-3 → LLM 总结 | 检索 + Rerank + 子链 |
| **2. Hybrid + 报告 + 行情** | [报告 ON] "帮我写一份英伟达（NVDA）的个股速评" | `hybrid_summarize`（本地行业 + Web 新闻）→ `get_stock_quote` → `get_stock_basics` → `compute_financial_metric` → 报告模板 | **多步工具编排（4-6 次工具调用）** |
| **3. 长对话记忆** | 第 1 轮"看下半导体行业" → 第 5 轮"那刚才聊的产业里 HBM 怎么定义？" | 滚动摘要保留"用户关心半导体" → RAG 检索把摘要作为 `dialogue_context` 喂入 | 最近窗口 + 滚动摘要 |

---

## MCP 工作流（对外标准化工具）

除 Streamlit 内的 **ReAct Agent** 外，本项目可作为 **MCP Server（stdio）** 运行：把同一套 RAG / 混合检索能力以 MCP `tools/list` + `tools/call` 暴露给 **Cursor、Claude Desktop** 等客户端，**不经过** LangGraph 推理层——由宿主应用决定是否调用、调用哪个工具。

### 与站内 Agent 的关系

| 路径 | 编排者 | 典型入口 |
|---|---|---|
| **Streamlit 工作台** | `ReactAgent` + LangGraph，模型自主选工具 | `streamlit run app.py` |
| **MCP** | 外部客户端（或用户显式点选工具） | `python mcp_server.py` |

业务逻辑仍落在 `tools/agent_tool.py`（LangChain `@tool`）；MCP 层只做 **协议适配、参数校验、线程卸载、日志**。

### 端到端数据流

```
┌─────────────────────────┐
│  MCP Client             │
│  （Cursor / Claude …）   │
└────────────┬────────────┘
             │  JSON-RPC over stdin/stdout
             ▼
┌─────────────────────────┐
│  mcp_server.py          │
│  固定工作目录 + 日志初始化 │
└────────────┬────────────┘
             ▼
┌─────────────────────────┐
│  mcp_impl/server.py     │
│  Server("rag-agent")    │
│  • list_tools           │
│  • call_tool            │
└────────────┬────────────┘
             │
             ▼
┌─────────────────────────┐
│  mcp_impl/handlers.py   │
│  JSON Schema 校验(MCP)  │
│  anyio.to_thread →      │
│  tool.invoke({...})     │
└────────────┬────────────┘
             ▼
┌─────────────────────────┐
│  tools/agent_tool.py    │
│  rag_* / hybrid_* /     │
│  web_search / 时间      │
└────────────┬────────────┘
             ▼
┌─────────────────────────┐
│  rag / HybridRAG / …    │
│  （与 Streamlit 相同链路）│
└─────────────────────────┘
```

### 已暴露的 MCP 工具（第一版）

与 `docs/mcp_implementation_guide.md` 对齐的 **6** 个原子工具（名称即 MCP `name`）：

| MCP 工具 | 说明 |
|---|---|
| `rag_summarize` | 本地语料检索 + LLM 总结（可选 `dialogue_context`） |
| `rag_retrieve` | 仅返回检索原文片段 |
| `hybrid_search` | 本地 + Web 多路召回，返回精排参考文本 |
| `hybrid_summarize` | 多路召回 + LLM 总结 |
| `web_search` | DuckDuckGo 联网搜索（`max_results` 默认 5，上限 10） |
| `get_local_datetime` | 指定 IANA 时区当前时间（实现对应 `get_market_datetime`） |

> 行情 / 基本面 / 汇率等金融工具仍在站内 Agent 的 `TOOLS` 中；若要在 MCP 中一并开放，可在 `mcp_impl` 中按同样模式扩展。

### 运行与客户端配置

**启动服务器（stdio，占用标准输入输出，宜由客户端子进程拉起）：**

```bash
pip install -r requirements.txt   # 含 mcp SDK
python mcp_server.py
```

**环境**：和主应用相同——`DASHSCOPE_API_KEY`（或 `TONGYI_API_KEY`）、已构建的向量库、`config/rag.yml` 等；详见实施文档 §12。

**Cursor 示例**（Windows 下建议解释器与脚本均用**绝对路径**；若用虚拟环境，将 `command` 换为 venv 的 `python.exe`）：

```json
{
  "mcpServers": {
    "rag-agent": {
      "command": "python",
      "args": ["C:/Users/you/Desktop/rag/agent/mcp_server.py"]
    }
  }
}
```

### 实现位置说明

- 代码放在 **`mcp_impl/`**，避免与 PyPI 包 **`mcp`** 同名目录冲突（否则 `from mcp.server import Server` 会导入失败）。
- 更细的落地步骤、验收清单与误区见 [`docs/mcp_implementation_guide.md`](docs/mcp_implementation_guide.md)。

---

## RAG 链路细节

### 配置要点（`config/rag.yml`）

| 配置项 | 默认 | 说明 |
|---|---|---|
| `rerank_model` | `qwen3-rerank` | DashScope Rerank 模型 |
| `rerank_search_k` | 20 | 向量粗排候选数 |
| `rerank_top_n` | 5 | 精排后保留 top-N |
| `rerank_score_threshold` | 0.25 | 分数阈值（低于此值丢弃） |
| `rerank_dedup_threshold` | 0.80 | 文本相似度去重阈值 |
| `rerank_instruct_mode` / `auto_instruct` | `qa` / `true` | 指令模式与自动选择 |
| `compression_enabled` | `true` | 是否启用上下文压缩 |
| `compression_max_tokens` | 3500 | 压缩后目标 token 上限（短输入自动跳过）|
| `compression_strategy` | `auto` | 压缩策略：`auto` / `extract` / `summarize` / `hybrid` |
| `query_expansion_enabled` | `false` | 是否用 Chat 模型生成多条检索 query（广度） |
| `query_expansion_variants` | 5 | LLM 生成的**额外**检索句式条数（不含原问） |
| `query_expansion_include_original` | `true` | 多查询时是否始终保留用户原问参与向量检索 |
| `query_expansion_max_coarse_docs` | 100 | 多路检索合并、去重后的子块上限，再父块展开与 Rerank |
| `query_expansion_max_workers` | 8 | 并行向量检索线程数 |
| `query_decompose_enabled` | `false` | 是否启用子问题分解（深度，需再命中触发词） |
| `query_decompose_max_subqueries` | 4 | 分解后的子问题上限 |
| `query_decompose_with_expansion` | `false` | 每个子问题是否再做多查询改写（延迟更高） |

### Query 扩展流程（可选）

检索前可扩展 **「送入向量库的 query 列表」**，不改变 **Rerank 与总结阶段所用的用户原问**（仍是一条字符串，含可选 `dialogue_context`）。实现见 [`rag/query_expand.py`](rag/query_expand.py)，在 [`rag/ragsummarize.py`](rag/ragsummarize.py) 的 `RAGSummarize._rerank_docs` 与 `HybridRAG._multi_retrieve`（**仅本地分支**）中接入。

**开启前提**：`query_expansion_enabled` 或（分解路径）`query_decompose_enabled` 打开，且 **`chat_model` 可用**；否则退化为单条原问检索，不额外调用扩展 LLM。

```
                    ┌─────────────────────────┐
  retrieval_input   │  build_search_queries()  │
  （用户原问±上下文）│  + rerank_config        │
                    └────────────┬────────────┘
                                 │
         ┌───────────────────────┼───────────────────────┐
         │                       │                       │
         ▼                       ▼                       ▼
 分解路径（深度）           多查询路径（广度）         关闭 / 无 LLM
  • query_decompose_         • query_expansion_       • 仅 [原问]
    enabled=true               enabled=true
  • 问题含触发词               • LLM 输出 JSON 数组
    （对比/优缺点/                  条检索短语
     哪些方面…）              │
  • LLM 拆子问题                   │
         │                       │                       │
         └───────────────────────┴───────────────────────┘
                                 │
                                 ▼
                    ┌─────────────────────────┐
                    │ coarse_retrieve_union() │
                    │  多条 query 并行 invoke   │
                    │  → 子块合并 → 去重 → cap  │
                    └────────────┬────────────┘
                                 │
                                 ▼
                    ┌─────────────────────────┐
                    │ 父块展开（Parent-Child）   │
                    └────────────┬────────────┘
                                 │
                                 ▼
                    ┌─────────────────────────┐
                    │ Rerank（query=原始原问）   │
                    └────────────┬────────────┘
                                 │
                                 ▼
                    ┌─────────────────────────┐
                    │ 上下文压缩（按需）→ LLM 总结│
                    └─────────────────────────┘
```

**Hybrid RAG**：Web 分支仍用**原始一条 query** 调用 DuckDuckGo，不把扩展后的多条 query 打到 Web，避免延迟与配额倍增。

**与 LCEL 子链的关系**：`rag_summarize` 内部仍是 `context = _rerank_docs(输入)` + `question = Passthrough(同一输入)`；扩展只发生在 `_rerank_docs` 粗排内部，**不必**改链的形状。

### 上下文压缩策略

| 策略 | 适用场景 | 实现 |
|---|---|---|
| `extract`（提取式）| 代码 / 事实查询 | 按查询相关性保留段落，纯本地计算（<100ms） |
| `summarize`（摘要式）| 超长 / 概括性查询 | 调用 LLM 生成查询相关摘要 |
| `hybrid`（混合式）| 极长文档 | 先 extract 减半，再 summarize 精练 |
| `auto`（默认）| 全部 | 按查询关键词与文档长度自动路由 |

**实测**：当前内置语料的 reranked 结果加起来 < 3500 token，压缩器统一选 `none`（**按需启用**，不对短文档过度压缩）。压缩力度模拟（300 tokens 紧预算）显示 extractive 策略最大单题压缩率 **82.4%**，证明能力在线、待真实长文档（年报 PDF）接入后激活。

---

## 入库流水线（`rag/vector_store.py`）

语料来源二选一（**写入同一套流水线**）：（1）直接把文件放入 `data/`（及子目录）；（2）在 Streamlit 工作台 **上传投研语料** → 保存到 `database_path`（默认 `data/`）→ 点击 **「执行入库（向量化）」**，内部同样调用 `VectorStoreService().load_data()`。

```
data/ 多级子目录递归扫描（支持 .md / .txt / .pdf / .docx / .xlsx / .pptx）
   │
   ├─ 跳过非语料文件（README.md / .gitkeep / CHANGELOG.md ...）
   ├─ MD5 ledger 增量（md5.txt）：按**原始文件二进制**判重；内容未变则跳过整文件
   │
   ▼
按后缀路由 Loader（utils/file_hander.py → LangChain loaders）
   │
   ▼
入库前正文清洗 ingestion_clean（rag/ingestion_clean.py）
   • 配置：`config/chrome.yml` → `ingestion_clean`（`enabled: false` 则跳过）
   • 仅改内存中的 `page_content`，不回写源文件；清洗后 `metadata.ingestion_cleaned = true`
   • 典型：BOM/零宽字符、Unicode NFKC、控制符、统一换行与行尾空格、压连续空行；
     可选：`drop_line_patterns`（整行正则）、`merge_soft_hyphens`、`collapse_duplicate_lines`
   • 过短片段可配置丢弃（`min_document_chars`）；改清洗规则后若要与旧向量一致需**全量重建索引**
   │
   ▼
结构化预分段（rag/structured_chunking.py）
   • 识别 Markdown # ~ ######
   • 识别中文 第 X 章 / 一二三 / 1.
   • 每段写入 metadata.section
   │
   ▼
RecursiveCharacterTextSplitter
   • chunk_size=512, chunk_overlap=80
   • 9 级分隔符回退（"\n\n" → "\n" → 中文句读 → 字符级）
   │
   ▼
Parent-Child 映射（由 chrome.yml 开关控制）
   • 子块入 Chroma 做向量检索
   • 父块（带 【章节】… 前缀）入 SQLite，命中后展开供 Rerank
   │
   ▼
Chroma.add_documents → db/chroma/
```

**命令行增量入库**（与 App 内「执行入库」等价，需已配置 `DASHSCOPE_API_KEY`）：

```bash
python rag/vector_store.py load    # 等价参数：ingest
```

**单文件预览清洗效果**（不写向量库、不写 md5）：

```bash
python scripts/preview_ingestion_clean.py path/to/your.pdf
python scripts/preview_ingestion_clean.py path/to/note.md --head 400
```

**重建索引**：

```bash
python scripts/rebuild_index.py --yes      # 清 chroma + parent_store + md5 → 重新入库
python scripts/rebuild_index.py --dry-run  # 仅打印将要清理 / 入库的内容
```

---

## 仓库结构

```
agent/
├── app.py                          # Streamlit 投研主题工作台（深蓝 + 金 + A 股涨跌色）
├── mcp_server.py                   # MCP stdio 启动入口（对外协议层）
├── mcp_impl/                       # MCP 适配（schemas / handlers / server；避免与 PyPI「mcp」包同名）
│   ├── schemas.py                  # 各工具 input JSON Schema
│   ├── handlers.py                 # invoke LangChain 工具 + 日志
│   └── server.py                   # MCP Server 注册 list_tools / call_tool
├── prompts/
│   ├── main_prompt.txt             # FinSight 投研 Copilot 角色卡（含合规免责）
│   ├── rag_prompt.txt              # RAG 子链 prompt（强调数据三要素）
│   └── report_prompt.txt           # 三种报告模板（个股 / 行业 / 晨会）
├── tools/
│   ├── reactagent.py               # ReAct Agent + 流式
│   ├── agent_tool.py               # RAG / Web / 算 / 时间工具
│   ├── finance_tool.py             # 行情 / 基本面 / 汇率工具（P1-1 新增）
│   └── mid_ware.py                 # 报告模式动态 prompt 切换
├── rag/
│   ├── vector_store.py             # Chroma 入库 + 父子块映射 + 调用 ingestion_clean
│   ├── ingestion_clean.py          # 入库前正文清洗（YAML 可配）
│   ├── structured_chunking.py      # 中文章节正则预分段
│   ├── ragsummarize.py             # RAGSummarize + HybridRAG
│   ├── query_expand.py             # 检索前多 query 扩展 / 子问题分解（可选）
│   ├── reranker_enhanced.py        # 阈值过滤 + 去重 + Instruct
│   ├── context_compressor.py       # 三策略上下文压缩
│   └── parent_store.py             # 父块 SQLite 存储
├── memory/
│   └── conversation_memory.py      # 最近窗口 + 滚动摘要 双层记忆
├── model/model.py                  # ChatTongyi + DashScopeEmbeddings + Rerank
├── data/                           # 语料根目录（含 README 数据采集指南）
│   ├── glossary/                   # 财经术语词典
│   ├── industry_kb/                # 行业百科
│   ├── _demo/                      # 演示样本（明标虚构）
│   ├── research_reports/           # 真实研报（待用户填充）
│   ├── company_filings/            # 公司公告（待用户填充）
│   └── policy/                     # 政策原文（待用户填充）
├── tests/
│   ├── golden_set.yml              # 30 题检索黄金集
│   ├── test_ingestion_clean.py     # 入库前清洗单元测试
│   └── eval_results.md             # 自动生成的评估报告
├── scripts/
│   ├── rebuild_index.py            # 一键重建 chroma + 父块 + md5
│   ├── preview_ingestion_clean.py  # 单文件清洗效果预览（不写库）
│   ├── smoke_test_rag.py           # RAG 冲烟（7 题）
│   ├── smoke_test_finance.py       # 金融工具冲烟（17 用例）
│   └── eval_retrieval_metrics.py   # 检索性能评估（三档对照 + 压缩力度模拟）
├── config/                         # YAML 配置（rag / chrome / agent / prompts）
├── db/                             # Chroma 持久化（运行后生成）
└── log/agent.log                   # 模型与工具调用轨迹
```

---

## 测试与评估

```bash
# 1. RAG 冲烟（7 题关键词命中检查）
python scripts/smoke_test_rag.py

# 2. 金融工具冲烟（17 用例：A/港/美股 + 汇率 + 错误处理）
python scripts/smoke_test_finance.py

# 3. 检索性能完整评估（30 题 + 三档对照 + 压缩力度模拟，~270s）
python scripts/eval_retrieval_metrics.py
# 报告自动写入 tests/eval_results.md，可贴进简历或 README

# 4. pytest 跑结构化分块 / 混合检索 / 上下文压缩等单元测试
pytest tests/ -q

# 5. MCP stdio 冒烟（子进程启动 mcp_server.py，校验 tools/list 与若干 tools/call）
python scripts/smoke_test_mcp.py

# 6. 入库前清洗单元测试
python -m pytest tests/test_ingestion_clean.py -v
```

**当前指标**：
- RAG 冲烟：✅ **7/7 通过**（Top-1 rerank 高分占比 6/7 ≥ 0.7，最高 0.977）
- 金融工具冲烟：✅ **17/17 通过**（含 A 股沪深、港股、美股 NASDAQ/NYSE 自动回退、错误 ticker / 错误币种 / 负数金额）
- 检索性能：✅ **Recall@5 = 100%，MRR = 1.000**，Rerank 减输入 token **−54.7%**

---

## 配置与日志

| 文件 | 内容 |
|---|---|
| `config/rag.yml` | 对话模型、embedding、Rerank、Hybrid 检索、上下文压缩 |
| `config/chrome.yml` | Chroma 路径、集合名、`k`、分块、结构化分块、`ingestion_clean` 清洗项 |
| `config/agent.yml` | Agent 迭代上限、长记忆参数 |
| `config/prompts.yml` | 提示词路径（指向 `prompts/*.txt`） |
| `log/agent.log` | 模型 / 工具调用轨迹（与 `ReactAgent.log_tool_calls` 配合） |

---

## 说明与限制

- **Web 搜索**：DuckDuckGo HTML 是非官方接口，适合演示与小流量；生产请换商业 API（Bing / Google CSE / Tavily 等）并注意合规与频控。
- **行情 / 基本面**：使用东方财富 push2 接口，**免 API Key 但非签约数据**；可能遇到偶发限速，工具内置 1+2 次重试 + 指数退避；**不建议**用于高频交易决策。
- **汇率**：open.er-api.com 约 24 小时刷新，宏观换算够用，**不实时**。
- **Embedding / Rerank**：DashScope 公网 API；未配置 `DASHSCOPE_API_KEY` 时向量服务会报错（**不会**默默回退到 Chroma 默认 ONNX 模型，避免出现意料之外的英文小模型行为）。
- **合规边界**：所有回答附"**仅基于公开信息整理，不构成投资建议**"；不做股价预测 / 买卖点判断 / 仓位建议。

---

## 简历项目段（可直接复制）

**一页纸建议**：项目名称 + 时间 + 技术栈占一行；下列 4～5 条按岗位取舍（算法岗强调 RAG/评测，工程岗强调 Agent/MCP/流水线）。

```text
FinSight · 中文投研 Copilot Agent（个人项目｜2026.04 – 至今）
技术栈：Python 3.10+ · LangChain / LangGraph（ReAct）· Streamlit · Chroma · DashScope（Qwen 对话 / Embedding / Qwen3-Rerank）

• 搭建面向 A 股/港美股的 LangGraph ReAct Agent：以模型语义决策路由 10 个领域工具（本地与混合 RAG、DuckDuckGo、
  行情/基本面快照、汇率换算、财务指标纯算术等），middleware 在对话模式与「个股/行业/晨会」三类报告模板间切换，
  Streamlit 侧流式展示最终回复。
• RAG 链路：财经语料章节感知结构化分块 + Parent-Child 向量检索；精排采用 Qwen3-Rerank，叠加分数阈值、
  近文本去重与 Instruct 模式。自建 30 题检索黄金集上 Recall@5=100%、MRR=1.0；在召回饱和前提下将送入总结的
  上下文由约 947 tokens 压至约 429（−54.7%），直接降低上游 LLM 成本。支持可配置 Query 扩展/子问题分解，
  以延迟换检索广度。
• Hybrid 设计：本地向量召回与 Web 结果汇入同一候选池后只跑一次 Rerank，规避双源分别加权带来的偏置；
  链尾按 query 特征自动路由 extract / summarize / hybrid 上下文压缩，短文档按需跳过避免过压。
• 长对话记忆：滚动摘要 + 固定近期窗口；扩展阶段由 LLM 维护 JSON 结构化长期事实（用户偏好、任务目标等），
  在轮次与 token 阈值触发下合并更新，平衡连贯性与上下文预算。
• 工程化落地：多格式文档入库 + MD5 增量跳过与可配置的 ingestion 清洗；东方财富 push2 行情/基本面封装
  （多市场 ticker、指数退避重试、进程内 TTL 缓存防抖）。另实现 MCP stdio Server，将核心 RAG 工具暴露给
  Cursor / Claude Desktop 等客户端；配套检索三档对照评估脚本、RAG/金融/MCP 冒烟测试与 pytest 回归。

GitHub: <你的仓库>    Demo: <可选链接>
```

---

## 相关文档

- **整体改造计划与进度**：[`FINSIGHT_PLAN.md`](FINSIGHT_PLAN.md)
- **数据采集指南**：[`data/README.md`](data/README.md)
- **检索性能完整报告**：[`tests/eval_results.md`](tests/eval_results.md)（自动生成）
- **上下文压缩完整指南**：`docs/context_compression_guide.md`
- **Rerank Instruct 指南**：`docs/rerank_instruct_guide.md`
- **Chunk 优化指南**：`docs/CHUNK_OPTIMIZATION.md`
- **长会话记忆指南**：`docs/conversation_memory_guide.md`
- **MCP 接入与工具清单**：[`docs/mcp_implementation_guide.md`](docs/mcp_implementation_guide.md)

---

> 本项目为个人作品集 demo，不用于商业用途。所有回答仅基于公开信息整理，**不构成任何投资建议**。
