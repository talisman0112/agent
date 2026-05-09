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
         │              └────────────┬─────────────────┘
         │                           │
         └────────────┬──────────────┘
                      ▼
            ┌──────────────────────┐
            │ 最终 AIMessage        │
            │ Streamlit 流式展示    │
            └──────────────────────┘
```

### Hybrid RAG 数据流（本地 + Web 合并精排）

`hybrid_summarize` / `hybrid_search` 内部把**两路召回合并到统一候选池**，**单次** Rerank 精排，避免双源加权偏置。

```
┌─────────────────────┐     ┌─────────────────────┐
│   Web 召回           │     │  本地向量检索         │
│  DuckDuckGo HTML    │     │  Chroma top-K        │
│  → Document(web)    │     │  → Document(local)   │
└──────────┬──────────┘     └──────────┬──────────┘
           │   同一 query 各自召回       │
           └──────────────┬─────────────┘
                          ▼
                ┌─────────────────────┐
                │   合并候选池          │
                │  source_channel 标记  │
                └──────────┬──────────┘
                           ▼
                ┌─────────────────────┐
                │  Qwen3-Rerank 精排   │
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

> 单路本地 RAG（`rag_summarize` / `rag_retrieve`）流程一致，只是合并池里只有 `Document(local)`，无 Web 一支。

### Demo 三连击（已可在内置数据集上跑通）

| Demo | 输入示例 | 走线 | 考点 |
|---|---|---|---|
| **1. 纯本地 RAG** | "什么是 ROE？请用杜邦分析拆解一下" | `rag_summarize` → 命中 `glossary/financial_terms.md` → Rerank Top-3 → LLM 总结 | 检索 + Rerank + 子链 |
| **2. Hybrid + 报告 + 行情** | [报告 ON] "帮我写一份英伟达（NVDA）的个股速评" | `hybrid_summarize`（本地行业 + Web 新闻）→ `get_stock_quote` → `get_stock_basics` → `compute_financial_metric` → 报告模板 | **多步工具编排（4-6 次工具调用）** |
| **3. 长对话记忆** | 第 1 轮"看下半导体行业" → 第 5 轮"那刚才聊的产业里 HBM 怎么定义？" | 滚动摘要保留"用户关心半导体" → RAG 检索把摘要作为 `dialogue_context` 喂入 | 最近窗口 + 滚动摘要 |

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

```
data/ 多级子目录递归扫描（支持 .md / .txt / .pdf / .docx / .xlsx / .pptx）
   │
   ├─ 跳过非语料文件（README.md / .gitkeep / CHANGELOG.md ...）
   ├─ MD5 ledger 增量入库（已入库则跳过）
   │
   ▼
按后缀路由 Loader
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
Parent-Child 映射
   • 子块入 Chroma 做向量检索
   • 父块（带 【章节】… 前缀）入 SQLite，命中后展开供 Rerank
   │
   ▼
Chroma.add_documents → db/chroma/
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
│   ├── vector_store.py             # Chroma 入库 + 父子块映射
│   ├── structured_chunking.py      # 中文章节正则预分段
│   ├── ragsummarize.py             # RAGSummarize + HybridRAG
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
│   └── eval_results.md             # 自动生成的评估报告
├── scripts/
│   ├── rebuild_index.py            # 一键重建 chroma + 父块 + md5
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
| `config/chrome.yml` | Chroma 路径、集合名、`k`、分块、结构化分块开关 |
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

```text
FinSight · 中文投研助理 Agent（个人项目，2026.4 – 至今）
技术栈：Python · LangChain · LangGraph · Streamlit · Chroma · Tongyi(Qwen) · DashScope Rerank

- 设计基于 LangGraph ReAct 的工具编排：本地 RAG / Hybrid RAG / 实时行情 / 基本面 /
  汇率换算 / 财务指标 / 交易日历，共 10 个 @tool，由 middleware 动态切换
  「对话 / 报告」双模式系统提示词。
- 自研结构化分块：识别中文章节（第 X 章 / 一二三 / Markdown #），配合
  RecursiveCharacterTextSplitter 多级分隔符回退 + 父子块映射；30 题黄金集
  评估 Recall@5 = 100%、MRR = 1.000。
- 实现 Hybrid RAG：本地研报库 + DuckDuckGo Web 召回进入统一候选池，
  Qwen3-Rerank 单次精排 + 0.25 阈值过滤 + 0.80 去重；将平均输入 token 从
  947 降到 429（−54.7%），平均返回文档数 4.93 → 2.30，Precision@actual 88.1%。
- 实现三策略上下文压缩器（extract / summarize / hybrid），按查询类型自动路由，
  在 300 token 紧预算下可拿到最大单题压缩率 82.4%（extractive 策略）。
- 实现「最近窗口 + 滚动摘要」两层长会话记忆，>20 轮对话 token 占用稳定。
- 自研金融数据工具集：股票行情 / 基本面（东财 push2，免 Key，多市场 ticker
  规范化 + 自动 NYSE 回退）+ 汇率（open.er-api.com，免 Key），冲烟测试 17/17 通过。
- 自研投研主题 Streamlit 工作台：A 股涨跌色 + 模式徽章 + 知识库实时统计 +
  示例提问引导 + 工具调用透明展开。
- 自动化评估脚本：30 题黄金集 + 三档 pipeline 对照（Vector-only / +Rerank /
  +Rerank+Compression）+ 多预算压缩力度模拟，单次端到端 ~270s。

GitHub: github.com/<you>/finsight    Demo: <streamlit / HF link>
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

---

> 本项目为个人作品集 demo，不用于商业用途。所有回答仅基于公开信息整理，**不构成任何投资建议**。
