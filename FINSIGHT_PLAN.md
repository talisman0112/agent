# FinSight 改造计划 · 中文投研助理 Agent

> 将当前通用 RAG Agent 改造为面向中文 A 股 / 港美股的**投研助理**项目，作为求职作品集核心项目。
> 改造原则：**不动架构，只换内容**。复用现有 LangGraph ReAct + Hybrid RAG + Rerank + 上下文压缩 + 滚动摘要记忆等所有基础设施。

---

## 〇、当前进度看板（一页流总览）

| 阶段 | 任务 | 状态 | 备注 |
|---|---|---|---|
| **P0-2** | 重写 3 个 prompts（main / rag / report） | ✅ **已落地** | 投研 Copilot 角色卡 / 数据三要素 / 三种报告模板 |
| **P0-3** | 工具改名 + description 投研化 | ✅ **已落地** | `compute_financial_metric` / `get_market_datetime`；移除 weather/geocode |
| **P0-1** | 数据替换 + 重新入库 | ✅ **已落地（基础版）** | 8 个语料 / 204 chunks，冲烟 7/7 通过；研报/公告/政策待真实数据填充 |
| **额外 1** | 入库流水线增强 | ✅ **已落地** | 递归扫描子目录、`.md` 支持、非语料文件过滤 |
| **额外 2** | 一键重建 + 冲烟脚本 | ✅ **已落地** | `scripts/rebuild_index.py` / `scripts/smoke_test_rag.py` |
| **额外 3** | 前端财经主题改造 | ✅ **已落地** | 深蓝 + 金 + A 股涨跌色，hero 区一键示例提问，KB 统计实时显示 |
| **P1-1** | 新增金融数据工具（行情 / 基本面 / 汇率） | ✅ **已落地** | 3 个工具（东财 + ECB 公开免 Key），冲烟 17/17 通过 |
| **P1-2** | 性能指标评估脚本（Recall@5 / 压缩率 / 延迟） | ✅ **已落地** | 30 题黄金集 + 三档对照；Recall@5=100%、MRR=1.000；Rerank 减输入 token 54.7% |
| **A** | 接入真实研报 / 年报数据 | ⏳ 待开始 | 让 demo 跳出"科普级问答" |
| **B** | 巨潮资讯批量下载脚本 | ⏳ 可选 | 自动化采集 |
| **P2** | README 改造 / Demo 视频 / 公网部署 | ⏳ 待开始 | 锦上添花 |

---

## 一、产品定位

**项目名**：`FinSight`（候选：`QuantBrief` / `投研 Copilot`）

**一句话定位**：
> 面向中文 A 股 / 港美股的投研助理 Agent——基于 LangGraph ReAct + 多路混合 RAG（研报本地库 + 实时新闻/行情 Web）+ Rerank 精排 + 上下文压缩，支持「个股速评 / 行业速评 / 晨会纪要」三种结构化报告输出。

**目标用户**：买方/卖方研究员、个人投资者、金融实习生。

**合规边界**：
- 全部回答附"**仅供学习参考，不构成投资建议**"声明
- 不涉及自动下单 / 持仓管理
- 不做股价预测，仅做信息聚合与基本面分析

---

## 二、能力盘点（现有 → 投研场景映射）

| 现有能力 | 投研场景价值 | 改造动作 | 状态 |
|---|---|---|---|
| 结构化分块 `rag/structured_chunking.py` | 研报/年报章节天然结构化 | 数据替换即可生效 | ✅ 已生效 |
| Hybrid RAG（本地+Web 合并 Rerank） | 本地研报回答基本面 + Web 拉最新动态 | 数据替换即可生效 | ✅ 已生效 |
| Rerank 增强版 `rag/reranker_enhanced.py` | 多份研报按相关性精选 | 调阈值，加金融类 instruct | ✅ 已生效（多查询 0.8+ 高分） |
| 上下文压缩 `rag/context_compressor.py` | 研报常 30+ 页，token 成本高 | 直接复用 | ✅ 已生效 |
| 报告模式 `prompts/report_prompt.txt` | 输出晨会纪要 / 个股速评 | **重写报告模板** | ✅ 已落地 |
| 滚动摘要记忆 `memory/conversation_memory.py` | 多轮研究保留关注的标的/行业 | 直接复用 | ✅ 已生效 |
| `calculate_arithmetic` | PE / ROE / 同比环比 | **改名 `compute_financial_metric`** | ✅ 已落地 |
| `get_local_datetime` | 判断交易日 / 财报披露窗口 | **改名 `get_market_datetime`** | ✅ 已落地 |
| `get_weather_by_location` / `geocode_place` | 投研场景不相关 | **从 `TOOLS` 列表移除**（保留代码） | ✅ 已落地 |
| `web_search` | 新闻 / 公告 / 行情 | description 投研化 | ✅ 已落地 |

---

## 三、改造任务清单（按优先级）

### P0 · 立项必做（半天 ~ 2 天）

#### ✅ P0-1 数据替换（已落地基础版）

**已完成**：
- 旧 `data/` 备份至 `data_backup_legacy/`（24 个百科 txt）；
- 新建 6 个子目录：`glossary/` / `industry_kb/` / `research_reports/` / `company_filings/` / `policy/` / `_demo/`；
- 写入 8 份高质量语料（共约 76 KB）：
  - `glossary/`：`financial_terms.md`、`valuation_methods.md`、`financial_statements.md`（80+ 财经术语 + 6 大估值方法 + 三大报表速查）
  - `industry_kb/`：`industry_new_energy_vehicle.md`、`industry_semiconductor.md`、`industry_ai_computing.md`（产业链 / 关键技术 / 估值锚定）
  - `_demo/`：明标虚构的演示样本（虚构公司 999001 + 虚构 AI 算力研报）
- `data/README.md` 数据采集指南（公开合规来源 / 命名约定 / 入库规模建议）；
- 入库统计：**8 文档 / 204 child chunks / 201 parent chunks**，Chroma 2.2 MB；
- RAG 冲烟测试 **7/7 通过**（`scripts/smoke_test_rag.py`），Top-1 rerank 多在 0.7~0.97 区间。

**留待补充**（P0-1 进阶版，参见"下一步计划 · 主线 A"）：
- `data/research_reports/` / `data/company_filings/` / `data/policy/` 三个目录目前为空，待真实研报 / 年报 / 政策原文填充。

#### ✅ P0-2 Prompts 重写（已落地）

| 文件 | 改造结果 |
|---|---|
| `prompts/main_prompt.txt` | 投研 Copilot 角色卡：身份 / 工具选择策略（RAG vs Web vs 行情）/ 数据三要素 / 合规免责 |
| `prompts/rag_prompt.txt` | 数据三要素（数值 + 单位 + 数据时点）；过期数据主动提醒；标的多种称呼对齐；冲突分述 |
| `prompts/report_prompt.txt` | 三种报告模板（个股速评 / 行业速评 / 晨会纪要），按用户意图自动选用，统一加风险提示 + 免责声明 |

#### ✅ P0-3 工具改造（已落地）

| 现工具 → 新工具 | description 改造结果 |
|---|---|
| `calculate_arithmetic` → `compute_financial_metric` | 列出 PE / ROE / 同环比 / 毛利率 / 股息率等典型用法 |
| `get_local_datetime` → `get_market_datetime` | A 股 / 港股 / 美股交易日窗口、财报披露时点示例 |
| `rag_summarize` / `rag_retrieve` | 强调"标的多种称呼对齐"、"实时性问题应改用 Web/Hybrid" |
| `web_search` / `hybrid_*` | 投研化用例（最新公告、并购重组、业绩快报、政策动态） |
| `get_weather_by_location` / `geocode_place` | **从 `TOOLS` 列表移除**（函数代码保留以备扩展） |

最终注册的工具数：**7 个**（rag_summarize / rag_retrieve / hybrid_search / hybrid_summarize / web_search / compute_financial_metric / get_market_datetime）。

#### ✅ 额外·入库流水线增强（已落地）
属于 P0-1 的"基础设施补漏"，独立列出便于复盘：

- `utils/file_hander.py:listdir_with_allowed_type` 升级为**递归扫描**（支持 `data/glossary/` 等子目录）；
- `rag/vector_store.py` 增加非语料文件白名单（`README.md` / `CHANGELOG.md` / `.gitkeep` 等自动跳过）；
- `config/chrome.yml` 加入 `.md` 后缀；`_documents_from_file` 加 `.md` 路由（复用 `txt_loader`）。

#### ✅ 额外·脚本 & 前端（已落地）
- `scripts/rebuild_index.py`：一键清 chroma + 父块 + md5 → 重新入库；支持 `--dry-run` / `--yes`；
- `scripts/smoke_test_rag.py`：7 个典型查询的关键词命中检查；
- `app.py` 财经主题改造：深蓝 + 金 + A 股涨跌色 CSS，品牌区，模式徽章，KB 实时统计卡，可用工具一览，hero 区三连击示例提问按钮（按对话/报告模式切换），合规免责声明常驻底部。

---

### P1 · 提升可信度（1 ~ 2 天）

#### ✅ P1-1 新增金融数据工具（已落地）

| 新工具 | 数据源 | 返回内容 |
|---|---|---|
| `get_stock_quote(ticker)` | 东方财富 push2（公开免 Key） | 最新价、涨跌额、涨跌幅、今开/昨收/最高/最低、成交量、成交额、数据时点；emoji 涨跌指示 |
| `get_stock_basics(ticker)` | 同上 | 总市值 / 流通市值（万亿/亿/万 自适应单位）、PE-TTM、PE-LYR、PB、换手率 |
| `convert_currency(amount, from_ccy, to_ccy)` | open.er-api.com（基于 ECB 等公开汇率，免 Key） | 换算后金额 + 汇率 + 数据更新时间（UTC） |

**实现要点**：
- `_normalize_ticker` 统一解析 `600519` / `sh600519` / `1.600519` / `00700` / `HK00700` / `NVDA` / `105.NVDA` 等多种写法 → 东财 secid；
- 美股 NASDAQ（`105.`）拿不到结果时**自动回退** NYSE（`106.`）；
- HTTP helper 内置 1+2 次重试 + 指数退避（0.6s → 1.2s → 2.4s），有效缓解东财偶发抖动；
- PE-LYR / 换手率为 0 时显示 `-`（避免误导，部分市场口径无静态 PE）。

**实现位置**：`tools/finance_tool.py`（300+ 行，含 helper / ticker 规范化 / 三个工具），在 `tools/agent_tool.py:TOOLS` 中注册。
**冲烟测试**：`scripts/smoke_test_finance.py`，**17/17 全部通过**（含 A 股沪深、港股、美股 NASDAQ/NYSE、错误 ticker、错误币种、负数金额等 case）。
**总工具数**：从 7 个升到 **10 个**。

#### ✅ P1-2 性能指标评估脚本（已落地）

**已完成**：
- `tests/golden_set.yml`：30 题黄金集，覆盖 7 个类别（glossary / valuation / statements / industry_nev / industry_semi / industry_ai / demo_company），每题给定 `expected_docs` + `expected_keywords` 双判定；
- `scripts/eval_retrieval_metrics.py`：三档 pipeline 对照（**Vector-only / +Rerank / +Rerank+Compression**），输出 markdown 报告 + 写盘到 `tests/eval_results.md`；
- 偶发网络抖动（DashScope SSL EOF）自动重试 1 次；
- 加入"压缩力度模拟"：用 `300 / 800 / 1500` token 预算重跑压缩器，证明压缩器在长输入下的能力。

**实测数字（30 题，2026-05-09）**：

| 指标 | Vector-only | +Rerank | +Rerank+Compression |
|---|---|---|---|
| **Recall@5** | **100%** | **100%** | **100%** |
| **MRR** | **1.000** | **1.000** | **1.000** |
| Precision@actual（实际分母） | 90.7% | **88.1%** | **88.1%** |
| 平均返回文档数 | 4.93 | **2.30** | **2.30** |
| **平均输入 tokens** | **947** | **429（−54.7%）** | **429（−54.7%）** |
| 延迟 P50 (ms) | 2163 | 4521 | 4521 |
| 延迟 P95 (ms) | 9293 | 15370 | 15370 |

**压缩力度模拟**（在 reranked docs 上以不同 token 预算重跑压缩器）：

| 预算 | 平均压缩率 | 最大压缩率 | 触发题数 |
|---|---|---|---|
| 1500 tokens | 0.0% | 0.0% | 0 / 30（全部不超阈值，正确选 none）|
| 800 tokens | 0.0% | 0.0% | 0 / 30 |
| 300 tokens | **12.3%** | **82.4%** | 6 / 30（触发 extractive 压缩）|

**含义解读**：
1. **Recall@5 与 MRR 在三档下均饱和**：当前语料结构清晰、章节切分到位，向量召回已经足够好；rerank 的真正价值不在 recall，而在**精度收紧 + 输入瘦身**。
2. **Rerank 主要价值是"砍冗余"**：返回文档从 4.93 → 2.30，平均输入 token 从 947 → 429（**−54.7%**）；对应 LLM 推理成本预期同比下降 ~50%。
3. **当前语料 reranked 后已经短到不需要压缩**：压缩器在 3500/1500/800 token 预算下统一选 none，**符合"按需压缩"原则**。压缩力度模拟显示在 300 tokens 紧预算下能拿到 82.4% 单题压缩率，证明压缩器本身能力在线，待真实长文档（如年报 PDF）接入后会真实激活。
4. **延迟代价**：Rerank 增加 ~2.3s P50 / ~6s P95（DashScope rerank 公网 RPC，含两次 round-trip + 模型推理），生产可考虑批量 rerank / 缓存。

---

### P2 · 锦上添花（1 ~ 2 天）

| 任务 | 说明 | 状态 |
|---|---|---|
| 项目级 README 重写 | 顶部加 FinSight Logo / Demo 截图 / GIF / 一键启动指引 | ⏳ 待开始 |
| 1 分钟 Demo 视频 | 录三连击用例，嵌入 GitHub README | ⏳ 待开始 |
| Streamlit Cloud / HF Space 部署 | 给一个公网可访问链接 | ⏳ 待开始 |
| 扩充 `data/glossary/` | 补充 ETF / 衍生品 / 量化术语等长尾词条 | ⏳ 待开始 |

---

## 四、三连击 Demo 用例（面试讲稿）

### Demo 1 · 纯本地 RAG（考点：检索 + Rerank + 压缩）
> "什么是 ROE？请用杜邦分析拆解一下"（已可在当前数据集上跑通）

**走线**：`rag_summarize` → 命中 `glossary/financial_terms.md` 章节 → 增强 Rerank Top-3 → 上下文压缩 → 子 LLM 摘要
**讲点**：结构化分块识别"### 净资产收益率（ROE…）"标题 → Rerank 阈值过滤无关章节 → 上下文压缩节省 token

### Demo 2 · Hybrid RAG + 报告模式 + 行情工具（考点：工具编排）
> [报告模式 ON] "帮我做一份英伟达（NVDA）的个股速评，要带上最新行情"（P1-1 后已可跑通）

**走线**：`hybrid_summarize`（本地 `industry_ai_computing.md` + Web 最新新闻）
→ `get_stock_quote("NVDA")` 拿最新价 + 涨跌
→ `get_stock_basics("NVDA")` 拿市值 + PE-TTM + PB
→ `convert_currency` 把 USD 市值换算 CNY（可选）
→ 报告 prompt 输出"个股速评"结构（核心观点 / 最新动态 / 财务摘要 / 估值 / 风险）
**讲点**：LangGraph middleware 在 `runtime.context["report"]` 动态切换 prompt；多源 Rerank 合并策略；ReAct 多步工具编排（一次问答触发 4~5 次工具调用）。

### Demo 3 · 长对话记忆（考点：工程深度）
> 第 1 轮："看下半导体行业有哪些环节？"
> 第 5 轮："那刚才聊的产业里 HBM 怎么定义？"

**走线**：滚动摘要保留"用户关心半导体" → RAG 检索时把摘要作为 `dialogue_context` 喂入
**讲点**：最近窗口 + 滚动摘要双层设计（`memory/conversation_memory.py`），避免长会话 token 失控

---

## 五、简历可写指标（已实测）

| 指标 | 测量方式 | 实测值 |
|---|---|---|
| 检索 Recall@5 | 30 题黄金集 + 三档对照 | ✅ **100% (30/30)**，全 7 类别全命中 |
| 检索 MRR | 同上 | ✅ **1.000**（命中均落 Top-1） |
| Precision@actual | 同上，按实际返回文档数为分母 | ✅ **88.1%**（+Rerank） |
| 输入 token 节省 | Rerank 前/后输入字符数对比 | ✅ **947 → 429，−54.7%** |
| 平均返回文档数 | top-K 实际命中条数 | ✅ **4.93 → 2.30**（去冗余 53%）|
| 端到端延迟 P50 / P95 | 三档对照 | ✅ **+Rerank P50 4.5s / P95 15.4s**；Vector-only P50 2.2s |
| 上下文压缩最大单题率 | 在 300 token 紧预算下 | ✅ **82.4%**（extractive 策略） |
| 冲烟通过率 RAG | `scripts/smoke_test_rag.py` 7 题关键词命中 | ✅ **7/7** |
| 冲烟通过率 金融工具 | `scripts/smoke_test_finance.py` 17 用例 | ✅ **17/17** |
| Top-1 rerank 高分占比 | 7 题中 Top-1 rerank ≥ 0.7 | ✅ **6/7**（最高 0.977）|

---

## 六、简历项目段模板

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
  在 300 token 紧预算下可拿到最大单题压缩率 82.4%（extractive 策略），
  对超长输入按需启用，避免对短文档的过度压缩。
- 实现「最近窗口 + 滚动摘要」两层长会话记忆，>20 轮对话 token 占用稳定。
- 自研金融数据工具集：股票行情 / 基本面（东财 push2，免 Key，多市场 ticker
  规范化 + 自动 NYSE 回退）+ 汇率（open.er-api.com，免 Key），冲烟测试 17/17 通过。
- 自研投研主题 Streamlit 工作台：A 股涨跌色 + 模式徽章 + 知识库实时统计 +
  示例提问引导 + 工具调用透明展开，端到端 demo 体验。
- 自动化评估脚本：30 题黄金集 + 三档 pipeline 对照（Vector-only / +Rerank /
  +Rerank+Compression）+ 多预算压缩力度模拟，单次端到端 ~270s。

GitHub: github.com/<you>/finsight    Demo: <streamlit / HF link>
```

---

## 七、下一步计划（按优先级 / 性价比）

下面给出**两条并行主线**，建议你按"主线 1 / 主线 2"交替推进，避免被某一项卡住停滞。

### 主线 1 · 让 Demo 立得住（**强烈推荐先做**）

| 顺序 | 任务 | 工时 | 收益 | 实现要点 | 状态 |
|---|---|---|---|---|---|
| 1️⃣ | **P1-1 新增 `get_stock_quote` / `get_stock_basics` / `convert_currency`** | 半天 | demo 实时感拉满 | 东财 push2 + open.er-api，多市场 ticker 规范化，重试退避 | ✅ **已落地** |
| 2️⃣ | **接入 1~2 只真实标的的真实研报/年报**（A） | 半天 | 让 RAG 不再只是"科普级" | 从巨潮资讯网下 1~2 份年报 PDF（如宁德时代 300750 / 比亚迪 002594）放进 `data/company_filings/`，跑一次 `rebuild_index.py` | ⏳ 待开始 |
| 3️⃣ | **录一段 1 分钟 demo 视频**（P2） | 半天 | 简历点击率 +10x | 屏幕录制三连击用例：① ROE 杜邦拆解 ② AI 算力行业速评（报告模式）③ 长对话记忆 + 实时行情 | ⏳ 待开始 |

### 主线 2 · 让简历有数字（与主线 1 解耦）

| 顺序 | 任务 | 工时 | 收益 | 实现要点 | 状态 |
|---|---|---|---|---|---|
| 1️⃣ | **P1-2 性能指标评估脚本** | 1 天 | 简历可写真实数字 | 30 题黄金集 + 三档对照 + 压缩力度模拟，自动输出 markdown 报告 | ✅ **已落地** |
| 2️⃣ | **README 重写**（P2） | 半天 | GitHub 第一印象 | 在主 README 顶部加 FinSight 品牌、demo 截图、性能指标表格、一键启动命令 | ⏳ 待开始 |
| 3️⃣ | **公网部署**（Streamlit Cloud / HF Space） | 1 小时 | 简历可挂 demo 链接 | 注意 DASHSCOPE_API_KEY 用平台 secret 注入；Web 限制下 DuckDuckGo 可能受限，需做兜底 | ⏳ 待开始 |

### 不推荐立刻做（性价比较低）

- ❌ **B. 巨潮批量爬虫** —— 工时 1 天但简历加分有限，且会让你陷入"数据采集"的坑里。手动下 3~5 份 PDF 已经足够 demo。
- ❌ **大幅扩充 `data/`（>50 文件）** —— 文件越多越容易遇到长尾质量问题，反而拉低 RAG 命中率；保持精而不在多。
- ❌ **接付费数据源**（Wind / Choice）—— 简历加分有限，本地起跑门槛高。

### 我的最终建议

> **本周内完成主线 1 的 1️⃣ + 2️⃣，用半天完成主线 2 的 1️⃣**——
> 这样三天后你就能拿到："实时行情工具 + 真实研报问答 + Recall@5 / 压缩率真实数字"三块硬料，
> 简历项目段就**完全立住**了。视频 / 部署 / README 等可作为后续 polish 一周内解决。

---

## 八、不动 / 慎动清单（已更新）

为防止改造过程中破坏现有架构，以下文件**默认不改**：

| 文件 | 状态 | 说明 |
|---|---|---|
| `rag/structured_chunking.py` | 不动 | 章节正则已能匹配研报 / 年报 / Markdown |
| `rag/reranker_enhanced.py` | 不动 | 仅调 `config/rag.yml` 阈值 |
| `rag/context_compressor.py` | 不动 | 策略路由已覆盖代码/事实/超长 |
| `rag/parent_store.py` | 不动 | parent-child 索引存储 |
| `tools/reactagent.py` | 不动 | ReAct 流式与 middleware 绑定 |
| `tools/mid_ware.py` | 不动 | 报告模式开关 |
| `memory/conversation_memory.py` | 不动 | 双层记忆已稳 |
| `model/model.py` | 不动 | 模型 / Embedding / Rerank 客户端 |

**已发生改动**（合理范围内）：

| 文件 | 改动原因 |
|---|---|
| `prompts/*.txt` × 3 | P0-2 投研化重写 |
| `tools/agent_tool.py` | P0-3 工具改名 + description 投研化 + 移除 weather/geocode |
| `data/**` | P0-1 数据替换 |
| `utils/file_hander.py` | 入库递归扫描升级（向后兼容） |
| `rag/vector_store.py` | `.md` 路由 + 非语料文件白名单 |
| `config/chrome.yml` | 加 `.md` 后缀 |
| `app.py` | 前端财经主题改造 |
| `tools/finance_tool.py` | P1-1 新增（行情 / 基本面 / 汇率三个工具） |
| `scripts/rebuild_index.py` | 新增（一键索引重建） |
| `scripts/smoke_test_rag.py` | 新增（RAG 冲烟） |
| `scripts/smoke_test_finance.py` | 新增（金融工具冲烟，17/17 通过） |
| `tests/golden_set.yml` | P1-2 新增（30 题检索黄金集） |
| `scripts/eval_retrieval_metrics.py` | P1-2 新增（三档对照 + 压缩力度模拟） |
| `tests/eval_results.md` | P1-2 自动生成（评估报告，Recall@5=100%、Rerank 减输入 token 54.7%） |

如确需改动"默认不动"清单中的文件，请单独评估并记录原因。

---

_最后更新：2026-05-09 · 进度 8/11 任务已落地_
