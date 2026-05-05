# Chunk 切分优化方案

本文档与当前仓库实现对应，便于后续按优先级落地改进。

## 1. 当前实现（基线）

| 项目 | 说明 |
|------|------|
| 切分器 | LangChain `RecursiveCharacterTextSplitter` |
| 配置位置 | `config/chrome.yml`：`chunk_size`、`chunk_overlap`、`separators`（列表）；可选旧键 `separator`（单字符串，会被包装为一项） |
| 代码位置 | `rag/vector_store.py`：`VectorStoreService` 通过 `_resolve_text_splitter_separators()` 解析配置并初始化 splitter，`load_data()` 中 `split_documents()` 后写入 Chroma |
| 计量单位 | **字符数**，非 token |
| 分隔策略 | **多级 `separators`**：`\\n\\n` → 单换行 → 中文句读 → 空格 → 字符兜底（与文档 2.1 一致） |
| 默认切分参数 | **2.2 已调优默认**：`chunk_size: 512`，`chunk_overlap: 80`（约 15.6% 重叠） |
| 结构化分段 | **2.3 已落地**：入库前按标题/章节预切 `Document`，写入 `section` metadata；可选为 chunk 增加 `【章节】` 前缀（见 `config/chrome.yml`、`rag/structured_chunking.py`） |

检索侧已有 **向量粗排 + Rerank**（见 `rag/ragsummarize.py`），chunk 质量直接影响粗排候选与 Rerank 上限。

---

## 2. 优化优先级（建议按顺序推进）

### 2.1 多级分隔符（低成本、高收益）

**问题**：仅用 `["\n\n"]` 时，段落很长或排版不规范的文档容易在不自然的位置被硬性截断。

**建议**：将 `separators` 扩展为多级回退，例如（中文知识库常用）：

```text
["\n\n", "\n", "。", "！", "？", "；", "，", " ", ""]
```

- 先按段落/空行切，再按句读，最后兜底到单字符。
- 具体顺序可按语料特点微调；代码与表格多的语料可加入 `"```"`、`"\n|"` 等。

**落地**：✅ 已在 `config/chrome.yml` 配置 `separators` 列表；`rag/vector_store.py` 中 `_resolve_text_splitter_separators()` 读取列表，若无 `separators`/`separator` 则使用与上文一致的默认列表。

---

### 2.2 调整 chunk_size / chunk_overlap

**仓库默认（已实现 2.2）**：`chunk_size: 512`、`chunk_overlap: 80`。可按语料与评测再调。

**历史典型值**：原先 `800` / `300`（重叠约 37.5%，索引冗余偏大）。

**经验方向**（需结合 embedding 模型上下文与实测召回）：

- `chunk_size`：中文可考虑约 **300～800 字** 或对齐 **256～512 tokens** 的量级（若后续改用按 token 切分）。
- `chunk_overlap`：约为 chunk 的 **10%～20%**，或 **50～120 字**，在“不断句”与“去重”之间折中。

**建议**：以小评测集（固定 query + 期望命中片段）扫参，而非单次拍脑袋。

---

### 2.3 结构化分块与 metadata（中成本）

**适用**：带标题、章节、FAQ、列表、多文件来源的知识库。

**做法**：

- 入库前按 **标题/章节** 先分段，再对每段做二级切分。
- 每个 chunk 的 `metadata` 写入：`source`、`section`/`title`、可选 `page`（PDF）等。
- 可选：将 **标题前缀** 拼入 `page_content` 或单独字段，提升向量与 Rerank 对“章节语义”的感知。

**落地**：✅ `rag/structured_chunking.py`：`split_documents_by_sections()`、`prepend_section_title_to_chunks()`；`rag/vector_store.py` 在 `split_documents` 前后接入。配置项：`config/chrome.yml` 中 `structured_chunking_enabled`、`structured_chunk_prepend_section`。

**识别规则（启发式）**：

- Markdown 标题行：`#`～`######` + 空格 + 标题；
- 中文独立行：`第…章/节/篇`（行长短阈值内，降低误报）；
- 序号小标题：`1.` `一、` 等形式的开头行（长度限制）。

PDF 等由加载器产生的 `page` 等 metadata 会随分段保留；无标题的纯文本仍按整段进入原有递归切块流程（无 `section` 字段时不做前缀拼接）。

---

### 2.4 Parent-child / Small-to-big（中高收益）

**思路**：

- **子块**（较小）：用于向量检索，提高命中率。
- **父块**（较大）：检索命中子块后，用其父块或邻接窗口拼接进 prompt，保证上下文完整。

**适用**：定义跨句、跨段的问题；单一固定长度切块容易丢上下文时。

**落地**：需在 Chroma 中存子块 id 与父块 id 的映射，检索后做一次 expand；或采用现成框架中的 sentence-window / parent-document 模式并适配当前 `VectorStoreService`。

---

### 2.5 语义分块（高成本、需评估）

**思路**：按 embedding 相似度变化检测“话题边界”，在边界处切分，而非固定长度。

**权衡**：预处理更慢、对 embedding 质量敏感、调试复杂；建议在 **多级分隔符 + 结构化 + parent-child** 仍不足时再考虑。

---

### 2.6 Token 级切分（与模型对齐）

**问题**：按字符切分与 **embedding / LLM 的 token 预算** 不完全一致，中英混排时偏差更大。

**做法**：使用基于 tokenizer 的 text splitter（如与所用 LLM 或 embedding 同族的 tokenizer），使 chunk 更贴近真实上下文窗口。

---

## 3. 业界常见分层（对照）

| 层级 | 典型组合 | 说明 |
|------|----------|------|
| 主流基线 | Recursive/Token chunk + overlap + rerank | 实现简单，多数场景足够 |
| 企业文档 | 结构化切块 + metadata + hybrid（向量 + 关键字/BM25）+ rerank | 复杂版式、合规文档 |
| 高阶 | Semantic chunking、parent-child、contextual chunk（块级上下文摘要后再 embed） | 长文档、跨章节问答 |

当前仓库已具备 **rerank** 与 **HybridRAG（本地 + Web）**；本地侧优先补足 **分块策略与 metadata**，通常比继续堆检索路数性价比更高。

---

## 4. 相关文件索引

- `config/chrome.yml` — `chunk_size`、`chunk_overlap`、`separators`（或旧键 `separator`）、`structured_chunking_enabled`、`structured_chunk_prepend_section`
- `rag/structured_chunking.py` — 标题预分段、metadata、`【章节】` 前缀
- `rag/vector_store.py` — `RecursiveCharacterTextSplitter`、`split_documents`、`add_documents`
- `rag/ragsummarize.py` — `RAGSummarize` / `HybridRAG` 检索与 Rerank 链路
- `utils/file_hander.py` — 各格式加载器（可扩展结构化解析入口）

---

## 5. 验收建议

- 固定一批 **query + 期望引用来源/段落**，对比改 chunk 前后的：命中率、Rerank 后 Top-1/Top-3 相关性、生成答案是否可溯源。
- 观察索引体积与去重：overlap 过大时，粗排列表易充斥近重复片段，可结合 Rerank 的去重阈值一起调（参见 `config/rag.yml` 中 rerank 相关项）。
