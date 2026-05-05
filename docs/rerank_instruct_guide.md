# Rerank Instruct 参数使用指南

## 概述

Instruct 参数是 `qwen3-rerank` 模型的核心功能，用于指导模型如何评估文档与查询的相关性。正确使用 instruct 可以显著提升检索精度。

## Instruct 工作原理

```
无 Instruct（默认语义匹配）          有 Instruct（任务导向）
├─ 模型判断：文本和查询像不像？        ├─ 模型判断：文本能不能回答问题？
└─ 结果：关键词匹配的文档虚高          └─ 结果：真正有用的文档得分突出
```

## 预设模式

### 1. QA 模式（默认）
```yaml
rerank_instruct_mode: qa
```
- **用途**：用户问具体问题，需要直接答案
- **Instruct**：`Given a web search query, retrieve relevant passages that answer the query.`
- **示例查询**：`什么是TCP三次握手？`, `怎么安装Python？`

### 2. 语义模式
```yaml
rerank_instruct_mode: semantic
```
- **用途**：查找主题相关资料，不追求直接回答
- **Instruct**：`Retrieve semantically similar text that shares the same core meaning.`
- **示例查询**：`找一些关于深度学习的资料`, `了解计算机网络相关概念`

### 3. 定义模式
```yaml
rerank_instruct_mode: definition
```
- **用途**：查找概念定义、术语解释
- **Instruct**：`Given a query about a concept or term, retrieve passages that define or explain it clearly.`
- **示例查询**：`什么是区块链？`, `解释下什么是ORM`

### 4. 代码模式
```yaml
rerank_instruct_mode: code
```
- **用途**：技术查询，找代码示例或 API 文档
- **Instruct**：`Given a technical query, retrieve relevant code snippets, API documentation, or technical explanations.`
- **示例查询**：`requests库怎么发送POST请求？`, `Java多线程怎么写？`

### 5. 事实模式
```yaml
rerank_instruct_mode: fact
```
- **用途**：查找具体事实、数据
- **Instruct**：`Given a factual query, retrieve passages that provide accurate, verifiable information.`
- **示例查询**：`2024年中国GDP是多少？`, `Python最新版本号`

### 6. 对比模式
```yaml
rerank_instruct_mode: comparison
```
- **用途**：对比两个事物
- **Instruct**：`Given a comparative query, retrieve passages that highlight differences, similarities, or relationships.`
- **示例查询**：`TCP和UDP有什么区别？`, `Python2和Python3的区别`

### 7. 摘要模式
```yaml
rerank_instruct_mode: summary
```
- **用途**：查找综述性、总结性内容
- **Instruct**：`Given a topic query, retrieve comprehensive passages that summarize key points.`
- **示例查询**：`总结下人工智能发展历程`, `机器学习的应用领域有哪些`

## 自动选择模式（推荐）

```yaml
rerank_auto_instruct: true  # 根据查询自动选择模式
```

当开启自动模式时，系统会根据查询内容智能选择 instruct：

| 查询特征 | 自动选择模式 |
|---------|-------------|
| 包含"代码/函数/API/报错" | code |
| 包含"什么是/什么叫/定义" | definition |
| 包含"区别/比较/vs" | comparison |
| 包含"多少/数据/时间" | fact |
| 其他 | qa |

## 配置示例

### 基础配置
```yaml
# config/rag.yml

# 启用增强版 Rerank
rerank_enhanced: true

# 使用自动模式（推荐）
rerank_instruct_mode: qa
rerank_auto_instruct: true

# 分数阈值（过滤低质量文档）
rerank_score_threshold: 0.4
```

### 代码检索专用配置
```yaml
# 针对技术文档库
rerank_instruct_mode: code
rerank_auto_instruct: false  # 固定使用代码模式
rerank_score_threshold: 0.35
```

### 自定义 Instruct（高级）
```yaml
# 完全自定义 instruct（会覆盖模式和自动选择）
rerank_custom_instruct: "Given a student query about computer networking, retrieve textbook passages that explain the concept clearly with examples."
```

## 代码中使用

### 方式1：使用配置文件
```python
from rag.ragsummarize import RAGSummarize

rag = RAGSummarize()  # 自动读取 config/rag.yml
answer = rag.summarize("什么是TCP三次握手？")
```

### 方式2：动态指定模式
```python
from rag.reranker_enhanced import get_enhanced_reranker, InstructMode

# 创建特定模式的 Reranker
reranker = get_enhanced_reranker(
    model="qwen3-rerank",
    instruct_mode=InstructMode.CODE,  # 固定代码模式
    auto_instruct=False,
    score_threshold=0.4,
)
```

### 方式3：单次调用自定义 Instruct
```python
# 针对某次查询使用特殊 instruct
docs = reranker.rerank(
    query="解释下什么是RESTful API",
    documents=candidate_docs,
    custom_instruct="Given a technical interview question, retrieve clear, structured explanations suitable for explaining to a junior developer."
)
```

## 常见问题

### Q1: 为什么我的计算机网络课程得分不高？

**原因**：课程文档通常很长，Rerank 模型只评估前 4000 token，且默认关注"语义相似"而非"回答问题"。

**解决**：
1. 确保开启 `rerank_auto_instruct: true`，让系统自动识别为 Q&A 查询
2. 检查文档切分大小（建议 800-1000 token）
3. 适当降低 `score_threshold`（如 0.3）避免过滤掉有用的长文档

### Q2: 自动选择错误怎么办？

**解决**：关闭自动选择，手动指定模式：
```yaml
rerank_auto_instruct: false
rerank_instruct_mode: definition  # 手动指定
```

### Q3: 如何知道当前使用了什么 instruct？

**方法**：查看日志，增强版 Reranker 会记录：
```
使用增强版 Reranker（阈值: 0.40, 模式: qa, 自动: True）
```

### Q4: 自定义 instruct 怎么写效果最好？

**原则**：
1. 明确任务：告诉模型要做什么（retrieve/find/locate）
2. 说明标准：说明好文档的特征（accurate/clear/structured）
3. 指定场景：说明使用场景（for a student/for a developer）

**示例**：
```
# 不好的 instruct
"Find good documents."

# 好的 instruct
"Given a student question about computer science, retrieve textbook passages that provide clear explanations with concrete examples."
```

## 调试技巧

### 查看实际使用的 Instruct
```python
from rag.reranker_enhanced import get_instruct, auto_select_instruct

query = "什么是TCP三次握手？"
instruct = auto_select_instruct(query)
print(f"查询: {query}")
print(f"自动选择的 instruct: {instruct}")
```

### 对比不同 Instruct 的效果
```python
from rag.reranker_enhanced import get_enhanced_reranker, InstructMode

docs = [...]  # 你的候选文档
query = "什么是TCP三次握手？"

for mode in [InstructMode.QA, InstructMode.DEFINITION, InstructMode.SEMANTIC]:
    reranker = get_enhanced_reranker(
        instruct_mode=mode,
        auto_instruct=False,
        top_n=3
    )
    result = reranker.rerank(query, docs)
    print(f"\n{mode.value} 模式:")
    for doc in result:
        print(f"  [{doc.metadata['rerank_score']:.4f}] {doc.page_content[:40]}")
```

## 最佳实践

1. **一般查询**：开启 `auto_instruct: true`，让系统自动选择
2. **专业领域**：如果是代码库/法律文档/医学文献，固定对应模式
3. **阈值调优**：
   - 文档质量高 → `threshold: 0.5`（严格过滤）
   - 文档质量参差 → `threshold: 0.3`（宽松保留）
4. **测试验证**：用 `tests/test_rerank_analysis.py` 工具验证效果
