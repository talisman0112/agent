# 上下文压缩方案：RAG 长文档优化

## 1. 方案概述

### 1.1 问题背景

当前系统面临的挑战：
- 父文档展开后内容过长，超出 LLM 上下文限制（常见 4k-8k token）
- 检索到的文档包含大量与查询无关的冗余信息
- 多路召回后文档总量累积，导致 prompt 臃肿

### 1.2 目标指标

| 指标 | 当前基线 | 目标值 | 测量方式 |
|------|---------|--------|---------|
| 平均上下文 Token 数 | 6000+ | ≤ 3500 | tiktoken 统计 |
| 信息密度（有效内容占比） | ~40% | ≥ 75% | 人工评估 + LLM 评分 |
| 答案质量（相关性评分） | 3.2/5 | ≥ 3.8/5 | 测试集评估 |
| 压缩率 | 0% | 40-60% | (原始-压缩后)/原始 |
| 端到端延迟 | 基准 | ≤ 基准+200ms | 实测 |

### 1.3 核心策略

采用**分层渐进式压缩**架构：

```
┌─────────────────────────────────────────────────────────────┐
│                    上下文压缩流水线                           │
├─────────────────────────────────────────────────────────────┤
│  Level 1: 粗粒度过滤                                         │
│    ├── 基于 Rerank 分数的文档筛选                            │
│    └── 冗余文档去重（已存在）                                 │
├─────────────────────────────────────────────────────────────┤
│  Level 2: 中粒度提取                                         │
│    ├── 查询感知段落提取（Query-Aware Extraction）              │
│    └── 关键句子/段落识别                                      │
├─────────────────────────────────────────────────────────────┤
│  Level 3: 细粒度压缩                                         │
│    ├── 摘要生成（Summarization）                              │
│    └── Token 预算动态分配                                     │
└─────────────────────────────────────────────────────────────┘
```

---

## 2. 技术实现

### 2.1 架构设计

```python
# 压缩模块核心接口
class ContextCompressor:
    """上下文压缩器：三级压缩策略"""
    
    def compress(
        self,
        query: str,
        documents: List[Document],
        max_tokens: int = 3500,
        strategy: CompressionStrategy = CompressionStrategy.AUTO
    ) -> CompressionResult:
        """
        Args:
            query: 用户查询
            documents: 待压缩文档列表（已通过 Rerank）
            max_tokens: 压缩后目标 Token 上限
            strategy: 压缩策略
        
        Returns:
            CompressionResult: 包含压缩后文档、统计信息、质量评分
        """
```

### 2.2 三级压缩策略

#### Level 1: 粗粒度过滤（已有基础增强）

```python
# rag/context_compressor.py

from typing import List, Dict, Tuple
from dataclasses import dataclass
from langchain_core.documents import Document
from enum import Enum
import numpy as np
from utils.log import logger

class CompressionStrategy(Enum):
    """压缩策略枚举"""
    AUTO = "auto"           # 自动选择
    EXTRACT = "extract"     # 提取模式（保留关键段落）
    SUMMARIZE = "summarize" # 摘要模式（生成新文本）
    HYBRID = "hybrid"       # 混合模式（提取+摘要）

@dataclass
class CompressionStats:
    """压缩统计信息"""
    original_tokens: int
    compressed_tokens: int
    compression_ratio: float
    docs_before: int
    docs_after: int
    method_used: str
    processing_time_ms: float

@dataclass
class CompressionResult:
    """压缩结果"""
    documents: List[Document]
    stats: CompressionStats
    quality_score: float  # 0-1，信息保留度评分


class ContextCompressor:
    """
    上下文压缩器：解决 RAG 检索结果过长问题
    
    核心功能：
    1. 基于查询相关性的段落提取
    2. 动态 Token 预算分配
    3. 自适应压缩策略选择
    4. 压缩质量评估
    """
    
    def __init__(
        self,
        llm_client=None,  # 用于生成摘要
        embedding_client=None,  # 用于语义相似度计算
        config: Dict = None
    ):
        self.llm = llm_client
        self.embeddings = embedding_client
        self.config = config or {}
        
        # 配置参数（可从 rag.yml 读取）
        self.max_tokens = self.config.get("compression_max_tokens", 3500)
        self.min_tokens_per_doc = self.config.get("compression_min_tokens", 200)
        self.extract_ratio = self.config.get("compression_extract_ratio", 0.6)
        self.quality_threshold = self.config.get("compression_quality_threshold", 0.7)
        
    def compress(
        self,
        query: str,
        documents: List[Document],
        max_tokens: int = None,
        strategy: CompressionStrategy = CompressionStrategy.AUTO
    ) -> CompressionResult:
        """
        主压缩入口
        
        流程：
        1. 估算原始 Token 数
        2. 如果未超限，直接返回
        3. 选择压缩策略
        4. 执行多级压缩
        5. 返回结果
        """
        import time
        start_time = time.time()
        
        max_tokens = max_tokens or self.max_tokens
        
        # 统计原始信息
        original_tokens = self._estimate_tokens(documents)
        docs_before = len(documents)
        
        # 如果未超限，无需压缩
        if original_tokens <= max_tokens:
            stats = CompressionStats(
                original_tokens=original_tokens,
                compressed_tokens=original_tokens,
                compression_ratio=0.0,
                docs_before=docs_before,
                docs_after=docs_before,
                method_used="none",
                processing_time_ms=(time.time() - start_time) * 1000
            )
            return CompressionResult(
                documents=documents,
                stats=stats,
                quality_score=1.0
            )
        
        # 自动选择策略
        if strategy == CompressionStrategy.AUTO:
            strategy = self._select_strategy(query, documents, max_tokens)
        
        # 执行压缩
        if strategy == CompressionStrategy.EXTRACT:
            compressed_docs = self._extractive_compress(query, documents, max_tokens)
            method_used = "extractive"
        elif strategy == CompressionStrategy.SUMMARIZE:
            compressed_docs = self._abstractive_compress(query, documents, max_tokens)
            method_used = "abstractive"
        else:  # HYBRID
            compressed_docs = self._hybrid_compress(query, documents, max_tokens)
            method_used = "hybrid"
        
        # 计算统计
        compressed_tokens = self._estimate_tokens(compressed_docs)
        compression_ratio = (original_tokens - compressed_tokens) / original_tokens
        
        # 质量评估
        quality_score = self._evaluate_quality(
            query, documents, compressed_docs
        )
        
        stats = CompressionStats(
            original_tokens=original_tokens,
            compressed_tokens=compressed_tokens,
            compression_ratio=compression_ratio,
            docs_before=docs_before,
            docs_after=len(compressed_docs),
            method_used=method_used,
            processing_time_ms=(time.time() - start_time) * 1000
        )
        
        logger.info(
            "上下文压缩完成: %d → %d tokens (压缩率 %.1f%%, 策略: %s, 质量: %.2f)",
            original_tokens, compressed_tokens,
            compression_ratio * 100, method_used, quality_score
        )
        
        return CompressionResult(
            documents=compressed_docs,
            stats=stats,
            quality_score=quality_score
        )
    
    def _estimate_tokens(self, documents: List[Document]) -> int:
        """估算文档列表的 Token 数（使用字符数/4 快速估算）"""
        total_chars = sum(len(doc.page_content) for doc in documents)
        # 中文每个字符约 1-2 token，英文每 4 字符约 1 token
        # 使用保守估计：每 2 字符 1 token
        return int(total_chars / 2)
    
    def _select_strategy(
        self,
        query: str,
        documents: List[Document],
        max_tokens: int
    ) -> CompressionStrategy:
        """
        根据查询类型和文档特征选择压缩策略
        
        规则：
        - 事实查询 → 提取模式（保留精确信息）
        - 总结/解释查询 → 摘要模式
        - 代码/技术 → 提取模式（保留代码块）
        - 超长文档 → 混合模式
        """
        original_tokens = self._estimate_tokens(documents)
        
        # 提取查询特征
        query_lower = query.lower()
        
        # 代码相关查询 → 提取模式（代码不能摘要）
        code_keywords = ['代码', 'code', '函数', 'function', 'api', '类', 'class', 
                        '报错', 'error', 'exception', 'bug', 'python', 'java']
        if any(kw in query_lower for kw in code_keywords):
            return CompressionStrategy.EXTRACT
        
        # 事实/定义查询 → 提取模式
        fact_patterns = ['什么是', '什么叫', '定义', '概念', '是什么意思',
                        '多少', '几', '数据', '统计', '时间', '日期']
        if any(p in query_lower for p in fact_patterns):
            return CompressionStrategy.EXTRACT
        
        # 超长文档 → 混合模式
        if original_tokens > max_tokens * 2:
            return CompressionStrategy.HYBRID
        
        # 默认 → 提取模式（信息损失最小）
        return CompressionStrategy.EXTRACT
```

#### Level 2: 提取式压缩（Query-Aware Extraction）

```python
    def _extractive_compress(
        self,
        query: str,
        documents: List[Document],
        max_tokens: int
    ) -> List[Document]:
        """
        提取式压缩：保留与查询最相关的段落/句子
        
        步骤：
        1. 文档分段（段落或句子）
        2. 计算每段与查询的相似度
        3. 按分数排序，保留高分段落直到达到 Token 预算
        4. 保持原文顺序重组
        """
        from langchain_core.documents import Document
        import re
        
        compressed_docs = []
        remaining_budget = max_tokens
        
        # 按 Rerank 分数排序（高分优先）
        sorted_docs = sorted(
            documents,
            key=lambda d: d.metadata.get("rerank_score", 0),
            reverse=True
        )
        
        for doc in sorted_docs:
            if remaining_budget <= self.min_tokens_per_doc:
                break
            
            # 分段（支持段落和句子两种粒度）
            segments = self._segment_document(doc)
            if len(segments) <= 1:
                # 文档本身已经很短，检查是否直接保留
                doc_tokens = self._estimate_tokens([doc])
                if doc_tokens <= remaining_budget:
                    compressed_docs.append(doc)
                    remaining_budget -= doc_tokens
                continue
            
            # 计算每个段落与查询的相关性
            segment_scores = self._score_segments(query, segments)
            
            # 按分数排序
            scored_segments = list(zip(segments, segment_scores))
            scored_segments.sort(key=lambda x: x[1], reverse=True)
            
            # 选择段落直到达到预算
            selected_segments = []
            current_tokens = 0
            
            for segment, score in scored_segments:
                segment_tokens = len(segment) // 2
                if current_tokens + segment_tokens > remaining_budget:
                    break
                
                selected_segments.append((segment, score))
                current_tokens += segment_tokens
            
            if not selected_segments:
                continue
            
            # 按原文顺序重组
            selected_segments.sort(key=lambda x: segments.index(x[0]))
            compressed_content = "\n\n".join([s for s, _ in selected_segments])
            
            # 创建压缩后的文档
            compressed_doc = Document(
                page_content=compressed_content,
                metadata={
                    **doc.metadata,
                    "compressed": True,
                    "original_length": len(doc.page_content),
                    "compressed_length": len(compressed_content),
                    "segments_kept": len(selected_segments),
                    "segments_total": len(segments),
                }
            )
            
            compressed_docs.append(compressed_doc)
            remaining_budget -= self._estimate_tokens([compressed_doc])
        
        return compressed_docs
    
    def _segment_document(self, doc: Document) -> List[str]:
        """
        将文档分段
        
        策略：
        - 优先按段落分割
        - 长段落进一步按句子分割
        - 保留代码块完整性
        """
        content = doc.page_content
        
        # 检测代码块（保持完整）
        code_block_pattern = r'```[\s\S]*?```'
        code_blocks = re.findall(code_block_pattern, content)
        
        # 移除代码块后分段
        text_without_code = re.sub(code_block_pattern, '\n[CODE_BLOCK]\n', content)
        
        # 按段落分割（支持中英文段落分隔符）
        paragraphs = re.split(r'\n\s*\n|\r\n\s*\r\n', text_without_code)
        paragraphs = [p.strip() for p in paragraphs if p.strip()]
        
        # 重组段落和代码块
        segments = []
        code_idx = 0
        for para in paragraphs:
            if '[CODE_BLOCK]' in para:
                if code_idx < len(code_blocks):
                    segments.append(code_blocks[code_idx])
                    code_idx += 1
            elif len(para) > 300:
                # 长段落按句子分割
                sentences = re.split(r'(?<=[。！？.!?])\s+', para)
                # 合并短句（每 2-3 句一组）
                for i in range(0, len(sentences), 2):
                    group = ' '.join(sentences[i:i+2])
                    if group.strip():
                        segments.append(group.strip())
            else:
                segments.append(para)
        
        return segments
    
    def _score_segments(self, query: str, segments: List[str]) -> List[float]:
        """
        计算段落与查询的相关性分数
        
        方法：
        1. 关键词匹配分数（BM25 风格）
        2. 语义相似度（如有 embedding 客户端）
        3. 位置加权（开头和结尾更重要）
        """
        scores = []
        query_words = set(query.lower().split())
        
        for i, segment in enumerate(segments):
            # 关键词匹配分数
            segment_words = set(segment.lower().split())
            keyword_overlap = len(query_words & segment_words)
            keyword_score = keyword_overlap / max(len(query_words), 1)
            
            # 位置加权（首段 +20%，尾段 +10%）
            position_weight = 1.0
            if i == 0:
                position_weight = 1.2
            elif i == len(segments) - 1:
                position_weight = 1.1
            
            # 综合分数
            final_score = keyword_score * position_weight
            
            # 如果有 embedding，增加语义相似度
            if self.embeddings and len(segment) > 50:
                try:
                    # 简化的语义相似度计算
                    semantic_score = self._calculate_semantic_similarity(query, segment)
                    final_score = 0.6 * final_score + 0.4 * semantic_score
                except:
                    pass
            
            scores.append(final_score)
        
        return scores
    
    def _calculate_semantic_similarity(self, text1: str, text2: str) -> float:
        """计算两段文本的语义相似度"""
        if not self.embeddings:
            return 0.5
        
        try:
            emb1 = self.embeddings.embed_query(text1[:500])  # 限制长度
            emb2 = self.embeddings.embed_query(text2[:500])
            
            # 余弦相似度
            import numpy as np
            emb1 = np.array(emb1)
            emb2 = np.array(emb2)
            
            similarity = np.dot(emb1, emb2) / (np.linalg.norm(emb1) * np.linalg.norm(emb2))
            return float(similarity)
        except Exception as e:
            logger.warning("语义相似度计算失败: %s", e)
            return 0.5
```

#### Level 3: 生成式摘要（Abstractive Summarization）

```python
    def _abstractive_compress(
        self,
        query: str,
        documents: List[Document],
        max_tokens: int
    ) -> List[Document]:
        """
        生成式压缩：使用 LLM 生成摘要
        
        适用场景：
        - 文档极长，提取式压缩仍超限
        - 查询需要概括性回答
        - 多文档需要融合信息
        
        注意：需要 LLM 客户端支持
        """
        if not self.llm:
            logger.warning("LLM 客户端未配置，回退到提取式压缩")
            return self._extractive_compress(query, documents, max_tokens)
        
        from langchain_core.documents import Document
        
        compressed_docs = []
        remaining_budget = max_tokens
        
        # 计算每个文档的预算
        budget_per_doc = max_tokens // len(documents)
        
        for doc in documents:
            if remaining_budget <= 0:
                break
            
            doc_tokens = self._estimate_tokens([doc])
            
            # 如果文档很短，直接保留
            if doc_tokens <= budget_per_doc:
                compressed_docs.append(doc)
                remaining_budget -= doc_tokens
                continue
            
            # 生成查询相关的摘要
            summary = self._generate_summary(query, doc.page_content, budget_per_doc)
            
            summary_doc = Document(
                page_content=summary,
                metadata={
                    **doc.metadata,
                    "compressed": True,
                    "compression_type": "summary",
                    "original_length": len(doc.page_content),
                    "compressed_length": len(summary),
                }
            )
            
            compressed_docs.append(summary_doc)
            remaining_budget -= self._estimate_tokens([summary_doc])
        
        return compressed_docs
    
    def _generate_summary(
        self,
        query: str,
        content: str,
        target_tokens: int
    ) -> str:
        """
        使用 LLM 生成查询相关的摘要
        
        Prompt 设计原则：
        1. 明确指定目标长度
        2. 强调保留与查询相关的信息
        3. 要求结构化输出
        """
        target_chars = target_tokens * 2  # 粗略估算
        
        summary_prompt = f"""请根据以下查询，从文档中提取关键信息并生成摘要。

查询：{query}

文档内容（长度：{len(content)} 字符）：
{content[:3000]}  # 限制输入长度，避免超出 LLM 上下文

要求：
1. 摘要长度控制在 {target_chars} 字符以内（约 {target_tokens} tokens）
2. 优先保留与查询直接相关的信息
3. 保留关键事实、数据、结论
4. 使用简洁的语言，去除冗余描述
5. 如有代码示例，保留核心逻辑

请输出结构化摘要："""

        try:
            # 调用 LLM 生成摘要
            if hasattr(self.llm, 'invoke'):
                response = self.llm.invoke(summary_prompt)
                if hasattr(response, 'content'):
                    return response.content.strip()
                return str(response).strip()
            else:
                # 兼容直接调用
                return str(self.llm(summary_prompt)).strip()
        except Exception as e:
            logger.error("摘要生成失败: %s", e)
            # 回退：返回原文前段
            return content[:target_chars]
    
    def _hybrid_compress(
        self,
        query: str,
        documents: List[Document],
        max_tokens: int
    ) -> List[Document]:
        """
        混合压缩：先提取，再摘要
        
        策略：
        1. 先用提取式压缩减少 50% 体积
        2. 如果仍超限，对提取结果进行摘要
        """
        # 第一阶段：提取式压缩（目标 50%）
        mid_tokens = max_tokens * 2
        extracted = self._extractive_compress(query, documents, mid_tokens)
        
        # 检查是否仍需要压缩
        current_tokens = self._estimate_tokens(extracted)
        if current_tokens <= max_tokens:
            return extracted
        
        # 第二阶段：摘要压缩
        return self._abstractive_compress(query, extracted, max_tokens)
    
    def _evaluate_quality(
        self,
        query: str,
        original_docs: List[Document],
        compressed_docs: List[Document]
    ) -> float:
        """
        评估压缩质量（信息保留度）
        
        启发式指标：
        1. 关键词保留率
        2. 实体保留率
        3. 文档覆盖率
        """
        if not compressed_docs:
            return 0.0
        
        # 提取原文关键词
        original_text = ' '.join([d.page_content for d in original_docs])
        compressed_text = ' '.join([d.page_content for d in compressed_docs])
        
        # 简单关键词匹配（中文分词简化版）
        import re
        
        # 提取中文字符串（词语近似）
        original_keywords = set(re.findall(r'[\u4e00-\u9fa5]{2,}', original_text))
        compressed_keywords = set(re.findall(r'[\u4e00-\u9fa5]{2,}', compressed_text))
        
        # 计算保留率
        if not original_keywords:
            return 1.0
        
        keyword_retention = len(compressed_keywords & original_keywords) / len(original_keywords)
        
        # 文档覆盖率
        doc_coverage = len(compressed_docs) / len(original_docs)
        
        # 综合质量分数
        quality_score = 0.7 * keyword_retention + 0.3 * doc_coverage
        
        return min(quality_score, 1.0)
```

### 2.3 与现有系统集成

```python
# rag/context_compressor.py（续）

def format_compressed_docs(docs: List[Document]) -> str:
    """
    格式化压缩后的文档供 prompt 使用
    
    包含压缩标记，让 LLM 知道内容已被优化
    """
    parts = []
    for i, doc in enumerate(docs, start=1):
        score = doc.metadata.get("rerank_score", 0.0)
        
        # 检查是否被压缩
        if doc.metadata.get("compressed"):
            original_len = doc.metadata.get("original_length", 0)
            compressed_len = doc.metadata.get("compressed_length", 0)
            compression_info = f"[已压缩 {original_len}→{compressed_len}字符]"
        else:
            compression_info = "[原文]"
        
        parts.append(
            f"参考资料{i} [相关性: {score:.3f}] {compression_info}:\n"
            f"{doc.page_content}"
        )
    
    return "\n\n".join(parts)


# 集成到 RAGSummarize
class RAGSummarize:
    """增强版 RAG：集成上下文压缩"""
    
    def __init__(self, ..., compression_enabled: bool = None):
        # ... 原有初始化代码 ...
        
        # 新增：上下文压缩器
        cfg = rerank_config
        self.compression_enabled = compression_enabled if compression_enabled is not None \
                                   else cfg.get("compression_enabled", True)
        
        if self.compression_enabled:
            from rag.context_compressor import ContextCompressor, CompressionStrategy
            self.compressor = ContextCompressor(
                llm_client=chat_model if cfg.get("compression_use_llm", False) else None,
                embedding_client=None,  # 可接入向量模型
                config={
                    "compression_max_tokens": cfg.get("compression_max_tokens", 3500),
                    "compression_min_tokens": cfg.get("compression_min_tokens", 200),
                    "compression_extract_ratio": cfg.get("compression_extract_ratio", 0.6),
                }
            )
            logger.info("上下文压缩已启用（目标 Token: %d）", 
                       cfg.get("compression_max_tokens", 3500))
    
    def _rerank_docs(self, query: str) -> list[Document]:
        """增强版检索流程：检索 → Rerank → 压缩"""
        # 1. 向量检索（粗排）
        coarse_docs = self.rag_retriever.invoke(query)
        coarse_docs = self.vector_store.expand_retrieval_to_parents(coarse_docs)
        
        if not coarse_docs:
            return []
        
        # 2. Rerank 精排
        reranked_docs = self.reranker.rerank(query, coarse_docs)
        
        # 3. 上下文压缩（新增）
        if self.compression_enabled and self.compressor:
            result = self.compressor.compress(
                query=query,
                documents=reranked_docs,
                max_tokens=self.compressor.max_tokens,
                strategy=CompressionStrategy.AUTO
            )
            
            # 记录压缩统计
            logger.info(
                "上下文压缩: %d→%d tokens (%.1f%%), 质量分: %.2f",
                result.stats.original_tokens,
                result.stats.compressed_tokens,
                result.stats.compression_ratio * 100,
                result.quality_score
            )
            
            return result.documents
        
        return reranked_docs
```

---

## 3. 配置说明

### 3.1 rag.yml 配置

```yaml
# config/rag.yml

# ==================== 上下文压缩配置 ====================

# 启用上下文压缩
compression_enabled: true

# 目标 Token 上限（根据 LLM 上下文窗口调整）
compression_max_tokens: 3500

# 单个文档最小保留 Token（避免过度压缩）
compression_min_tokens: 200

# 提取式压缩保留比例（默认 60%）
compression_extract_ratio: 0.6

# 是否使用 LLM 进行生成式摘要（需要更多 API 调用）
compression_use_llm: false

# 压缩质量阈值（低于此值会记录警告）
compression_quality_threshold: 0.7

# 压缩策略（auto/extract/summarize/hybrid）
compression_strategy: auto
```

### 3.2 不同场景的推荐配置

| 场景 | 目标 Token | 策略 | 使用 LLM | 说明 |
|------|-----------|------|---------|------|
| 短文档问答（<2k token） | 3500 | auto | false | 轻度压缩 |
| 长文档问答（5k-10k） | 3000 | extract | false | 提取关键段落 |
| 超长文档（>10k） | 2500 | hybrid | true | 提取+摘要 |
| 代码检索 | 4000 | extract | false | 保留代码完整 |
| 快速响应 | 3000 | extract | false | 减少 LLM 调用 |

---

## 4. 评估验证

### 4.1 自动化测试

```python
# tests/test_context_compression.py

import pytest
from rag.context_compressor import ContextCompressor, CompressionStrategy
from langchain_core.documents import Document


class TestContextCompression:
    """上下文压缩测试套件"""
    
    def test_no_compression_needed(self):
        """测试：Token 未超限时不应压缩"""
        compressor = ContextCompressor()
        
        docs = [
            Document(page_content="短文档内容", metadata={"rerank_score": 0.9})
        ]
        
        result = compressor.compress("查询", docs, max_tokens=5000)
        
        assert result.stats.compression_ratio == 0.0
        assert result.stats.method_used == "none"
        assert len(result.documents) == 1
    
    def test_extractive_compression(self):
        """测试：提取式压缩功能"""
        compressor = ContextCompressor()
        
        # 构造长文档
        long_content = "\n\n".join([
            f"第{i}段内容：这是关于人工智能的讨论，包含机器学习和深度学习的基本概念。"
            for i in range(20)
        ])
        
        docs = [
            Document(page_content=long_content, metadata={"rerank_score": 0.9})
        ]
        
        result = compressor.compress(
            "人工智能",
            docs,
            max_tokens=500,
            strategy=CompressionStrategy.EXTRACT
        )
        
        # 验证压缩发生了
        assert result.stats.compression_ratio > 0.3
        assert result.stats.compressed_tokens < result.stats.original_tokens
        assert all(d.metadata.get("compressed") for d in result.documents)
    
    def test_compression_quality(self):
        """测试：压缩质量评估"""
        compressor = ContextCompressor()
        
        docs = [
            Document(
                page_content="Python 是一种高级编程语言，支持面向对象编程。",
                metadata={"rerank_score": 0.9}
            ),
            Document(
                page_content="Java 也是一种编程语言，广泛应用于企业开发。",
                metadata={"rerank_score": 0.8}
            )
        ]
        
        result = compressor.compress("编程语言", docs, max_tokens=100)
        
        # 质量分数应在合理范围
        assert 0 <= result.quality_score <= 1.0
        assert result.quality_score > 0.5  # 至少保留一半信息
    
    def test_code_preservation(self):
        """测试：代码块完整性保留"""
        compressor = ContextCompressor()
        
        code_doc = Document(
            page_content="""
这里是一些说明文字。

```python
def hello():
    print("Hello World")
```

更多说明文字。
            """,
            metadata={"rerank_score": 0.9}
        )
        
        result = compressor.compress(
            "代码示例",
            [code_doc],
            max_tokens=200,
            strategy=CompressionStrategy.EXTRACT
        )
        
        # 验证代码块被保留
        compressed = result.documents[0].page_content
        assert "```python" in compressed or "def hello" in compressed


# 性能基准测试
def test_compression_performance():
    """测试：压缩处理时间"""
    import time
    
    compressor = ContextCompressor()
    
    # 构造批量文档
    docs = [
        Document(page_content=f"文档{i}内容" * 100, metadata={"rerank_score": 0.9 - i*0.1})
        for i in range(10)
    ]
    
    start = time.time()
    result = compressor.compress("查询", docs, max_tokens=1000)
    elapsed = (time.time() - start) * 1000
    
    # 应在 500ms 内完成（纯提取式）
    assert elapsed < 500
    print(f"压缩耗时: {elapsed:.1f}ms")
```

### 4.2 人工评估指南

```python
# tests/evaluate_compression_manual.py

"""
人工评估脚本：对比压缩前后的回答质量

运行方式：
1. 准备 20-50 个测试查询
2. 分别运行压缩版和无压缩版
3. 人工评分（1-5分）
4. 统计对比
"""

import json
from rag.ragsummarize import RAGSummarize

def run_comparison_eval():
    test_queries = [
        "什么是TCP三次握手？",
        "Python 的 GIL 是什么？",
        "解释Transformer架构",
        "RESTful API 设计原则",
        # ... 更多查询
    ]
    
    # 无压缩版本
    rag_no_compress = RAGSummarize(compression_enabled=False)
    
    # 压缩版本
    rag_with_compress = RAGSummarize(compression_enabled=True)
    
    results = []
    for query in test_queries:
        answer_no_compress = rag_no_compress.summarize(query)
        answer_with_compress = rag_with_compress.summarize(query)
        
        results.append({
            "query": query,
            "no_compress": answer_no_compress,
            "with_compress": answer_with_compress,
        })
    
    # 输出对比文件供人工评估
    with open("compression_eval.json", "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    
    print("评估文件已生成: compression_eval.json")
    print("请人工评分后使用 calculate_score() 计算最终指标")


def calculate_score(evaluation_file: str):
    """计算评估指标"""
    with open(evaluation_file, "r", encoding="utf-8") as f:
        data = json.load(f)
    
    scores_no_compress = [d["score_no_compress"] for d in data]
    scores_with_compress = [d["score_with_compress"] for d in data]
    
    print(f"无压缩平均分: {sum(scores_no_compress)/len(scores_no_compress):.2f}")
    print(f"压缩后平均分: {sum(scores_with_compress)/len(scores_with_compress):.2f}")
    
    # 计算质量保留率
    retention = sum(scores_with_compress) / sum(scores_no_compress)
    print(f"质量保留率: {retention*100:.1f}%")
```

### 4.3 监控指标

```python
# 在应用中添加压缩监控

class CompressionMetrics:
    """压缩指标收集器"""
    
    def __init__(self):
        self.total_requests = 0
        self.compression_triggered = 0
        self.total_original_tokens = 0
        self.total_compressed_tokens = 0
        self.method_counts = {}
        self.quality_scores = []
    
    def record(self, result: CompressionResult):
        self.total_requests += 1
        
        if result.stats.compression_ratio > 0:
            self.compression_triggered += 1
        
        self.total_original_tokens += result.stats.original_tokens
        self.total_compressed_tokens += result.stats.compressed_tokens
        
        method = result.stats.method_used
        self.method_counts[method] = self.method_counts.get(method, 0) + 1
        
        self.quality_scores.append(result.quality_score)
    
    def report(self) -> dict:
        """生成监控报告"""
        if self.total_requests == 0:
            return {}
        
        return {
            "compression_rate": self.compression_triggered / self.total_requests,
            "avg_compression_ratio": (
                (self.total_original_tokens - self.total_compressed_tokens) 
                / self.total_original_tokens
            ),
            "avg_quality_score": sum(self.quality_scores) / len(self.quality_scores),
            "method_distribution": self.method_counts,
            "total_requests": self.total_requests,
        }
```

---

## 5. 实施路线图

### Phase 1: 基础实现（1-2 天）

- [x] 创建 `rag/context_compressor.py`
- [ ] 实现 `ContextCompressor` 基础类
- [ ] 实现 `_estimate_tokens` 和 `_select_strategy`
- [ ] 集成到 `RAGSummarize._rerank_docs`

### Phase 2: 提取式压缩（2-3 天）

- [ ] 实现 `_segment_document`（支持段落/句子分割）
- [ ] 实现 `_score_segments`（关键词+位置加权）
- [ ] 实现 `_extractive_compress`
- [ ] 添加代码块保护逻辑
- [ ] 编写单元测试

### Phase 3: 生成式压缩（3-5 天，可选）

- [ ] 实现 `_generate_summary`
- [ ] 设计摘要生成 Prompt
- [ ] 实现 `_abstractive_compress`
- [ ] 实现 `_hybrid_compress`
- [ ] 性能优化（缓存、并发）

### Phase 4: 评估优化（2-3 天）

- [ ] 编写测试套件 `tests/test_context_compression.py`
- [ ] 创建人工评估数据集
- [ ] 运行 A/B 测试（压缩 vs 无压缩）
- [ ] 调整默认参数
- [ ] 编写实施文档

### Phase 5: 生产部署（1-2 天）

- [ ] 添加监控指标
- [ ] 配置灰度发布
- [ ] 设置告警规则（质量分数低于阈值）
- [ ] 编写运维文档

---

## 6. 风险与应对

| 风险 | 影响 | 应对措施 |
|------|------|---------|
| 压缩过度导致信息丢失 | 高 | 设置最小 Token 限制；质量评分告警 |
| 代码块被截断 | 高 | 代码块检测与保护逻辑 |
| 处理延迟增加 | 中 | 异步处理；缓存策略；可选 LLM 摘要 |
| 摘要生成成本 | 中 | 默认使用提取式；LLM 摘要可配置关闭 |
| 多语言支持问题 | 低 | 分词器适配；中日韩字符特殊处理 |

---

## 7. 预期收益

基于参考数据（参考 LLMLingua 和 RAG 优化实践）：

| 指标 | 保守估计 | 乐观估计 |
|------|---------|---------|
| Token 消耗减少 | 30-40% | 50-60% |
| LLM API 成本降低 | 30% | 50% |
| 响应速度提升 | 10-20% | 30% |
| 答案质量保持 | >90% | >95% |

---

## 附录：参考资源

1. **LLMLingua**: 微软的 prompt 压缩框架
   - 论文: https://arxiv.org/abs/2310.05736
   - 代码: https://github.com/microsoft/LLMLingua

2. **LongLLMLingua**: 面向长文档的压缩
   - 论文: https://arxiv.org/abs/2312.05885

3. **RAPTOR**: 递归抽象检索树
   - 论文: https://arxiv.org/abs/2401.18059

4. **LangChain Contextual Compression**
   - 文档: https://python.langchain.com/docs/modules/data_connection/retrievers/contextual_compression/
