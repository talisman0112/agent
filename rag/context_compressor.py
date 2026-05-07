"""
上下文压缩模块：RAG 检索结果优化

解决 RAG 系统中检索结果过长、超出 LLM 上下文限制的问题。
采用三级压缩策略：粗粒度过滤 → 中粒度提取 → 细粒度摘要

Author: AI Assistant
Date: 2026-05-06
"""

import re
import time
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass
from enum import Enum

from langchain_core.documents import Document

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

    使用示例：
        >>> compressor = ContextCompressor()
        >>> result = compressor.compress(
        ...     query="什么是TCP三次握手？",
        ...     documents=reranked_docs,
        ...     max_tokens=3500
        ... )
        >>> print(f"压缩率: {result.stats.compression_ratio:.1%}")
        >>> compressed_docs = result.documents
    """

    def __init__(
        self,
        llm_client=None,
        embedding_client=None,
        config: Optional[Dict] = None
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
        max_tokens: Optional[int] = None,
        strategy: CompressionStrategy = CompressionStrategy.AUTO
    ) -> CompressionResult:
        """
        主压缩入口

        Args:
            query: 用户查询
            documents: 待压缩文档列表（已通过 Rerank）
            max_tokens: 压缩后目标 Token 上限
            strategy: 压缩策略

        Returns:
            CompressionResult: 包含压缩后文档、统计信息、质量评分

        流程：
            1. 估算原始 Token 数
            2. 如果未超限，直接返回
            3. 选择压缩策略
            4. 执行多级压缩
            5. 返回结果
        """
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
        compression_ratio = (original_tokens - compressed_tokens) / original_tokens if original_tokens > 0 else 0

        # 质量评估
        quality_score = self._evaluate_quality(query, documents, compressed_docs)

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
        """
        估算文档列表的 Token 数

        使用字符数/2 快速估算（中文每个字符约 1-2 token，英文每 4 字符约 1 token）
        """
        total_chars = sum(len(doc.page_content) for doc in documents)
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
        query_lower = query.lower()

        # 代码相关查询 → 提取模式（代码不能摘要）
        code_keywords = ['代码', 'code', '函数', 'function', 'api', '类', 'class',
                        '报错', 'error', 'exception', 'bug', 'python', 'java', 'c++']
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

            # 分段
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
                    "compression_type": "extract",
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
        placeholder = '\n[CODE_BLOCK_{}]\n'
        text_without_code = content
        for i, _ in enumerate(code_blocks):
            text_without_code = re.sub(
                code_block_pattern,
                placeholder.format(i),
                text_without_code,
                count=1
            )

        # 按段落分割（支持中英文段落分隔符）
        paragraphs = re.split(r'\n\s*\n|\r\n\s*\r\n', text_without_code)
        paragraphs = [p.strip() for p in paragraphs if p.strip()]

        # 重组段落和代码块
        segments = []
        for para in paragraphs:
            code_match = re.search(r'\[CODE_BLOCK_(\d+)\]', para)
            if code_match:
                idx = int(code_match.group(1))
                if idx < len(code_blocks):
                    segments.append(code_blocks[idx])
            elif len(para) > 300:
                # 长段落按句子分割（支持中英文）
                sentences = re.split(r'(?<=[。！？.!?])\s+', para)
                # 合并短句（每 2 句一组）
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
        query_words = set(self._extract_words(query.lower()))

        for i, segment in enumerate(segments):
            # 关键词匹配分数
            segment_words = set(self._extract_words(segment.lower()))
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
                    semantic_score = self._calculate_semantic_similarity(query, segment)
                    final_score = 0.6 * final_score + 0.4 * semantic_score
                except Exception:
                    pass

            scores.append(final_score)

        return scores

    def _extract_words(self, text: str) -> List[str]:
        """提取有效词汇（中英文混合）"""
        # 提取中文字符串（2字以上）
        chinese_words = re.findall(r'[\u4e00-\u9fa5]{2,}', text)
        # 提取英文单词
        english_words = re.findall(r'[a-zA-Z]+', text)
        return chinese_words + english_words

    def _calculate_semantic_similarity(self, text1: str, text2: str) -> float:
        """计算两段文本的语义相似度"""
        if not self.embeddings:
            return 0.5

        try:
            emb1 = self.embeddings.embed_query(text1[:500])
            emb2 = self.embeddings.embed_query(text2[:500])

            import numpy as np
            emb1 = np.array(emb1)
            emb2 = np.array(emb2)

            similarity = np.dot(emb1, emb2) / (np.linalg.norm(emb1) * np.linalg.norm(emb2) + 1e-8)
            return float(similarity)
        except Exception as e:
            logger.warning("语义相似度计算失败: %s", e)
            return 0.5

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
        """
        if not self.llm:
            logger.warning("LLM 客户端未配置，回退到提取式压缩")
            return self._extractive_compress(query, documents, max_tokens)

        compressed_docs = []

        # 计算每个文档的预算
        budget_per_doc = max_tokens // max(len(documents), 1)

        for doc in documents:
            doc_tokens = self._estimate_tokens([doc])

            # 如果文档很短，直接保留
            if doc_tokens <= budget_per_doc * 1.2:  # 允许 20% 弹性
                compressed_docs.append(doc)
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

        return compressed_docs

    def _generate_summary(
        self,
        query: str,
        content: str,
        target_tokens: int
    ) -> str:
        """
        使用 LLM 生成查询相关的摘要

        Args:
            query: 用户查询
            content: 原始文档内容
            target_tokens: 目标 Token 数

        Returns:
            摘要文本
        """
        target_chars = target_tokens * 2  # 粗略估算

        # 截断输入，避免超出 LLM 上下文
        content_input = content[:3000] if len(content) > 3000 else content

        summary_prompt = f"""请根据以下查询，从文档中提取关键信息并生成摘要。

查询：{query}

文档内容（长度：{len(content)} 字符）：
{content_input}

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
        # 第一阶段：提取式压缩（目标 150% 预算，为第二阶段留空间）
        mid_tokens = int(max_tokens * 1.5)
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

        # 关键词匹配
        original_keywords = set(self._extract_words(original_text))
        compressed_keywords = set(self._extract_words(compressed_text))

        if not original_keywords:
            return 1.0

        keyword_retention = len(compressed_keywords & original_keywords) / len(original_keywords)

        # 文档覆盖率
        doc_coverage = len(compressed_docs) / len(original_docs) if original_docs else 1.0

        # 综合质量分数
        quality_score = 0.7 * keyword_retention + 0.3 * doc_coverage

        return min(quality_score, 1.0)


def format_compressed_docs(docs: List[Document]) -> str:
    """
    格式化压缩后的文档供 prompt 使用

    包含压缩标记，让 LLM 知道内容已被优化

    Args:
        docs: 文档列表

    Returns:
        格式化字符串
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


# ============== 测试 ==============

if __name__ == "__main__":
    # 单元测试
    print("=" * 70)
    print("上下文压缩器测试")
    print("=" * 70)

    # 构造测试文档
    test_docs = [
        Document(
            page_content="""TCP协议是一种面向连接的、可靠的、基于字节流的传输层通信协议。

TCP三次握手是建立TCP连接的过程。第一次握手：客户端发送SYN包到服务器。第二次握手：服务器收到SYN包，确认客户的SYN，同时自己也发送一个SYN包。第三次握手：客户端收到服务器的SYN包，向服务器发送确认包ACK。

TCP协议的特点包括：
1. 面向连接：通信双方必须先建立连接
2. 可靠传输：通过确认和重传机制保证数据可靠
3. 流量控制：使用滑动窗口机制
4. 拥塞控制：防止网络过载

UDP协议与TCP不同，它是无连接的，不保证数据可靠传输。""",
            metadata={"rerank_score": 0.95, "source": "network.txt"}
        ),
        Document(
            page_content="""HTTP协议是基于TCP的应用层协议，用于Web数据传输。

HTTP/1.0 每次请求都需要建立新的TCP连接，效率较低。
HTTP/1.1 引入了持久连接，可以在一个TCP连接上发送多个请求。
HTTP/2 引入了多路复用，解决了队头阻塞问题。

HTTP方法包括：GET、POST、PUT、DELETE等。
状态码包括：200成功、404未找到、500服务器错误等。""",
            metadata={"rerank_score": 0.85, "source": "http.txt"}
        ),
        Document(
            page_content="""Python是一种高级编程语言，支持多种编程范式。

```python
def hello_world():
    print("Hello, World!")
    return True
```

Python的特点：
- 简洁优雅
- 丰富的标准库
- 强大的第三方生态
- 跨平台支持

广泛应用于Web开发、数据分析、人工智能等领域。""",
            metadata={"rerank_score": 0.75, "source": "python.txt"}
        )
    ]

    compressor = ContextCompressor()

    # 测试 1: 无需压缩
    print("\n【测试 1】Token 未超限（max_tokens=10000）")
    result = compressor.compress("TCP协议", test_docs, max_tokens=10000)
    print(f"  压缩方法: {result.stats.method_used}")
    print(f"  压缩率: {result.stats.compression_ratio:.1%}")
    print(f"  质量分: {result.quality_score:.2f}")

    # 测试 2: 提取式压缩
    print("\n【测试 2】提取式压缩（max_tokens=1000）")
    result = compressor.compress(
        "TCP三次握手",
        test_docs,
        max_tokens=1000,
        strategy=CompressionStrategy.EXTRACT
    )
    print(f"  压缩方法: {result.stats.method_used}")
    print(f"  原始 Token: {result.stats.original_tokens}")
    print(f"  压缩后 Token: {result.stats.compressed_tokens}")
    print(f"  压缩率: {result.stats.compression_ratio:.1%}")
    print(f"  质量分: {result.quality_score:.2f}")
    print(f"  处理时间: {result.stats.processing_time_ms:.1f}ms")

    print("\n  压缩后文档:")
    for i, doc in enumerate(result.documents, 1):
        meta = doc.metadata
        print(f"  文档{i}: {meta.get('compressed_length', 0)}字符 "
              f"(原{meta.get('original_length', 0)}字符) "
              f"保留{meta.get('segments_kept', 0)}/{meta.get('segments_total', 0)}段")

    # 测试 3: 格式化输出
    print("\n【测试 3】格式化输出")
    formatted = format_compressed_docs(result.documents)
    print(formatted[:500] + "..." if len(formatted) > 500 else formatted)

    # 测试 4: 代码查询（应使用 EXTRACT 策略）
    print("\n【测试 4】自动策略选择（代码查询）")
    result = compressor.compress(
        "Python 代码示例",
        test_docs,
        max_tokens=800,
        strategy=CompressionStrategy.AUTO
    )
    print(f"  自动选择策略: {result.stats.method_used}")

    print("\n" + "=" * 70)
    print("测试完成")
    print("=" * 70)
