"""增强版 RAG Rerank 模块 - 解决评分区分度问题"""

import os
import re
from typing import List, Callable, Optional
from dataclasses import dataclass, field
from enum import Enum

import dashscope
from dashscope import TextReRank
from langchain_core.documents import Document

from utils.log import logger


class InstructMode(Enum):
    """预设的 Instruct 模式"""
    QA = "qa"                           # 问答检索
    SEMANTIC = "semantic"               # 语义相似
    DEFINITION = "definition"           # 定义/概念查找
    CODE = "code"                       # 代码/技术文档
    FACT = "fact"                       # 事实核查
    COMPARISON = "comparison"           # 对比分析
    SUMMARY = "summary"                 # 摘要总结


# 预定义的 Instruct 模板
INSTRUCT_TEMPLATES = {
    InstructMode.QA: "Given a web search query, retrieve relevant passages that answer the query.",
    
    InstructMode.SEMANTIC: "Retrieve semantically similar text that shares the same core meaning.",
    
    InstructMode.DEFINITION: "Given a query about a concept or term, retrieve passages that define or explain it clearly.",
    
    InstructMode.CODE: "Given a technical query, retrieve relevant code snippets, API documentation, or technical explanations.",
    
    InstructMode.FACT: "Given a factual query, retrieve passages that provide accurate, verifiable information.",
    
    InstructMode.COMPARISON: "Given a comparative query, retrieve passages that highlight differences, similarities, or relationships.",
    
    InstructMode.SUMMARY: "Given a topic query, retrieve comprehensive passages that summarize key points.",
}


def auto_select_instruct(query: str) -> str:
    """根据查询内容自动选择合适的 instruct。
    
    Args:
        query: 用户查询
        
    Returns:
        合适的 instruct 字符串
    """
    query_lower = query.lower()
    
    # 代码相关
    code_keywords = ['代码', 'code', '函数', 'function', 'api', '类', 'class', 
                     '报错', 'error', 'exception', 'bug', 'python', 'java', 'c++']
    if any(kw in query_lower for kw in code_keywords):
        return INSTRUCT_TEMPLATES[InstructMode.CODE]
    
    # 定义/概念相关
    definition_patterns = [
        r'什么是', r'什么叫', r'定义', r'概念', r'是什么意思',
        r'what is', r'what does', r'define', r'meaning of'
    ]
    if any(re.search(p, query_lower) for p in definition_patterns):
        return INSTRUCT_TEMPLATES[InstructMode.DEFINITION]
    
    # 对比相关
    comparison_patterns = [
        r'区别', r'差异', r'比较', r'对比', r'vs', r'versus',
        r'difference', r'compare', r'contrast', r'vs\.?'
    ]
    if any(re.search(p, query_lower) for p in comparison_patterns):
        return INSTRUCT_TEMPLATES[InstructMode.COMPARISON]
    
    # 事实/数据相关
    fact_patterns = [
        r'多少', r'几', r'数据', r'统计', r'时间', r'日期',
        r'how many', r'how much', r'when', r'what time', r'数据'
    ]
    if any(re.search(p, query_lower) for p in fact_patterns):
        return INSTRUCT_TEMPLATES[InstructMode.FACT]
    
    # 默认使用 QA 模式
    return INSTRUCT_TEMPLATES[InstructMode.QA]


def get_instruct(
    mode: Optional[InstructMode] = None,
    custom_instruct: Optional[str] = None,
    query: Optional[str] = None,
    auto: bool = False
) -> str:
    """获取 instruct 的统一接口。
    
    优先级：custom_instruct > mode > auto_select > default(QA)
    
    Args:
        mode: 使用预设模式
        custom_instruct: 自定义 instruct（最高优先级）
        query: 用户查询（用于 auto 模式）
        auto: 是否根据查询自动选择
        
    Returns:
        instruct 字符串
    """
    # 优先级1: 自定义 instruct
    if custom_instruct:
        return custom_instruct
    
    # 优先级2: 预设模式
    if mode and mode in INSTRUCT_TEMPLATES:
        return INSTRUCT_TEMPLATES[mode]
    
    # 优先级3: 自动选择
    if auto and query:
        return auto_select_instruct(query)
    
    # 默认: QA 模式
    return INSTRUCT_TEMPLATES[InstructMode.QA]


@dataclass
class RerankConfig:
    """Rerank 配置参数"""
    model: str = "qwen3-rerank"
    top_n: int = 5
    score_threshold: float = 0.25  # 分数阈值，低于此值的文档会被过滤
    
    # Instruct 配置
    instruct_mode: InstructMode = InstructMode.QA  # 预设模式
    custom_instruct: Optional[str] = None  # 自定义 instruct
    auto_instruct: bool = True  # 是否根据查询自动选择 instruct
    use_instruct: bool = True  # 是否使用指令引导
    
    # 去重配置
    dedup_threshold: float = 0.80  # 去重相似度阈值，默认0.80（越小越严格）
    
    min_score_gap: float = 0.1  # 期望的最小分数差距
    
    def get_instruct_for_query(self, query: str) -> str:
        """根据查询获取合适的 instruct"""
        return get_instruct(
            mode=self.instruct_mode if not self.auto_instruct else None,
            custom_instruct=self.custom_instruct,
            query=query,
            auto=self.auto_instruct
        )


class EnhancedDashScopeReranker:
    """增强版 DashScope Reranker，解决评分区分度问题。

    优化策略：
    1. 使用 instruct 参数指导模型关注"回答问题"而非"语义相似"
    2. 分数阈值过滤，剔除低质量文档
    3. 分数归一化，增强显示区分度
    4. 混合打分：结合 Rerank 分数和向量相似度分数
    5. 动态 Top-N：根据分数分布自适应调整返回数量
    """

    def __init__(
        self,
        config: RerankConfig = None,
        api_key: str = None,
    ):
        self.config = config or RerankConfig()
        self.api_key = api_key or os.getenv("DASHSCOPE_API_KEY") or os.getenv("TONGYI_API_KEY")
        if self.api_key:
            dashscope.api_key = self.api_key

    def rerank(
        self,
        query: str,
        documents: List[Document],
        vector_scores: List[float] = None,  # 可选：传入向量检索分数进行混合
        custom_instruct: Optional[str] = None,  # 本次调用的临时自定义 instruct
    ) -> List[Document]:
        """增强版 Rerank，支持混合打分和阈值过滤。
        
        Args:
            query: 用户查询
            documents: 候选文档
            vector_scores: 向量检索分数（可选）
            custom_instruct: 临时自定义 instruct（覆盖配置）
        """
        if not documents:
            return []

        if not self.api_key:
            logger.warning("DASHSCOPE_API_KEY 未设置，跳过 Rerank")
            return documents[: self.config.top_n]

        texts = [doc.page_content for doc in documents]

        try:
            # 获取本次使用的 instruct
            if custom_instruct:
                instruct = custom_instruct
            else:
                instruct = self.config.get_instruct_for_query(query)
            
            logger.debug("使用 instruct: %s", instruct[:50] + "..." if len(instruct) > 50 else instruct)

            # 构建请求参数
            kwargs = {
                "model": self.config.model,
                "query": query,
                "documents": texts,
                "top_n": min(len(documents), max(self.config.top_n * 2, 10)),  # 多取一些用于过滤
                "return_documents": True,
            }
            
            # qwen3-rerank 支持 instruct 参数，可引导评分偏好
            if self.config.use_instruct and self.config.model.startswith("qwen3"):
                kwargs["instruct"] = instruct

            resp = TextReRank.call(**kwargs)

            if resp.status_code != 200:
                logger.error("Rerank API 错误: %s - %s", resp.status_code, resp.message)
                return documents[: self.config.top_n]

            results = resp.output.results if hasattr(resp.output, "results") else resp.output.get("results", [])
            
            # 处理结果
            reranked_docs = self._process_results(
                documents, results, vector_scores
            )
            
            logger.info(
                "Rerank 完成: %d docs → %d docs (阈值: %.2f)",
                len(documents), len(reranked_docs), self.config.score_threshold
            )
            return reranked_docs

        except Exception as e:
            logger.error("Rerank 调用异常: %s", str(e), exc_info=True)
            return documents[: self.config.top_n]

    def _process_results(
        self,
        documents: List[Document],
        results: List[dict],
        vector_scores: List[float] = None,
    ) -> List[Document]:
        """处理 Rerank 结果：混合打分、阈值过滤、分数归一化。"""
        processed = []
        
        for r in results:
            idx = r.get("index", 0)
            rerank_score = r.get("relevance_score", 0.0)
            
            # 混合打分（可选）
            if vector_scores and idx < len(vector_scores):
                # 加权融合：Rerank 占 70%，向量分数占 30%
                vec_score = vector_scores[idx]
                combined_score = 0.7 * rerank_score + 0.3 * vec_score
            else:
                combined_score = rerank_score
            
            doc = documents[idx].copy()
            doc.metadata["rerank_score"] = rerank_score
            doc.metadata["combined_score"] = combined_score
            
            # 阈值过滤
            if combined_score >= self.config.score_threshold:
                processed.append((doc, combined_score))
        
        if not processed:
            # 如果没有通过阈值的，取最高分的 2 个保底
            logger.warning("无文档通过阈值 %.2f，取 Top-2 保底", self.config.score_threshold)
            for r in results[:2]:
                idx = r.get("index", 0)
                doc = documents[idx].copy()
                doc.metadata["rerank_score"] = r.get("relevance_score", 0.0)
                processed.append((doc, doc.metadata["rerank_score"]))
        
        # 按综合分数排序
        processed.sort(key=lambda x: x[1], reverse=True)
        
        # 动态调整 Top-N：如果分数差距大就多取，差距小就少取
        final_docs = self._adaptive_top_n(processed)
        
        return final_docs

    def _adaptive_top_n(self, sorted_docs: List[tuple[Document, float]]) -> List[Document]:
        """根据分数分布自适应调整返回数量。"""
        if not sorted_docs:
            return []
        
        scores = [s for _, s in sorted_docs]
        top_n = self.config.top_n
        
        # 计算头部分数差距
        if len(scores) >= 3:
            gap1 = scores[0] - scores[1]  # 第一与第二的差距
            gap2 = scores[1] - scores[2]  # 第二与第三的差距
            
            # 如果头部文档分数很接近，减少返回数量（只取最确定的）
            if gap1 < 0.05 and gap2 < 0.05:
                top_n = max(2, top_n - 2)
                logger.debug("头部分数接近，减少返回数量至 %d", top_n)
            # 如果第一文档明显领先，置信度高，可以多取一些
            elif gap1 > 0.2:
                top_n = min(len(sorted_docs), top_n + 2)
                logger.debug("第一文档明显领先，增加返回数量至 %d", top_n)
        
        # 去重处理（使用配置中的阈值）
        unique_docs = self._deduplicate_documents(
            [doc for doc, _ in sorted_docs[:top_n]], 
            similarity_threshold=self.config.dedup_threshold
        )
        return unique_docs
    
    def _deduplicate_documents(self, documents: List[Document], similarity_threshold: float = 0.80) -> List[Document]:
        """基于内容相似度和来源的去重（更严格版本）。

        Args:
            documents: 待去重的文档列表
            similarity_threshold: 相似度阈值，超过此值认为是重复（默认0.80）
            
        Returns:
            去重后的文档列表
        """
        if not documents:
            return []
        
        unique_docs = []
        seen_sources = {}  # 记录已处理的来源和最高分数
        
        for doc in documents:
            is_duplicate = False
            doc_content = doc.page_content.strip()
            doc_source = doc.metadata.get("source", "")  # 获取文档来源
            doc_score = doc.metadata.get("combined_score", doc.metadata.get("rerank_score", 0.0))
            
            # 策略1: 同一来源文档，只保留最高分的那个
            if doc_source:
                if doc_source in seen_sources:
                    if doc_score > seen_sources[doc_source]["score"]:
                        logger.debug("同一来源重复文档，保留高分版本: %s (%.4f > %.4f)", 
                                   doc_source, doc_score, seen_sources[doc_source]["score"])
                        # 移除旧的
                        unique_docs.remove(seen_sources[doc_source]["doc"])
                        # 更新记录
                        seen_sources[doc_source] = {"score": doc_score, "doc": doc}
                    else:
                        logger.debug("同一来源重复文档，丢弃低分版本: %s (%.4f <= %.4f)",
                                   doc_source, doc_score, seen_sources[doc_source]["score"])
                        is_duplicate = True
                        continue
                else:
                    seen_sources[doc_source] = {"score": doc_score, "doc": doc}
            
            # 策略2: 内容相似度去重（更严格的阈值）
            for kept_doc in unique_docs:
                kept_content = kept_doc.page_content.strip()
                
                # 计算相似度
                similarity = self._calculate_similarity(doc_content, kept_content)
                
                if similarity >= similarity_threshold:
                    kept_score = kept_doc.metadata.get("combined_score", kept_doc.metadata.get("rerank_score", 0.0))
                    
                    if doc_score > kept_score:
                        logger.debug("内容重复，保留高分版本 (%.4f > %.4f, 相似度: %.2f%%)", 
                                   doc_score, kept_score, similarity * 100)
                        if kept_doc in unique_docs:
                            unique_docs.remove(kept_doc)
                            # 更新来源记录
                            kept_source = kept_doc.metadata.get("source", "")
                            if kept_source and kept_source in seen_sources:
                                del seen_sources[kept_source]
                        unique_docs.append(doc)
                        if doc_source:
                            seen_sources[doc_source] = {"score": doc_score, "doc": doc}
                    else:
                        logger.debug("内容重复，丢弃低分版本 (%.4f <= %.4f, 相似度: %.2f%%)",
                                   doc_score, kept_score, similarity * 100)
                    
                    is_duplicate = True
                    break
            
            if not is_duplicate:
                unique_docs.append(doc)
        
        if len(unique_docs) < len(documents):
            logger.info("去重处理: %d docs → %d docs (过滤了 %d 个重复)", 
                       len(documents), len(unique_docs), len(documents) - len(unique_docs))
        
        return unique_docs
    
    def _calculate_similarity(self, text1: str, text2: str) -> float:
        """计算两段文本的相似度（基于字符级Jaccard）。
        
        Args:
            text1: 第一段文本
            text2: 第二段文本
            
        Returns:
            相似度 (0.0 - 1.0)
        """
        if not text1 or not text2:
            return 0.0
        
        # 使用滑动窗口字符n-gram计算相似度
        def get_ngrams(text: str, n: int = 3) -> set:
            """获取字符n-gram集合"""
            text = text.lower().replace(" ", "").replace("\n", "")
            return set(text[i:i+n] for i in range(len(text) - n + 1))
        
        ngrams1 = get_ngrams(text1)
        ngrams2 = get_ngrams(text2)
        
        if not ngrams1 or not ngrams2:
            return 0.0
        
        # Jaccard相似度
        intersection = len(ngrams1 & ngrams2)
        union = len(ngrams1 | ngrams2)
        
        return intersection / union if union > 0 else 0.0

    def rerank_with_detailed_scores(
        self, query: str, documents: List[Document]
    ) -> List[tuple[Document, float, float]]:
        """返回详细的分数信息：(Document, rerank_score, combined_score)"""
        docs = self.rerank(query, documents)
        return [
            (doc, doc.metadata.get("rerank_score", 0.0), doc.metadata.get("combined_score", 0.0))
            for doc in docs
        ]


class HybridRetriever:
    """混合检索器：BM25 + 向量检索 + Rerank，解决召回质量问题。"""

    def __init__(
        self,
        vector_store,
        reranker: EnhancedDashScopeReranker,
        bm25_weight: float = 0.3,
        vector_weight: float = 0.7,
    ):
        self.vector_store = vector_store
        self.reranker = reranker
        self.bm25_weight = bm25_weight
        self.vector_weight = vector_weight

    def retrieve(self, query: str, k: int = 20) -> List[Document]:
        """混合检索：结合向量检索和 BM25（如有）。"""
        # 1. 向量检索
        vector_docs = self.vector_store.similarity_search_with_score(query, k=k)
        
        # 2. 如果有 BM25 索引，可以合并结果（此处简化，仅用向量）
        # TODO: 可接入 Elasticsearch/OpenSearch BM25
        
        docs = [doc for doc, _ in vector_docs]
        vector_scores = [score for _, score in vector_docs]
        
        # 3. Rerank 精排（传入向量分数做混合）
        reranked = self.reranker.rerank(query, docs, vector_scores)
        
        return reranked


def get_enhanced_reranker(
    model: str = "qwen3-rerank",
    top_n: int = 5,
    score_threshold: float = 0.3,
    instruct_mode: InstructMode = InstructMode.QA,
    custom_instruct: str = None,
    auto_instruct: bool = True,
    dedup_threshold: float = 0.80,
) -> EnhancedDashScopeReranker:
    """工厂函数：创建增强版 Reranker。

    Args:
        model: 模型名称
        top_n: 返回数量
        score_threshold: 分数阈值（重要！过滤低质量文档）
        instruct_mode: 预设 instruct 模式
        custom_instruct: 自定义 instruct（最高优先级）
        auto_instruct: 是否根据查询自动选择 instruct
        dedup_threshold: 去重相似度阈值（默认0.80，越小越严格）
    """
    config = RerankConfig(
        model=model,
        top_n=top_n,
        score_threshold=score_threshold,
        instruct_mode=instruct_mode,
        custom_instruct=custom_instruct,
        auto_instruct=auto_instruct,
        use_instruct=True,
        dedup_threshold=dedup_threshold,
    )
    return EnhancedDashScopeReranker(config)


# ============== 测试 ==============
if __name__ == "__main__":
    # 模拟测试：混合质量的文档
    test_docs = [
        Document(page_content="TCP协议是一种面向连接的、可靠的、基于字节流的传输层通信协议。"),  # 高度相关
        Document(page_content="TCP三次握手是建立TCP连接的过程，包括SYN、SYN-ACK、ACK三个步骤。"),  # 最相关
        Document(page_content="HTTP协议是基于TCP的应用层协议，用于Web数据传输。"),  # 中等相关
        Document(page_content="UDP协议是无连接的，不保证数据可靠传输。"),  # 弱相关
        Document(page_content="Python是一种高级编程语言，支持多种编程范式。"),  # 无关
        Document(page_content="机器学习是人工智能的一个分支，专注于数据学习。"),  # 无关
    ]

    print("=" * 70)
    print("测试：Instruct 参数设计")
    print("=" * 70)

    # 展示可用的 instruct 模式
    print("\n【可用的 Instruct 模式】")
    for mode, template in INSTRUCT_TEMPLATES.items():
        print(f"  {mode.value}: {template[:60]}...")

    # 测试自动选择
    print("\n【自动选择 Instruct】")
    test_queries = [
        "什么是TCP三次握手？",
        "Python requests库怎么用？",
        "TCP和UDP有什么区别？",
        "2025年中国GDP多少？",
    ]
    for query in test_queries:
        selected = auto_select_instruct(query)
        print(f"  查询: {query[:25]}...")
        print(f"  → 选择: {selected[:50]}...")
        print()

    # 对比：普通 vs 增强
    from rag.reranker import get_reranker
    
    print("\n" + "=" * 70)
    print("【Rerank 效果对比】")
    print("=" * 70)
    
    print("\n1. 普通 Rerank（无语义引导）")
    normal = get_reranker(enabled=True, model="qwen3-rerank", top_n=4)
    result1 = normal.rerank("什么是TCP三次握手？", test_docs)
    for i, doc in enumerate(result1, 1):
        score = doc.metadata.get("rerank_score", 0.0)
        print(f"  {i}. [{score:.4f}] {doc.page_content[:40]}...")

    print("\n2. 增强版 - Q&A模式（回答问题导向）")
    enhanced_qa = get_enhanced_reranker(
        model="qwen3-rerank",
        top_n=4,
        score_threshold=0.0,
        instruct_mode=InstructMode.QA,
        auto_instruct=False,
    )
    result2 = enhanced_qa.rerank("什么是TCP三次握手？", test_docs)
    for i, doc in enumerate(result2, 1):
        r_score = doc.metadata.get("rerank_score", 0.0)
        print(f"  {i}. [R:{r_score:.4f}] {doc.page_content[:40]}...")

    print("\n3. 增强版 - 自动模式（根据查询自动选择）")
    enhanced_auto = get_enhanced_reranker(
        model="qwen3-rerank",
        top_n=4,
        score_threshold=0.0,
        auto_instruct=True,
    )
    result3 = enhanced_auto.rerank("什么是TCP三次握手？", test_docs)
    for i, doc in enumerate(result3, 1):
        r_score = doc.metadata.get("rerank_score", 0.0)
        print(f"  {i}. [R:{r_score:.4f}] {doc.page_content[:40]}...")

    print("\n4. 增强版 - 带阈值过滤（阈值=0.4）")
    enhanced_filter = get_enhanced_reranker(
        model="qwen3-rerank",
        top_n=4,
        score_threshold=0.4,
        auto_instruct=True,
    )
    result4 = enhanced_filter.rerank("什么是TCP三次握手？", test_docs)
    print(f"  过滤后保留: {len(result4)}/{len(test_docs)} 个文档")
    for i, doc in enumerate(result4, 1):
        c_score = doc.metadata.get("combined_score", 0.0)
        print(f"  {i}. [C:{c_score:.4f}] {doc.page_content[:40]}...")

    print("\n" + "=" * 70)
