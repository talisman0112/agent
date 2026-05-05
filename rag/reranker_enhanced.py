"""增强版 RAG Rerank 模块 - 解决评分区分度问题"""

import os
from typing import List, Callable
from dataclasses import dataclass

import dashscope
from dashscope import TextReRank
from langchain_core.documents import Document

from utils.log import logger


@dataclass
class RerankConfig:
    """Rerank 配置参数"""
    model: str = "qwen3-rerank"
    top_n: int = 5
    score_threshold: float = 0.25  # 分数阈值，低于此值的文档会被过滤
    instruct: str = "Given a web search query, retrieve relevant passages that answer the query."
    use_instruct: bool = True  # 是否使用指令引导
    min_score_gap: float = 0.1  # 期望的最小分数差距


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
    ) -> List[Document]:
        """增强版 Rerank，支持混合打分和阈值过滤。"""
        if not documents:
            return []

        if not self.api_key:
            logger.warning("DASHSCOPE_API_KEY 未设置，跳过 Rerank")
            return documents[: self.config.top_n]

        texts = [doc.page_content for doc in documents]

        try:
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
                kwargs["instruct"] = self.config.instruct

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
        
        return [doc for doc, _ in sorted_docs[:top_n]]

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
    instruct: str = None,
) -> EnhancedDashScopeReranker:
    """工厂函数：创建增强版 Reranker。

    Args:
        model: 模型名称
        top_n: 返回数量
        score_threshold: 分数阈值（重要！过滤低质量文档）
        instruct: 自定义指令，控制评分偏好
    """
    # 默认使用 Q&A 检索指令，让模型关注"是否回答问题"
    default_instruct = "Given a web search query, retrieve relevant passages that answer the query."
    
    config = RerankConfig(
        model=model,
        top_n=top_n,
        score_threshold=score_threshold,
        instruct=instruct or default_instruct,
        use_instruct=True,
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

    print("=" * 60)
    print("测试：增强版 Rerank 精度")
    print("=" * 60)

    # 对比：普通 vs 增强
    from rag.reranker import get_reranker
    
    print("\n【1. 普通 Rerank】")
    normal = get_reranker(enabled=True, model="qwen3-rerank", top_n=4)
    result1 = normal.rerank("什么是TCP三次握手？", test_docs)
    for i, doc in enumerate(result1, 1):
        score = doc.metadata.get("rerank_score", 0.0)
        print(f"  {i}. [{score:.4f}] {doc.page_content[:40]}...")

    print("\n【2. 增强版 Rerank（带阈值和指令）】")
    enhanced = get_enhanced_reranker(
        model="qwen3-rerank",
        top_n=4,
        score_threshold=0.4,  # 设置阈值过滤低分
    )
    result2 = enhanced.rerank("什么是TCP三次握手？", test_docs)
    for i, doc in enumerate(result2, 1):
        r_score = doc.metadata.get("rerank_score", 0.0)
        c_score = doc.metadata.get("combined_score", 0.0)
        print(f"  {i}. [R:{r_score:.4f}, C:{c_score:.4f}] {doc.page_content[:40]}...")

    print("\n" + "=" * 60)
