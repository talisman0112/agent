"""RAG Rerank 模块 - 基于 DashScope Cohere Rerank API"""

import os
from typing import List

import dashscope
from dashscope import TextReRank
from langchain_core.documents import Document

from utils.log import logger


class DashScopeReranker:
    """基于 DashScope Text ReRank API 的精排器。

    使用方式：
        reranker = DashScopeReranker(model="qwen3-rerank", top_n=5)
        docs = vectorstore.similarity_search(query, k=20)  # 粗排取更多
        reranked = reranker.rerank(query, docs)  # 精排取 Top-5

    可用模型：
        - qwen3-rerank: 推荐，支持100+语言，新加坡区域
        - gte-rerank-v2: 支持50+语言，北京区域
        - qwen3-vl-rerank: 多模态，支持图文视频（北京区域）
    """

    DEFAULT_MODEL = "qwen3-rerank"  # 推荐模型，支持100+语言
    FALLBACK_MODEL = "gte-rerank-v2"  # 备选模型

    def __init__(
        self,
        model: str = None,
        top_n: int = 5,
        return_documents: bool = True,
        api_key: str = None,
    ):
        """初始化 Reranker。

        Args:
            model: 使用的 rerank 模型，默认 cohere-rerank-v3.5
            top_n: 精排后返回的文档数量
            return_documents: 是否返回原始文档内容
            api_key: DashScope API Key，默认从环境变量读取
        """
        self.model = model or self.DEFAULT_MODEL
        self.top_n = top_n
        self.return_documents = return_documents

        # 设置 API Key
        self.api_key = api_key or os.getenv("DASHSCOPE_API_KEY") or os.getenv("TONGYI_API_KEY")
        if self.api_key:
            dashscope.api_key = self.api_key

    def rerank(self, query: str, documents: List[Document]) -> List[Document]:
        """对文档进行重排序。

        Args:
            query: 用户查询
            documents: 待重排的文档列表（通常来自向量检索的粗排结果）

        Returns:
            重排后的文档列表，按相关性降序排列
        """
        if not documents:
            return []

        if not self.api_key:
            logger.warning("DASHSCOPE_API_KEY 未设置，跳过 Rerank，返回原顺序")
            return documents[: self.top_n]

        # 提取纯文本
        texts = [doc.page_content for doc in documents]

        try:
            resp = TextReRank.call(
                model=self.model,
                query=query,
                documents=texts,
                top_n=self.top_n,
                return_documents=self.return_documents,
            )

            if resp.status_code != 200:
                logger.error("Rerank API 错误: %s - %s", resp.status_code, resp.message)
                # 降级：返回原 Top-N
                return documents[: self.top_n]

            # 解析结果
            results = resp.output.results if hasattr(resp.output, "results") else resp.output.get("results", [])

            reranked_docs = []
            for r in results:
                idx = r.get("index", 0)
                score = r.get("relevance_score", 0.0)
                # 取出对应原文档，并附加 rerank 分数到 metadata
                doc = documents[idx].copy()
                doc.metadata["rerank_score"] = score
                reranked_docs.append(doc)

            logger.info("Rerank 完成: %d docs → %d docs", len(documents), len(reranked_docs))
            return reranked_docs

        except Exception as e:
            logger.error("Rerank 调用异常: %s", str(e), exc_info=True)
            # 异常降级
            return documents[: self.top_n]

    def rerank_with_scores(
        self, query: str, documents: List[Document]
    ) -> List[tuple[Document, float]]:
        """返回带分数的文档列表。

        Returns:
            [(Document, score), ...] 按相关性降序
        """
        docs = self.rerank(query, documents)
        return [(doc, doc.metadata.get("rerank_score", 0.0)) for doc in docs]


class NoOpReranker:
    """空实现，用于关闭 Rerank 时的占位。"""

    def __init__(self, top_n: int = 5):
        self.top_n = top_n

    def rerank(self, query: str, documents: List[Document]) -> List[Document]:
        return documents[: self.top_n]


def get_reranker(
    enabled: bool = True,
    model: str = None,
    top_n: int = 5,
) -> DashScopeReranker | NoOpReranker:
    """工厂函数：创建 Reranker 实例。

    Args:
        enabled: 是否启用 Rerank
        model: 模型名称，None 则用默认
        top_n: 精排后返回数量
    """
    if not enabled:
        return NoOpReranker(top_n=top_n)
    return DashScopeReranker(model=model, top_n=top_n)


# ============== 快速测试 ==============
if __name__ == "__main__":
    # 测试代码
    test_docs = [
        Document(page_content="机器学习是人工智能的一个分支，专注于让计算机从数据中学习。"),
        Document(page_content="深度学习是机器学习的一种方法，使用多层神经网络。"),
        Document(page_content="Python 是一种流行的编程语言，广泛用于数据科学。"),
        Document(page_content="大语言模型如 GPT-4 可以理解和生成自然语言。"),
        Document(page_content="强化学习通过试错来学习策略，常用于游戏和机器人控制。"),
    ]

    # 使用 qwen3-rerank 模型（推荐）
    reranker = get_reranker(enabled=True, model="qwen3-rerank", top_n=3)
    result = reranker.rerank("什么是深度学习？", test_docs)

    print("Rerank 结果:")
    for i, doc in enumerate(result, 1):
        score = doc.metadata.get("rerank_score", 0.0)
        print(f"{i}. [{score:.4f}] {doc.page_content[:50]}...")
