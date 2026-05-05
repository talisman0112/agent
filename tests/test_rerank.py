"""Rerank 效果对比测试"""

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from rag.ragsummarize import RAGSummarize
from rag.vector_store import VectorStoreService


def test_rerank_comparison(query: str):
    """对比：有/无 Rerank 的检索结果"""
    print(f"\n{'='*60}")
    print(f"查询: {query}")
    print(f"{'='*60}")

    # 1. 无 Rerank（直接取向量检索 Top-5）
    print("\n【1. 无 Rerank - 向量检索 Top-5】")
    rag_no_rerank = RAGSummarize(rerank_enabled=False, rerank_top_n=5)
    rag_no_rerank.vector_k = 5  # 向量只取 5 个
    rag_no_rerank.rag_retriever = rag_no_rerank.vector_store.chroma.as_retriever(
        search_kwargs={"k": 5}
    )
    rag_no_rerank.chain = rag_no_rerank._create_chain()

    docs_no_rerank = rag_no_rerank.retrieve_docs(query)
    for i, doc in enumerate(docs_no_rerank, 1):
        print(f"  {i}. {doc.page_content[:100]}...")

    # 2. 有 Rerank（向量召回 20 → Rerank 取 5）
    print("\n【2. 有 Rerank - 向量召回 20 → 精排取 5】")
    rag_with_rerank = RAGSummarize(rerank_enabled=True, vector_search_k=20, rerank_top_n=5)
    docs_with_rerank = rag_with_rerank.retrieve_docs_with_scores(query)
    for i, (doc, score) in enumerate(docs_with_rerank, 1):
        print(f"  {i}. [得分: {score:.4f}] {doc.page_content[:100]}...")

    print("\n" + "="*60)


def test_rerank_generation(query: str):
    """测试完整生成链路"""
    print(f"\n{'='*60}")
    print(f"完整 RAG 生成测试")
    print(f"{'='*60}")

    rag = RAGSummarize(rerank_enabled=True, vector_search_k=20, rerank_top_n=5)
    answer = rag.summarize(query)

    print(f"\n【回答】\n{answer}")
    print("="*60)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Rerank 对比测试")
    parser.add_argument("--query", "-q", type=str, help="测试查询语句")
    parser.add_argument("--test", "-t", choices=["compare", "generate", "all"], default="compare")
    args = parser.parse_args()

    # 默认测试查询
    test_query = args.query or "什么是深度学习，它和机器学习有什么关系？"

    if args.test in ("compare", "all"):
        test_rerank_comparison(test_query)

    if args.test in ("generate", "all"):
        test_rerank_generation(test_query)
