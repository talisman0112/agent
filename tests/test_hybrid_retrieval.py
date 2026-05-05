"""HybridRAG 多路召回测试（Web + 本地 + Rerank）。"""

from __future__ import annotations

import sys
import types
from pathlib import Path

from langchain_core.documents import Document

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from rag.ragsummarize import HybridRAG


class _DummyRetriever:
    """模拟本地向量检索器。"""

    def __init__(self, docs: list[Document]):
        self._docs = docs

    def invoke(self, query: str) -> list[Document]:
        _ = query
        return self._docs


class _DummyReranker:
    """模拟 Reranker，记录输入并返回可预测排序。"""

    def __init__(self):
        self.last_query = ""
        self.last_input_docs: list[Document] = []

    def rerank(self, query: str, docs: list[Document]) -> list[Document]:
        self.last_query = query
        self.last_input_docs = list(docs)

        # 模拟“Web 结果更相关”的排序结果，便于验证 rerank 生效
        ranked = sorted(
            docs,
            key=lambda d: 1 if d.metadata.get("source_channel") == "web" else 0,
            reverse=True,
        )
        for i, doc in enumerate(ranked, 1):
            doc.metadata["rerank_score"] = 1.0 - i * 0.1
        return ranked[:3]


def test_parse_web_results_to_documents():
    hybrid = HybridRAG.__new__(HybridRAG)
    raw = (
        "DuckDuckGo 搜索结果（2条）：\n\n"
        "1. RAG 基础\n"
        "   链接: https://example.com/rag\n"
        "   摘要: 介绍 RAG 的核心流程。\n\n"
        "2. 向量检索实践\n"
        "   链接: https://example.com/vector\n"
        "   摘要: 讲解向量库与召回策略。\n"
    )

    docs = hybrid._parse_web_results(raw)

    assert len(docs) == 2
    assert docs[0].metadata["source_channel"] == "web"
    assert docs[0].metadata["web_rank"] == 1
    assert "标题: RAG 基础" in docs[0].page_content
    assert docs[1].metadata["source"] == "https://example.com/vector"


def test_multi_retrieve_merges_web_and_local(monkeypatch):
    hybrid = HybridRAG.__new__(HybridRAG)
    hybrid.web_max_results = 2
    hybrid.rag_retriever = _DummyRetriever(
        [
            Document(page_content="本地文档A", metadata={"source": "local_a.txt"}),
            Document(page_content="本地文档B", metadata={"source": "local_b.txt"}),
        ]
    )

    fake_module = types.ModuleType("tools.agent_tool")
    fake_module.web_search = lambda query, max_results=5: (
        "DuckDuckGo 搜索结果（2条）：\n\n"
        "1. Web 文档A\n"
        "   链接: https://web-a.example\n"
        "   摘要: 与查询高度相关。\n\n"
        "2. Web 文档B\n"
        "   链接: https://web-b.example\n"
        "   摘要: 相关性一般。\n"
    )
    monkeypatch.setitem(sys.modules, "tools.agent_tool", fake_module)

    docs = hybrid._multi_retrieve("什么是多路召回")

    assert len(docs) == 4
    assert sum(d.metadata.get("source_channel") == "local" for d in docs) == 2
    assert sum(d.metadata.get("source_channel") == "web" for d in docs) == 2


def test_rerank_docs_uses_combined_candidates(monkeypatch):
    hybrid = HybridRAG.__new__(HybridRAG)
    hybrid.web_max_results = 2
    hybrid.rag_retriever = _DummyRetriever(
        [
            Document(page_content="本地: 旧版本资料", metadata={"source": "local_old.txt"}),
            Document(page_content="本地: 通用知识", metadata={"source": "local_common.txt"}),
        ]
    )
    hybrid.reranker = _DummyReranker()

    fake_module = types.ModuleType("tools.agent_tool")
    fake_module.web_search = lambda query, max_results=5: (
        "DuckDuckGo 搜索结果（2条）：\n\n"
        "1. Web: 最新发布说明\n"
        "   链接: https://web-new.example\n"
        "   摘要: 包含最新版本信息。\n\n"
        "2. Web: 新闻解读\n"
        "   链接: https://web-news.example\n"
        "   摘要: 对新特性进行分析。\n"
    )
    monkeypatch.setitem(sys.modules, "tools.agent_tool", fake_module)

    reranked = hybrid._rerank_docs("这个功能最近有什么更新")

    # 断言：Reranker 输入了“多路召回的所有候选”
    assert len(hybrid.reranker.last_input_docs) == 4
    assert any(d.metadata.get("source_channel") == "local" for d in hybrid.reranker.last_input_docs)
    assert any(d.metadata.get("source_channel") == "web" for d in hybrid.reranker.last_input_docs)

    # 断言：rerank 输出顺序生效（Dummy 逻辑会把 web 放前面）
    assert len(reranked) == 3
    assert reranked[0].metadata.get("source_channel") == "web"
    assert reranked[0].metadata.get("rerank_score", 0) > reranked[-1].metadata.get("rerank_score", 0)
