"""query_expand 单元测试（Mock LLM / Retriever，无需外网）。"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import pytest
from langchain_core.documents import Document

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from rag import query_expand as qe


@pytest.fixture(autouse=True)
def _clear_query_expand_ui():
    qe.clear_query_expand_ui_records()
    yield
    qe.clear_query_expand_ui_records()


class _FakeLLM:
    def __init__(self, texts: list[str]) -> None:
        self._texts = list(texts)
        self.calls = 0

    def invoke(self, messages: Any) -> Any:
        self.calls += 1
        payload = self._texts.pop(0) if self._texts else "[]"

        class R:
            content = payload

        return R()


class _FakeRetriever:
    def __init__(self) -> None:
        self.queries: list[str] = []

    def invoke(self, q: str) -> list[Document]:
        self.queries.append(q)
        pid = str(len(self.queries))
        return [
            Document(
                page_content=f"hit-{q}-{pid}",
                metadata={"parent_id": f"p{pid}"},
            )
        ]


def test_parse_json_string_list_from_fence():
    raw = '\nHere\n```json\n["a", "b"]\n```\n'
    assert qe.parse_json_string_list(raw) == ["a", "b"]


def test_parse_json_string_list_invalid_returns_empty():
    assert qe.parse_json_string_list("not json") == []


def test_dedupe_documents_by_parent():
    docs = [
        Document(page_content="x", metadata={"parent_id": "1"}),
        Document(page_content="y", metadata={"parent_id": "1"}),
        Document(page_content="z", metadata={"parent_id": "2"}),
    ]
    deduped = qe.dedupe_documents_preserve_order(docs)
    assert len(deduped) == 2
    assert deduped[0].metadata["parent_id"] == "1"


def test_coarse_retrieve_union_dedupe_and_cap():
    r = _FakeRetriever()
    out = qe.coarse_retrieve_union(
        r,
        ["q1", "q1", "q2"],
        max_coarse_docs=1,
        max_workers=4,
    )
    assert len(out) == 1
    assert len(r.queries) == 2


def test_build_search_queries_expansion(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(qe.logger, "info", lambda *a, **k: None)

    variant_json = '["宁德时代 储能", "300750 CATL battery"]'

    qs = qe.build_search_queries(
        retrieval_input="宁德时代业务",
        cfg={
            "query_expansion_enabled": True,
            "query_expansion_variants": 3,
            "query_expansion_include_original": True,
            "query_decompose_enabled": False,
        },
        llm=_FakeLLM([variant_json]),
    )
    assert "宁德时代业务" in qs
    assert "宁德时代 储能" in qs


def test_build_search_queries_fallback_no_llm():
    qs = qe.build_search_queries(
        retrieval_input="仅本地",
        cfg={"query_expansion_enabled": True},
        llm=None,
    )
    assert qs == ["仅本地"]


def test_build_search_queries_decompose_path(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(qe.logger, "info", lambda *a, **k: None)

    decompose_json = '["子问题A", "子问题B"]'
    llm = _FakeLLM([decompose_json])
    q = "对比 A 与 B 的优缺点"
    assert qe.should_decompose_for_depth(q)

    qs = qe.build_search_queries(
        retrieval_input=q,
        cfg={
            "query_expansion_enabled": False,
            "query_decompose_enabled": True,
            "query_decompose_max_subqueries": 4,
            "query_decompose_with_expansion": False,
        },
        llm=llm,
    )
    assert q in qs
    assert "子问题A" in qs
    assert "子问题B" in qs


def test_ui_force_off_overrides_config(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(qe.logger, "info", lambda *a, **k: None)
    tok = qe.push_ui_query_expand_force_off()
    try:
        qs = qe.build_search_queries(
            retrieval_input="宁德时代 产能",
            cfg={
                "query_expansion_enabled": True,
                "query_expansion_variants": 5,
                "query_expansion_include_original": True,
                "query_decompose_enabled": False,
            },
            llm=_FakeLLM(['["应忽略"]']),
        )
        assert qs == ["宁德时代 产能"]
        recs = qe.take_query_expand_ui_records()
        assert len(recs) == 1
        assert "工作台" in recs[0].remark
    finally:
        qe.reset_ui_query_expand_force_off(tok)
