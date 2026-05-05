"""Parent-child 父块存储与展开。"""
from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path

from langchain_core.documents import Document

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from rag.parent_store import ParentChunkStore, expand_child_hits_to_parents


def test_parent_store_roundtrip():
    td = tempfile.TemporaryDirectory(ignore_cleanup_errors=True)
    try:
        db = os.path.join(td.name, "p.sqlite")
        store = ParentChunkStore(db)
        store.put("pid1", "父全文", {"source": "a.txt", "section": "一"})
        got = store.get("pid1")
        assert got is not None
        text, meta = got
        assert text == "父全文"
        assert meta["section"] == "一"
    finally:
        td.cleanup()


def test_expand_dedupes_parents_preserves_order():
    td = tempfile.TemporaryDirectory(ignore_cleanup_errors=True)
    try:
        store = ParentChunkStore(os.path.join(td.name, "p.sqlite"))
        store.put("p1", "FULL ONE", {"source": "s"})
        store.put("p2", "FULL TWO", {"source": "s"})

        children = [
            Document(page_content="c1", metadata={"parent_id": "p1"}),
            Document(page_content="c1b", metadata={"parent_id": "p1"}),
            Document(page_content="c2", metadata={"parent_id": "p2"}),
        ]
        out = expand_child_hits_to_parents(children, store)
        assert len(out) == 2
        assert out[0].page_content == "FULL ONE"
        assert out[1].page_content == "FULL TWO"
    finally:
        td.cleanup()


def test_expand_no_store_returns_children():
    kids = [Document(page_content="x", metadata={})]
    assert expand_child_hits_to_parents(kids, None) is kids


def test_expand_child_without_parent_id_passthrough():
    td = tempfile.TemporaryDirectory(ignore_cleanup_errors=True)
    try:
        store = ParentChunkStore(os.path.join(td.name, "p.sqlite"))
        kids = [
            Document(page_content="legacy", metadata={"source": "z"}),
        ]
        out = expand_child_hits_to_parents(kids, store)
        assert len(out) == 1
        assert out[0].page_content == "legacy"
    finally:
        td.cleanup()
