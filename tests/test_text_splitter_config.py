"""chrome 分块 separators 解析（2.1 / vector_store）。"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from rag.vector_store import _DEFAULT_TEXT_SPLIT_SEPARATORS, _resolve_text_splitter_separators


def test_resolve_separators_prefers_list():
    cfg = {"separators": ["\n\n", "。"]}
    assert _resolve_text_splitter_separators(cfg) == ["\n\n", "。"]


def test_resolve_separators_legacy_separator():
    cfg = {"separator": "\t"}
    assert _resolve_text_splitter_separators(cfg) == ["\t"]


def test_resolve_separators_empty_list_falls_back_to_default():
    cfg = {"separators": []}
    assert _resolve_text_splitter_separators(cfg) == list(_DEFAULT_TEXT_SPLIT_SEPARATORS)


def test_resolve_separators_none_in_list_becomes_empty_string():
    cfg = {"separators": ["\n\n", None]}
    assert _resolve_text_splitter_separators(cfg) == ["\n\n", ""]
