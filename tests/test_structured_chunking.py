"""结构化分块（2.3）单测。"""

from __future__ import annotations

import sys
from pathlib import Path

from langchain_core.documents import Document

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from rag.structured_chunking import (
    _classify_heading,
    _split_text_by_headings,
    prepend_section_title_to_chunks,
    split_documents_by_sections,
)


def test_classify_markdown_heading():
    assert _classify_heading("# 标题 A") == "标题 A"
    assert _classify_heading("## 子标题") == "子标题"
    assert _classify_heading("正文一段话") is None


def test_classify_cn_chapter_line():
    assert _classify_heading("第一章 绪论") == "第一章 绪论"
    assert _classify_heading("第三节：概述") == "第三节 概述"


def test_split_text_preamble_then_sections():
    text = """前言段落。

## 第一节

内容一。

## 第二节

内容二。
"""
    parts = _split_text_by_headings(text)
    assert len(parts) == 3
    assert parts[0] == ("", "前言段落。")
    assert parts[1][0] == "第一节"
    assert "内容一" in parts[1][1]
    assert parts[2][0] == "第二节"


def test_split_documents_by_sections_metadata():
    docs = [
        Document(
            page_content="# A\n\nIntro.\n\n## B\n\nMore.",
            metadata={"source": "x.txt"},
        )
    ]
    out = split_documents_by_sections(docs)
    assert len(out) == 2
    assert out[0].metadata["source"] == "x.txt"
    assert out[0].metadata["section"] == "A"
    assert "Intro" in out[0].page_content
    assert out[1].metadata["section"] == "B"


def test_prepend_section_title():
    chunks = [
        Document(page_content="body", metadata={"section": "S1", "source": "f.txt"}),
        Document(page_content="no sec", metadata={"source": "f.txt"}),
    ]
    out = prepend_section_title_to_chunks(chunks)
    assert out[0].page_content.startswith("【章节】S1\n\nbody")
    assert out[1].page_content == "no sec"
