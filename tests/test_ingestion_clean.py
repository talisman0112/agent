"""入库前正文清洗 ingestion_clean。"""
from __future__ import annotations

import sys
from pathlib import Path

from langchain_core.documents import Document

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from rag.ingestion_clean import (
    clean_documents,
    clean_page_content,
    resolve_ingestion_clean_config,
)
from utils.config_hander import chroma_config  # noqa: E402


def _base_cfg(**overrides):
    merged = resolve_ingestion_clean_config({})
    merged.update(overrides)
    return merged


def test_resolve_merges_chrome_subset():
    cfg = resolve_ingestion_clean_config({"ingestion_clean": {"min_document_chars": 5}})
    assert cfg["min_document_chars"] == 5
    assert cfg.get("unicode_form") == "NFKC"


def test_resolve_uses_project_chrome_yml_ingestion_clean():
    """与仓库内真实 ``config/chrome.yml`` 合并后，清洗仍产生稳定、可预期的结果。"""
    cfg = resolve_ingestion_clean_config(chroma_config)
    assert cfg.get("enabled") is True
    assert isinstance(cfg.get("drop_line_patterns"), list)
    sample = "\ufeff\u200b第一行\n\n\n\n第二行（测试）"
    out = clean_page_content(sample, cfg)
    assert "第一行" in out and "第二行" in out
    assert "\ufeff" not in out and "\u200b" not in out
    assert "(" in out  # NFKC：全角括号已规范化


def test_phase_a_unicode_bom_zw_controls():
    text = "\ufeff\u200ba\x01bc（note）"
    cfg = _base_cfg()
    out = clean_page_content(text, cfg)
    assert out.startswith("abc(")
    assert "\x01" not in out
    assert "\u200b" not in out
    assert "\ufeff" not in out


def test_phase_a_blank_line_compression_and_strip_lines():
    raw = "line one  \r\n\r\n\r\n\r\n\r\n\r\n\nline two"
    cfg = _base_cfg(max_consecutive_blank_lines=2, strip_lines=True)
    out = clean_page_content(raw, cfg)
    assert "line one" in out
    assert "line two" in out
    blanks = sum(1 for blk in out.split("\n") if blk == "")
    assert blanks <= 2


def test_drop_line_patterns_fullmatch():
    raw = "正文一段\n\n第 3 页\n后续"
    cfg = _base_cfg(drop_line_patterns=[r"^第\s*\d+\s*页$"])
    out = clean_page_content(raw, cfg)
    assert "正文" in out
    assert "后续" in out
    assert "第 3 页" not in out


def test_merge_soft_hyphens():
    raw = "We study multi-\nfactor models in equity markets."
    cfg = _base_cfg(merge_soft_hyphens=True)
    out = clean_page_content(raw, cfg)
    assert "multi-\n" not in out
    assert "multifactor" in out.replace(" ", "").lower()


def test_collapse_duplicate_short_lines():
    raw = "header\nfooter\nfooter\nfooter\nreal content line"
    cfg = _base_cfg(collapse_duplicate_lines=True, collapse_duplicate_max_line_len=80, collapse_duplicate_min_repeats=2)
    out = clean_page_content(raw, cfg)
    assert out.count("footer") == 1
    assert "real content line" in out


def test_clean_documents_drops_short():
    docs = [
        Document(page_content="short", metadata={"source": "x"}),
        Document(page_content="x" * 30, metadata={"source": "y"}),
    ]
    cfg = _base_cfg(min_document_chars=20, drop_empty_documents=True)
    out = clean_documents(docs, cfg, source_hint="test")
    assert len(out) == 1
    assert len(out[0].page_content) >= 20
    assert out[0].metadata.get("ingestion_cleaned") is True


def test_clean_documents_disabled_passthrough():
    docs = [Document(page_content="hi", metadata={})]
    cfg = _base_cfg(enabled=False)
    out = clean_documents(docs, cfg)
    assert len(out) == 1
    assert out[0].page_content == "hi"
    assert "ingestion_cleaned" not in out[0].metadata
