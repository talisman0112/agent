"""按标题/章节预分段并写入 metadata（CHUNK_OPTIMIZATION.md 2.3）。"""

from __future__ import annotations

import re

from langchain_core.documents import Document

# Markdown：`#`～`######` 开头行
_MD_HEADING = re.compile(r"^(#{1,6})\s+(.+)$")
# 中文独立行：第…章/节/篇（整段标题不宜过长，避免正文误匹配）
_CN_CHAPTER = re.compile(
    r"^(第[一二三四五六七八九十百千零〇\d]+[章节篇])[：:\-\s]*(\S.*)?$"
)
# 数字 / 中文序号开头的小标题：1. xxx、一、xxx
_NUM_HEADING = re.compile(r"^(\d{1,3})[\.\、．]\s+(\S.+)$")
_CN_ENUM_HEADING = re.compile(r"^([一二三四五六七八九十百千]+)[、．]\s*(\S.+)$")

# 视为「标题行」的最大长度（超长行更可能是正文）
_MAX_CN_CHAPTER_LINE = 100
_MAX_ENUM_LINE = 120


def _classify_heading(stripped: str) -> str | None:
    """若该行是标题，返回写入 metadata 的章节名；否则返回 None。"""
    if not stripped:
        return None

    m = _MD_HEADING.match(stripped)
    if m:
        return m.group(2).strip()

    m = _CN_CHAPTER.match(stripped)
    if m and len(stripped) <= _MAX_CN_CHAPTER_LINE:
        tail = (m.group(2) or "").strip()
        head = m.group(1).strip()
        return f"{head} {tail}".strip() if tail else head

    m = _NUM_HEADING.match(stripped)
    if m and len(stripped) <= _MAX_ENUM_LINE:
        return stripped

    m = _CN_ENUM_HEADING.match(stripped)
    if m and len(stripped) <= _MAX_ENUM_LINE:
        return stripped

    return None


def _split_text_by_headings(text: str) -> list[tuple[str, str]]:
    """按标题切成 (section_title, body)；无标题时返回单段 ('', full_text)。"""
    if not (text or "").strip():
        return []

    lines = text.splitlines()
    current_title = ""
    buf: list[str] = []
    out: list[tuple[str, str]] = []

    def flush() -> None:
        nonlocal buf
        body = "\n".join(buf).strip()
        buf = []
        if body:
            out.append((current_title, body))

    for line in lines:
        stripped = line.strip()
        title = _classify_heading(stripped)
        if title is not None:
            flush()
            current_title = title
        else:
            buf.append(line)
    flush()

    if not out and text.strip():
        return [("", text.strip())]
    return out


def split_documents_by_sections(documents: list[Document]) -> list[Document]:
    """将每个 Document 按标题拆成多段，保留原有 metadata，并增加 `section`（若有标题）。"""
    out: list[Document] = []
    for doc in documents:
        base = dict(doc.metadata) if doc.metadata else {}
        sections = _split_text_by_headings(doc.page_content or "")
        for sec_title, body in sections:
            meta = dict(base)
            if sec_title:
                meta["section"] = sec_title
            out.append(Document(page_content=body, metadata=meta))
    return out


def prepend_section_title_to_chunks(chunks: list[Document]) -> list[Document]:
    """对最终 chunk 在正文前拼接章节标题，增强向量语义（仅当 metadata 含 section）。"""
    result: list[Document] = []
    for d in chunks:
        sec = (d.metadata or {}).get("section") if d.metadata else None
        if not sec:
            result.append(d)
            continue
        prefix = f"【章节】{sec}\n\n"
        result.append(
            Document(
                page_content=prefix + d.page_content,
                metadata=dict(d.metadata),
            )
        )
    return result
