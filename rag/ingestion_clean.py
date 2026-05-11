"""RAG 入库前正文清洗（Loader 之后、结构化分块之前）。"""

from __future__ import annotations

import re
import unicodedata

from langchain_core.documents import Document

from utils.log import logger

_DEFAULT_INGESTION_CLEAN: dict = {
    "enabled": True,
    "unicode_form": "NFKC",
    "max_consecutive_blank_lines": 2,
    "strip_lines": True,
    "drop_empty_documents": True,
    "min_document_chars": 20,
    "drop_line_patterns": [],
    "collapse_duplicate_lines": False,
    "collapse_duplicate_max_line_len": 80,
    "collapse_duplicate_min_repeats": 2,
    "merge_soft_hyphens": False,
}

# Zero-width / bidi formatting that often survives copy-paste and PDF extraction
_ZW_PATTERN = re.compile(
    "[\u200b\u200c\u200d\u2060\uFEFF]"
)

# ASCII controls except tab and newline
_CTRL_PATTERN = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")


def resolve_ingestion_clean_config(chroma_config: dict) -> dict:
    """合并 chrome.yml 中的 ``ingestion_clean`` 与内置默认值。"""
    user = chroma_config.get("ingestion_clean")
    if not isinstance(user, dict):
        user = {}
    merged = dict(_DEFAULT_INGESTION_CLEAN)
    merged.update(user)
    return merged


def _strip_bom(text: str) -> str:
    if text.startswith("\ufeff"):
        return text[1:]
    return text


def _normalize_unicode(text: str, form: str) -> str:
    if form not in ("NFC", "NFD", "NFKC", "NFKD"):
        form = "NFKC"
    return unicodedata.normalize(form, text)


def _merge_soft_hyphens(text: str) -> str:
    """行尾连字符 + 换行 + 小写ASCII起头：合并为连续词（常见英文 PDF 断词）。"""
    return re.sub(r"-\r?\n\s*([a-z])", r"\1", text)


def _drop_lines_matching(text: str, patterns: list[re.Pattern[str]]) -> str:
    """按正则整行剔除（与 strip 后的行做 ``fullmatch``，便于书写 ``^...$``）。"""
    if not patterns:
        return text
    lines = text.split("\n")
    kept: list[str] = []
    for line in lines:
        cand = line.strip()
        if cand and any(p.fullmatch(cand) for p in patterns):
            continue
        kept.append(line)
    return "\n".join(kept)


def _compile_drop_patterns(raw: list[str]) -> list[re.Pattern[str]]:
    out: list[re.Pattern[str]] = []
    for s in raw:
        if not s or not isinstance(s, str):
            continue
        try:
            out.append(re.compile(s))
        except re.error as e:
            logger.warning("ingestion_clean: invalid drop_line_patterns regex %r: %s", s, e)
    return out


def _collapse_duplicate_consecutive_lines(
    lines: list[str],
    *,
    max_line_len: int,
    min_repeats: int,
) -> list[str]:
    if min_repeats < 2:
        min_repeats = 2
    out: list[str] = []
    i = 0
    n = len(lines)
    while i < n:
        line = lines[i]
        stripped = line.strip()
        if not stripped or len(line) > max_line_len:
            out.append(line)
            i += 1
            continue
        j = i
        while j < n and lines[j] == line:
            j += 1
        count = j - i
        if count >= min_repeats:
            out.append(line)
        else:
            out.extend(lines[i:j])
        i = j
    return out


def _compress_blank_lines(lines: list[str], max_blank: int) -> list[str]:
    """连续空行最多保留 ``max_blank`` 条空行记录。"""
    if max_blank < 0:
        max_blank = 0
    out: list[str] = []
    i = 0
    n = len(lines)
    while i < n:
        line = lines[i]
        if line.strip():
            out.append(line)
            i += 1
            continue
        j = i
        while j < n and not lines[j].strip():
            j += 1
        run = j - i
        take = min(run, max_blank)
        out.extend([""] * take)
        i = j
    return out


def clean_page_content(text: str, cfg: dict) -> str:
    """对单段正文做清洗；``cfg`` 为 ``resolve_ingestion_clean_config`` 的合并结果。"""
    if not text:
        return ""

    t = _strip_bom(text)
    t = _ZW_PATTERN.sub("", t)

    form = str(cfg.get("unicode_form") or "NFKC")
    t = _normalize_unicode(t, form)
    t = _CTRL_PATTERN.sub("", t)

    t = t.replace("\r\n", "\n").replace("\r", "\n")

    if cfg.get("merge_soft_hyphens"):
        t = _merge_soft_hyphens(t)

    patterns = _compile_drop_patterns(list(cfg.get("drop_line_patterns") or []))
    t = _drop_lines_matching(t, patterns)

    lines = t.split("\n")
    if cfg.get("strip_lines"):
        lines = [ln.rstrip() for ln in lines]

    if cfg.get("collapse_duplicate_lines"):
        lines = _collapse_duplicate_consecutive_lines(
            lines,
            max_line_len=int(cfg.get("collapse_duplicate_max_line_len") or 80),
            min_repeats=int(cfg.get("collapse_duplicate_min_repeats") or 2),
        )

    max_blank = int(cfg.get("max_consecutive_blank_lines") or 2)
    lines = _compress_blank_lines(lines, max_blank)

    t = "\n".join(lines)
    return t.strip()


def clean_documents(
    docs: list[Document],
    cfg: dict,
    *,
    source_hint: str = "",
) -> list[Document]:
    """逐条清洗 ``page_content``；必要时丢弃过短文档。metadata 浅拷贝并标记 ``ingestion_cleaned``。"""
    if not cfg.get("enabled", True):
        return list(docs)

    out: list[Document] = []
    dropped = 0
    total_before = 0
    total_after = 0
    min_c = int(cfg.get("min_document_chars") or 20)
    drop_empty = bool(cfg.get("drop_empty_documents", True))

    for doc in docs:
        raw = doc.page_content or ""
        total_before += len(raw)
        cleaned = clean_page_content(raw, cfg)
        total_after += len(cleaned)

        if drop_empty and len(cleaned) < min_c:
            dropped += 1
            logger.warning(
                "ingestion_clean: dropped short document (%d chars < min_document_chars=%d) %s",
                len(cleaned),
                min_c,
                source_hint or "",
            )
            continue

        meta = dict(doc.metadata) if doc.metadata else {}
        meta["ingestion_cleaned"] = True
        out.append(Document(page_content=cleaned, metadata=meta))

    logger.info(
        "ingestion_clean: %s docs_in=%d docs_out=%d dropped=%d chars %d->%d",
        source_hint or "(no path)",
        len(docs),
        len(out),
        dropped,
        total_before,
        total_after,
    )
    return out
