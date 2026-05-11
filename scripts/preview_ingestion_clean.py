"""预览单文件在 ``ingestion_clean`` 规则下的清洗效果（不改向量库、不写 md5 ledger）。

使用：
    python scripts/preview_ingestion_clean.py path/to/file.pdf
    python scripts/preview_ingestion_clean.py path/to/file.md --head 400

与 ``rag/vector_store.py`` 使用相同的加载器与 ``config/chrome.yml`` 中的 ``ingestion_clean`` 配置。
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_THIS = Path(__file__).resolve()
_PROJECT_ROOT = _THIS.parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from langchain_core.documents import Document  # noqa: E402

from rag.ingestion_clean import clean_documents, clean_page_content, resolve_ingestion_clean_config  # noqa: E402
from utils.config_hander import chroma_config  # noqa: E402
from utils.file_hander import (  # noqa: E402
    doc_loader,
    docx_loader,
    pdf_loader,
    ppt_loader,
    pptx_loader,
    txt_loader,
    xls_loader,
    xlsx_loader,
)
from utils.log import logger  # noqa: E402


def _documents_from_file(file_path: str) -> list[Document]:
    if file_path.endswith((".txt", ".md")):
        return txt_loader(file_path).load()
    if file_path.endswith(".pdf"):
        return pdf_loader(file_path).load()
    if file_path.endswith(".docx"):
        return docx_loader(file_path).load()
    if file_path.endswith(".doc"):
        return doc_loader(file_path).load()
    if file_path.endswith(".xlsx"):
        return xlsx_loader(file_path).load()
    if file_path.endswith(".xls"):
        return xls_loader(file_path).load()
    if file_path.endswith(".ppt"):
        return ppt_loader(file_path).load()
    if file_path.endswith(".pptx"):
        return pptx_loader(file_path).load()
    logger.error("Unsupported file type: %s", file_path)
    return []


def _snippet(text: str, head: int) -> str:
    t = text.replace("\n", "\\n")
    if len(t) <= head * 2:
        return t
    return t[:head] + "\n... (" + str(len(text)) + " chars total) ...\n" + t[-head:]


def main() -> None:
    parser = argparse.ArgumentParser(description="Preview RAG ingestion_clean on one file.")
    parser.add_argument("file", help="Path to a knowledge file (pdf, md, txt, office, ...)")
    parser.add_argument("--head", type=int, default=320, help="Chars to show from start/end of each doc (escaped newlines)")
    args = parser.parse_args()
    path = os_path = str(Path(args.file).resolve())
    cfg = resolve_ingestion_clean_config(chroma_config)

    docs = _documents_from_file(os_path)
    if not docs:
        print(f"[error] No documents loaded: {path}")
        sys.exit(1)

    print("ingestion_clean config (merged defaults):")
    for k, v in sorted(cfg.items()):
        print(f"  {k}: {v!r}")
    print()

    total_in = sum(len(d.page_content or "") for d in docs)
    cleaned = clean_documents(docs, cfg, source_hint=path)
    total_out = sum(len(d.page_content or "") for d in cleaned)

    print(f"documents: {len(docs)} -> {len(cleaned)}  chars: {total_in} -> {total_out}")
    for i, d in enumerate(cleaned):
        print(f"\n--- cleaned doc {i + 1} metadata keys: {list((d.metadata or {}).keys())} ---")
        print(_snippet(d.page_content or "", args.head))

    if not cleaned and docs:
        print("\n(note) All documents were dropped (e.g. shorter than min_document_chars).")
        print("Raw first doc snippet (before per-doc drop):")
        for i, d in enumerate(docs):
            raw = d.page_content or ""
            c = clean_page_content(raw, cfg)
            print(f"  doc {i + 1}: cleaned len={len(c)}  snippet: {_snippet(c, args.head)}")


if __name__ == "__main__":
    main()
