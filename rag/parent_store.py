"""父块全文存储（2.4 parent-child）：子块走向量库，父块仅存 SQLite 供展开。"""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path


class ParentChunkStore:
    """按 parent_id 存 page_content + metadata，供向量命中子块后换回完整父文本。"""

    def __init__(self, db_path: str) -> None:
        self.db_path = db_path
        parent = Path(db_path).parent
        if str(parent):
            parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _init_db(self) -> None:
        conn = sqlite3.connect(self.db_path)
        try:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS parents (
                    parent_id TEXT PRIMARY KEY,
                    page_content TEXT NOT NULL,
                    metadata_json TEXT NOT NULL
                )
                """
            )
            conn.commit()
        finally:
            conn.close()

    def put(self, parent_id: str, page_content: str, metadata: dict) -> None:
        payload = json.dumps(metadata, ensure_ascii=False, default=str)
        conn = sqlite3.connect(self.db_path)
        try:
            conn.execute(
                "INSERT OR REPLACE INTO parents (parent_id, page_content, metadata_json) VALUES (?, ?, ?)",
                (parent_id, page_content, payload),
            )
            conn.commit()
        finally:
            conn.close()

    def get(self, parent_id: str) -> tuple[str, dict] | None:
        conn = sqlite3.connect(self.db_path)
        try:
            row = conn.execute(
                "SELECT page_content, metadata_json FROM parents WHERE parent_id = ?",
                (parent_id,),
            ).fetchone()
        finally:
            conn.close()
        if not row:
            return None
        text, meta_raw = row
        try:
            meta = json.loads(meta_raw) if meta_raw else {}
        except json.JSONDecodeError:
            meta = {}
        if not isinstance(meta, dict):
            meta = {}
        return text, meta


def expand_child_hits_to_parents(
    child_docs: list,
    store: ParentChunkStore | None,
    *,
    fallback_to_child: bool = True,
):
    """按子块检索顺序展开为父块列表；同一 parent_id 只保留首次出现。"""
    if store is None:
        return child_docs

    from langchain_core.documents import Document

    out: list[Document] = []
    seen: set[str] = set()

    for d in child_docs:
        meta = d.metadata or {}
        pid = meta.get("parent_id")
        if not pid:
            out.append(d)
            continue
        if pid in seen:
            continue
        seen.add(pid)
        got = store.get(pid)
        if got is None:
            if fallback_to_child:
                out.append(d)
            continue
        text, pmeta = got
        out.append(Document(page_content=text, metadata=pmeta))

    return out

