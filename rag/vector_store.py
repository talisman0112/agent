import os
import sys
import uuid
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
_root_str = str(_ROOT)
if _root_str not in sys.path:
    sys.path.insert(0, _root_str)

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from model.model import embedding_model
from utils.config_hander import chroma_config
from utils.file_hander import (
    doc_loader,
    docx_loader,
    get_file_md5,
    listdir_with_allowed_type,
    pdf_loader,
    ppt_loader,
    pptx_loader,
    txt_loader,
    xlsx_loader,
    xls_loader,
)
from utils.log import logger
from utils.path_pool import get_abs_path

from rag.ingestion_clean import clean_documents, resolve_ingestion_clean_config
from rag.parent_store import ParentChunkStore, expand_child_hits_to_parents
from rag.structured_chunking import (
    prepend_section_title_to_chunks,
    split_documents_by_sections,
)


# 未配置 separators / separator 时的默认回退顺序（与 CHUNK_OPTIMIZATION.md 2.1 一致）
_DEFAULT_TEXT_SPLIT_SEPARATORS = ["\n\n", "\n", "。", "！", "？", "；", "，", " ", ""]


# 在 data/ 下属于"项目元数据"而非"知识语料"的常见文件名，入库时统一跳过。
# 大小写不敏感；路径末段精确匹配（如 ``data/README.md``、``data/glossary/README.md``）。
_NON_CORPUS_BASENAMES = {
    "readme.md",
    "readme.txt",
    "changelog.md",
    "license.md",
    "license.txt",
    "todo.md",
    ".gitkeep",
}


def _is_non_corpus_file(file_path: str) -> bool:
    return os.path.basename(file_path).lower() in _NON_CORPUS_BASENAMES


def _resolve_text_splitter_separators(cfg: dict) -> list[str]:
    """从 chrome 配置解析 RecursiveCharacterTextSplitter 的 separators 列表。"""
    raw_list = cfg.get("separators")
    if isinstance(raw_list, list) and raw_list:
        return ["" if s is None else str(s) for s in raw_list]
    legacy = cfg.get("separator")
    if isinstance(legacy, str) and legacy:
        return [legacy]
    return list(_DEFAULT_TEXT_SPLIT_SEPARATORS)


def _parent_document_for_store(doc: Document, prepend_section: bool) -> str:
    """写入父块库的完整正文（与结构化章节前缀策略一致）。"""
    body = doc.page_content or ""
    sec = (doc.metadata or {}).get("section")
    if prepend_section and sec:
        return f"【章节】{sec}\n\n{body}"
    return body


_EMBED_DISABLED_MSG = (
    "embedding 未就绪：请先 pip install dashscope langchain-community，并设置环境变量 DASHSCOPE_API_KEY。\n"
    "若不传 embedding_function，Chroma 会改用本机 ONNX（all-MiniLM-L6-v2），从外网下载大文件，国内常见超时。"
)


def _documents_from_file(file_path: str) -> list[Document]:
    # .md 与 .txt 走同一文本加载器；结构化分块（``rag/structured_chunking.py``）
    # 已能识别 Markdown ``#``～``######`` 标题和中文章节标题，无需特殊 markdown loader。
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


class VectorStoreService:
    def __init__(self):
        if embedding_model is None:
            raise RuntimeError(_EMBED_DISABLED_MSG)
        # 确保 persist_directory 为绝对路径，避免从不同工作目录启动时路径不一致
        persist_dir = get_abs_path(chroma_config["persist_directory"])
        logger.info("Chroma 持久化路径: %s", persist_dir)
        self.chroma = Chroma(
            collection_name=chroma_config["collection_name"],
            embedding_function=embedding_model,
            persist_directory=persist_dir,
        )
        split_seps = _resolve_text_splitter_separators(chroma_config)
        self.spliter = RecursiveCharacterTextSplitter(
            chunk_size=chroma_config["chunk_size"],
            chunk_overlap=chroma_config["chunk_overlap"],
            separators=split_seps,
        )
        logger.info(
            "Text splitter: chunk_size=%s chunk_overlap=%s separators=%d级",
            chroma_config["chunk_size"],
            chroma_config["chunk_overlap"],
            len(split_seps),
        )
        self.parent_store: ParentChunkStore | None = None
        if chroma_config.get("parent_child_enabled", False):
            sqlite_rel = chroma_config.get("parent_store_sqlite", "db/parent_store.sqlite")
            sqlite_abs = get_abs_path(sqlite_rel)
            self.parent_store = ParentChunkStore(sqlite_abs)
            logger.info("Parent-child：父块 SQLite %s", sqlite_abs)

    def expand_retrieval_to_parents(self, documents: list[Document]) -> list[Document]:
        """将子块向量命中展开为父块（2.4）；未启用或无 store 时原样返回。"""
        if not chroma_config.get("parent_child_enabled", False):
            return documents
        return expand_child_hits_to_parents(documents, self.parent_store)

    def get_retriever(self):
        return self.chroma.as_retriever(search_kwargs={"k": chroma_config["k"]})

    def load_data(self) -> None:
        """读取 `database_path` 下允许后缀的文件，分块并写入 Chroma。

        每条文件在 Loader 之后、结构化分块之前会经 ``ingestion_clean`` 清洗正文
        （见 ``config/chrome.yml`` 中 ``ingestion_clean``，`enabled: false` 则跳过）。
        `md5_path` 记录已成功入库的文件内容 MD5；内容不变的文件下次运行会跳过。
        """
        data_dir = get_abs_path(chroma_config["database_path"])
        md5_path = get_abs_path(chroma_config["md5_path"])
        allowed = tuple(chroma_config["allow_knowledge_file_types"])

        def digest_seen(digest: str) -> bool:
            if not os.path.isfile(md5_path):
                return False
            with open(md5_path, encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    existing = line.split()[0]
                    if existing == digest:
                        return True
            return False

        def append_digest(digest: str) -> None:
            parent = os.path.dirname(md5_path)
            if parent:
                os.makedirs(parent, exist_ok=True)
            with open(md5_path, "a", encoding="utf-8") as f:
                f.write(digest + "\n")

        ingestion_clean_cfg = resolve_ingestion_clean_config(chroma_config)
        logger.info(
            "ingestion_clean: enabled=%s (see config/chrome.yml ingestion_clean)",
            ingestion_clean_cfg.get("enabled", True),
        )
        paths = listdir_with_allowed_type(data_dir, allowed)
        for file_path in paths:
            if _is_non_corpus_file(file_path):
                logger.info("Skip non-corpus file: %s", file_path)
                continue
            digest = get_file_md5(file_path)
            if digest_seen(digest):
                logger.info("Skip unchanged file (already in md5 ledger): %s", file_path)
                continue
            try:
                document = _documents_from_file(file_path)
                if not document:
                    logger.error("No documents loaded: %s", file_path)
                    continue
                document = clean_documents(document, ingestion_clean_cfg, source_hint=file_path)
                if not document:
                    logger.error("No documents left after ingestion_clean: %s", file_path)
                    continue
                if chroma_config.get("structured_chunking_enabled", True):
                    document = split_documents_by_sections(document)
                prepend_cfg = chroma_config.get("structured_chunking_enabled", True) and chroma_config.get(
                    "structured_chunk_prepend_section", True
                )
                use_pc = chroma_config.get("parent_child_enabled", False) and self.parent_store is not None

                if use_pc:
                    split_documents = []
                    parent_count = 0
                    for p_doc in document:
                        parent_id = str(uuid.uuid4())
                        parent_text = _parent_document_for_store(p_doc, prepend_cfg)
                        pmeta = dict(p_doc.metadata or {})
                        pmeta["parent_id"] = parent_id
                        self.parent_store.put(parent_id, parent_text, pmeta)
                        parent_count += 1
                        kids = self.spliter.split_documents([p_doc])
                        if prepend_cfg:
                            kids = prepend_section_title_to_chunks(kids)
                        for k in kids:
                            km = {**(k.metadata or {}), "parent_id": parent_id}
                            split_documents.append(
                                Document(page_content=k.page_content, metadata=km)
                            )
                else:
                    split_documents = self.spliter.split_documents(document)
                    if prepend_cfg:
                        split_documents = prepend_section_title_to_chunks(split_documents)
                if not split_documents:
                    logger.error("Split produced no chunks: %s", file_path)
                    continue
                self.chroma.add_documents(split_documents)
                append_digest(digest)
                if use_pc:
                    logger.info(
                        "Indexed %s (%d child chunks, %d parents)",
                        file_path,
                        len(split_documents),
                        parent_count,
                    )
                else:
                    logger.info("Indexed %s (%d chunks)", file_path, len(split_documents))
            except Exception:
                logger.error("Error loading file %s", file_path, exc_info=True)
        logger.info("Ingest scan finished (%d paths under %s)", len(paths), data_dir)


def main():
    """Basic smoke test for VectorStoreService wiring."""
    try:
        service = VectorStoreService()
        retriever = service.get_retriever()
        print("[OK] VectorStoreService 初始化成功")
        print(f"collection_name = {chroma_config['collection_name']}")
        print(f"persist_directory = {chroma_config['persist_directory']}")
        print(f"retriever_type = {type(retriever).__name__}")
    except Exception as e:
        print(f"[FAIL] VectorStoreService 测试失败: {e}")
        raise


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1].lower() in ("load", "ingest"):
        VectorStoreService().load_data()
    else:
        main()
