import csv
import hashlib
import json
import os
from collections.abc import Iterator
from typing import Any

from langchain_core.documents import Document

from langchain_community.document_loaders import (
    Docx2txtLoader,
    PyPDFLoader,
    TextLoader,
    UnstructuredPowerPointLoader,
    UnstructuredExcelLoader,
)
from utils.log import logger


def get_file_md5(file_path):
    if not os.path.exists(file_path):
        logger.error(f"File {file_path} does not exist")
        raise FileNotFoundError(f"File {file_path} does not exist")
    if os.path.isdir(file_path):
        logger.error(f"File {file_path} is a directory")
        raise IsADirectoryError(f"File {file_path} is a directory")
    chunk_size = 8192
    md5_obj = hashlib.md5()
    with open(file_path, "rb") as f:
        while True:
            data = f.read(chunk_size)
            if not data:
                break
            md5_obj.update(data)
    return md5_obj.hexdigest()


def listdir_with_allowed_type(dir_path: str, allowed_type: tuple[str]):
    """递归扫描 ``dir_path`` 下所有匹配 ``allowed_type`` 后缀的文件。

    递归是为了支持按业务维度组织子目录（例如 FinSight 投研语料库下的
    ``glossary/`` / ``industry_kb/`` / ``research_reports/`` / ``company_filings/`` /
    ``policy/`` 等）；顶层文件依然会被扫到，向后兼容旧的"全部平铺"结构。

    返回值按目录树深度优先排序，便于复现的入库顺序与 MD5 ledger 对齐。
    """
    if not os.path.exists(dir_path):
        logger.error(f"Directory {dir_path} does not exist")
        raise FileNotFoundError(f"Directory {dir_path} does not exist")
    if not os.path.isdir(dir_path):
        logger.error(f"Directory {dir_path} is not a directory")
        raise IsADirectoryError(f"Directory {dir_path} is not a directory")

    file_list: list[str] = []
    for current_root, sub_dirs, files in os.walk(dir_path):
        sub_dirs.sort()
        for file in sorted(files):
            if not file.endswith(allowed_type):
                continue
            file_list.append(os.path.join(current_root, file))
    return file_list


def pdf_loader(file_path: str):
    return PyPDFLoader(file_path)


def txt_loader(file_path: str):
    return TextLoader(file_path, encoding="utf-8")


def docx_loader(file_path: str):
    return Docx2txtLoader(file_path)


def doc_loader(file_path: str):
    return Docx2txtLoader(file_path)


def xlsx_loader(file_path: str):
    return UnstructuredExcelLoader(file_path)


def xls_loader(file_path: str):
    return UnstructuredExcelLoader(file_path)


def ppt_loader(file_path: str):
    return UnstructuredPowerPointLoader(file_path)


def pptx_loader(file_path: str):
    return UnstructuredPowerPointLoader(file_path)


def _csv_row_text(row: dict[str, str]) -> str:
    parts = [
        f"{k}: {v}"
        for k, v in row.items()
        if v is not None and str(v).strip()
    ]
    return "\n".join(parts)


def _read_text_with_encodings(
    file_path: str,
    *,
    encoding_candidates: tuple[str, ...],
) -> str | None:
    last_err: Exception | None = None
    for enc in encoding_candidates:
        try:
            with open(file_path, encoding=enc) as f:
                return f.read()
        except UnicodeDecodeError as e:
            last_err = e
            continue
    logger.error("文本解码失败 %s: %s", file_path, last_err)
    return None


def _yield_batched_line_documents(
    lines_iter: Iterator[str],
    *,
    file_path: str,
    loader: str,
    max_batch_chars: int,
) -> Iterator[Document]:
    """将若干行字符串按字符预算合并为 ``Document``（与 CSV / JSON 攒批逻辑一致）。"""
    batch_lines: list[str] = []
    batch_chars = 0
    for line in lines_iter:
        if not line:
            continue
        add_len = len(line) + 1
        if batch_lines and batch_chars + add_len > max_batch_chars:
            body = "\n\n".join(batch_lines)
            yield Document(
                page_content=body,
                metadata={"source": file_path, "loader": loader},
            )
            batch_lines = []
            batch_chars = 0
        batch_lines.append(line)
        batch_chars += add_len
    if batch_lines:
        body = "\n\n".join(batch_lines)
        yield Document(
            page_content=body,
            metadata={"source": file_path, "loader": loader},
        )


def iter_csv_documents(
    file_path: str,
    *,
    max_batch_chars: int = 6000,
    encoding_candidates: tuple[str, ...] = ("utf-8-sig", "utf-8", "gb18030"),
):
    """逐条产出 CSV 合并后的 ``Document``（按字符预算攒批），供入库流式写入，避免超大 CSV 一次性进内存。"""
    last_err: Exception | None = None
    for enc in encoding_candidates:
        try:
            with open(file_path, newline="", encoding=enc) as f:
                reader = csv.DictReader(f)
                if reader.fieldnames is None:
                    logger.warning("CSV 无表头，跳过: %s", file_path)
                    return

                def row_lines() -> Iterator[str]:
                    for row in reader:
                        line = _csv_row_text(row)
                        if line:
                            yield line

                yield from _yield_batched_line_documents(
                    row_lines(),
                    file_path=file_path,
                    loader="csv",
                    max_batch_chars=max_batch_chars,
                )
                return
        except UnicodeDecodeError as e:
            last_err = e
            continue
    logger.error("CSV 解码失败 %s: %s", file_path, last_err)


# 超过此大小的 JSON 不再整文件 read()；用 raw_decode 流式读顶层数组或按行 NDJSON
_LARGE_JSON_BYTES = 16 * 1024 * 1024


class _NotTopLevelJsonArray(Exception):
    """文件首段非 ``[`` 开头的顶层数组，换其它策略。"""


def _iter_json_stdlib_top_array(
    file_path: str,
    *,
    encoding_candidates: tuple[str, ...],
    max_batch_chars: int,
) -> Iterator[Document]:
    """用 ``JSONDecoder.raw_decode`` 分块读取顶层 ``[ ... ]``（不依赖第三方库）。"""
    for enc in encoding_candidates:
        try:

            def element_strings() -> Iterator[str]:
                with open(file_path, "r", encoding=enc) as f:
                    decoder = json.JSONDecoder()
                    buf = ""
                    seen_bracket = False
                    while True:
                        if not seen_bracket:
                            chunk = f.read(65536)
                            if not chunk and not buf.strip():
                                return
                            buf += chunk
                            head = buf.lstrip()
                            if not head:
                                if not chunk:
                                    return
                                continue
                            if head[0] != "[":
                                raise _NotTopLevelJsonArray()
                            buf = head[1:].lstrip()
                            seen_bracket = True
                            continue

                        buf = buf.lstrip()
                        while not buf:
                            chunk = f.read(65536)
                            if not chunk:
                                return
                            buf = chunk.lstrip()
                        if buf.startswith("]"):
                            return
                        if buf.startswith(","):
                            buf = buf[1:].lstrip()
                            continue
                        try:
                            obj, idx = decoder.raw_decode(buf)
                        except json.JSONDecodeError:
                            chunk = f.read(65536)
                            if not chunk:
                                logger.warning(
                                    "顶层 JSON 数组元素不完整或未闭合: %s",
                                    file_path,
                                )
                                return
                            buf += chunk
                            continue
                        yield json.dumps(obj, ensure_ascii=False)
                        buf = buf[idx:].lstrip()

            yield from _yield_batched_line_documents(
                element_strings(),
                file_path=file_path,
                loader="json",
                max_batch_chars=max_batch_chars,
            )
            return
        except _NotTopLevelJsonArray:
            continue
        except UnicodeDecodeError:
            continue


def _iter_json_ndjson_lines_stream(
    file_path: str,
    *,
    encoding_candidates: tuple[str, ...],
    max_batch_chars: int,
) -> Iterator[Document]:
    """不加载整文件：逐行 NDJSON。"""
    last_err: Exception | None = None
    for enc in encoding_candidates:
        try:
            with open(file_path, encoding=enc) as f:

                def ndjson_lines() -> Iterator[str]:
                    bad = 0
                    for raw_line in f:
                        line = raw_line.strip()
                        if not line:
                            continue
                        try:
                            obj = json.loads(line)
                        except json.JSONDecodeError:
                            bad += 1
                            if bad <= 3:
                                logger.warning(
                                    "跳过无法解析的 NDJSON 行 (%s): %.120s…",
                                    file_path,
                                    line,
                                )
                            continue
                        yield json.dumps(obj, ensure_ascii=False)
                    if bad:
                        logger.warning("NDJSON 共跳过 %d 行无效 JSON: %s", bad, file_path)

                yield from _yield_batched_line_documents(
                    ndjson_lines(),
                    file_path=file_path,
                    loader="json",
                    max_batch_chars=max_batch_chars,
                )
                return
        except UnicodeDecodeError as e:
            last_err = e
            continue
    logger.error("NDJSON 文本解码失败 %s: %s", file_path, last_err)


def iter_json_documents(
    file_path: str,
    *,
    max_batch_chars: int = 6000,
    encoding_candidates: tuple[str, ...] = ("utf-8-sig", "utf-8", "gb18030"),
) -> Iterator[Document]:
    """逐条产出 JSON 合并后的 ``Document``。

    - 小文件：整文件 ``json.loads``；失败则按内存中的文本做 NDJSON 分行解析。
    - 大文件（>{size} MB）：**不** ``read()`` 全文件；先用 **stdlib** ``raw_decode`` 流式读顶层数组 ``[...]``，
      若无产出再按行 NDJSON。

    依赖：无；若为超大顶层对象 ``{{...}}`` 而非数组，请改为 NDJSON 或拆文件。
    """.format(size=_LARGE_JSON_BYTES // (1024 * 1024))

    try:
        nbytes = os.path.getsize(file_path)
    except OSError as e:
        logger.error("无法读取 JSON 大小 %s: %s", file_path, e)
        return

    if nbytes > _LARGE_JSON_BYTES:
        sent_array = False
        for doc in _iter_json_stdlib_top_array(
            file_path,
            encoding_candidates=encoding_candidates,
            max_batch_chars=max_batch_chars,
        ):
            sent_array = True
            yield doc
        if sent_array:
            return
        yield from _iter_json_ndjson_lines_stream(
            file_path,
            encoding_candidates=encoding_candidates,
            max_batch_chars=max_batch_chars,
        )
        return

    text = _read_text_with_encodings(file_path, encoding_candidates=encoding_candidates)
    if text is None:
        return
    stripped = text.strip()
    if not stripped:
        logger.warning("JSON 文件为空: %s", file_path)
        return

    def records_from_parse(data: Any) -> Iterator[str]:
        if isinstance(data, list):
            for item in data:
                yield json.dumps(item, ensure_ascii=False)
        else:
            yield json.dumps(data, ensure_ascii=False)

    try:
        parsed = json.loads(stripped)
    except json.JSONDecodeError:
        def ndjson_lines() -> Iterator[str]:
            bad = 0
            for raw_line in text.splitlines():
                line = raw_line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError:
                    bad += 1
                    if bad <= 3:
                        logger.warning(
                            "跳过无法解析的 NDJSON 行 (%s): %.120s…",
                            file_path,
                            line,
                        )
                    continue
                yield json.dumps(obj, ensure_ascii=False)
            if bad:
                logger.warning("NDJSON 共跳过 %d 行无效 JSON: %s", bad, file_path)

        yield from _yield_batched_line_documents(
            ndjson_lines(),
            file_path=file_path,
            loader="json",
            max_batch_chars=max_batch_chars,
        )
        return

    yield from _yield_batched_line_documents(
        records_from_parse(parsed),
        file_path=file_path,
        loader="json",
        max_batch_chars=max_batch_chars,
    )
