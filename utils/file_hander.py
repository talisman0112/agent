import hashlib
import os
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