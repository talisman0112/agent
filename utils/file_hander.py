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
    file_list = []
    if not os.path.exists(dir_path):
        logger.error(f"Directory {dir_path} does not exist")
        raise FileNotFoundError(f"Directory {dir_path} does not exist")
    if not os.path.isdir(dir_path):
        logger.error(f"Directory {dir_path} is not a directory")
        raise IsADirectoryError(f"Directory {dir_path} is not a directory")
    for file in os.listdir(dir_path):
        if not file.endswith(allowed_type):
            continue
        file_list.append(os.path.join(dir_path, file))
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