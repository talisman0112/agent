from utils.path_pool import get_abs_path
import os
import logging
log_root = get_abs_path("log")

os.makedirs(log_root, exist_ok=True)
DEFAULT_LOG_FORMAT = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")


def get_logger(name):
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    if logger.handlers:
        return logger
    handler = logging.StreamHandler()
    handler.setLevel(logging.INFO)
    handler.setFormatter(DEFAULT_LOG_FORMAT)
    logger.addHandler(handler)
    log_file = os.path.join(log_root, f"{name}.log")
    file_handler = logging.FileHandler(log_file, encoding="utf-8")
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(DEFAULT_LOG_FORMAT)
    logger.addHandler(file_handler)
    return logger


def setup_logging():
    logging.basicConfig(level=logging.INFO)
    logging.getLogger("langchain").setLevel(logging.WARNING)
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("openai").setLevel(logging.WARNING)
    logging.getLogger("chromadb").setLevel(logging.WARNING)
    logging.getLogger("langchain_core").setLevel(logging.WARNING)


logger = get_logger("agent")