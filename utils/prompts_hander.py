from utils.config_hander import prompt_config
from utils.path_pool import get_abs_path
from utils.log import logger
def get_main_prompt():
    try:
        with open(get_abs_path(prompt_config["main_prompt"]), "r", encoding="utf-8") as f:
            return f.read()
    except Exception as e:
        logger.error(f"Failed to get main prompt: {e}")
        raise e
def get_rag_prompt():
    try:
        with open(get_abs_path(prompt_config["rag_prompt"]), "r", encoding="utf-8") as f:
            return f.read()
    except Exception as e:
        logger.error(f"Failed to get rag prompt: {e}")
        raise e
def get_report_prompt():
    try:
        with open(get_abs_path(prompt_config["report_prompt"]), "r", encoding="utf-8") as f:
            return f.read()
    except Exception as e:
        logger.error(f"Failed to get report prompt: {e}")
        raise e