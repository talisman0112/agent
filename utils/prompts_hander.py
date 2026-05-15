from utils.config_hander import prompt_config, rag_config
from utils.path_pool import get_abs_path
from utils.log import logger


_STRICT_GROUNDING_TAIL = """
---
【系统追加·仅用参考资料作答】正文开头允许用 **1～2 句**完整话复述用户问题及本次作答将依据的范围（不引入参考资料之外的新事实）。之后不得包含「以下为通识性归纳」段或脱离参考资料的教科书式长篇；仅在 **【依据摘录】/【可查结论】** 中写有出处的内容，子问题缺口简短说明即可。**禁止捏造**数字、主体公司、时点与未经资料支撑的买卖/监管断言。
"""


def _rag_strict_grounding_enabled() -> bool:
    """强约束：见 config/rag.yml 的 rag_strict_grounding；在统一主提示词后追加禁用通识兜底段的尾注。"""
    return bool(rag_config.get("rag_strict_grounding", False))


def get_main_prompt():
    try:
        with open(get_abs_path(prompt_config["main_prompt"]), "r", encoding="utf-8") as f:
            return f.read()
    except Exception as e:
        logger.error(f"Failed to get main prompt: {e}")
        raise e


def get_rag_prompt():
    rel = prompt_config.get("rag_prompt", "prompts/rag_prompt.txt")
    try:
        with open(get_abs_path(rel), "r", encoding="utf-8") as f:
            body = f.read()
        if _rag_strict_grounding_enabled():
            body = body.rstrip() + _STRICT_GROUNDING_TAIL
            logger.info("RAG 已追加强约束尾注（rag_strict_grounding=true）")
        return body
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