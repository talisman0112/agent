import yaml
from utils.path_pool import get_abs_path

rag_config_path = get_abs_path("config/rag.yml")
agent_config_path = get_abs_path("config/agent.yml")
prompt_config_path = get_abs_path("config/prompts.yml")
chroma_config_path = get_abs_path("config/chrome.yml")


def load_rag_config():
    with open(rag_config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    return config


def load_agent_config():
    with open(agent_config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    return config


def load_prompt_config():
    with open(prompt_config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    return config


def load_chroma_config():
    with open(chroma_config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    return config


rag_config = load_rag_config()
agent_config = load_agent_config()
prompt_config = load_prompt_config()
chroma_config = load_chroma_config()