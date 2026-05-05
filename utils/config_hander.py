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

# Rerank 配置（从 rag.yml 读取，提供默认值）
rerank_config = {
    "enabled": rag_config.get("rerank_enabled", True),
    "model": rag_config.get("rerank_model", "qwen3-rerank"),
    "top_n": rag_config.get("rerank_top_n", 5),
    "vector_search_k": rag_config.get("rerank_search_k", 20),
    # 增强版配置
    "enhanced": rag_config.get("rerank_enhanced", True),
    "score_threshold": rag_config.get("rerank_score_threshold", 0.4),
    "instruct": rag_config.get("rerank_instruct"),
}