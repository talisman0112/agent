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
    "dedup_threshold": rag_config.get("rerank_dedup_threshold", 0.80),
    # Instruct 配置
    "instruct_mode": rag_config.get("rerank_instruct_mode", "qa"),
    "auto_instruct": rag_config.get("rerank_auto_instruct", True),
    "custom_instruct": rag_config.get("rerank_custom_instruct"),
    # 上下文压缩配置
    "compression_enabled": rag_config.get("compression_enabled", True),
    "compression_max_tokens": rag_config.get("compression_max_tokens", 3500),
    "compression_min_tokens": rag_config.get("compression_min_tokens", 200),
    "compression_extract_ratio": rag_config.get("compression_extract_ratio", 0.6),
    "compression_use_llm": rag_config.get("compression_use_llm", False),
    "compression_quality_threshold": rag_config.get("compression_quality_threshold", 0.7),
    "compression_strategy": rag_config.get("compression_strategy", "auto"),
    # Query 扩展（多查询 / 分解）
    "query_expansion_enabled": rag_config.get("query_expansion_enabled", False),
    "query_expansion_variants": rag_config.get("query_expansion_variants", 5),
    "query_expansion_include_original": rag_config.get(
        "query_expansion_include_original", True
    ),
    "query_expansion_max_coarse_docs": rag_config.get(
        "query_expansion_max_coarse_docs", 100
    ),
    "query_expansion_max_workers": rag_config.get("query_expansion_max_workers", 8),
    "query_decompose_enabled": rag_config.get("query_decompose_enabled", False),
    "query_decompose_max_subqueries": rag_config.get(
        "query_decompose_max_subqueries", 4
    ),
    "query_decompose_with_expansion": rag_config.get(
        "query_decompose_with_expansion", False
    ),
}