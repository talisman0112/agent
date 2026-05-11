"""MCP tools 的 JSON Schema（手写，与 docs/mcp_implementation_guide.md 一致）。

客户端发 ``tools/call`` 时，MCP SDK 会按这里的 schema 校验 arguments；
字段含义也会出现在 Cursor 等 IDE 的工具说明里。
"""

from __future__ import annotations

# 本地 RAG：检索 + 大模型总结
RAG_SUMMARIZE_SCHEMA = {
    "type": "object",
    "properties": {
        "query": {
            "type": "string",
            "description": "用户问题或检索查询",
        },
        "dialogue_context": {
            "type": "string",
            "description": "与本轮问题相关的极简对话上下文",
            "default": "",
        },
    },
    "required": ["query"],
}

# 本地 RAG：只检索片段，不调 LLM
RAG_RETRIEVE_SCHEMA = {
    "type": "object",
    "properties": {
        "query": {
            "type": "string",
            "description": "检索关键词或完整问题",
        },
    },
    "required": ["query"],
}

# 混合检索：本地向量库 + Web，返回精排后的参考文本（无最终「问答」段落时可当材料）
HYBRID_SEARCH_SCHEMA = {
    "type": "object",
    "properties": {
        "query": {
            "type": "string",
            "description": "本地知识库 + Web 多路召回检索用查询",
        },
    },
    "required": ["query"],
}

# 混合检索后再由 LLM 生成完整回答
HYBRID_SUMMARIZE_SCHEMA = {
    "type": "object",
    "properties": {
        "query": {
            "type": "string",
            "description": "多路召回后生成答案所依据的用户问题",
        },
    },
    "required": ["query"],
}

# DuckDuckGo 联网搜索（依赖外网）
WEB_SEARCH_SCHEMA = {
    "type": "object",
    "properties": {
        "query": {
            "type": "string",
            "description": "搜索关键词",
        },
        "max_results": {
            "type": "integer",
            "description": "返回结果条数，建议 1-10",
            "default": 5,
            "minimum": 1,
            "maximum": 10,
        },
    },
    "required": ["query"],
}

# MCP 对外名 get_local_datetime，底层绑定 agent_tool 的 get_market_datetime
GET_LOCAL_DATETIME_SCHEMA = {
    "type": "object",
    "properties": {
        "timezone_name": {
            "type": "string",
            "description": 'IANA 时区名，如 Asia/Shanghai、America/New_York、UTC',
            "default": "Asia/Shanghai",
        },
    },
}
