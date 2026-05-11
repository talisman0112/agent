"""调用 ``tools/agent_tool`` 中的 LangChain 工具，统一日志与异常文案。

流程要点：
1. MCP ``call_tool`` 在异步句柄里收到 name + arguments；
2. ``run_mcp_tool`` 把真正调用丢进线程池（``to_thread``），避免阻塞 asyncio 事件循环；
3. 各 ``_run_*`` 从 arguments 取参 → ``tool.invoke({...})`` → 字符串结果；
4. ``_wrap_tool_call`` 打耗时日志，异常时转为对用户可读的中文句。
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import Any, Callable

import anyio
import mcp.types as types

from mcp_impl.schemas import (
    GET_LOCAL_DATETIME_SCHEMA,
    HYBRID_SEARCH_SCHEMA,
    HYBRID_SUMMARIZE_SCHEMA,
    RAG_RETRIEVE_SCHEMA,
    RAG_SUMMARIZE_SCHEMA,
    WEB_SEARCH_SCHEMA,
)
from tools.agent_tool import (
    get_market_datetime,
    hybrid_search,
    hybrid_summarize,
    rag_retrieve,
    rag_summarize,
    web_search,
)

_log = logging.getLogger("agent")


@dataclass(frozen=True)
class ToolEntry:
    """单个 MCP 工具在 Python 侧的静态描述：对外名称、说明、入参 schema、执行函数。"""

    name: str  # MCP tools/list 与 tools/call 使用的工具名
    description: str  # 展示给客户端的人类可读说明
    input_schema: dict[str, Any]  # 与 mcp_impl/schemas 中常量一致
    run: Callable[[dict[str, Any]], str]  # 接收原始 arguments 字典，返回文本


def _friendly_error(tool_name: str, err: BaseException) -> str:
    """把底层异常消息粗分类，返回统一前缀「工具执行失败：…」，便于 IDE 里阅读。"""
    msg = str(err).lower()
    if "api" in msg and "key" in msg:
        return (
            "工具执行失败：未检测到可用的模型 API Key（如 DASHSCOPE_API_KEY / "
            "TONGYI_API_KEY），请先完成环境变量配置。"
        )
    if "timeout" in msg or "timed out" in msg:
        if tool_name == "web_search":
            return "工具执行失败：联网搜索当前不可用（超时），请检查网络连接或稍后再试。"
        return f"工具执行失败：请求超时（{err}）。"
    if "name or service not known" in msg or "getaddrinfo" in msg:
        return "工具执行失败：网络或 DNS 不可用，请检查连接后重试。"
    return f"工具执行失败：{err}"


def _wrap_tool_call(
    tool_name: str, fn: Callable[[], str], *, args_summary: str
) -> str:
    """包一层同步调用：打日志、计时；成功返回字符串，异常则交给 ``_friendly_error``。"""
    t0 = time.perf_counter()
    _log.info("MCP 调用工具: %s args=%s", tool_name, args_summary)
    try:
        out = fn()
        elapsed_ms = (time.perf_counter() - t0) * 1000
        _log.info("MCP 工具完成: %s elapsed_ms=%.0f", tool_name, elapsed_ms)
        return out if isinstance(out, str) else str(out)
    except Exception as e:  # noqa: BLE001
        elapsed_ms = (time.perf_counter() - t0) * 1000
        _log.warning(
            "MCP 工具失败: %s error=%r elapsed_ms=%.0f",
            tool_name,
            e,
            elapsed_ms,
        )
        return _friendly_error(tool_name, e)


def _run_rag_summarize(args: dict[str, Any]) -> str:
    """MCP ``rag_summarize``：可选多轮摘要进 ``dialogue_context``，与 Streamlit 工具入参一致。"""
    q = (args.get("query") or "").strip()
    if not q:
        return "提问为空，请提供具体问题。"
    ctx = args.get("dialogue_context") or ""
    summary = {"query": q[:200], "dialogue_context": (ctx or "")[:120]}
    return _wrap_tool_call(
        "rag_summarize",
        lambda: rag_summarize.invoke({"query": q, "dialogue_context": ctx}),
        args_summary=str(summary),
    )


def _run_rag_retrieve(args: dict[str, Any]) -> str:
    """MCP ``rag_retrieve``：只做向量检索 + 格式化片段，不调用总结模型。"""
    q = (args.get("query") or "").strip()
    if not q:
        return "提问为空，请提供检索关键词或问题。"
    return _wrap_tool_call(
        "rag_retrieve",
        lambda: rag_retrieve.invoke({"query": q}),
        args_summary=str({"query": q[:200]}),
    )


def _run_hybrid_search(args: dict[str, Any]) -> str:
    """MCP ``hybrid_search``：本地 + Web 召回、合并 Rerank 后的文本材料。"""
    q = (args.get("query") or "").strip()
    if not q:
        return "提问为空，请提供检索查询。"
    return _wrap_tool_call(
        "hybrid_search",
        lambda: hybrid_search.invoke({"query": q}),
        args_summary=str({"query": q[:200]}),
    )


def _run_hybrid_summarize(args: dict[str, Any]) -> str:
    """MCP ``hybrid_summarize``：在 hybrid 检索链末尾接上 LLM 生成最终回答。"""
    q = (args.get("query") or "").strip()
    if not q:
        return "提问为空，请提供具体问题。"
    return _wrap_tool_call(
        "hybrid_summarize",
        lambda: hybrid_summarize.invoke({"query": q}),
        args_summary=str({"query": q[:200]}),
    )


def _run_web_search(args: dict[str, Any]) -> str:
    """MCP ``web_search``：``max_results`` 钳制在 1–10，防止客户端传离谱数值。"""
    q = (args.get("query") or "").strip()
    if not q:
        return "搜索关键词为空，请提供要查询的内容。"
    max_results = args.get("max_results", 5)
    try:
        mr = int(max_results)
    except (TypeError, ValueError):
        mr = 5
    mr = max(1, min(mr, 10))
    return _wrap_tool_call(
        "web_search",
        lambda: web_search.invoke({"query": q, "max_results": mr}),
        args_summary=str({"query": q[:200], "max_results": mr}),
    )


def _run_get_local_datetime(args: dict[str, Any]) -> str:
    """MCP 工具名是 ``get_local_datetime``，这里转调 ``get_market_datetime``（项目内命名）。"""
    tz = args.get("timezone_name", "Asia/Shanghai") or "Asia/Shanghai"
    return _wrap_tool_call(
        "get_local_datetime",
        lambda: get_market_datetime.invoke({"timezone_name": tz}),
        args_summary=str({"timezone_name": tz}),
    )


# 对外工具列表的唯一真相源：增删工具时只改这一处 + schemas.py
TOOL_ENTRIES: tuple[ToolEntry, ...] = (
    ToolEntry(
        "rag_summarize",
        "基于本地向量知识库（投研语料）回答问题。",
        RAG_SUMMARIZE_SCHEMA,
        _run_rag_summarize,
    ),
    ToolEntry(
        "rag_retrieve",
        "仅从本地知识库检索相关原文片段，不调用大模型总结。",
        RAG_RETRIEVE_SCHEMA,
        _run_rag_retrieve,
    ),
    ToolEntry(
        "hybrid_search",
        "本地知识库与 Web 多路召回检索，返回结构化检索结果文本。",
        HYBRID_SEARCH_SCHEMA,
        _run_hybrid_search,
    ),
    ToolEntry(
        "hybrid_summarize",
        "多路召回后由大模型生成回答（本地 + Web + Rerank）。",
        HYBRID_SUMMARIZE_SCHEMA,
        _run_hybrid_summarize,
    ),
    ToolEntry(
        "web_search",
        "联网搜索（DuckDuckGo），适合时效性信息；依赖外网。",
        WEB_SEARCH_SCHEMA,
        _run_web_search,
    ),
    ToolEntry(
        "get_local_datetime",
        "返回指定 IANA 时区的当前本地日期与时间。",
        GET_LOCAL_DATETIME_SCHEMA,
        _run_get_local_datetime,
    ),
)

# 按名称 O(1) 查找，供 ``run_mcp_tool`` 路由
_TOOL_BY_NAME = {t.name: t for t in TOOL_ENTRIES}


def list_tool_definitions() -> list[types.Tool]:
    """供 MCP ``tools/list``：把本文件的 ToolEntry 转成 SDK 的 ``types.Tool``。"""
    return [
        types.Tool(
            name=t.name,
            description=t.description,
            inputSchema=t.input_schema,
        )
        for t in TOOL_ENTRIES
    ]


async def run_mcp_tool(name: str, arguments: dict[str, Any]) -> str:
    """执行一次 MCP 工具：未知名称抛 ``ValueError``（SDK 会转成带 isError 的响应）。"""
    entry = _TOOL_BY_NAME.get(name)
    if entry is None:
        raise ValueError(f"未知工具: {name}")

    def _sync() -> str:
        return entry.run(arguments)

    # LangChain 工具内部多为同步阻塞 IO/模型调用，必须放到线程里以免卡住 MCP 会话
    return await anyio.to_thread.run_sync(_sync)
