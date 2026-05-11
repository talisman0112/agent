"""构建并运行基于 stdio 的 MCP Server。

stdio 表示：不监听端口，只通过标准输入 / 标准输出与父进程里的 MCP 客户端交换 JSON-RPC；
Cursor、Claude Desktop 等一般会 ``subprocess`` 拉起 ``python mcp_server.py`` 并接管其管道。
"""

from __future__ import annotations

import mcp.types as types
from mcp.server import Server
from mcp.server.stdio import stdio_server

from mcp_impl.handlers import list_tool_definitions, run_mcp_tool


def build_server() -> Server:
    """创建一个已注册 ``list_tools`` / ``call_tool`` 的服务器实例（尚未开始读写 stdio）。"""
    server = Server(
        "rag-agent",
        version="1.0.0",
        instructions=(
            "FinSight 投研 RAG Agent MCP：本地语料检索与总结、混合检索、联网搜索与时区时间。"
            "需配置 DASHSCOPE_API_KEY 或 TONGYI_API_KEY；RAG 依赖已构建的向量库。"
        ),
    )

    @server.list_tools()
    async def list_tools() -> list[types.Tool]:
        """响应客户端 ``tools/list``：返回名称、描述、inputSchema，供 UI 展示与参数校验。"""
        return list_tool_definitions()

    @server.call_tool()
    async def call_tool_handler(
        name: str,
        arguments: dict | None,
    ) -> list[types.TextContent]:
        """响应 ``tools/call``：按 name 路由到 handlers，结果统一封装为一段 ``text`` 内容。"""
        text = await run_mcp_tool(name, arguments or {})
        return [types.TextContent(type="text", text=text)]

    return server


async def run_stdio() -> None:
    """挂载 stdio 传输并阻塞运行，直到客户端断开或进程结束。"""
    server = build_server()
    # SDK 根据已注册的 handler 自动填写 capabilities（例如声明支持 tools）
    init = server.create_initialization_options()
    async with stdio_server() as (read_stream, write_stream):
        await server.run(read_stream, write_stream, init)
