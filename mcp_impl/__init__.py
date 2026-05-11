"""MCP 协议适配层：将本项目 RAG / 混合检索等能力以 MCP tools 暴露。

目录名使用 ``mcp_impl`` 而非 ``mcp``，避免与 PyPI 上的官方 ``mcp`` SDK 包冲突。

对外常用入口：
- ``run_stdio``：被 ``mcp_server.py`` 调用，跑完整 stdio 服务；
- ``build_server``：仅在测试或自定义传输层时需要预先拿到 ``Server`` 实例。
"""

from mcp_impl.server import build_server, run_stdio

__all__ = ["build_server", "run_stdio"]
