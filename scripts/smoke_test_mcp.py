"""
MCP 端到端冒烟：以子进程启动 ``mcp_server.py``，走真实 JSON-RPC stdio。

用法（在项目根）::

    python scripts/smoke_test_mcp.py

依赖：已安装 ``mcp``；``rag_retrieve`` / ``rag_summarize`` 需向量库与（总结时）API Key；
``web_search`` 需外网，失败时记为 SKIP 而非整脚本失败。
"""

from __future__ import annotations

import os
import sys
from datetime import timedelta
from pathlib import Path

import anyio
import mcp.types as types
from mcp.client.session import ClientSession
from mcp.client.stdio import StdioServerParameters, stdio_client


def _project_root() -> Path:
    return Path(__file__).resolve().parent.parent


def _text_from_result(result: types.CallToolResult) -> str:
    parts: list[str] = []
    for block in result.content:
        if isinstance(block, types.TextContent):
            parts.append(block.text)
    return "\n".join(parts) if parts else ""


def _fail(msg: str) -> None:
    print(f"[FAIL] {msg}", file=sys.stderr)
    raise SystemExit(1)


def _ok(msg: str) -> None:
    print(f"[OK]   {msg}")


def _warn(msg: str) -> None:
    print(f"[SKIP] {msg}", file=sys.stderr)


async def _run() -> None:
    root = _project_root()
    server_script = root / "mcp_server.py"
    if not server_script.is_file():
        _fail(f"未找到 {server_script}")

    expected = {
        "rag_summarize",
        "rag_retrieve",
        "hybrid_search",
        "hybrid_summarize",
        "web_search",
        "get_local_datetime",
    }

    params = StdioServerParameters(
        command=sys.executable,
        args=[str(server_script)],
        cwd=str(root),
        # 子进程需继承当前环境（DASHSCOPE_API_KEY 等）；SDK 默认 env 不含这些变量
        env=dict(os.environ),
    )

    async with stdio_client(params) as (read_stream, write_stream):
        async with ClientSession(read_stream, write_stream) as session:
            await session.initialize()
            _ok("session.initialize")

            listed = await session.list_tools()
            names = {t.name for t in listed.tools}
            if names != expected:
                _fail(f"tools/list 名称集合不符。\n  期望: {sorted(expected)}\n  实际: {sorted(names)}")
            _ok(f"tools/list 共 {len(listed.tools)} 个工具，名称符合预期")

            # 1) 无时区模型/API 的轻量调用
            r0 = await session.call_tool("get_local_datetime", {"timezone_name": "UTC"})
            if r0.isError:
                _fail(f"get_local_datetime: { _text_from_result(r0) }")
            if "UTC" not in _text_from_result(r0) and "GMT" not in _text_from_result(
                r0
            ) and "202" not in _text_from_result(r0):
                _fail(f"get_local_datetime 返回异常: {_text_from_result(r0)[:200]}")
            _ok("call_tool get_local_datetime")

            # 2) 仅检索，不调 LLM
            r1 = await session.call_tool("rag_retrieve", {"query": "ROE 杜邦"})
            if r1.isError:
                _fail(f"rag_retrieve: {_text_from_result(r1)}")
            body = _text_from_result(r1)
            if len(body) < 10:
                _fail(f"rag_retrieve 返回过短: {body!r}")
            _ok("call_tool rag_retrieve（返回非空文本）")

            # 3) 总结：可能因缺 Key / 网络报错 → 记 SKIP
            r2 = await session.call_tool(
                "rag_summarize",
                {"query": "什么是市盈率", "dialogue_context": ""},
                read_timeout_seconds=timedelta(seconds=120),
            )
            if r2.isError or "工具执行失败" in _text_from_result(r2):
                _warn(
                    "rag_summarize 未通过（多为未配置 DASHSCOPE_API_KEY 或模型不可用），"
                    f"详情: {_text_from_result(r2)[:300]}"
                )
            else:
                _ok("call_tool rag_summarize")

            # 4) 联网搜索：可选
            r3 = await session.call_tool(
                "web_search",
                {"query": "Python MCP protocol", "max_results": 2},
                read_timeout_seconds=timedelta(seconds=45),
            )
            if r3.isError:
                _warn(f"web_search: {_text_from_result(r3)[:200]}")
            else:
                _ok("call_tool web_search")


def main() -> None:
    anyio.run(_run, backend="asyncio")


if __name__ == "__main__":
    main()
