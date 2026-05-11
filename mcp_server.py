"""
MCP stdio 启动入口（见 docs/mcp_implementation_guide.md）。

实现代码位于 ``mcp_impl/``（避免与官方 PyPI 包 ``mcp`` 同名目录冲突）。

运行::

    python mcp_server.py

在 Cursor 等客户端中可将 command 指向本仓库解释器与上述脚本绝对路径。
"""

from __future__ import annotations

import os
import sys


def _ensure_project_root() -> None:
    """把进程工作目录和 import 路径固定在项目根，便于找到 config/、db/ 与各包。"""
    root = os.path.dirname(os.path.abspath(__file__))
    os.chdir(root)
    if root not in sys.path:
        sys.path.insert(0, root)


def main() -> None:
    """初始化日志后启动 stdio MCP：客户端通过标准输入输出与本进程对话。"""
    _ensure_project_root()
    import anyio

    from utils.log import setup_logging

    setup_logging()
    from mcp_impl.server import run_stdio

    # anyio.run：在事件循环里跑异步的 MCP 会话（读写 stdin/stdout）
    anyio.run(run_stdio)


if __name__ == "__main__":
    main()
