"""Agent 运行时参数（P2）：recursion_limit 与工具错误 middleware。"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from tools.mid_ware import build_middleware, handle_tool_errors_middleware
from utils.agent_runtime import get_agent_runtime_settings


def test_recursion_limit_from_max_iterations():
    s = get_agent_runtime_settings({"max_iterations": 18})
    assert s["max_iterations"] == 18
    assert s["recursion_limit"] == 37  # 18 * 2 + 1


def test_handle_parsing_errors_default_true():
    s = get_agent_runtime_settings({})
    assert s["handle_parsing_errors"] is True


def test_build_middleware_without_error_handler():
    stack = build_middleware(handle_tool_errors=False)
    assert handle_tool_errors_middleware not in stack


def test_build_middleware_with_error_handler():
    stack = build_middleware(handle_tool_errors=True)
    assert handle_tool_errors_middleware in stack


def test_handle_tool_errors_middleware_returns_tool_message():
    class Boom(Exception):
        pass

    def bad_handler(_request):
        raise Boom("invalid args")

    class Req:
        tool_call = {"id": "tc1", "name": "rag_summarize", "args": {}}

    msg = handle_tool_errors_middleware.wrap_tool_call(Req(), bad_handler)
    assert msg.status == "error"
    assert "invalid args" in msg.content
    assert msg.tool_call_id == "tc1"


def test_react_agent_reads_runtime_settings():
    from tools.reactagent import ReactAgent

    agent = ReactAgent(
        runtime_config={
            "max_iterations": 10,
            "handle_parsing_errors": False,
        }
    )
    assert agent._max_iterations == 10
    assert agent._recursion_limit == 21
    assert agent._handle_parsing_errors is False
