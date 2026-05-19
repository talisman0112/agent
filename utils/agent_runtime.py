"""ReAct Agent 运行时参数：将 agent.yml 映射为 LangGraph 可消费的配置。"""

from __future__ import annotations

from typing import Any


def get_agent_runtime_settings(agent_config: dict[str, Any] | None = None) -> dict[str, Any]:
    """从 ``config/agent.yml`` 读取并规范化运行时参数。

    Returns:
        recursion_limit: LangGraph ``stream``/``invoke`` 的图步数上限；
            按 ``max_iterations * 2 + 1`` 估算（一轮 model+tools ≈ 2 步）。
        max_iterations: yml 原始值，供日志/UI。
        handle_parsing_errors: 是否在工具执行异常时回写 ToolMessage 供模型重试。
    """
    cfg = agent_config or {}
    max_iterations = int(cfg.get("max_iterations", 18))
    max_iterations = max(1, max_iterations)
    # 每轮 ReAct 通常含 model 与 tools 两个 superstep
    recursion_limit = max(4, max_iterations * 2 + 1)
    return {
        "max_iterations": max_iterations,
        "recursion_limit": recursion_limit,
        "handle_parsing_errors": bool(cfg.get("handle_parsing_errors", True)),
    }
