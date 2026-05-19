from langgraph.runtime import Runtime

from langchain.agents.middleware.types import (
    AgentState,
    ModelRequest,
    after_model,
    before_model,
    dynamic_prompt,
    wrap_tool_call,
)
from langchain_core.messages import ToolMessage

from utils.log import logger
from utils.prompts_hander import get_main_prompt, get_report_prompt


@before_model
def log_before_model(state: AgentState, runtime: Runtime) -> None:
    logger.info("Model call request: %s", state)
    logger.info("[log_before_model] 即将调用模型，带有 %s 条消息", len(state["messages"]))
    if state["messages"]:
        logger.debug("[log_before_model] 最后一条: %s", state["messages"][-1].content)


@after_model
def log_after_model(state: AgentState, runtime: Runtime) -> None:
    logger.info("Model call response: %s", state)


@dynamic_prompt
def report_prompt_switch(request: ModelRequest) -> str:
    is_report = request.runtime.context.get("report", False)
    if is_report:
        return get_report_prompt()
    return get_main_prompt()


@wrap_tool_call
def handle_tool_errors_middleware(request, handler):
    """工具执行异常时回写错误 ToolMessage，避免整轮崩溃（对应 agent.yml handle_parsing_errors）。"""
    try:
        return handler(request)
    except Exception as e:  # noqa: BLE001
        tc = request.tool_call or {}
        tool_call_id = tc.get("id") or ""
        name = tc.get("name") or "?"
        logger.warning(
            "[handle_tool_errors_middleware] tool=%s error=%s",
            name,
            e,
        )
        return ToolMessage(
            content=(
                f"工具调用失败（{name}）：{e}\n"
                "请检查参数格式后重试，或改用其他工具。"
            ),
            tool_call_id=tool_call_id,
            name=name,
            status="error",
        )


def build_middleware(*, handle_tool_errors: bool = True) -> list:
    """组装 Agent middleware；``handle_tool_errors=False`` 时不注入错误兜底层。"""
    stack = [log_before_model, log_after_model, report_prompt_switch]
    if handle_tool_errors:
        stack.append(handle_tool_errors_middleware)
    return stack


middleware = build_middleware(handle_tool_errors=True)
