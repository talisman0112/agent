from langgraph.runtime import Runtime

from langchain.agents.middleware.types import (
    AgentState,
    ModelRequest,
    after_model,
    before_model,
    dynamic_prompt,
)
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


middleware = [log_before_model, log_after_model, report_prompt_switch]
