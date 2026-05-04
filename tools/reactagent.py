from langchain.agents import create_agent
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from model.model import chat_model
from tools.agent_tool import TOOLS
from tools.mid_ware import middleware
from utils.log import logger


def _conversation_history_to_messages(
    conversation_history: list[dict],
    *,
    max_turns: int | None = 10,
) -> list[HumanMessage | AIMessage]:
    """将会话历史（role/content）转为 LangChain 消息列表，用作短期上下文。

    仅还原用户与助手**可见的最终文本**，不包含本轮用户输入；一般由调用方传入
    「当前提问之前」的 ``st.session_state.messages`` 等。
    """
    if not conversation_history:
        return []
    out: list[HumanMessage | AIMessage] = []
    for item in conversation_history:
        role = item.get("role")
        content = (item.get("content") or "").strip()
        if not content:
            continue
        if role == "user":
            out.append(HumanMessage(content=content))
        elif role == "assistant":
            out.append(AIMessage(content=content))
    if max_turns is None or max_turns <= 0:
        return out
    limit = max_turns * 2
    return out[-limit:] if len(out) > limit else out


def _message_text(msg) -> str:
    c = getattr(msg, "content", None)
    if isinstance(c, str):
        return c
    if isinstance(c, list):
        parts = []
        for block in c:
            if isinstance(block, str):
                parts.append(block)
            elif isinstance(block, dict) and block.get("type") == "text":
                parts.append(block.get("text", ""))
            else:
                parts.append(str(block))
        return "".join(parts)
    return "" if c is None else str(c)


class ReactAgent:
    def __init__(self):
        if chat_model is None:
            raise RuntimeError(
                "未配置对话模型：请安装 langchain-community 并设置 DASHSCOPE_API_KEY 或 TONGYI_API_KEY。"
            )
        self.agent = create_agent(
            chat_model,
            TOOLS,
            middleware=middleware,
        )
        # 最近一次 execute 的工具调用摘要，供页面展示或调试（每个元素: name, content 预览）
        self.last_tool_calls: list[dict[str, str]] = []

    def execute(
        self,
        user_input: str,
        *,
        conversation_history: list[dict] | None = None,
        short_term_turns: int | None = 20,
        log_tool_calls: bool = True,
        report_mode: bool = False,
    ):
        """流式输出：按 values 事件中最后一条 AI 消息的增量 yield 文本片段。

        ``conversation_history`` 为短期记忆：每项 ``{"role": "user"|"assistant", "content": str}``，
        应为**不含本轮用户提问**的上文；不传则与本实现原先行为一致（单轮）。
        ``short_term_turns`` 表示最多保留的对话轮数上限（每轮约含一条用户消息与一条助手消息）。
        ``report_mode`` 控制使用的提示词策略：True 时使用报告模式（结构化、可沉淀的回答），
        False 时使用主对话模式（常规对话）。

        判断是否调用了工具，可以：
        1. 看本方法执行后的 ``self.last_tool_calls``（本次 run 的非空表示调用过工具）；
        2. 看日志文件 ``log/agent.log`` 中的 ``Agent 调用工具`` / ``模型请求工具``；
           若整轮没有任何工具执行，会打一行的 ``本轮未调用任何工具``（在 ``log_tool_calls=True`` 时）。
        3. 将 ``stream_mode`` 设为 ``\"updates\"`` 自行解析（本实现已合并 ``values`` + ``updates`` 并打点日志）。
        """
        self.last_tool_calls = []
        prev = ""
        prior = _conversation_history_to_messages(
            conversation_history or [],
            max_turns=short_term_turns,
        )
        messages_to_send = [*prior, HumanMessage(content=user_input)]
        stream = self.agent.stream(
            {"messages": messages_to_send},
            stream_mode=["values", "updates"],
            context={"report": report_mode},
        )
        try:
            for mode, event in stream:
                if mode == "updates" and isinstance(event, dict) and log_tool_calls:
                    if "model" in event:
                        patch = event["model"]
                        if isinstance(patch, dict):
                            for msg in patch.get("messages") or []:
                                if isinstance(msg, AIMessage) and msg.tool_calls:
                                    for tc in msg.tool_calls:
                                        if isinstance(tc, dict):
                                            name = tc.get("name", "?")
                                            args = tc.get("args")
                                        else:
                                            name = getattr(tc, "name", None) or "?"
                                            args = getattr(tc, "args", None)
                                        if log_tool_calls:
                                            logger.info("模型请求工具: %s args=%s", name, args)
                    if "tools" in event:
                        patch = event["tools"]
                        if isinstance(patch, dict):
                            for msg in patch.get("messages") or []:
                                if isinstance(msg, ToolMessage):
                                    preview = (msg.content or "")[:500]
                                    rec = {
                                        "name": msg.name or "?",
                                        "content_preview": preview
                                        + ("…" if len(str(msg.content or "")) > 500 else ""),
                                    }
                                    self.last_tool_calls.append(rec)
                                    if log_tool_calls:
                                        logger.info(
                                            "Agent 调用工具: name=%s tool_call_id=%s content_preview=%s",
                                            rec["name"],
                                            getattr(msg, "tool_call_id", ""),
                                            preview[:200],
                                        )
                    continue

                if mode != "values":
                    continue
                messages = event.get("messages") or []
                if not messages:
                    continue
                last = messages[-1]
                if not isinstance(last, AIMessage):
                    continue
                text = _message_text(last)
                if not text:
                    continue
                if text.startswith(prev):
                    delta = text[len(prev) :]
                    prev = text
                    if delta:
                        yield delta
                else:
                    prev = text
                    yield text
        finally:
            if log_tool_calls:
                if not self.last_tool_calls:
                    logger.info("本轮未调用任何工具")
                else:
                    logger.info(
                        "本轮工具调用结束，共 %s 次: %s",
                        len(self.last_tool_calls),
                        [c["name"] for c in self.last_tool_calls],
                    )


if __name__ == "__main__":
    agent = ReactAgent()
    print("".join(agent.execute("你好，用一句话自我介绍。")))
