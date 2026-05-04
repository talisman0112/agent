"""会话历史：保证写入 session 后 ReactAgent 能将多轮消息传入图状态。"""

from unittest.mock import MagicMock, patch

from langchain_core.messages import AIMessage, HumanMessage


def test_conversation_history_to_messages_order_and_limit():
    from tools.reactagent import _conversation_history_to_messages

    hist = [
        {"role": "user", "content": "a"},
        {"role": "assistant", "content": "b"},
        {"role": "user", "content": "c"},
        {"role": "assistant", "content": "d"},
    ]
    ms = _conversation_history_to_messages(hist, max_turns=1)
    assert len(ms) == 2
    assert isinstance(ms[0], HumanMessage) and ms[0].content == "c"
    assert isinstance(ms[1], AIMessage) and ms[1].content == "d"


def test_execute_passes_prior_messages_into_agent_stream():
    """execute 应把历史 Human/AI + 本轮 user 一并交给 agent.stream（图中 before_model 才能看到多条）。"""

    def fake_init(self):
        self.last_tool_calls = []
        mock_graph = MagicMock()
        mock_graph.stream.return_value = iter([])
        self.agent = mock_graph

    import tools.reactagent as reactagent_mod

    with patch.object(reactagent_mod.ReactAgent, "__init__", fake_init):
        agent = reactagent_mod.ReactAgent()
        list(
            agent.execute(
                "second",
                conversation_history=[
                    {"role": "user", "content": "first q"},
                    {"role": "assistant", "content": "first a"},
                ],
                log_tool_calls=False,
            )
        )
        call_kw = agent.agent.stream.call_args
        assert call_kw is not None
        payload = call_kw[0][0]
        assert "messages" in payload
        assert len(payload["messages"]) == 3
        assert payload["messages"][0].content == "first q"
        assert payload["messages"][1].content == "first a"
        assert payload["messages"][2].content == "second"


def test_skips_empty_content_entries():
    from tools.reactagent import _conversation_history_to_messages

    hist = [
        {"role": "user", "content": "  hi  "},
        {"role": "assistant", "content": "   "},
        {"role": "user", "content": "next"},
    ]
    ms = _conversation_history_to_messages(hist, max_turns=None)
    assert len(ms) == 2
    assert ms[0].content == "hi"
    assert ms[1].content == "next"
