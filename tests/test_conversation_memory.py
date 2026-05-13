from unittest.mock import MagicMock, patch


class _FakeResponse:
    def __init__(self, content: str):
        self.content = content


class _FakeLLM:
    def __init__(self, content: str):
        self.content = content

    def invoke(self, prompt: str):
        return _FakeResponse(self.content)


def _build_history(turns: int) -> list[dict]:
    history = []
    for i in range(1, turns + 1):
        history.append({"role": "user", "content": f"user-{i}"})
        history.append({"role": "assistant", "content": f"assistant-{i}"})
    return history


def test_should_compact_when_turn_threshold_reached():
    from memory.conversation_memory import ConversationMemoryManager

    manager = ConversationMemoryManager(
        llm=None,
        config={"recent_turns": 1, "summary_trigger_turns": 2, "summary_increment_turns": 1},
    )
    history = _build_history(3)

    assert manager.should_compact(history, manager.init_summary_state()) is True


def test_should_not_compact_below_threshold():
    from memory.conversation_memory import ConversationMemoryManager

    manager = ConversationMemoryManager(
        llm=None,
        config={"recent_turns": 2, "summary_trigger_turns": 5, "summary_increment_turns": 2},
    )
    history = _build_history(3)

    assert manager.should_compact(history, manager.init_summary_state()) is False


def test_update_summary_updates_state_and_covered_count():
    from memory.conversation_memory import ConversationMemoryManager

    manager = ConversationMemoryManager(
        llm=_FakeLLM("- 当前任务：继续排查长对话 memory"),
        config={"recent_turns": 1, "summary_trigger_turns": 1, "summary_increment_turns": 1},
    )
    history = _build_history(3)

    updated = manager.update_summary(history, manager.init_summary_state())

    assert updated["summary_text"] == "- 当前任务：继续排查长对话 memory"
    assert updated["covered_message_count"] == 4
    assert updated["last_update_turn"] == 3


def test_execute_includes_memory_summary_before_recent_history():
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
                "third",
                conversation_history=[
                    {"role": "user", "content": "first q"},
                    {"role": "assistant", "content": "first a"},
                    {"role": "user", "content": "second q"},
                    {"role": "assistant", "content": "second a"},
                ],
                short_term_turns=1,
                memory_summary="- 已讨论 earlier context",
                log_tool_calls=False,
            )
        )

        call_kw = agent.agent.stream.call_args
        assert call_kw is not None
        payload = call_kw[0][0]
        messages = payload["messages"]
        assert len(messages) == 4
        assert "已讨论 earlier context" in messages[0].content
        assert messages[1].content == "second q"
        assert messages[2].content == "second a"
        assert messages[3].content == "third"


def test_execute_memory_facts_before_summary():
    """结构化长期记忆应插在滚动摘要之前。"""

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
                "current",
                conversation_history=[],
                short_term_turns=20,
                memory_facts_text="【长期记忆】\n- 目标：测试",
                memory_summary="- 较早摘要要点",
                log_tool_calls=False,
            )
        )

        payload = agent.agent.stream.call_args[0][0]
        messages = payload["messages"]
        assert len(messages) == 3
        assert "长期记忆" in messages[0].content
        assert "历史摘要" in messages[1].content
        assert messages[2].content == "current"


def test_extract_facts_merges_json():
    from memory.conversation_memory import ConversationMemoryManager

    payload = (
        '{"user_profile": {"name": "张三", "language": "zh", "preferences": ["偏好简洁"]}, '
        '"task_state": {"current_goal": "分析茅台", "repo": "600519", '
        '"constraints": [], "decisions": [], "open_questions": []}}'
    )
    manager = ConversationMemoryManager(
        llm=_FakeLLM(payload),
        config={"memory_facts_enabled": True},
    )
    out = manager.extract_facts(
        None,
        summary_text="- 任务：分析茅台基本面",
        old_messages_excerpt=[],
    )
    assert out["user_profile"]["name"] == "张三"
    assert out["task_state"]["repo"] == "600519"
    txt = manager.format_memory_facts_text(out)
    assert "张三" in txt
    assert "600519" in txt

