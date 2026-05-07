"""长对话记忆管理：最近窗口 + 滚动摘要（Phase 1）。"""

from __future__ import annotations

from copy import deepcopy
from typing import Any

from langchain_core.messages import HumanMessage

from utils.log import logger


DEFAULT_SUMMARY_STATE = {
    "summary_text": "",
    "covered_message_count": 0,
    "last_update_turn": 0,
}


class ConversationMemoryManager:
    """管理长对话的摘要记忆。

    Phase 1 仅实现：
    1. 最近窗口 recent window
    2. 滚动摘要 rolling summary
    """

    def __init__(self, llm=None, config: dict[str, Any] | None = None):
        self.llm = llm
        self.config = config or {}
        self.recent_turns = int(self.config.get("recent_turns", 6))
        self.summary_trigger_turns = int(self.config.get("summary_trigger_turns", 12))
        self.summary_increment_turns = int(self.config.get("summary_increment_turns", 4))
        self.max_history_tokens_before_summary = int(
            self.config.get("max_history_tokens_before_summary", 3000)
        )
        self.summary_max_chars = int(self.config.get("summary_max_chars", 1200))

    def init_summary_state(self, state: dict[str, Any] | None = None) -> dict[str, Any]:
        base = deepcopy(DEFAULT_SUMMARY_STATE)
        if state:
            base.update(state)
        return base

    def normalize_history(self, conversation_history: list[dict] | None) -> list[dict[str, str]]:
        if not conversation_history:
            return []
        out: list[dict[str, str]] = []
        for item in conversation_history:
            role = item.get("role")
            content = (item.get("content") or "").strip()
            if role not in {"user", "assistant"} or not content:
                continue
            out.append({"role": role, "content": content})
        return out

    def estimate_tokens(self, conversation_history: list[dict] | None) -> int:
        history = self.normalize_history(conversation_history)
        total_chars = sum(len(item["content"]) for item in history)
        return total_chars // 2

    def estimate_turns(self, conversation_history: list[dict] | None) -> int:
        history = self.normalize_history(conversation_history)
        user_count = sum(1 for item in history if item["role"] == "user")
        if user_count:
            return user_count
        return len(history) // 2

    def get_recent_messages(self, conversation_history: list[dict] | None) -> list[dict[str, str]]:
        history = self.normalize_history(conversation_history)
        if self.recent_turns <= 0:
            return history
        limit = self.recent_turns * 2
        return history[-limit:] if len(history) > limit else history

    def get_messages_for_summary(self, conversation_history: list[dict] | None) -> list[dict[str, str]]:
        history = self.normalize_history(conversation_history)
        if self.recent_turns <= 0:
            return []
        limit = self.recent_turns * 2
        if len(history) <= limit:
            return []
        return history[:-limit]

    def should_compact(
        self,
        conversation_history: list[dict] | None,
        summary_state: dict[str, Any] | None,
    ) -> bool:
        history = self.normalize_history(conversation_history)
        summary_state = self.init_summary_state(summary_state)
        if not history:
            return False

        old_messages = self.get_messages_for_summary(history)
        if not old_messages:
            return False

        total_turns = self.estimate_turns(history)
        estimated_tokens = self.estimate_tokens(history)
        covered = min(summary_state.get("covered_message_count", 0), len(old_messages))
        unsummarized = old_messages[covered:]
        if not unsummarized:
            return False

        has_summary = bool((summary_state.get("summary_text") or "").strip())
        if not has_summary:
            return (
                total_turns >= self.summary_trigger_turns
                or estimated_tokens >= self.max_history_tokens_before_summary
            )

        unsummarized_turns = self.estimate_turns(unsummarized)
        return (
            unsummarized_turns >= self.summary_increment_turns
            or estimated_tokens >= self.max_history_tokens_before_summary
        )

    def update_summary(
        self,
        conversation_history: list[dict] | None,
        summary_state: dict[str, Any] | None,
    ) -> dict[str, Any]:
        history = self.normalize_history(conversation_history)
        summary_state = self.init_summary_state(summary_state)
        old_messages = self.get_messages_for_summary(history)
        if not old_messages:
            return summary_state

        covered = min(summary_state.get("covered_message_count", 0), len(old_messages))
        new_old_messages = old_messages[covered:]
        if not new_old_messages:
            return summary_state

        previous_summary = (summary_state.get("summary_text") or "").strip()
        summary_text = self._summarize(previous_summary, new_old_messages)

        updated = self.init_summary_state(summary_state)
        updated["summary_text"] = summary_text
        updated["covered_message_count"] = len(old_messages)
        updated["last_update_turn"] = self.estimate_turns(history)
        logger.info(
            "长对话摘要已更新：覆盖 %d 条历史消息，摘要长度 %d 字符",
            updated["covered_message_count"],
            len(summary_text),
        )
        return updated

    def build_summary_message(self, summary_state: dict[str, Any] | None) -> HumanMessage | None:
        summary_state = self.init_summary_state(summary_state)
        summary_text = (summary_state.get("summary_text") or "").strip()
        if not summary_text:
            return None
        return HumanMessage(
            content=(
                "以下是当前会话中较早历史的压缩摘要，请将其视为延续当前对话的重要背景：\n"
                f"【历史摘要】\n{summary_text}"
            )
        )

    def _summarize(self, previous_summary: str, new_old_messages: list[dict[str, str]]) -> str:
        history_text = self._render_history(new_old_messages)
        if self.llm is not None:
            prompt = self._build_summary_prompt(previous_summary, history_text)
            try:
                response = self.llm.invoke(prompt)
                text = getattr(response, "content", None)
                summary = text if isinstance(text, str) else str(response)
                summary = summary.strip()
                if summary:
                    return summary[: self.summary_max_chars]
            except Exception as e:
                logger.warning("长对话摘要生成失败，回退到规则摘要: %s", e)
        return self._fallback_summary(previous_summary, new_old_messages)

    def _build_summary_prompt(self, previous_summary: str, history_text: str) -> str:
        previous_block = previous_summary or "（暂无历史摘要）"
        return f"""请将以下历史对话压缩为后续多轮对话可复用的记忆摘要。

保留：
1. 用户偏好与长期设定
2. 当前任务目标
3. 已确认的约束条件
4. 已做出的关键决策
5. 尚未解决的问题
6. 对未来回答仍然重要的信息

不要保留：
1. 寒暄
2. 重复表达
3. 无关闲聊
4. 已解决且不再需要的局部细节

已有摘要：
{previous_block}

新增历史：
{history_text}

请输出简洁、结构化的要点摘要，总长度控制在 {self.summary_max_chars} 字符以内。"""

    def _render_history(self, messages: list[dict[str, str]]) -> str:
        lines = []
        for item in messages:
            role = "用户" if item["role"] == "user" else "助手"
            lines.append(f"{role}: {item['content']}")
        return "\n".join(lines)

    def _fallback_summary(self, previous_summary: str, new_old_messages: list[dict[str, str]]) -> str:
        lines: list[str] = []
        if previous_summary:
            lines.append("已有摘要：")
            lines.append(previous_summary)
        lines.append("新增历史要点：")
        for item in new_old_messages[-8:]:
            role = "用户" if item["role"] == "user" else "助手"
            content = item["content"].replace("\n", " ").strip()
            if len(content) > 120:
                content = content[:120] + "..."
            lines.append(f"- {role}: {content}")
        return "\n".join(lines)[: self.summary_max_chars]

