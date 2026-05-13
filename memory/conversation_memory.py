"""长对话记忆管理：最近窗口 + 滚动摘要（Phase 1）+ 结构化事实（Phase 2）。"""

from __future__ import annotations

import json
import re
from copy import deepcopy
from typing import Any

from langchain_core.messages import HumanMessage

from utils.log import logger


DEFAULT_SUMMARY_STATE = {
    "summary_text": "",
    "covered_message_count": 0,
    "last_update_turn": 0,
}

DEFAULT_MEMORY_FACTS = {
    "user_profile": {
        "name": None,
        "language": None,
        "preferences": [],
    },
    "task_state": {
        "current_goal": None,
        "repo": None,
        "constraints": [],
        "decisions": [],
        "open_questions": [],
    },
}


class ConversationMemoryManager:
    """管理长对话的摘要记忆。

    Phase 1：
    1. 最近窗口 recent window
    2. 滚动摘要 rolling summary

    Phase 2：
    3. 结构化长期事实 memory_facts（摘要更新时抽取并合并）
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
        self.memory_facts_enabled = bool(self.config.get("memory_facts_enabled", True))
        self.memory_facts_max_chars = int(self.config.get("memory_facts_max_chars", 800))

    def init_summary_state(self, state: dict[str, Any] | None = None) -> dict[str, Any]:
        base = deepcopy(DEFAULT_SUMMARY_STATE)
        if state:
            base.update(state)
        return base

    def init_memory_facts(self, state: dict[str, Any] | None = None) -> dict[str, Any]:
        base: dict[str, Any] = deepcopy(DEFAULT_MEMORY_FACTS)
        if not state:
            return base
        up = state.get("user_profile") if isinstance(state.get("user_profile"), dict) else {}
        ts = state.get("task_state") if isinstance(state.get("task_state"), dict) else {}
        for k in ("name", "language"):
            if k in up and up.get(k) is not None:
                s = str(up[k]).strip()
                base["user_profile"][k] = s or None
        if isinstance(up.get("preferences"), list):
            base["user_profile"]["preferences"] = self._dedupe_str_list(
                [str(x).strip() for x in up["preferences"] if str(x).strip()][:20]
            )
        for k in ("current_goal", "repo"):
            if k in ts and ts.get(k) is not None:
                s = str(ts[k]).strip()
                base["task_state"][k] = s or None
        for key in ("constraints", "decisions", "open_questions"):
            if isinstance(ts.get(key), list):
                base["task_state"][key] = self._dedupe_str_list(
                    [str(x).strip() for x in ts[key] if str(x).strip()][:20]
                )
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

    def format_memory_facts_text(self, memory_facts: dict[str, Any] | None) -> str:
        """将结构化事实格式化为模型可读文本；无有效内容时返回空串。"""
        facts = self.init_memory_facts(memory_facts)
        if not self._facts_has_content(facts):
            return ""
        up = facts["user_profile"]
        ts = facts["task_state"]
        lines: list[str] = [
            "以下是基于当前会话整理的**长期结构化记忆**，与下文摘要互为补充，请一并参考：",
            "【长期记忆】",
        ]
        name = up.get("name")
        lang = up.get("language")
        if name:
            lines.append(f"- 用户称呼或姓名线索：{name}")
        if lang:
            lines.append(f"- 输出语言偏好：{lang}")
        prefs = up.get("preferences") or []
        if prefs:
            lines.append("- 其他偏好：")
            for p in prefs[:12]:
                lines.append(f"  - {p}")
        goal = ts.get("current_goal")
        repo = ts.get("repo")
        if goal:
            lines.append(f"- 当前任务/目标：{goal}")
        if repo:
            lines.append(f"- 关注标的/代码/范围：{repo}")
        for label, key in (
            ("约束条件", "constraints"),
            ("已确认决策", "decisions"),
            ("未解决问题", "open_questions"),
        ):
            items = ts.get(key) or []
            if items:
                lines.append(f"- {label}：")
                for it in items[:12]:
                    lines.append(f"  - {it}")
        text = "\n".join(lines).strip()
        if len(text) > self.memory_facts_max_chars:
            return text[: self.memory_facts_max_chars] + "…"
        return text

    def build_memory_facts_message(self, memory_facts: dict[str, Any] | None) -> HumanMessage | None:
        text = self.format_memory_facts_text(memory_facts)
        if not text:
            return None
        return HumanMessage(content=text)

    def extract_facts(
        self,
        memory_facts: dict[str, Any] | None,
        *,
        summary_text: str,
        old_messages_excerpt: list[dict[str, str]] | None = None,
    ) -> dict[str, Any]:
        """根据滚动摘要（及可选旧消息摘录）抽取/合并结构化长期事实。"""
        if not self.memory_facts_enabled:
            return self.init_memory_facts(memory_facts)

        summary_text = (summary_text or "").strip()
        old_messages_excerpt = old_messages_excerpt or []
        if not summary_text and not old_messages_excerpt:
            return self.init_memory_facts(memory_facts)

        previous = self.init_memory_facts(memory_facts)
        if self.llm is None:
            return previous

        excerpt = self._render_history(old_messages_excerpt[-50:])
        if len(excerpt) > 6000:
            excerpt = excerpt[-6000:]

        prompt = self._build_extract_facts_prompt(previous, summary_text, excerpt)
        try:
            response = self.llm.invoke(prompt)
            raw = getattr(response, "content", None)
            raw = raw if isinstance(raw, str) else str(response)
            parsed = self._parse_json_object(raw)
            if parsed:
                merged = self._validate_and_merge_facts(previous, parsed)
                logger.info("结构化长期记忆已更新")
                return merged
        except Exception as e:
            logger.warning("结构化记忆抽取失败，保留上一版: %s", e)
        return previous

    def _build_extract_facts_prompt(
        self,
        previous_facts: dict[str, Any],
        summary_text: str,
        old_excerpt: str,
    ) -> str:
        prev_json = json.dumps(previous_facts, ensure_ascii=False, indent=2)
        excerpt_block = old_excerpt.strip() or "（无单独摘录，请仅依据摘要与已有事实推断）"
        return f"""你是会话长期记忆整理助手。请在「已有结构化记忆」基础上，结合「滚动摘要」与「较早对话摘录」更新事实。

要求：
1. 输出**仅包含**一个 JSON 对象，不要 Markdown、不要代码围栏以外的文字。
2. JSON 必须严格符合以下结构（键名一致，缺失的列表用 []，缺失的标量用 null）：
{{
  "user_profile": {{
    "name": null,
    "language": null,
    "preferences": []
  }},
  "task_state": {{
    "current_goal": null,
    "repo": null,
    "constraints": [],
    "decisions": [],
    "open_questions": []
  }}
}}
3. 与已有事实矛盾时，以**更新后的对话与摘要**为准。
4. `repo` 可填股票代码、标的名称、或当前关注范围（投研场景）。
5. 列表型字段每项为简短字符串，各自最多保留 12 条，去重。

已有结构化记忆（JSON）：
{prev_json}

当前滚动摘要：
{summary_text or "（空）"}

较早对话摘录（与摘要窗口可能重叠，供核对）：
{excerpt_block}

请输出合并后的完整 JSON。"""

    def _parse_json_object(self, text: str) -> dict[str, Any] | None:
        text = (text or "").strip()
        if not text:
            return None
        m = re.search(r"```(?:json)?\s*([\s\S]*?)```", text, re.I)
        if m:
            text = m.group(1).strip()
        try:
            obj = json.loads(text)
            return obj if isinstance(obj, dict) else None
        except json.JSONDecodeError:
            pass
        i = text.find("{")
        j = text.rfind("}")
        if i >= 0 and j > i:
            try:
                obj = json.loads(text[i : j + 1])
                return obj if isinstance(obj, dict) else None
            except json.JSONDecodeError:
                return None
        return None

    def _validate_and_merge_facts(
        self,
        previous: dict[str, Any],
        parsed: dict[str, Any],
    ) -> dict[str, Any]:
        base = self.init_memory_facts(previous)
        up = parsed.get("user_profile") if isinstance(parsed.get("user_profile"), dict) else {}
        ts = parsed.get("task_state") if isinstance(parsed.get("task_state"), dict) else {}
        for k in ("name", "language"):
            v = up.get(k)
            if v is not None and str(v).strip():
                base["user_profile"][k] = str(v).strip()
        if isinstance(up.get("preferences"), list):
            merged = self._dedupe_str_list(
                (base["user_profile"].get("preferences") or [])
                + [str(x).strip() for x in up["preferences"] if str(x).strip()]
            )[:20]
            base["user_profile"]["preferences"] = merged
        for k in ("current_goal", "repo"):
            v = ts.get(k)
            if v is not None and str(v).strip():
                base["task_state"][k] = str(v).strip()
        for key in ("constraints", "decisions", "open_questions"):
            if isinstance(ts.get(key), list):
                merged = self._dedupe_str_list(
                    (base["task_state"].get(key) or [])
                    + [str(x).strip() for x in ts[key] if str(x).strip()]
                )[:20]
                base["task_state"][key] = merged
        return base

    def _facts_has_content(self, facts: dict[str, Any]) -> bool:
        up = facts.get("user_profile") or {}
        ts = facts.get("task_state") or {}
        if up.get("name") or up.get("language"):
            return True
        if isinstance(up.get("preferences"), list) and up["preferences"]:
            return True
        if ts.get("current_goal") or ts.get("repo"):
            return True
        for key in ("constraints", "decisions", "open_questions"):
            if isinstance(ts.get(key), list) and ts[key]:
                return True
        return False

    @staticmethod
    def _dedupe_str_list(items: list[str]) -> list[str]:
        seen: set[str] = set()
        out: list[str] = []
        for x in items:
            if x not in seen:
                seen.add(x)
                out.append(x)
        return out

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

