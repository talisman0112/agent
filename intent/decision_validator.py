"""Intent Guard P1：对照任务契约校验 Agent 行为。"""

from __future__ import annotations

import re
from enum import Enum
from typing import Any

from intent.intent_router import IntentContract, IntentTag


class Violation(str, Enum):
    MISSING_REQUIRED_TOOL = "MISSING_REQUIRED_TOOL"
    FORBIDDEN_TOOL_USED = "FORBIDDEN_TOOL_USED"
    NUMBERS_WITHOUT_TOOLS = "NUMBERS_WITHOUT_TOOLS"
    STALE_SOURCE_ONLY = "STALE_SOURCE_ONLY"


# R3 / R4：视为已满足「实时/行情」类要求的工具
_DATA_TOOLS = frozenset(
    {
        "get_stock_quote",
        "get_stock_basics",
        "get_stock_kline",
        "web_search",
        "hybrid_search",
        "hybrid_summarize",
        "get_market_datetime",
    }
)

_LOCAL_ONLY_TOOLS = frozenset({"rag_summarize", "rag_retrieve"})

_INTENTS_STRICT_NUMBERS = frozenset(
    {
        IntentTag.QUOTE_SNAPSHOT,
        IntentTag.INVESTMENT_ADVICE,
        IntentTag.REALTIME_NEWS,
    }
)

# 保守：疑似未经验证的具体数字表述
_NUMBER_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(r"\d+(?:\.\d+)?\s*(?:元|万元|亿元|%|倍|点)", re.I),
    re.compile(r"(?:股价|现价|市值|市盈率|市净率|PE|PB)\s*[:：]?\s*\d", re.I),
    re.compile(r"(?:涨|跌)\s*(?:了|幅)?\s*\d+(?:\.\d+)?%?", re.I),
    re.compile(r"\d{4,6}\s*(?:元|点)", re.I),
)


def _tool_names_from_calls(tool_calls: list[dict[str, Any]] | None) -> set[str]:
    names: set[str] = set()
    for c in tool_calls or []:
        n = (c.get("name") or "").strip()
        if n and n != "?":
            names.add(n)
    return names


def reply_has_unverified_numbers(reply: str) -> bool:
    text = (reply or "").strip()
    if not text or len(text) < 8:
        return False
    return any(p.search(text) for p in _NUMBER_PATTERNS)


def validate(
    contract: IntentContract,
    tool_calls: list[dict[str, Any]] | None,
    reply: str,
    *,
    strict_numbers: bool = True,
) -> list[Violation]:
    """对照 IntentContract 检查工具调用与回复，返回违规列表（空表示通过）。"""
    violations: list[Violation] = []
    called = _tool_names_from_calls(tool_calls)
    required = set(contract.required_tools or [])
    forbidden = set(contract.forbidden_tools or [])
    primary = contract.primary

    if required and not called.intersection(required):
        violations.append(Violation.MISSING_REQUIRED_TOOL)

    if forbidden and called.intersection(forbidden):
        violations.append(Violation.FORBIDDEN_TOOL_USED)

    if strict_numbers and primary in _INTENTS_STRICT_NUMBERS:
        if reply_has_unverified_numbers(reply) and not called.intersection(_DATA_TOOLS):
            violations.append(Violation.NUMBERS_WITHOUT_TOOLS)

    if primary == IntentTag.REALTIME_NEWS and called and called <= _LOCAL_ONLY_TOOLS:
        violations.append(Violation.STALE_SOURCE_ONLY)

    return violations


_VIOLATION_HINTS: dict[Violation, str] = {
    Violation.MISSING_REQUIRED_TOOL: (
        "上一轮未调用任务契约要求的工具。请立即调用契约中的「建议至少调用」列表里的工具之一，"
        "并基于工具返回重新组织回答。"
    ),
    Violation.FORBIDDEN_TOOL_USED: (
        "上一轮使用了契约不建议单独依赖的工具。请改用契约允许的工具组合后重新作答。"
    ),
    Violation.NUMBERS_WITHOUT_TOOLS: (
        "回答中出现了具体行情/估值类数字，但未调用行情、Web 或 hybrid 工具。"
        "请先调用相关工具获取数据，再输出数字；若工具失败须如实说明，勿编造。"
    ),
    Violation.STALE_SOURCE_ONLY: (
        "时效性问题仅使用了本地 RAG，未检索 Web。请调用 web_search 或 hybrid_search / hybrid_summarize 后重新作答。"
    ),
}


def format_correction_hint(
    violations: list[Violation],
    contract: IntentContract,
) -> str:
    """生成纠正轮注入给模型的提示（不含用户可见历史污染）。"""
    codes = ", ".join(v.value for v in violations)
    req = "、".join(f"`{t}`" for t in contract.required_tools) or "（见契约）"
    parts = [
        f"上一轮未满足任务契约，违规项：{codes}。",
        f"意图类型：{contract.primary.value}。",
        f"请至少调用以下工具之一：{req}。",
    ]
    for v in violations:
        parts.append(_VIOLATION_HINTS.get(v, ""))
    parts.append("请基于最新工具返回完整重答用户问题，勿重复未验证的数字。")
    return "\n".join(p for p in parts if p)


def validation_status_label(
    violations: list[Violation],
    *,
    corrected: bool = False,
) -> str:
    if corrected and not violations:
        return "纠正后通过"
    if corrected and violations:
        return f"已纠正仍存疑 ({', '.join(v.value for v in violations)})"
    if violations:
        return f"未通过 ({', '.join(v.value for v in violations)})"
    return "通过"
