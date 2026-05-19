"""规则型意图分类与 IntentContract 生成（Intent Guard P0）。"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class IntentTag(str, Enum):
    LOCAL_KB_QA = "LOCAL_KB_QA"
    REALTIME_NEWS = "REALTIME_NEWS"
    QUOTE_SNAPSHOT = "QUOTE_SNAPSHOT"
    FIN_CALC = "FIN_CALC"
    MARKET_TIME = "MARKET_TIME"
    INVESTMENT_ADVICE = "INVESTMENT_ADVICE"
    REPORT_WRITE = "REPORT_WRITE"
    CHITCHAT = "CHITCHAT"
    MULTI_TURN_REF = "MULTI_TURN_REF"


# 多标签命中时按此顺序取主意图（靠前优先）
_INTENT_PRIORITY: tuple[IntentTag, ...] = (
    IntentTag.QUOTE_SNAPSHOT,
    IntentTag.INVESTMENT_ADVICE,
    IntentTag.REALTIME_NEWS,
    IntentTag.MARKET_TIME,
    IntentTag.FIN_CALC,
    IntentTag.LOCAL_KB_QA,
    IntentTag.CHITCHAT,
)

_PATTERNS: dict[IntentTag, re.Pattern[str]] = {
    IntentTag.QUOTE_SNAPSHOT: re.compile(
        r"(股价|现价|多少钱|涨跌|涨幅|跌幅|市值|市盈率|市净率|"
        r"PE\b|PB\b|报价|最新价|行情快照|换手)",
        re.I,
    ),
    IntentTag.INVESTMENT_ADVICE: re.compile(
        r"(推荐|看好|配置|潜力股|买入|卖出|选股|标的推荐|"
        r"值得买|加仓|减仓|仓位|板块推荐|主题投资)",
        re.I,
    ),
    IntentTag.REALTIME_NEWS: re.compile(
        r"(最新|今天|今日|昨日|近期|刚刚|刚才发布|"
        r"Q[1-4]|20\d{2}\s*年?(上|下)半年|本周|本月|要闻|快讯)",
        re.I,
    ),
    IntentTag.MARKET_TIME: re.compile(
        r"(开盘|收盘|交易日|几点|美东|时区|开市|休市|"
        r"America/New_York|Asia/Shanghai)",
        re.I,
    ),
    IntentTag.FIN_CALC: re.compile(
        r"(同比|环比|计算|换算|估值倍数|算术|公式|"
        r"convert_currency|compute_financial)",
        re.I,
    ),
    IntentTag.LOCAL_KB_QA: re.compile(
        r"(什么是|解释|含义|定义|年报|季报|研报|公告|"
        r"政策原文|杜邦|ROE|毛利率|术语|百科)",
        re.I,
    ),
    IntentTag.CHITCHAT: re.compile(
        r"^(你好|您好|嗨|hello|hi|在吗|你是谁|自我介绍)[\s!！?？。]*$",
        re.I,
    ),
}

_MULTI_TURN_REF = re.compile(
    r"(他|她|它|那只|这家|该公司|刚才|上文|之前说的|"
    r"前面提到的|同上|前述)",
    re.I,
)

_STOCK_CODE = re.compile(r"\b[036]\d{5}\b|\b\d{6}\b")

# 主意图 → 契约模板（required_tools：至少应调用其中任一）
_CONTRACT_BY_TAG: dict[IntentTag, dict[str, Any]] = {
    IntentTag.REPORT_WRITE: {
        "required_tools": [
            "hybrid_summarize",
            "hybrid_search",
            "get_stock_quote",
            "get_stock_basics",
            "get_stock_kline",
        ],
        "forbidden_tools": [],
        "constraints": [
            "报告模式：须结合本地语料与 Web/行情，按报告模板结构化输出。",
            "涉及估值或现价须调用行情/基本面工具，勿凭记忆编造数字。",
        ],
    },
    IntentTag.QUOTE_SNAPSHOT: {
        "required_tools": ["get_stock_quote", "get_stock_basics", "get_stock_kline"],
        "forbidden_tools": [],
        "constraints": [
            "未调用行情或基本面工具前，不得在正文中写出具体股价、PE、市值等数字。",
        ],
    },
    IntentTag.INVESTMENT_ADVICE: {
        "required_tools": [
            "get_market_datetime",
            "web_search",
            "hybrid_search",
            "hybrid_summarize",
        ],
        "forbidden_tools": [],
        "constraints": [
            "投资建议须锚定当前时点：先查市场时间，再检索近期公开信息。",
            "不可仅依赖本地静态语料完成「当下」推荐。",
        ],
    },
    IntentTag.REALTIME_NEWS: {
        "required_tools": ["web_search", "hybrid_search", "hybrid_summarize"],
        "forbidden_tools": [],
        "constraints": [
            "时效性问题须使用 Web 或 hybrid 工具；不要仅用 rag_summarize/rag_retrieve。",
        ],
    },
    IntentTag.MARKET_TIME: {
        "required_tools": ["get_market_datetime"],
        "forbidden_tools": [],
        "constraints": [],
    },
    IntentTag.FIN_CALC: {
        "required_tools": ["compute_financial_metric", "convert_currency"],
        "forbidden_tools": [],
        "constraints": ["财务比率与换算请用专用计算工具，避免心算。"],
    },
    IntentTag.LOCAL_KB_QA: {
        "required_tools": ["rag_summarize", "rag_retrieve"],
        "forbidden_tools": [],
        "constraints": [
            "构造检索 query 时尽量包含公司多种称呼（中文名/代码/英文名）。",
        ],
    },
    IntentTag.MULTI_TURN_REF: {
        "required_tools": [],
        "forbidden_tools": [],
        "constraints": [
            "用户存在指代（他/那只/刚才等），须结合对话上文或长期记忆锚定具体标的后再检索或调工具。",
        ],
    },
    IntentTag.CHITCHAT: {
        "required_tools": [],
        "forbidden_tools": [],
        "constraints": ["寒暄类问题可直接简短回答，无需调用投研工具。"],
    },
}


@dataclass
class IntentContract:
    primary: IntentTag
    confidence: float = 1.0
    source: str = "rule"
    required_tools: list[str] = field(default_factory=list)
    forbidden_tools: list[str] = field(default_factory=list)
    query_hints: list[str] = field(default_factory=list)
    constraints: list[str] = field(default_factory=list)

    def to_trace_dict(self) -> dict[str, Any]:
        return {
            "intent": self.primary.value,
            "confidence": self.confidence,
            "source": self.source,
            "required_tools": list(self.required_tools),
            "forbidden_tools": list(self.forbidden_tools),
            "query_hints": list(self.query_hints),
            "constraints": list(self.constraints),
        }


def _normalize_text(user_input: str, history: list[dict] | None) -> str:
    parts = [(user_input or "").strip()]
    if history:
        for item in history[-4:]:
            c = (item.get("content") or "").strip()
            if c:
                parts.append(c)
    return "\n".join(p for p in parts if p)


def _match_tags(text: str) -> set[IntentTag]:
    matched: set[IntentTag] = set()
    for tag, pattern in _PATTERNS.items():
        if pattern.search(text):
            matched.add(tag)
    if _STOCK_CODE.search(text) and not matched:
        matched.add(IntentTag.QUOTE_SNAPSHOT)
    return matched


def _pick_primary(matched: set[IntentTag]) -> IntentTag:
    if not matched:
        return IntentTag.LOCAL_KB_QA
    for tag in _INTENT_PRIORITY:
        if tag in matched:
            return tag
    return next(iter(matched))


def _query_hints_from_text(text: str, memory_facts: dict | None) -> list[str]:
    hints: list[str] = []
    codes = _STOCK_CODE.findall(text)
    hints.extend(codes[:3])
    if memory_facts:
        ts = memory_facts.get("task_state") if isinstance(memory_facts.get("task_state"), dict) else {}
        goal = ts.get("current_goal")
        if goal and str(goal).strip():
            hints.append(str(goal).strip()[:200])
    snippet = text.strip().replace("\n", " ")[:120]
    if snippet and snippet not in hints:
        hints.insert(0, snippet)
    seen: set[str] = set()
    out: list[str] = []
    for h in hints:
        if h and h not in seen:
            seen.add(h)
            out.append(h)
    return out[:5]


def _merge_contract(
    primary: IntentTag,
    *,
    multi_turn: bool,
    query_hints: list[str],
) -> IntentContract:
    base = _CONTRACT_BY_TAG.get(primary, _CONTRACT_BY_TAG[IntentTag.LOCAL_KB_QA])
    constraints = list(base.get("constraints") or [])
    if multi_turn and primary != IntentTag.MULTI_TURN_REF:
        ref_extra = _CONTRACT_BY_TAG[IntentTag.MULTI_TURN_REF]["constraints"]
        constraints.extend(ref_extra)
    return IntentContract(
        primary=primary,
        confidence=0.92 if primary != IntentTag.LOCAL_KB_QA else 0.75,
        source="rule",
        required_tools=list(base.get("required_tools") or []),
        forbidden_tools=list(base.get("forbidden_tools") or []),
        query_hints=query_hints,
        constraints=constraints,
    )


def classify_intent(
    user_input: str,
    history: list[dict] | None = None,
    memory_facts: dict | None = None,
    *,
    report_mode: bool = False,
) -> IntentContract:
    """根据规则将用户输入分类为主意图并生成任务契约。"""
    text = _normalize_text(user_input, history)

    if report_mode:
        return _merge_contract(
            IntentTag.REPORT_WRITE,
            multi_turn=bool(_MULTI_TURN_REF.search(text)),
            query_hints=_query_hints_from_text(text, memory_facts),
        )

    matched = _match_tags(text)
    if not matched and len((user_input or "").strip()) <= 12:
        matched.add(IntentTag.CHITCHAT)

    primary = _pick_primary(matched)
    return _merge_contract(
        primary,
        multi_turn=bool(_MULTI_TURN_REF.search(text)),
        query_hints=_query_hints_from_text(text, memory_facts),
    )


def format_contract_message(contract: IntentContract) -> str:
    """将 IntentContract 格式化为注入模型的辅助 HumanMessage 正文。"""
    req = contract.required_tools
    req_line = "、".join(f"`{t}`" for t in req) if req else "（无强制工具，按场景自选）"
    forbid = contract.forbidden_tools
    forbid_line = "、".join(f"`{t}`" for t in forbid) if forbid else "（无）"
    hints = contract.query_hints
    hints_line = "；".join(hints) if hints else "（无额外提示）"
    constraints = contract.constraints
    constraint_lines = "\n".join(f"- {c}" for c in constraints) if constraints else "- （无附加约束）"

    return (
        "【本轮任务契约·由系统根据用户意图生成，请严格遵守】\n"
        f"- 意图类型：{contract.primary.value}\n"
        f"- 置信度：{contract.confidence:.2f}（来源：{contract.source}）\n"
        f"- 建议至少调用以下工具之一：{req_line}\n"
        f"- 不建议单独依赖：{forbid_line}\n"
        f"- 检索/标的提示：{hints_line}\n"
        "- 附加约束：\n"
        f"{constraint_lines}"
    )
