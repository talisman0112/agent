"""Intent Guard P0：规则 classify_intent 与契约文案（无需 API Key）。"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from intent.decision_validator import (
    Violation,
    format_correction_hint,
    reply_has_unverified_numbers,
    validate,
    validation_status_label,
)
from intent.intent_router import (
    IntentContract,
    IntentTag,
    classify_intent,
    format_contract_message,
)


def test_quote_snapshot_intent():
    c = classify_intent("贵州茅台现在多少钱")
    assert c.primary == IntentTag.QUOTE_SNAPSHOT
    assert "get_stock_quote" in c.required_tools


def test_realtime_news_intent():
    c = classify_intent("宁德时代最新公告有哪些")
    assert c.primary == IntentTag.REALTIME_NEWS
    assert "web_search" in c.required_tools or "hybrid_summarize" in c.required_tools


def test_local_kb_intent():
    c = classify_intent("什么是 ROE？请用杜邦分析拆解")
    assert c.primary == IntentTag.LOCAL_KB_QA
    assert "rag_summarize" in c.required_tools


def test_chitchat_intent():
    c = classify_intent("你好")
    assert c.primary == IntentTag.CHITCHAT
    assert c.required_tools == []


def test_report_mode_intent():
    c = classify_intent("写一份英伟达个股速评", report_mode=True)
    assert c.primary == IntentTag.REPORT_WRITE
    assert "hybrid_summarize" in c.required_tools


def test_format_contract_message_non_empty():
    c = classify_intent("600519 现价")
    msg = format_contract_message(c)
    assert "本轮任务契约" in msg
    assert "QUOTE_SNAPSHOT" in msg
    assert "get_stock_quote" in msg


def test_multi_turn_adds_constraint():
    c = classify_intent("他现在的股价是多少", history=[{"role": "user", "content": "聊宁德时代"}])
    assert c.primary == IntentTag.QUOTE_SNAPSHOT
    assert any("指代" in x for x in c.constraints)


def test_validate_missing_required_tool():
    c = classify_intent("贵州茅台现在多少钱")
    v = validate(c, [], "茅台大概一千八左右。")
    assert Violation.MISSING_REQUIRED_TOOL in v


def test_validate_numbers_without_tools():
    c = classify_intent("贵州茅台现在多少钱")
    v = validate(
        c,
        [],
        "贵州茅台现价约为 1688.00 元，市盈率 28 倍。",
        strict_numbers=True,
    )
    assert Violation.NUMBERS_WITHOUT_TOOLS in v


def test_validate_pass_with_quote_tool():
    c = classify_intent("贵州茅台现在多少钱")
    calls = [{"name": "get_stock_quote", "content_preview": "..."}]
    v = validate(c, calls, "根据行情工具，现价 1688 元。")
    assert v == []


def test_validate_stale_source_only():
    c = classify_intent("宁德时代最新公告有哪些")
    calls = [{"name": "rag_summarize", "content_preview": "..."}]
    v = validate(c, calls, "根据本地资料…")
    assert Violation.STALE_SOURCE_ONLY in v


def test_format_correction_hint():
    c = classify_intent("贵州茅台现在多少钱")
    hint = format_correction_hint([Violation.MISSING_REQUIRED_TOOL], c)
    assert "MISSING_REQUIRED_TOOL" in hint
    assert "get_stock_quote" in hint


def test_validation_status_label():
    assert validation_status_label([]) == "通过"
    assert validation_status_label([], corrected=True) == "纠正后通过"


def test_reply_has_unverified_numbers():
    assert reply_has_unverified_numbers("PE：28 倍，市值约 2.1 万亿元")
    assert not reply_has_unverified_numbers("ROE 是净资产收益率的概念")
