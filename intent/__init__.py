"""Intent Guard：规则意图分类与任务契约。"""

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

__all__ = [
    "IntentContract",
    "IntentTag",
    "Violation",
    "classify_intent",
    "format_contract_message",
    "format_correction_hint",
    "reply_has_unverified_numbers",
    "validate",
    "validation_status_label",
]
