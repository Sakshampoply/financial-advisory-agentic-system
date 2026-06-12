import re

from langchain_core.messages import HumanMessage

from app.agents.state import GraphState
from app.observability.langfuse_setup import traced_node

# --- PII patterns -------------------------------------------------------

_SSN = re.compile(r"\b\d{3}-\d{2}-\d{4}\b")
_PHONE = re.compile(r"\b(?:\+?1[-.\s]?)?\(?\d{3}\)?[-.\s]\d{3}[-.\s]\d{4}\b")
_EMAIL = re.compile(r"\b[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}\b")
_CREDIT_CARD = re.compile(r"\b(?:\d[ -]?){13,16}\b")

# --- Injection patterns -------------------------------------------------

_INJECTION_PATTERNS = [
    re.compile(p, re.IGNORECASE)
    for p in [
        r"ignore\s+(all\s+)?(previous|prior|above)\s+instructions?",
        r"disregard\s+(all\s+)?(previous|prior|above)\s+instructions?",
        r"forget\s+(all\s+)?(previous|prior|above)\s+instructions?",
        r"you\s+are\s+now\s+(?:a|an)\s+",
        r"act\s+as\s+(?:if\s+you\s+are\s+)?(?:a|an)\s+",
        r"pretend\s+(?:you\s+are|to\s+be)\s+",
        r"your\s+new\s+instructions?\s+are",
        r"system\s*:\s*you\s+are",
        r"<\s*system\s*>",
        r"\[\s*system\s*\]",
        r"###\s*instruction",
    ]
]


def mask_pii(text: str) -> str:
    text = _SSN.sub("[SSN REDACTED]", text)
    text = _PHONE.sub("[PHONE REDACTED]", text)
    text = _EMAIL.sub("[EMAIL REDACTED]", text)
    text = _CREDIT_CARD.sub("[CC REDACTED]", text)
    return text


def is_injection(text: str) -> bool:
    return any(p.search(text) for p in _INJECTION_PATTERNS)


@traced_node("guardrail_input")
async def guardrail_input_node(state: GraphState) -> dict:
    messages = list(state.get("messages") or [])
    if not messages:
        return {}

    last = messages[-1]
    if not isinstance(last, HumanMessage):
        return {}

    original = last.content if isinstance(last.content, str) else str(last.content)

    if is_injection(original):
        # Signal error; error_handler_node will produce the user-facing message
        return {"error": "injection_detected"}

    masked = mask_pii(original)
    if masked == original:
        return {}

    # Replace last message in-place by returning same id — add_messages updates by id
    return {"messages": [HumanMessage(content=masked, id=last.id)]}
