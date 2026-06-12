"""Intent classifier tests.

Unit (deterministic, mocked LLM):  uv run pytest evals/test_intent_classifier.py -m unit -v
LLM eval (real LLM calls):          uv run pytest evals/test_intent_classifier.py -m llm_eval -v
"""
import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from langchain_core.messages import AIMessage, HumanMessage
from evals.conftest import skip_if_no_openrouter

_VALID_INTENTS = {"general", "risk_analysis", "score_portfolio", "full_analysis"}


def _make_state(
    messages: list,
    intent: str | None = None,
    intake_complete: bool = False,
) -> dict:
    return {
        "messages": messages,
        "session_id": "test",
        "intake_complete": intake_complete,
        "user_profile": None,
        "documents_uploaded": [],
        "documents_extracted": False,
        "risk_metrics": None,
        "allocation_result": None,
        "scoring_result": None,
        "iteration_count": 0,
        "error": None,
        "intent": intent,
        "advisor_report_generated": False,
    }


def _fake_llm(response_text: str):
    """Mock LLM that returns a fixed plaintext response."""
    resp = MagicMock()
    resp.content = response_text
    resp.tool_calls = []
    mock = MagicMock()
    mock.ainvoke = AsyncMock(return_value=resp)
    return mock


# ---------------------------------------------------------------------------
# Unit tests — mocked LLM, no network
# ---------------------------------------------------------------------------

@pytest.mark.unit
@pytest.mark.asyncio
async def test_intent_classifier_returns_dict_with_intent_key():
    """intent_classifier_node always returns a dict with an 'intent' key."""
    from app.agents.intent_classifier import intent_classifier_node

    state = _make_state([HumanMessage(content="What is an ETF?")])
    with patch("app.agents.intent_classifier.get_chat_model", return_value=_fake_llm("general")):
        result = await intent_classifier_node(state)

    assert isinstance(result, dict)
    assert "intent" in result


@pytest.mark.unit
@pytest.mark.asyncio
async def test_intent_classifier_returns_valid_intent():
    """Classifier maps LLM output to one of the 4 valid intents."""
    from app.agents.intent_classifier import intent_classifier_node

    state = _make_state([HumanMessage(content="Build me an investment plan")])
    with patch("app.agents.intent_classifier.get_chat_model", return_value=_fake_llm("full_analysis")):
        result = await intent_classifier_node(state)

    assert result["intent"] in _VALID_INTENTS


@pytest.mark.unit
@pytest.mark.asyncio
@pytest.mark.parametrize("raw_response,expected_intent", [
    ("general", "general"),
    ("risk_analysis", "risk_analysis"),
    ("score_portfolio", "score_portfolio"),
    ("full_analysis", "full_analysis"),
    ("GENERAL", "general"),                # uppercase → normalised
    ("  risk_analysis  ", "risk_analysis"),  # whitespace → stripped
    ("unknown_garbage", "full_analysis"),    # fallback → full_analysis
    ("", "full_analysis"),                   # empty → fallback
])
async def test_intent_classifier_handles_all_llm_outputs(raw_response, expected_intent):
    """Classifier correctly maps or falls back for all LLM response variants."""
    from app.agents.intent_classifier import intent_classifier_node

    state = _make_state([HumanMessage(content="test query")])
    with patch("app.agents.intent_classifier.get_chat_model", return_value=_fake_llm(raw_response)):
        result = await intent_classifier_node(state)

    assert result.get("intent") == expected_intent, (
        f"LLM returned {raw_response!r}, expected intent={expected_intent}, got={result.get('intent')}"
    )


@pytest.mark.unit
@pytest.mark.asyncio
async def test_intent_classifier_skips_when_last_message_is_ai():
    """When last message is AIMessage (not HumanMessage), classifier returns empty dict."""
    from app.agents.intent_classifier import intent_classifier_node

    state = _make_state([
        HumanMessage(content="How risky is my portfolio?"),
        AIMessage(content="What is your investment horizon?", name="intake"),
    ])
    with patch("app.agents.intent_classifier.get_chat_model") as mock_llm:
        result = await intent_classifier_node(state)

    mock_llm.assert_not_called()
    assert result == {}


@pytest.mark.unit
@pytest.mark.asyncio
async def test_intent_classifier_skips_reclassification_during_intake():
    """During intake flow (intake not done, intake message exists, intent already set),
    classifier must NOT reclassify — it returns empty dict preserving original intent."""
    from app.agents.intent_classifier import intent_classifier_node

    state = _make_state(
        messages=[
            HumanMessage(content="I want to invest $50,000"),
            AIMessage(content="What is your risk tolerance?", name="intake"),
            HumanMessage(content="moderate risk"),
        ],
        intent="full_analysis",
        intake_complete=False,
    )
    with patch("app.agents.intent_classifier.get_chat_model") as mock_llm:
        result = await intent_classifier_node(state)

    mock_llm.assert_not_called()
    assert result == {}


@pytest.mark.unit
@pytest.mark.asyncio
async def test_intent_classifier_reclassifies_after_intake_complete():
    """Once intake_complete is True, classifier runs normally for new HumanMessages."""
    from app.agents.intent_classifier import intent_classifier_node

    state = _make_state(
        messages=[
            HumanMessage(content="Score my current portfolio"),
        ],
        intent="full_analysis",
        intake_complete=True,
    )
    with patch("app.agents.intent_classifier.get_chat_model", return_value=_fake_llm("score_portfolio")):
        result = await intent_classifier_node(state)

    assert result.get("intent") == "score_portfolio"


@pytest.mark.unit
@pytest.mark.asyncio
async def test_intent_classifier_no_messages_returns_empty():
    """With no messages, classifier returns empty dict without calling LLM."""
    from app.agents.intent_classifier import intent_classifier_node

    state = _make_state(messages=[])
    with patch("app.agents.intent_classifier.get_chat_model") as mock_llm:
        result = await intent_classifier_node(state)

    mock_llm.assert_not_called()
    assert result == {}


# ---------------------------------------------------------------------------
# LLM eval tests — real LLM calls
# ---------------------------------------------------------------------------

@pytest.mark.llm_eval
@pytest.mark.asyncio
async def test_intent_classifies_etf_question_as_general(langfuse_scorer):
    """'What is an ETF?' → 'general' (educational, no personal data needed)."""
    skip_if_no_openrouter()
    from app.agents.intent_classifier import intent_classifier_node

    state = _make_state([HumanMessage(content="What is an ETF and how does it work?")])
    result = await intent_classifier_node(state)
    intent = result.get("intent")

    if langfuse_scorer:
        try:
            langfuse_scorer.create_score(
                name="intent_classifier.accuracy",
                value=1.0 if intent == "general" else 0.0,
                data_type="NUMERIC",
                comment=f"'What is an ETF?' → expected=general, got={intent}",
            )
        except Exception:
            pass

    assert intent == "general", f"Expected 'general', got {intent!r}"


@pytest.mark.llm_eval
@pytest.mark.asyncio
async def test_intent_classifies_risk_question_as_risk_analysis(langfuse_scorer):
    """'How risky is my portfolio?' → 'risk_analysis'."""
    skip_if_no_openrouter()
    from app.agents.intent_classifier import intent_classifier_node

    state = _make_state([HumanMessage(content="How risky is my portfolio? What is my Sharpe ratio?")])
    result = await intent_classifier_node(state)
    intent = result.get("intent")

    if langfuse_scorer:
        try:
            langfuse_scorer.create_score(
                name="intent_classifier.accuracy",
                value=1.0 if intent == "risk_analysis" else 0.0,
                data_type="NUMERIC",
                comment=f"Risk question → expected=risk_analysis, got={intent}",
            )
        except Exception:
            pass

    assert intent == "risk_analysis", f"Expected 'risk_analysis', got {intent!r}"


@pytest.mark.llm_eval
@pytest.mark.asyncio
async def test_intent_classifies_full_plan_as_full_analysis(langfuse_scorer):
    """'Build me a complete investment plan with allocation' → 'full_analysis'."""
    skip_if_no_openrouter()
    from app.agents.intent_classifier import intent_classifier_node

    state = _make_state([HumanMessage(content="Build me a complete investment plan with recommended allocation")])
    result = await intent_classifier_node(state)
    intent = result.get("intent")

    if langfuse_scorer:
        try:
            langfuse_scorer.create_score(
                name="intent_classifier.accuracy",
                value=1.0 if intent == "full_analysis" else 0.0,
                data_type="NUMERIC",
                comment=f"Plan request → expected=full_analysis, got={intent}",
            )
        except Exception:
            pass

    assert intent == "full_analysis", f"Expected 'full_analysis', got {intent!r}"


@pytest.mark.llm_eval
@pytest.mark.asyncio
async def test_intent_classifies_score_request_as_score_portfolio(langfuse_scorer):
    """'Can you score and grade my current portfolio?' → 'score_portfolio'."""
    skip_if_no_openrouter()
    from app.agents.intent_classifier import intent_classifier_node

    state = _make_state([HumanMessage(content="Can you score and grade my current portfolio out of 100?")])
    result = await intent_classifier_node(state)
    intent = result.get("intent")

    if langfuse_scorer:
        try:
            langfuse_scorer.create_score(
                name="intent_classifier.accuracy",
                value=1.0 if intent == "score_portfolio" else 0.0,
                data_type="NUMERIC",
                comment=f"Score request → expected=score_portfolio, got={intent}",
            )
        except Exception:
            pass

    assert intent == "score_portfolio", f"Expected 'score_portfolio', got {intent!r}"


@pytest.mark.llm_eval
@pytest.mark.asyncio
async def test_intent_classifies_rebalancing_as_full_or_score(langfuse_scorer):
    """'Should I rebalance my portfolio?' is ambiguous — accepts full_analysis or score_portfolio."""
    skip_if_no_openrouter()
    from app.agents.intent_classifier import intent_classifier_node

    state = _make_state([HumanMessage(content="Should I rebalance my portfolio?")])
    result = await intent_classifier_node(state)
    intent = result.get("intent")

    acceptable = {"full_analysis", "score_portfolio"}
    if langfuse_scorer:
        try:
            langfuse_scorer.create_score(
                name="intent_classifier.accuracy",
                value=1.0 if intent in acceptable else 0.0,
                data_type="NUMERIC",
                comment=f"Rebalancing → expected one of {acceptable}, got={intent}",
            )
        except Exception:
            pass

    assert intent in acceptable, (
        f"Expected one of {acceptable} for rebalancing query, got {intent!r}"
    )
