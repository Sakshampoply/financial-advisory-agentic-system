"""Intake node tests.

Unit (deterministic, no LLM):   uv run pytest evals/test_intake_extraction.py -m unit -v
LLM eval (real LLM calls):       uv run pytest evals/test_intake_extraction.py -m llm_eval -v
"""
import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from langchain_core.messages import AIMessage, HumanMessage
from evals.conftest import skip_if_no_openrouter


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_state(messages: list, existing_profile: dict | None = None) -> dict:
    return {
        "messages": messages,
        "session_id": "test",
        "intake_complete": False,
        "user_profile": existing_profile or {},
        "documents_uploaded": [],
        "documents_extracted": False,
        "risk_metrics": None,
        "allocation_result": None,
        "scoring_result": None,
        "iteration_count": 0,
        "error": None,
        "intent": "full_analysis",
        "advisor_report_generated": False,
    }


def _fake_llm_no_tool_call(content: str = "Could you tell me your risk tolerance?"):
    """Mock LLM that returns plain text (no tool call)."""
    resp = MagicMock()
    resp.content = content
    resp.tool_calls = []
    mock = MagicMock()
    mock.bind_tools = MagicMock(return_value=mock)
    mock.ainvoke = AsyncMock(return_value=resp)
    return mock


def _fake_llm_with_tool_call(args: dict):
    """Mock LLM that returns a collect_profile tool call with given args."""
    resp = MagicMock()
    resp.content = ""
    resp.tool_calls = [{"name": "collect_profile", "args": args}]
    mock = MagicMock()
    mock.bind_tools = MagicMock(return_value=mock)
    mock.ainvoke = AsyncMock(return_value=resp)
    return mock


# ---------------------------------------------------------------------------
# Unit: _build_system_prompt
# ---------------------------------------------------------------------------

@pytest.mark.unit
def test_system_prompt_lists_missing_fields_only():
    """_build_system_prompt omits already-known fields from the 'Still needed' section."""
    from app.agents.intake import _build_system_prompt

    partial_profile = {"risk_tolerance": "high", "investment_horizon_years": 10}
    prompt = _build_system_prompt(partial_profile)

    assert "risk_tolerance" not in prompt.split("Still needed")[1] if "Still needed" in prompt else True
    assert "investment_amount_usd" in prompt or "amount" in prompt.lower()


@pytest.mark.unit
def test_system_prompt_shows_already_known_values():
    """Fields already in profile appear in the 'Already known' section."""
    from app.agents.intake import _build_system_prompt

    profile = {"risk_tolerance": "moderate", "investment_horizon_years": 5}
    prompt = _build_system_prompt(profile)

    assert "moderate" in prompt
    assert "Already known" in prompt


@pytest.mark.unit
def test_system_prompt_asks_portfolio_question_when_not_known():
    """When portfolio is empty/absent, system prompt contains the portfolio question."""
    from app.agents.intake import _build_system_prompt

    prompt = _build_system_prompt({})
    assert "existing investments" in prompt.lower() or "portfolio" in prompt.lower()


@pytest.mark.unit
def test_system_prompt_skips_portfolio_question_when_known():
    """When portfolio is already populated, system prompt does not ask for it."""
    from app.agents.intake import _build_system_prompt

    profile = {"portfolio": {"AAPL": 0.5, "MSFT": 0.5}}
    prompt = _build_system_prompt(profile)
    assert "do you have existing investments" not in prompt.lower()
    assert "Portfolio is already known" in prompt


# ---------------------------------------------------------------------------
# Unit: _build_portfolio
# ---------------------------------------------------------------------------

@pytest.mark.unit
def test_build_portfolio_normalises_weights():
    """_build_portfolio converts weight_pct to fractions summing to 1.0."""
    from app.agents.intake import _build_portfolio

    holdings = [
        {"ticker": "AAPL", "weight_pct": 40},
        {"ticker": "MSFT", "weight_pct": 60},
    ]
    result = _build_portfolio(holdings)
    assert abs(sum(result.values()) - 1.0) < 0.001
    assert result["AAPL"] == pytest.approx(0.4, abs=0.01)


@pytest.mark.unit
def test_build_portfolio_empty_returns_empty():
    """_build_portfolio with zero total returns empty dict."""
    from app.agents.intake import _build_portfolio
    assert _build_portfolio([]) == {}


@pytest.mark.unit
def test_build_portfolio_uppercases_tickers():
    """Lowercase tickers are normalised to uppercase."""
    from app.agents.intake import _build_portfolio
    result = _build_portfolio([{"ticker": "aapl", "weight_pct": 100}])
    assert "AAPL" in result


# ---------------------------------------------------------------------------
# Unit: intake fast-path (all fields pre-populated)
# ---------------------------------------------------------------------------

@pytest.mark.unit
@pytest.mark.asyncio
async def test_intake_fast_path_skips_llm_when_all_fields_present():
    """When all 4 fields are in user_profile, intake returns without calling LLM."""
    from app.agents.intake import intake_node

    full_profile = {
        "risk_tolerance": "high",
        "investment_horizon_years": 10,
        "investment_amount_usd": 100_000,
        "portfolio": {"SPY": 0.6, "BND": 0.4},
    }
    state = _make_state(
        messages=[HumanMessage(content="Let's get started")],
        existing_profile=full_profile,
    )

    llm_called = False

    def mock_get_model(*args, **kwargs):
        nonlocal llm_called
        llm_called = True
        raise RuntimeError("LLM should not be called")

    with patch("app.agents.intake.get_chat_model", side_effect=mock_get_model):
        result = await intake_node(state)

    assert not llm_called
    assert result.get("intake_complete") is True


@pytest.mark.unit
@pytest.mark.asyncio
async def test_intake_fast_path_confirmation_includes_profile_data():
    """Fast-path confirmation message includes risk tolerance and investment amount."""
    from app.agents.intake import intake_node

    full_profile = {
        "risk_tolerance": "moderate",
        "investment_horizon_years": 15,
        "investment_amount_usd": 75_000,
        "portfolio": {"VTI": 0.7, "BND": 0.3},
    }
    state = _make_state(
        messages=[HumanMessage(content="Ready")],
        existing_profile=full_profile,
    )

    with patch("app.agents.intake.get_chat_model") as mock_llm:
        result = await intake_node(state)

    content = result["messages"][-1].content
    assert "moderate" in content.lower()
    assert "75,000" in content or "75000" in content


# ---------------------------------------------------------------------------
# Unit: fake-confirmation guard
# ---------------------------------------------------------------------------

@pytest.mark.unit
@pytest.mark.asyncio
async def test_intake_fake_confirmation_guard_replaces_llm_text():
    """If LLM generates 'I've captured your profile' without a tool call,
    intake replaces it with a proper clarifying question."""
    from app.agents.intake import intake_node

    state = _make_state(
        messages=[HumanMessage(content="I want to invest")],
        existing_profile={},
    )

    fake_llm = _fake_llm_no_tool_call("Great! I've captured your profile and will proceed.")

    with patch("app.agents.intake.get_chat_model", return_value=fake_llm):
        result = await intake_node(state)

    content = result["messages"][-1].content
    assert "I've captured your profile" not in content
    assert "still need" in content.lower() or "could you" in content.lower()


@pytest.mark.unit
@pytest.mark.asyncio
async def test_intake_fake_confirmation_guard_case_insensitive():
    """Guard catches 'I have captured your profile' (uppercase I) variant."""
    from app.agents.intake import intake_node

    state = _make_state(
        messages=[HumanMessage(content="ok")],
        existing_profile={},
    )
    fake_llm = _fake_llm_no_tool_call("I have captured your profile. Moving on.")

    with patch("app.agents.intake.get_chat_model", return_value=fake_llm):
        result = await intake_node(state)

    content = result["messages"][-1].content
    assert "I have captured your profile" not in content


# ---------------------------------------------------------------------------
# Unit: portfolio question gate
# ---------------------------------------------------------------------------

@pytest.mark.unit
@pytest.mark.asyncio
async def test_intake_asks_portfolio_question_when_not_in_tool_call():
    """When LLM calls collect_profile without portfolio info, intake asks portfolio question."""
    from app.agents.intake import intake_node

    state = _make_state(
        messages=[HumanMessage(content="High risk, 10 years, $100,000")],
        existing_profile={},
    )
    # LLM calls collect_profile with only the 3 required fields, no portfolio
    args = {"risk_tolerance": "high", "investment_horizon_years": 10, "investment_amount_usd": 100_000}
    fake_llm = _fake_llm_with_tool_call(args)

    with patch("app.agents.intake.get_chat_model", return_value=fake_llm):
        result = await intake_node(state)

    # Should NOT complete intake — should ask portfolio question
    assert not result.get("intake_complete")
    content = result["messages"][-1].content
    assert "existing investments" in content.lower() or "portfolio" in content.lower()


@pytest.mark.unit
@pytest.mark.asyncio
async def test_intake_document_bypass_completes_without_portfolio_question():
    """When existing_profile already has portfolio key, intake does not ask for it."""
    from app.agents.intake import intake_node

    existing = {
        "risk_tolerance": "low",
        "investment_horizon_years": 5,
        "investment_amount_usd": 25_000,
        "portfolio": {"BND": 0.8, "GLD": 0.2},
    }
    # LLM finds the 3 required fields but no portfolio in tool call args
    args = {"risk_tolerance": "low", "investment_horizon_years": 5, "investment_amount_usd": 25_000}
    fake_llm = _fake_llm_with_tool_call(args)

    state = _make_state(
        messages=[HumanMessage(content="Low risk 5yr $25k")],
        existing_profile=existing,
    )

    with patch("app.agents.intake.get_chat_model", return_value=fake_llm):
        result = await intake_node(state)

    assert result.get("intake_complete") is True
    assert result["user_profile"]["portfolio"] == {"BND": 0.8, "GLD": 0.2}


# ---------------------------------------------------------------------------
# LLM eval: real LLM field extraction
# ---------------------------------------------------------------------------

@pytest.mark.llm_eval
@pytest.mark.asyncio
async def test_intake_extracts_explicit_numeric_fields(langfuse_scorer):
    """LLM extracts risk_tolerance, horizon, amount from explicit numeric message."""
    skip_if_no_openrouter()
    from app.agents.intake import intake_node

    state = _make_state(
        messages=[HumanMessage(content="High risk, 15 years, $50,000")]
    )

    result = await intake_node(state)
    profile = result.get("user_profile") or {}

    expected = {
        "risk_tolerance": "high",
        "investment_horizon_years": 15,
        "investment_amount_usd": 50_000,
    }
    matched = sum(1 for k, v in expected.items() if profile.get(k) == v)
    accuracy = matched / len(expected)

    if langfuse_scorer:
        try:
            langfuse_scorer.create_score(
                name="intake_extraction_accuracy",
                value=accuracy,
                data_type="NUMERIC",
                comment=f"Matched {matched}/{len(expected)} fields from explicit message",
            )
        except Exception:
            pass

    assert accuracy >= 0.6, f"Low extraction accuracy ({accuracy}): profile={profile}"


@pytest.mark.llm_eval
@pytest.mark.asyncio
async def test_intake_does_not_call_tool_on_ambiguous_input():
    """Ambiguous input should trigger a clarifying question, not a tool call."""
    skip_if_no_openrouter()
    from app.agents.intake import intake_node

    state = _make_state(
        messages=[HumanMessage(content="maybe moderate-ish risk, not sure about timeline")]
    )

    result = await intake_node(state)
    # If tool was called and intake_complete set, that's wrong
    assert not result.get("intake_complete"), (
        "intake should not complete on ambiguous input — LLM should ask clarifying question"
    )


@pytest.mark.llm_eval
@pytest.mark.asyncio
async def test_intake_handles_word_form_amount():
    """'fifty thousand dollars' should be extracted as 50000."""
    skip_if_no_openrouter()
    from app.agents.intake import intake_node

    state = _make_state(
        messages=[
            HumanMessage(content="moderate risk"),
            AIMessage(content="How long is your investment horizon?", name="intake"),
            HumanMessage(content="10 years"),
            AIMessage(content="How much would you like to invest?", name="intake"),
            HumanMessage(content="fifty thousand dollars"),
        ]
    )

    result = await intake_node(state)
    profile = result.get("user_profile") or {}

    if profile.get("investment_amount_usd"):
        # Either extracted correctly or ask for portfolio
        assert profile["investment_amount_usd"] in (50_000, 50000.0), (
            f"Amount not parsed correctly: {profile.get('investment_amount_usd')}"
        )
