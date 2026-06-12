"""Deterministic supervisor routing tests — no LLM calls, no API keys needed."""
import pytest
from langchain_core.messages import AIMessage, HumanMessage

from app.agents.supervisor import route_supervisor
from app.agents.state import AllocationResult, RiskMetrics, ScoringResult, UserProfile


def _state(
    *,
    intent: str | None = None,
    intake_complete: bool = False,
    user_profile: UserProfile | None = None,
    risk_metrics: RiskMetrics | None = None,
    allocation_result: AllocationResult | None = None,
    scoring_result: ScoringResult | None = None,
    advisor_report_generated: bool = False,
    documents_uploaded: list[str] | None = None,
    documents_extracted: bool = False,
    messages: list | None = None,
    iteration_count: int = 0,
    error: str | None = None,
) -> dict:
    return {
        "intent": intent,
        "intake_complete": intake_complete,
        "user_profile": user_profile,
        "risk_metrics": risk_metrics,
        "allocation_result": allocation_result,
        "scoring_result": scoring_result,
        "advisor_report_generated": advisor_report_generated,
        "documents_uploaded": documents_uploaded or [],
        "documents_extracted": documents_extracted,
        "messages": messages or [HumanMessage(content="hi")],
        "iteration_count": iteration_count,
        "error": error,
        "session_id": "test",
    }


_RISK: RiskMetrics = {
    "sharpe_ratio": 0.8,
    "volatility": 0.15,
    "max_drawdown": -0.12,
    "risk_flags": ["Equity heavy"],
}
_ALLOC: AllocationResult = {
    "weights": {"SPY": 0.6, "BND": 0.4},
    "expected_return": 0.08,
    "expected_volatility": 0.12,
    "strategy_rationale": "Balanced.",
}
_SCORE: ScoringResult = {"composite_score": 72.0, "breakdown": {}}
_PROFILE: UserProfile = {
    "risk_tolerance": "moderate",
    "investment_horizon_years": 10,
    "investment_amount_usd": 50000,
    "portfolio": {"SPY": 1.0},
}


# ---------------------------------------------------------------------------
# general intent
# ---------------------------------------------------------------------------

@pytest.mark.unit
def test_general_goes_to_advisor_copilot():
    state = _state(intent="general")
    assert route_supervisor(state) == "advisor_copilot"


@pytest.mark.unit
def test_general_skips_intake_even_if_incomplete():
    state = _state(intent="general", intake_complete=False)
    assert route_supervisor(state) == "advisor_copilot"


# ---------------------------------------------------------------------------
# risk_analysis intent
# ---------------------------------------------------------------------------

@pytest.mark.unit
def test_risk_analysis_triggers_intake_when_profile_missing():
    state = _state(intent="risk_analysis", intake_complete=False)
    assert route_supervisor(state) == "intake"


@pytest.mark.unit
def test_risk_analysis_triggers_risk_assessment_after_intake():
    state = _state(intent="risk_analysis", intake_complete=True, user_profile=_PROFILE)
    assert route_supervisor(state) == "risk_assessment"


@pytest.mark.unit
def test_risk_analysis_skips_strategy():
    state = _state(
        intent="risk_analysis",
        intake_complete=True,
        user_profile=_PROFILE,
        risk_metrics=_RISK,
    )
    # No strategy or scoring needed — should go straight to advisor_copilot
    assert route_supervisor(state) == "advisor_copilot"


@pytest.mark.unit
def test_risk_analysis_does_not_run_strategy_even_if_missing():
    state = _state(
        intent="risk_analysis",
        intake_complete=True,
        user_profile=_PROFILE,
        risk_metrics=_RISK,
        allocation_result=None,  # explicitly absent
        scoring_result=None,
    )
    assert route_supervisor(state) == "advisor_copilot"


# ---------------------------------------------------------------------------
# score_portfolio intent
# ---------------------------------------------------------------------------

@pytest.mark.unit
def test_score_portfolio_triggers_intake_when_profile_missing():
    state = _state(intent="score_portfolio", intake_complete=False)
    assert route_supervisor(state) == "intake"


@pytest.mark.unit
def test_score_portfolio_triggers_risk_assessment_before_scoring():
    state = _state(intent="score_portfolio", intake_complete=True, user_profile=_PROFILE)
    assert route_supervisor(state) == "risk_assessment"


@pytest.mark.unit
def test_score_portfolio_triggers_scoring_after_risk():
    state = _state(
        intent="score_portfolio",
        intake_complete=True,
        user_profile=_PROFILE,
        risk_metrics=_RISK,
    )
    assert route_supervisor(state) == "scoring"


@pytest.mark.unit
def test_score_portfolio_skips_strategy():
    state = _state(
        intent="score_portfolio",
        intake_complete=True,
        user_profile=_PROFILE,
        risk_metrics=_RISK,
        scoring_result=_SCORE,
    )
    # strategy was never run — still should not be triggered
    assert route_supervisor(state) == "advisor_copilot"


# ---------------------------------------------------------------------------
# full_analysis intent
# ---------------------------------------------------------------------------

@pytest.mark.unit
def test_full_analysis_triggers_intake_when_missing():
    state = _state(intent="full_analysis", intake_complete=False)
    assert route_supervisor(state) == "intake"


@pytest.mark.unit
def test_full_analysis_runs_complete_pipeline_in_order():
    # Step 1: risk
    s = _state(intent="full_analysis", intake_complete=True, user_profile=_PROFILE)
    assert route_supervisor(s) == "risk_assessment"

    # Step 2: strategy
    s = _state(
        intent="full_analysis",
        intake_complete=True,
        user_profile=_PROFILE,
        risk_metrics=_RISK,
    )
    assert route_supervisor(s) == "strategy"

    # Step 3: scoring
    s = _state(
        intent="full_analysis",
        intake_complete=True,
        user_profile=_PROFILE,
        risk_metrics=_RISK,
        allocation_result=_ALLOC,
    )
    assert route_supervisor(s) == "scoring"

    # Step 4: advisor_copilot
    s = _state(
        intent="full_analysis",
        intake_complete=True,
        user_profile=_PROFILE,
        risk_metrics=_RISK,
        allocation_result=_ALLOC,
        scoring_result=_SCORE,
    )
    assert route_supervisor(s) == "advisor_copilot"


# ---------------------------------------------------------------------------
# Cross-intent upgrade: risk_analysis → full_analysis
# ---------------------------------------------------------------------------

@pytest.mark.unit
def test_cross_intent_upgrade_adds_missing_strategy():
    # risk_analysis was run previously; now user asks for full_analysis
    state = _state(
        intent="full_analysis",
        intake_complete=True,
        user_profile=_PROFILE,
        risk_metrics=_RISK,      # already computed
        allocation_result=None,  # not yet computed
        scoring_result=None,
    )
    assert route_supervisor(state) == "strategy"


# ---------------------------------------------------------------------------
# Guard conditions
# ---------------------------------------------------------------------------

@pytest.mark.unit
def test_max_iterations_returns_end():
    state = _state(intent="full_analysis", iteration_count=50)
    assert route_supervisor(state) == "end"


@pytest.mark.unit
def test_error_routes_to_error_handler():
    state = _state(intent="general", error="something went wrong")
    assert route_supervisor(state) == "error_handler"


@pytest.mark.unit
def test_document_pipeline_runs_before_intent_routing():
    state = _state(
        intent="general",
        documents_uploaded=["doc.pdf"],
        documents_extracted=False,
    )
    assert route_supervisor(state) == "document_intelligence"


@pytest.mark.unit
def test_profile_builder_runs_after_extraction():
    state = _state(
        intent="general",
        documents_uploaded=["doc.pdf"],
        documents_extracted=True,
        user_profile={},  # no "portfolio" key
    )
    assert route_supervisor(state) == "profile_builder"


@pytest.mark.unit
def test_report_regenerated_on_new_human_message():
    state = _state(
        intent="full_analysis",
        intake_complete=True,
        user_profile=_PROFILE,
        risk_metrics=_RISK,
        allocation_result=_ALLOC,
        scoring_result=_SCORE,
        advisor_report_generated=True,
        messages=[HumanMessage(content="tell me more")],
    )
    assert route_supervisor(state) == "advisor_copilot"


@pytest.mark.unit
def test_end_when_report_generated_and_no_new_message():
    state = _state(
        intent="full_analysis",
        intake_complete=True,
        user_profile=_PROFILE,
        risk_metrics=_RISK,
        allocation_result=_ALLOC,
        scoring_result=_SCORE,
        advisor_report_generated=True,
        messages=[AIMessage(content="Here is your plan.", name="advisor_copilot")],
    )
    assert route_supervisor(state) == "end"


# ---------------------------------------------------------------------------
# effective_intake_complete — derived from profile fields even without flag
# ---------------------------------------------------------------------------

_FULL_PROFILE: UserProfile = {
    "risk_tolerance": "high",
    "investment_horizon_years": 10,
    "investment_amount_usd": 100_000,
    "portfolio": {"SPY": 0.6, "BND": 0.4},
}


@pytest.mark.unit
def test_effective_intake_complete_from_profile_fields():
    """Supervisor routes past intake when all 3 fields + portfolio key are present,
    even if intake_complete flag is False."""
    state = _state(
        intent="full_analysis",
        intake_complete=False,
        user_profile=_FULL_PROFILE,
    )
    # Should route to risk_assessment, not intake
    assert route_supervisor(state) == "risk_assessment"


@pytest.mark.unit
def test_effective_intake_complete_false_when_risk_tolerance_missing():
    """Portfolio present but risk_tolerance absent → effective_intake_complete is False."""
    profile: UserProfile = {
        "investment_horizon_years": 10,
        "investment_amount_usd": 100_000,
        "portfolio": {"SPY": 1.0},
        "risk_tolerance": "",  # empty string is falsy
    }
    state = _state(
        intent="full_analysis",
        intake_complete=False,
        user_profile=profile,
        messages=[HumanMessage(content="hi")],
    )
    assert route_supervisor(state) == "intake"


@pytest.mark.unit
def test_effective_intake_complete_false_when_portfolio_key_absent():
    """All 3 required fields present but no 'portfolio' key → not considered complete."""
    profile: UserProfile = {
        "risk_tolerance": "moderate",
        "investment_horizon_years": 10,
        "investment_amount_usd": 50_000,
        # no 'portfolio' key
    }
    state = _state(
        intent="full_analysis",
        intake_complete=False,
        user_profile=profile,
        messages=[HumanMessage(content="hi")],
    )
    assert route_supervisor(state) == "intake"


@pytest.mark.unit
def test_effective_intake_complete_false_when_horizon_zero():
    """investment_horizon_years = 0 is falsy → effective_intake_complete stays False."""
    profile: UserProfile = {
        "risk_tolerance": "low",
        "investment_horizon_years": 0,
        "investment_amount_usd": 50_000,
        "portfolio": {"BND": 1.0},
    }
    state = _state(
        intent="full_analysis",
        intake_complete=False,
        user_profile=profile,
        messages=[HumanMessage(content="hi")],
    )
    assert route_supervisor(state) == "intake"


@pytest.mark.unit
def test_effective_intake_bypasses_intake_and_reaches_advisor():
    """Full pipeline completes via effective_intake_complete without intake_complete flag."""
    state = _state(
        intent="full_analysis",
        intake_complete=False,
        user_profile=_FULL_PROFILE,
        risk_metrics=_RISK,
        allocation_result=_ALLOC,
        scoring_result=_SCORE,
    )
    assert route_supervisor(state) == "advisor_copilot"
