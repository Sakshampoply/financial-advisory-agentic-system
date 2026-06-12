from typing import Annotated, TypedDict

from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages


class UserProfile(TypedDict, total=False):
    risk_tolerance: str
    investment_horizon_years: int
    investment_amount_usd: float
    annual_income_usd: float | None
    tax_bracket: str | None
    liquidity_needs: str | None
    portfolio: dict[str, float]  # ticker → weight


class RiskMetrics(TypedDict):
    sharpe_ratio: float
    volatility: float
    max_drawdown: float
    risk_flags: list[str]


class AllocationResult(TypedDict):
    weights: dict[str, float]
    expected_return: float
    expected_volatility: float
    strategy_rationale: str


class ScoringResult(TypedDict):
    composite_score: float
    breakdown: dict[str, float]


class GraphState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]
    session_id: str
    intake_complete: bool
    user_profile: UserProfile | None
    documents_uploaded: list[str]
    documents_extracted: bool
    risk_metrics: RiskMetrics | None
    allocation_result: AllocationResult | None
    scoring_result: ScoringResult | None
    iteration_count: int
    error: str | None
    # Intent of the latest human message: "general" | "risk_analysis" | "score_portfolio" | "full_analysis" | None
    intent: str | None
    # True after advisor_copilot has generated the first post-pipeline report
    advisor_report_generated: bool


def make_initial_state(session_id: str) -> dict:
    return {
        "messages": [],
        "session_id": session_id,
        "intake_complete": False,
        "user_profile": None,
        "documents_uploaded": [],
        "documents_extracted": False,
        "risk_metrics": None,
        "allocation_result": None,
        "scoring_result": None,
        "iteration_count": 0,
        "error": None,
        "intent": None,
        "advisor_report_generated": False,
    }
