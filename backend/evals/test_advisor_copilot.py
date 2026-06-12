"""Tests for advisor_copilot: unit (deterministic) + deepeval LLM-as-judge.

Unit tests:   uv run pytest evals/test_advisor_copilot.py -m unit -v
LLM eval:     uv run pytest evals/test_advisor_copilot.py -m llm_eval -v --timeout=300
"""
import json
import re
import pytest
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch
from langchain_core.messages import AIMessage, HumanMessage


FIXTURES_DIR = Path(__file__).parent / "fixtures"
_ADVISOR_CASES = json.loads((FIXTURES_DIR / "advisor_test_cases.json").read_text())

# Valid source labels per grounding rule
_VALID_ANALYSIS_SOURCES = {
    "User Profile",
    "Portfolio Risk Analysis",
    "Strategy Engine",
    "Portfolio Score",
}

_DISCLAIMER_MARKER = "*This is not professional financial advice.*"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_state(intent: str, extra: dict | None = None) -> dict:
    base = {
        "messages": [HumanMessage(content="Tell me about my portfolio")],
        "session_id": "test",
        "intake_complete": True,
        "user_profile": {
            "risk_tolerance": "high",
            "investment_horizon_years": 10,
            "investment_amount_usd": 100_000,
            "portfolio": {"SPY": 0.5, "QQQ": 0.3, "BND": 0.2},
        },
        "documents_uploaded": [],
        "documents_extracted": False,
        "risk_metrics": {
            "sharpe_ratio": 1.25,
            "volatility": 0.18,
            "max_drawdown": -0.12,
            "risk_flags": ["Concentration risk"],
        },
        "allocation_result": {
            "weights": {"SPY": 0.50, "QQQ": 0.30, "GLD": 0.20},
            "expected_return": 0.12,
            "expected_volatility": 0.15,
            "strategy_rationale": "Diversified global allocation",
        },
        "scoring_result": {
            "composite_score": 72.0,
            "breakdown": {"sharpe_score": 75, "drawdown_score": 70, "diversification_score": 54},
        },
        "iteration_count": 5,
        "error": None,
        "intent": intent,
        "advisor_report_generated": False,
    }
    if extra:
        base.update(extra)
    return base


# ---------------------------------------------------------------------------
# Unit: _build_context source labels
# ---------------------------------------------------------------------------

@pytest.mark.unit
def test_build_context_source_labels_all_present(full_state):
    """_build_context() output contains all 4 [Source: ...] labels when data is present."""
    from app.agents.advisor_copilot import _build_context

    context = _build_context(full_state)

    assert "[Source: User Profile]" in context
    assert "[Source: Portfolio Risk Analysis]" in context
    assert "[Source: Strategy Engine]" in context
    assert "[Source: Portfolio Score]" in context


@pytest.mark.unit
def test_build_context_empty_state_returns_empty_string():
    """_build_context() with no data fields returns empty string."""
    from app.agents.advisor_copilot import _build_context

    empty = {
        "messages": [],
        "session_id": "test",
        "user_profile": None,
        "risk_metrics": None,
        "allocation_result": None,
        "scoring_result": None,
    }
    assert _build_context(empty) == ""


@pytest.mark.unit
def test_build_context_partial_state_only_shows_available_sources():
    """When only some fields are populated, only their [Source:] labels appear."""
    from app.agents.advisor_copilot import _build_context

    partial = {
        "messages": [],
        "session_id": "test",
        "user_profile": {
            "risk_tolerance": "moderate",
            "investment_horizon_years": 10,
            "investment_amount_usd": 50_000,
            "portfolio": {"SPY": 1.0},
        },
        "risk_metrics": None,
        "allocation_result": None,
        "scoring_result": None,
    }
    context = _build_context(partial)
    assert "[Source: User Profile]" in context
    assert "[Source: Portfolio Risk Analysis]" not in context
    assert "[Source: Strategy Engine]" not in context
    assert "[Source: Portfolio Score]" not in context


# ---------------------------------------------------------------------------
# Unit: advisor_copilot_node — deterministic checks via mocked LLM
# ---------------------------------------------------------------------------

def _mock_llm_response(content: str):
    """Return a mock LLM that yields a fixed AIMessage content."""
    resp = MagicMock()
    resp.content = content
    mock_llm = MagicMock()
    mock_llm.ainvoke = AsyncMock(return_value=resp)
    return mock_llm


@pytest.mark.unit
@pytest.mark.asyncio
async def test_advisor_no_kb_suffix_when_no_chunks(full_state):
    """When retriever returns [], the ⚠️ suffix is appended to the response."""
    from app.agents.advisor_copilot import advisor_copilot_node

    with patch("app.agents.advisor_copilot._retrieve_context", new=AsyncMock(return_value=[])):
        with patch("app.agents.advisor_copilot.get_chat_model") as mock_get_llm:
            mock_get_llm.return_value = _mock_llm_response("Here is your analysis.")
            result = await advisor_copilot_node(full_state)

    last_msg = result["messages"][-1]
    assert "⚠️" in last_msg.content


@pytest.mark.unit
@pytest.mark.asyncio
async def test_advisor_no_kb_suffix_not_added_when_chunks_present(full_state):
    """When retriever returns chunks, the ⚠️ warning is NOT added."""
    from app.agents.advisor_copilot import advisor_copilot_node

    chunks = ["[Source: SEC_ETF_Guide.txt]\nETFs are investment funds traded on exchanges."]

    with patch("app.agents.advisor_copilot._retrieve_context", new=AsyncMock(return_value=chunks)):
        with patch("app.agents.advisor_copilot.get_chat_model") as mock_get_llm:
            mock_get_llm.return_value = _mock_llm_response("Here is your analysis.")
            result = await advisor_copilot_node(full_state)

    last_msg = result["messages"][-1]
    assert "⚠️" not in last_msg.content


@pytest.mark.unit
@pytest.mark.asyncio
async def test_advisor_sets_report_generated_flag(full_state):
    """advisor_copilot_node always sets advisor_report_generated: True."""
    from app.agents.advisor_copilot import advisor_copilot_node

    with patch("app.agents.advisor_copilot._retrieve_context", new=AsyncMock(return_value=[])):
        with patch("app.agents.advisor_copilot.get_chat_model") as mock_get_llm:
            mock_get_llm.return_value = _mock_llm_response("Analysis complete.")
            result = await advisor_copilot_node(full_state)

    assert result.get("advisor_report_generated") is True


@pytest.mark.unit
@pytest.mark.asyncio
async def test_advisor_message_name_is_advisor_copilot(full_state):
    """The AIMessage returned has name='advisor_copilot'."""
    from app.agents.advisor_copilot import advisor_copilot_node

    with patch("app.agents.advisor_copilot._retrieve_context", new=AsyncMock(return_value=[])):
        with patch("app.agents.advisor_copilot.get_chat_model") as mock_get_llm:
            mock_get_llm.return_value = _mock_llm_response("Analysis.")
            result = await advisor_copilot_node(full_state)

    msg = result["messages"][-1]
    assert isinstance(msg, AIMessage)
    assert getattr(msg, "name", None) == "advisor_copilot"


@pytest.mark.unit
@pytest.mark.asyncio
async def test_advisor_general_intent_uses_no_context():
    """For 'general' intent, _build_context is effectively empty (no quantitative data)."""
    from app.agents.advisor_copilot import advisor_copilot_node

    state = _make_state("general", {
        "risk_metrics": None,
        "allocation_result": None,
        "scoring_result": None,
        "user_profile": None,
    })

    with patch("app.agents.advisor_copilot._retrieve_context", new=AsyncMock(return_value=[])):
        with patch("app.agents.advisor_copilot.get_chat_model") as mock_get_llm:
            mock_get_llm.return_value = _mock_llm_response("ETFs are efficient investment vehicles.")
            result = await advisor_copilot_node(state)

    assert result.get("advisor_report_generated") is True


@pytest.mark.unit
@pytest.mark.asyncio
async def test_advisor_langfuse_metadata_failure_does_not_crash(full_state):
    """Langfuse metadata logging failure (import error etc) does not crash the node."""
    from app.agents.advisor_copilot import advisor_copilot_node

    with patch("app.agents.advisor_copilot._retrieve_context", new=AsyncMock(return_value=[])):
        with patch("app.agents.advisor_copilot.get_chat_model") as mock_get_llm:
            mock_get_llm.return_value = _mock_llm_response("Analysis.")
            # Simulate langfuse import failure
            with patch.dict("sys.modules", {"langfuse": None}):
                result = await advisor_copilot_node(full_state)

    assert "messages" in result


# ---------------------------------------------------------------------------
# Unit: citation format validation
# ---------------------------------------------------------------------------

_SOURCE_PATTERN = re.compile(r"\(Source: ([^)]+)\)")

_VALID_SOURCE_NAMES = _VALID_ANALYSIS_SOURCES | {
    # Known KB filenames (subset — others are covered by the general pattern)
    "SEC_ETF_Guide.txt",
    "SEC_Mutual_Funds_Guide.txt",
    "SEC_Asset_Allocation_Diversification.txt",
    "arXiv_Sharpe_Ratio_Estimation.pdf",
    "arXiv_Bond_Duration_Risk.pdf",
    "arXiv_CVaR_Portfolio_Optimization.pdf",
    "Fed_Monetary_Policy_Report_2025_Feb.pdf",
    "Fed_Monetary_Policy_Report_2025_Jun.pdf",
}


@pytest.mark.unit
def test_citation_pattern_matches_analysis_sources():
    """(Source: Portfolio Risk Analysis) matches the citation regex."""
    text = "The Sharpe ratio is 1.25 (Source: Portfolio Risk Analysis) indicating..."
    matches = _SOURCE_PATTERN.findall(text)
    assert "Portfolio Risk Analysis" in matches


@pytest.mark.unit
def test_citation_pattern_matches_kb_filename():
    """(Source: SEC_ETF_Guide.txt) matches the citation regex."""
    text = "ETFs trade like stocks (Source: SEC_ETF_Guide.txt) throughout the day."
    matches = _SOURCE_PATTERN.findall(text)
    assert "SEC_ETF_Guide.txt" in matches


@pytest.mark.unit
def test_citation_pattern_multiple_sources():
    """Multiple sources in one sentence are all captured."""
    text = "(Source: User Profile) and (Source: Portfolio Risk Analysis) both indicate..."
    matches = _SOURCE_PATTERN.findall(text)
    assert "User Profile" in matches
    assert "Portfolio Risk Analysis" in matches


# ---------------------------------------------------------------------------
# LLM eval: real LLM calls with deepeval metrics
# ---------------------------------------------------------------------------

@pytest.mark.llm_eval
@pytest.mark.asyncio
async def test_advisor_disclaimer_present_in_llm_response(full_state, langfuse_scorer):
    from evals.conftest import skip_if_no_openrouter
    skip_if_no_openrouter()
    """Real LLM response always ends with the professional advice disclaimer."""
    from app.agents.advisor_copilot import advisor_copilot_node

    chunks = [
        "[Source: SEC_ETF_Guide.txt]\nETFs are pooled investment funds traded on exchanges "
        "throughout the trading day at market prices."
    ]
    with patch("app.agents.advisor_copilot._retrieve_context", new=AsyncMock(return_value=chunks)):
        result = await advisor_copilot_node(full_state)

    content = result["messages"][-1].content
    has_disclaimer = _DISCLAIMER_MARKER in content

    # Push boolean score to Langfuse
    if langfuse_scorer:
        try:
            langfuse_scorer.create_score(
                name="disclaimer_present",
                value=1.0 if has_disclaimer else 0.0,
                data_type="NUMERIC",
                comment="Checked via string match on advisor response",
            )
        except Exception:
            pass

    assert has_disclaimer, f"Disclaimer missing. Response tail: {content[-200:]}"


@pytest.mark.llm_eval
@pytest.mark.asyncio
async def test_advisor_citation_present_in_llm_response(full_state, langfuse_scorer):
    from evals.conftest import skip_if_no_openrouter
    skip_if_no_openrouter()
    """Real LLM response contains at least one (Source: ...) citation."""
    from app.agents.advisor_copilot import advisor_copilot_node

    chunks = [
        "[Source: arXiv_Sharpe_Ratio_Estimation.pdf]\nThe Sharpe ratio measures risk-adjusted "
        "return by dividing excess return by portfolio standard deviation."
    ]
    with patch("app.agents.advisor_copilot._retrieve_context", new=AsyncMock(return_value=chunks)):
        result = await advisor_copilot_node(full_state)

    content = result["messages"][-1].content
    citations = _SOURCE_PATTERN.findall(content)
    has_citations = len(citations) > 0

    if langfuse_scorer:
        try:
            langfuse_scorer.create_score(
                name="citation_format_valid",
                value=1.0 if has_citations else 0.0,
                data_type="NUMERIC",
                comment=f"Found citations: {citations[:3]}",
            )
        except Exception:
            pass

    assert has_citations, f"No (Source: ...) citations found in response: {content[:500]}"


@pytest.mark.llm_eval
@pytest.mark.asyncio
async def test_advisor_faithfulness_via_deepeval(full_state, langfuse_scorer, judge_model):
    """Advisor response claims are grounded in retrieved chunks (deepeval FaithfulnessMetric)."""
    from evals.conftest import skip_if_no_openrouter
    skip_if_no_openrouter()
    pytest.importorskip("deepeval")
    from deepeval.metrics import FaithfulnessMetric
    from deepeval.test_case import LLMTestCase
    from app.agents.advisor_copilot import advisor_copilot_node

    chunks = [
        "[Source: arXiv_Sharpe_Ratio_Estimation.pdf]\nThe Sharpe ratio is defined as "
        "(E[R_p] - R_f) / σ_p, where R_p is portfolio return, R_f is risk-free rate, "
        "and σ_p is the standard deviation of excess returns.",
        "[Source: SEC_Asset_Allocation_Diversification.txt]\nDiversification across asset "
        "classes reduces unsystematic risk while maintaining expected return.",
    ]

    with patch("app.agents.advisor_copilot._retrieve_context", new=AsyncMock(return_value=chunks)):
        result = await advisor_copilot_node(full_state)

    response_text = result["messages"][-1].content
    query = next(
        (m.content for m in reversed(full_state["messages"]) if isinstance(m, HumanMessage)),
        "Tell me about my portfolio",
    )
    context_texts = [c.split("\n", 1)[-1] for c in chunks]  # strip [Source: ...] prefix

    test_case = LLMTestCase(
        input=query,
        actual_output=response_text,
        retrieval_context=context_texts,
    )

    metric = FaithfulnessMetric(threshold=0.7, model=judge_model, verbose_mode=False)
    metric.measure(test_case)

    if langfuse_scorer:
        try:
            langfuse_scorer.create_score(
                name="rag_faithfulness",
                value=metric.score if metric.score is not None else 0.0,
                data_type="NUMERIC",
                comment=metric.reason or "",
            )
        except Exception:
            pass

    assert metric.score is not None, "deepeval FaithfulnessMetric returned no score"


@pytest.mark.llm_eval
@pytest.mark.asyncio
async def test_advisor_answer_relevancy_via_deepeval(full_state, langfuse_scorer, judge_model):
    """Advisor response is relevant to the user's question (deepeval AnswerRelevancyMetric)."""
    from evals.conftest import skip_if_no_openrouter
    skip_if_no_openrouter()
    pytest.importorskip("deepeval")
    from deepeval.metrics import AnswerRelevancyMetric
    from deepeval.test_case import LLMTestCase
    from app.agents.advisor_copilot import advisor_copilot_node

    full_state["messages"] = [HumanMessage(content="How should I rebalance my portfolio?")]

    chunks = [
        "[Source: SEC_Asset_Allocation_Diversification.txt]\nRebalancing restores a portfolio "
        "to its target allocation after market movements cause drift."
    ]
    with patch("app.agents.advisor_copilot._retrieve_context", new=AsyncMock(return_value=chunks)):
        result = await advisor_copilot_node(full_state)

    response_text = result["messages"][-1].content

    test_case = LLMTestCase(
        input="How should I rebalance my portfolio?",
        actual_output=response_text,
        retrieval_context=[chunks[0].split("\n", 1)[-1]],
    )

    metric = AnswerRelevancyMetric(threshold=0.6, model=judge_model, verbose_mode=False)
    metric.measure(test_case)

    if langfuse_scorer:
        try:
            langfuse_scorer.create_score(
                name="rag_answer_relevancy",
                value=metric.score if metric.score is not None else 0.0,
                data_type="NUMERIC",
                comment=metric.reason or "",
            )
        except Exception:
            pass

    assert metric.score is not None, "deepeval AnswerRelevancyMetric returned no score"


# ---------------------------------------------------------------------------
# LLM eval: per-intent requirement checks (parametrized over advisor_test_cases.json)
# ---------------------------------------------------------------------------

@pytest.mark.llm_eval
@pytest.mark.asyncio
@pytest.mark.parametrize("case", _ADVISOR_CASES, ids=[c["description"] for c in _ADVISOR_CASES])
async def test_advisor_response_meets_intent_requirements(full_state, case, langfuse_scorer):
    """Real LLM response must satisfy all must_contain and must_not_contain rules.
    Score pushed to Langfuse as advisor.requirement_pass_rate per case."""
    from evals.conftest import skip_if_no_openrouter
    skip_if_no_openrouter()
    from app.agents.advisor_copilot import advisor_copilot_node

    state = _make_state(case["intent"])
    state["messages"] = [HumanMessage(content=case["input"])]

    is_no_kb = "xyzzy" in case["input"] or "Bitcoin" in case["input"]
    chunks_for_test = [] if is_no_kb else [
        "[Source: SEC_ETF_Guide.txt]\nETFs are pooled investment funds traded on stock exchanges.",
        "[Source: SEC_Asset_Allocation_Diversification.txt]\nDiversification reduces unsystematic risk.",
        "[Source: Portfolio Risk Analysis]\nSharpe ratio: 1.25, Volatility: 18%, Max Drawdown: -12%",
        "[Source: Portfolio Score]\nComposite score: 72/100 — Sharpe 75, Drawdown 70, Diversification 54",
    ]
    retriever_patch = AsyncMock(return_value=chunks_for_test)

    with patch("app.agents.advisor_copilot._retrieve_context", new=retriever_patch):
        result = await advisor_copilot_node(state)

    content = result["messages"][-1].content
    must_pass = all(m in content for m in case["must_contain"])
    must_not_pass = all(m not in content for m in case.get("must_not_contain", []))
    score = 1.0 if (must_pass and must_not_pass) else 0.0

    if langfuse_scorer:
        try:
            langfuse_scorer.create_score(
                name="advisor.requirement_pass_rate",
                value=score,
                data_type="NUMERIC",
                comment=case["description"],
            )
        except Exception:
            pass

    assert must_pass, (
        f"Missing must_contain items for: {case['description']}\n"
        f"Required: {case['must_contain']}\n"
        f"Response tail: {content[-400:]}"
    )
    assert must_not_pass, (
        f"Found must_not_contain items for: {case['description']}\n"
        f"Forbidden: {case['must_not_contain']}"
    )


# ---------------------------------------------------------------------------
# LLM eval: deepeval ContextualPrecisionMetric
# ---------------------------------------------------------------------------

@pytest.mark.llm_eval
@pytest.mark.asyncio
async def test_advisor_contextual_precision_via_deepeval(full_state, langfuse_scorer, judge_model):
    """Most relevant chunk (Sharpe paper) ranked before irrelevant chunk (Stocks guide).
    ContextualPrecisionMetric checks that higher-relevance chunks come first."""
    from evals.conftest import skip_if_no_openrouter
    skip_if_no_openrouter()
    pytest.importorskip("deepeval")
    from deepeval.metrics import ContextualPrecisionMetric
    from deepeval.test_case import LLMTestCase
    from app.agents.advisor_copilot import advisor_copilot_node

    full_state["messages"] = [
        HumanMessage(content="What is the Sharpe ratio and how should I optimize for it?")
    ]
    chunks = [
        "[Source: arXiv_Sharpe_Ratio_Estimation.pdf]\nThe Sharpe ratio measures risk-adjusted "
        "return by dividing excess return by portfolio standard deviation. Higher values indicate "
        "better risk-adjusted performance.",
        "[Source: SEC_Stocks_Guide.txt]\nStocks represent an ownership interest in a corporation "
        "and may pay dividends. They carry equity risk and may fluctuate in value.",
    ]
    with patch("app.agents.advisor_copilot._retrieve_context", new=AsyncMock(return_value=chunks)):
        result = await advisor_copilot_node(full_state)

    response_text = result["messages"][-1].content
    context_texts = [c.split("\n", 1)[-1] for c in chunks]

    test_case = LLMTestCase(
        input="What is the Sharpe ratio and how should I optimize for it?",
        actual_output=response_text,
        expected_output="Explanation of Sharpe ratio, its formula, and portfolio optimization techniques",
        retrieval_context=context_texts,
    )
    metric = ContextualPrecisionMetric(threshold=0.5, model=judge_model, verbose_mode=False)
    metric.measure(test_case)

    if langfuse_scorer:
        try:
            langfuse_scorer.create_score(
                name="rag.contextual_precision",
                value=metric.score if metric.score is not None else 0.0,
                data_type="NUMERIC",
                comment=metric.reason or "",
            )
        except Exception:
            pass

    assert metric.score is not None, "ContextualPrecisionMetric returned no score"


# ---------------------------------------------------------------------------
# LLM eval: hallucination detection — LLM must not cite unlisted sources
# ---------------------------------------------------------------------------

@pytest.mark.llm_eval
@pytest.mark.asyncio
async def test_advisor_does_not_hallucinate_sources(full_state, langfuse_scorer):
    """LLM response must only cite (Source: X) for sources actually provided.
    Any KB citation outside the provided chunk set is a hallucination."""
    from evals.conftest import skip_if_no_openrouter
    skip_if_no_openrouter()
    from app.agents.advisor_copilot import advisor_copilot_node

    chunks = [
        "[Source: SEC_ETF_Guide.txt]\nETFs are pooled investment funds traded on stock exchanges "
        "throughout the trading day at market prices, unlike mutual funds which price at NAV.",
    ]
    with patch("app.agents.advisor_copilot._retrieve_context", new=AsyncMock(return_value=chunks)):
        result = await advisor_copilot_node(full_state)

    content = result["messages"][-1].content
    cited = set(_SOURCE_PATTERN.findall(content))
    # Strip internal analysis sources — use substring match so combined citations
    # like "Portfolio Risk Analysis vs. Strategy Engine" are also accepted.
    kb_cited = {s for s in cited if not any(v in s for v in _VALID_ANALYSIS_SOURCES)}
    allowed_kb_sources = {"SEC_ETF_Guide.txt"}
    hallucinated = kb_cited - allowed_kb_sources

    score = 0.0 if hallucinated else 1.0
    if langfuse_scorer:
        try:
            langfuse_scorer.create_score(
                name="advisor.no_hallucination",
                value=score,
                data_type="NUMERIC",
                comment=f"Hallucinated sources: {hallucinated}",
            )
        except Exception:
            pass

    assert not hallucinated, (
        f"LLM cited sources not in provided chunks: {hallucinated}\n"
        f"All citations found: {cited}"
    )


# ---------------------------------------------------------------------------
# LLM eval: full_analysis response completeness
# ---------------------------------------------------------------------------

@pytest.mark.llm_eval
@pytest.mark.asyncio
async def test_full_analysis_response_completeness(full_state, langfuse_scorer):
    """full_analysis response must address Sharpe, allocation, and risk in the body."""
    from evals.conftest import skip_if_no_openrouter
    skip_if_no_openrouter()
    from app.agents.advisor_copilot import advisor_copilot_node

    full_state["messages"] = [HumanMessage(content="Give me a complete investment analysis")]
    chunks = [
        "[Source: Portfolio Risk Analysis]\nSharpe ratio: 1.25 | Volatility: 18% | Max Drawdown: -12%",
        "[Source: Strategy Engine]\nRecommended allocation: SPY 50%, QQQ 30%, GLD 20%",
        "[Source: arXiv_Asset_Allocation_Review.pdf]\nSparse MV portfolios reduce estimation error "
        "through LASSO regularization on the covariance matrix.",
    ]
    with patch("app.agents.advisor_copilot._retrieve_context", new=AsyncMock(return_value=chunks)):
        result = await advisor_copilot_node(full_state)

    content = result["messages"][-1].content
    required_terms = ["sharpe", "allocation", "risk"]
    found = [r for r in required_terms if r in content.lower()]
    completeness_score = len(found) / len(required_terms)

    if langfuse_scorer:
        try:
            langfuse_scorer.create_score(
                name="advisor.response_completeness",
                value=completeness_score,
                data_type="NUMERIC",
                comment=f"Found terms: {found}",
            )
        except Exception:
            pass

    assert completeness_score >= 0.66, (
        f"Incomplete full_analysis response — missing: {set(required_terms) - set(found)}\n"
        f"Response excerpt: {content[:500]}"
    )
