"""End-to-end integration tests for the full LangGraph pipeline.

These tests run the full graph against a live PostgreSQL DB (for checkpointer)
but mock all external network calls (yfinance, FRED, Alpha Vantage).

Run:  uv run pytest evals/test_e2e.py -m integration -v --timeout=120

Requirements:
  - DATABASE_URL env var pointing to running PostgreSQL
  - LangGraph checkpoint tables created (uv run alembic upgrade head)
  - OPENROUTER_API_KEY env var (for LLM calls in non-mocked tests)
"""
import uuid
import pytest
import pandas as pd
import numpy as np
from unittest.mock import AsyncMock, MagicMock, patch
from langchain_core.messages import AIMessage, HumanMessage


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _make_synthetic_df(n: int = 120) -> pd.DataFrame:
    dates = pd.date_range("2023-01-01", periods=n, freq="B")
    close = np.cumprod(1 + np.random.randn(n) * 0.01) * 100
    return pd.DataFrame(
        {"Open": close, "High": close * 1.01, "Low": close * 0.99,
         "Close": close, "Volume": 1_000_000},
        index=dates,
    )


@pytest.fixture
async def checkpointer():
    """Real async PostgreSQL checkpointer — skips the test if DB is not reachable.

    Function-scoped to avoid cross-event-loop asyncpg errors in pytest-asyncio auto mode.
    Requires: docker compose up -d && uv run alembic upgrade head
    """
    from app.config import settings
    from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
    try:
        async with AsyncPostgresSaver.from_conn_string(settings.DATABASE_URL) as saver:
            await saver.setup()
            yield saver
    except Exception as exc:
        pytest.skip(
            f"PostgreSQL not reachable — run 'docker compose up -d' first. ({type(exc).__name__}: {exc})"
        )


@pytest.fixture
async def graph(checkpointer):
    """Compiled LangGraph with real checkpointer."""
    from app.agents.graph import create_graph
    return create_graph(checkpointer)


@pytest.fixture
def thread_config():
    """Unique thread config per test to avoid state pollution."""
    return {"configurable": {"thread_id": str(uuid.uuid4())}}


def _patch_market_data():
    """Return a context manager that patches all external market data calls."""
    import contextlib

    @contextlib.asynccontextmanager
    async def _ctx():
        df = _make_synthetic_df()
        with patch("app.market_data.yfinance_client._download_ohlcv", return_value=df):
            with patch("app.market_data.fred_client._fetch_fred", new=AsyncMock(return_value=5.0)):
                with patch("app.market_data.alpha_vantage_client.get_fundamentals",
                           new=AsyncMock(return_value=None)):
                    with patch("app.market_data.alpha_vantage_client.get_sentiment",
                               new=AsyncMock(return_value=None)):
                        yield

    return _ctx()


async def _collect_nodes(graph, state: dict, config: dict) -> list[str]:
    """Run graph and return list of node names visited."""
    visited = []
    async for event in graph.astream_events(state, config=config, version="v2"):
        if event["event"] == "on_chain_start" and event.get("name") not in (
            "LangGraph", "ChannelWrite", "ChannelRead", "RunnableSequence",
        ):
            name = event.get("name", "")
            if name and name not in visited:
                visited.append(name)
    return visited


# ---------------------------------------------------------------------------
# Integration tests
# ---------------------------------------------------------------------------

@pytest.mark.integration
@pytest.mark.asyncio
async def test_general_query_skips_pipeline(graph, thread_config):
    """'What is an ETF?' routes to advisor_copilot without running intake/risk/strategy."""
    from app.agents.state import make_initial_state

    state = make_initial_state("e2e-test-session")
    state["messages"] = [HumanMessage(content="What is an ETF?")]
    state["intent"] = "general"  # pre-set to bypass intent_classifier LLM call

    nodes = await _collect_nodes(graph, state, thread_config)

    assert "advisor_copilot" in nodes or "guardrail_output" in nodes
    assert "risk_assessment" not in nodes
    assert "strategy" not in nodes
    assert "intake" not in nodes


@pytest.mark.integration
@pytest.mark.asyncio
async def test_full_analysis_pipeline_runs_all_stages(graph, thread_config):
    """full_analysis with complete profile runs risk → strategy → scoring → advisor."""
    from app.agents.state import make_initial_state

    state = make_initial_state("e2e-full-analysis")
    state.update({
        "messages": [HumanMessage(content="Build me an investment plan")],
        "intent": "full_analysis",
        "intake_complete": True,
        "user_profile": {
            "risk_tolerance": "high",
            "investment_horizon_years": 10,
            "investment_amount_usd": 100_000,
            "portfolio": {"SPY": 0.5, "BND": 0.5},
        },
    })

    async with _patch_market_data():
        nodes = await _collect_nodes(graph, state, thread_config)

    assert "risk_assessment" in nodes
    assert "strategy" in nodes
    assert "scoring" in nodes
    assert "advisor_copilot" in nodes
    assert "guardrail_output" in nodes


@pytest.mark.integration
@pytest.mark.asyncio
async def test_risk_analysis_skips_strategy_and_scoring(graph, thread_config):
    """risk_analysis intent runs only risk_assessment, skips strategy and scoring."""
    from app.agents.state import make_initial_state

    state = make_initial_state("e2e-risk-only")
    state.update({
        "messages": [HumanMessage(content="How risky is my portfolio?")],
        "intent": "risk_analysis",
        "intake_complete": True,
        "user_profile": {
            "risk_tolerance": "moderate",
            "investment_horizon_years": 5,
            "investment_amount_usd": 50_000,
            "portfolio": {"SPY": 0.6, "BND": 0.4},
        },
    })

    async with _patch_market_data():
        nodes = await _collect_nodes(graph, state, thread_config)

    assert "risk_assessment" in nodes
    assert "strategy" not in nodes
    assert "scoring" not in nodes
    assert "advisor_copilot" in nodes


@pytest.mark.integration
@pytest.mark.asyncio
async def test_iteration_limit_terminates_graph(graph, thread_config):
    """Graph with iteration_count near MAX_ITERATIONS terminates gracefully."""
    from app.agents.state import make_initial_state

    state = make_initial_state("e2e-iter-limit")
    state.update({
        "messages": [HumanMessage(content="Test iteration limit")],
        "intent": "full_analysis",
        "iteration_count": 49,  # one step from limit
    })

    # Should terminate quickly without hanging
    nodes = await _collect_nodes(graph, state, thread_config)
    # With iteration_count=49, supervisor should route to END after one tick
    assert "advisor_copilot" not in nodes or len(nodes) < 5


@pytest.mark.integration
@pytest.mark.asyncio
async def test_error_state_routes_to_error_handler(graph, thread_config):
    """When error field is set, supervisor routes to error_handler."""
    from app.agents.state import make_initial_state

    state = make_initial_state("e2e-error-route")
    state.update({
        "messages": [HumanMessage(content="Test error routing")],
        "intent": "general",
        "error": "injection_detected",
    })

    nodes = await _collect_nodes(graph, state, thread_config)
    assert "error_handler" in nodes


@pytest.mark.integration
@pytest.mark.asyncio
async def test_advisor_report_not_regenerated_without_new_message(graph, thread_config):
    """After advisor runs once, calling with no new HumanMessage terminates at END."""
    from app.agents.state import make_initial_state

    # First pass: complete state, advisor already ran
    state = make_initial_state("e2e-no-regen")
    state.update({
        "messages": [
            HumanMessage(content="Build me a plan"),
            AIMessage(content="Here is your plan.", name="advisor_copilot"),
        ],
        "intent": "full_analysis",
        "intake_complete": True,
        "user_profile": {
            "risk_tolerance": "high",
            "investment_horizon_years": 10,
            "investment_amount_usd": 100_000,
            "portfolio": {"SPY": 1.0},
        },
        "risk_metrics": {
            "sharpe_ratio": 1.0,
            "volatility": 0.15,
            "max_drawdown": -0.10,
            "risk_flags": [],
        },
        "allocation_result": {
            "weights": {"SPY": 1.0},
            "expected_return": 0.09,
            "expected_volatility": 0.15,
            "strategy_rationale": "100% equity",
        },
        "scoring_result": {"composite_score": 65.0, "breakdown": {}},
        "advisor_report_generated": True,
    })

    nodes = await _collect_nodes(graph, state, thread_config)
    # With advisor already done and last message is AI, graph should reach END immediately
    assert "advisor_copilot" not in nodes


@pytest.mark.integration
@pytest.mark.asyncio
async def test_effective_intake_complete_bypasses_intake(graph, thread_config):
    """Profile with all 3 fields + portfolio key bypasses intake even if flag is False."""
    from app.agents.state import make_initial_state

    state = make_initial_state("e2e-effective-intake")
    state.update({
        "messages": [HumanMessage(content="Score my portfolio")],
        "intent": "score_portfolio",
        "intake_complete": False,  # flag not set
        "user_profile": {
            "risk_tolerance": "low",
            "investment_horizon_years": 3,
            "investment_amount_usd": 30_000,
            "portfolio": {"BND": 0.8, "GLD": 0.2},  # portfolio key present
        },
    })

    async with _patch_market_data():
        nodes = await _collect_nodes(graph, state, thread_config)

    # Should have gone to risk_assessment, skipping intake
    assert "intake" not in nodes
    assert "risk_assessment" in nodes
