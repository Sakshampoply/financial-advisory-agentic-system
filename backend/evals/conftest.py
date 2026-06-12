"""Shared fixtures, pytest marks, and Langfuse scorer for the eval suite."""
import numpy as np
import pandas as pd
import pytest
from unittest.mock import AsyncMock, MagicMock
from langchain_core.messages import AIMessage, HumanMessage


# ---------- pytest marks ----------

def pytest_configure(config):
    config.addinivalue_line("markers", "unit: fast deterministic, no network or LLM")
    config.addinivalue_line("markers", "rag: requires live PostgreSQL with seeded KB")
    config.addinivalue_line("markers", "llm_eval: calls LLM-as-judge via OpenRouter")
    config.addinivalue_line("markers", "integration: full graph run, mocked network")


# ---------- OpenRouter API key guard ----------

def skip_if_no_openrouter():
    """Call at the top of llm_eval tests to skip when no API key is configured."""
    from app.config import settings
    if not settings.OPENROUTER_API_KEY:
        pytest.skip("OPENROUTER_API_KEY not configured — set it in ../.env to run LLM evals")


# ---------- deepeval judge model (OpenRouter) ----------

@pytest.fixture(scope="session")
def judge_model():
    """deepeval GPTModel wired to OpenRouter so no OPENAI_API_KEY is required.

    Falls back to None (deepeval uses its default) if OpenRouter key is absent.
    Tests that use this fixture should call skip_if_no_openrouter() themselves.

    Uses openai/gpt-4o-mini: cheapest model on OpenRouter that reliably emits
    the structured JSON schemas deepeval metrics require (FaithfulnessMetric,
    ContextualPrecisionMetric, etc.). Conversational models like deepseek-chat
    fail deepeval's JSON extraction schema non-deterministically.
    """
    try:
        from deepeval.models import GPTModel
        from app.config import settings
        if not settings.OPENROUTER_API_KEY:
            return None
        return GPTModel(
            model="openai/gpt-4o-mini",
            api_key=settings.OPENROUTER_API_KEY,
            base_url="https://openrouter.ai/api/v1",
        )
    except Exception:
        return None


# ---------- Langfuse scorer ----------

class _EvalScorer:
    """Wraps the Langfuse client for score submission compatible with v4 SDK.

    Langfuse v4 moved to an OTEL-based API — lf.trace() no longer exists.
    The correct path is: create_trace_id() → create_score(trace_id=...).
    Each eval score gets its own lightweight trace so it's visible in the
    Langfuse Scores tab and linked to a traceable entry.

    The interface is identical to what the tests already call:
        scorer.create_score(name=..., value=..., data_type=..., comment=...)
    """

    def __init__(self, lf):
        self._lf = lf

    def create_score(self, *, name: str, value: float, data_type: str | None = None, comment: str = "") -> None:
        try:
            tid = self._lf.create_trace_id()
            self._lf.create_score(
                trace_id=tid,
                name=name,
                value=value,
                comment=str(comment)[:500],
            )
            self._lf.flush()
        except Exception as e:
            print(f"\n  [Langfuse] score submission failed for '{name}': {e}")


@pytest.fixture(scope="session")
def langfuse_scorer():
    """Return _EvalScorer for score submission. Returns None if Langfuse is not configured.

    Uses explicit credentials from app.config.settings so that the Langfuse SDK
    does not need LANGFUSE_PUBLIC_KEY in os.environ — pydantic_settings loads
    ../.env into the settings model but does not write back to os.environ.
    """
    try:
        from langfuse import Langfuse
        from app.config import settings
        if not settings.LANGFUSE_PUBLIC_KEY:
            return None
        lf = Langfuse(
            public_key=settings.LANGFUSE_PUBLIC_KEY,
            secret_key=settings.LANGFUSE_SECRET_KEY,
            host=settings.LANGFUSE_HOST,
        )
        return _EvalScorer(lf)
    except Exception:
        return None


# ---------- Mock DB session ----------

@pytest.fixture
def mock_db():
    db = AsyncMock()
    db.execute = AsyncMock(return_value=MagicMock(fetchall=lambda: []))
    return db


# ---------- Mock Redis ----------

@pytest.fixture
def mock_redis():
    redis = AsyncMock()
    redis.get = AsyncMock(return_value=None)
    redis.set = AsyncMock(return_value=True)
    return redis


# ---------- Synthetic OHLCV ----------

@pytest.fixture(scope="session")
def synthetic_prices():
    """500 business days of seeded synthetic OHLCV for 5 tickers."""
    np.random.seed(42)
    dates = pd.date_range("2023-01-01", periods=500, freq="B")
    tickers = ["SPY", "QQQ", "BND", "GLD", "VNQ"]
    prices = {}
    for t in tickers:
        close = 100 * np.cumprod(1 + np.random.randn(500) * 0.01)
        prices[t] = pd.DataFrame(
            {
                "Open": close * 0.999,
                "High": close * 1.005,
                "Low": close * 0.995,
                "Close": close,
                "Volume": np.random.randint(1_000_000, 5_000_000, 500),
            },
            index=dates,
        )
    return prices


# ---------- Shared GraphState ----------

@pytest.fixture
def full_state():
    """A complete post-pipeline GraphState with all analysis fields populated."""
    return {
        "messages": [
            HumanMessage(content="Create an investment plan for me"),
            AIMessage(content="Thanks!", name="intake"),
        ],
        "session_id": "test-session-id",
        "intake_complete": True,
        "user_profile": {
            "risk_tolerance": "high",
            "investment_horizon_years": 10,
            "investment_amount_usd": 100_000,
            "portfolio": {"RELIANCE.NS": 0.20, "TCS.NS": 0.30, "INFY.NS": 0.50},
        },
        "documents_uploaded": [],
        "documents_extracted": False,
        "risk_metrics": {
            "sharpe_ratio": 1.25,
            "volatility": 0.18,
            "max_drawdown": -0.12,
            "risk_flags": ["Concentration in Indian equities"],
        },
        "allocation_result": {
            "weights": {"SPY": 0.50, "QQQ": 0.30, "GLD": 0.20},
            "expected_return": 0.12,
            "expected_volatility": 0.15,
            "strategy_rationale": "Diversified global allocation",
        },
        "scoring_result": {
            "composite_score": 72.0,
            "breakdown": {
                "sharpe_score": 75,
                "drawdown_score": 70,
                "diversification_score": 54,
            },
        },
        "iteration_count": 5,
        "error": None,
        "intent": "full_analysis",
        "advisor_report_generated": False,
    }


# ---------- Score helper ----------

def push_langfuse_score(scorer, trace_id: str, name: str, value: float, comment: str = ""):
    """Submit a numeric score to Langfuse. No-ops if scorer is None or submission fails."""
    if scorer is None:
        return
    try:
        scorer.create_score(
            trace_id=trace_id,
            name=name,
            value=value,
            comment=comment,
            data_type="NUMERIC",
        )
    except Exception:
        pass
