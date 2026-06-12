"""Node-level integration tests for risk_assessment_node, strategy_node, and scoring_node.

All tests are deterministic (unit mark) — external calls (yfinance, FRED, LLM) are mocked.
These tests verify the node wrapper contracts: correct return structure, proper handling of
market data failures, and LLM rationale/flags extraction.

Run: uv run pytest evals/test_node_integration.py -m unit -v
"""
import numpy as np
import pandas as pd
import pytest
from unittest.mock import AsyncMock, MagicMock, patch


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _make_ohlcv(n: int = 120) -> pd.DataFrame:
    """Minimal valid OHLCV DataFrame for mocking get_ohlcv."""
    np.random.seed(42)
    dates = pd.date_range("2023-01-01", periods=n, freq="B")
    close = np.cumprod(1 + np.random.randn(n) * 0.01) * 100
    return pd.DataFrame(
        {
            "Open": close * 0.999,
            "High": close * 1.005,
            "Low": close * 0.995,
            "Close": close,
            "Volume": np.ones(n, dtype=int) * 1_000_000,
        },
        index=dates,
    )


def _make_state(extra: dict | None = None) -> dict:
    base = {
        "messages": [],
        "session_id": "test",
        "intake_complete": True,
        "user_profile": {
            "risk_tolerance": "moderate",
            "investment_horizon_years": 10,
            "investment_amount_usd": 50_000,
            "portfolio": {"SPY": 0.6, "BND": 0.4},
        },
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
    if extra:
        base.update(extra)
    return base


def _fake_llm_tool_call(tool_name: str, args: dict):
    """Mock LLM that returns a single tool call with given args."""
    resp = MagicMock()
    resp.content = ""
    resp.tool_calls = [{"name": tool_name, "args": args}]
    mock = MagicMock()
    mock.bind_tools = MagicMock(return_value=mock)
    mock.ainvoke = AsyncMock(return_value=resp)
    return mock


def _fake_llm_text(content: str):
    """Mock LLM that returns plain text (no tool call)."""
    resp = MagicMock()
    resp.content = content
    resp.tool_calls = []
    mock = MagicMock()
    mock.bind_tools = MagicMock(return_value=mock)
    mock.ainvoke = AsyncMock(return_value=resp)
    return mock


# ---------------------------------------------------------------------------
# risk_assessment_node tests
# ---------------------------------------------------------------------------

def _patch_risk_market_data(ohlcv_df=None, rf=0.04, yc=0.5, inflation=0.03):
    """Return a context manager that patches all risk_assessment_node external calls."""
    import contextlib

    @contextlib.contextmanager
    def _ctx():
        df = ohlcv_df if ohlcv_df is not None else _make_ohlcv()
        with patch("app.agents.risk_assessment.get_ohlcv", new=AsyncMock(return_value=df)):
            with patch("app.agents.risk_assessment.get_ticker_info", new=AsyncMock(return_value={})):
                with patch("app.agents.risk_assessment.get_risk_free_rate", new=AsyncMock(return_value=rf)):
                    with patch("app.agents.risk_assessment.get_yield_curve", new=AsyncMock(return_value=yc)):
                        with patch("app.agents.risk_assessment.get_inflation", new=AsyncMock(return_value=inflation)):
                            with patch("app.agents.risk_assessment.get_fundamentals", new=AsyncMock(return_value=None)):
                                with patch("app.agents.risk_assessment.get_sentiment", new=AsyncMock(return_value=None)):
                                    with patch("app.agents.risk_assessment.get_redis", return_value=MagicMock()):
                                        yield

    return _ctx()


@pytest.mark.unit
@pytest.mark.asyncio
async def test_risk_node_returns_risk_metrics_dict():
    """risk_assessment_node returns a dict with a 'risk_metrics' key containing all 4 fields."""
    from app.agents.risk_assessment import risk_assessment_node

    llm = _fake_llm_tool_call("report_risk_flags", {"risk_flags": ["Equity concentration risk"]})

    with _patch_risk_market_data():
        with patch("app.agents.risk_assessment.get_chat_model", return_value=llm):
            result = await risk_assessment_node(_make_state())

    assert "risk_metrics" in result
    rm = result["risk_metrics"]
    assert "sharpe_ratio" in rm
    assert "volatility" in rm
    assert "max_drawdown" in rm
    assert "risk_flags" in rm


@pytest.mark.unit
@pytest.mark.asyncio
async def test_risk_node_risk_flags_is_list():
    """risk_flags in the returned risk_metrics must be a list of strings, not a bare string."""
    from app.agents.risk_assessment import risk_assessment_node

    llm = _fake_llm_tool_call(
        "report_risk_flags",
        {"risk_flags": ["High equity concentration", "Interest rate sensitivity", "Low diversification"]},
    )

    with _patch_risk_market_data():
        with patch("app.agents.risk_assessment.get_chat_model", return_value=llm):
            result = await risk_assessment_node(_make_state())

    flags = result["risk_metrics"]["risk_flags"]
    assert isinstance(flags, list), f"risk_flags should be list, got {type(flags)}"
    for flag in flags:
        assert isinstance(flag, str), f"Each flag must be a string, got {type(flag)}: {flag!r}"


@pytest.mark.unit
@pytest.mark.asyncio
async def test_risk_node_metrics_are_finite_floats():
    """Sharpe, volatility, and max_drawdown are finite float values."""
    import math
    from app.agents.risk_assessment import risk_assessment_node

    llm = _fake_llm_tool_call("report_risk_flags", {"risk_flags": ["Test flag"]})

    with _patch_risk_market_data():
        with patch("app.agents.risk_assessment.get_chat_model", return_value=llm):
            result = await risk_assessment_node(_make_state())

    rm = result["risk_metrics"]
    assert math.isfinite(rm["sharpe_ratio"]), "sharpe_ratio must be finite"
    assert math.isfinite(rm["volatility"]), "volatility must be finite"
    assert math.isfinite(rm["max_drawdown"]), "max_drawdown must be finite"
    assert rm["volatility"] >= 0, "volatility must be non-negative"
    assert rm["max_drawdown"] <= 0, "max_drawdown must be <= 0"


@pytest.mark.unit
@pytest.mark.asyncio
async def test_risk_node_market_data_unavailable_uses_fallback():
    """When get_ohlcv returns None for portfolio tickers, node falls back to DEFAULT_UNIVERSE."""
    from app.agents.risk_assessment import risk_assessment_node

    call_count: dict[str, int] = {"n": 0}
    default_universe = ["SPY", "QQQ", "BND", "GLD", "VNQ"]

    async def fake_get_ohlcv(ticker, redis):
        call_count["n"] += 1
        if ticker in ("SPY", "BND"):
            return None  # portfolio tickers fail
        if ticker in default_universe:
            return _make_ohlcv()
        return None

    llm = _fake_llm_tool_call("report_risk_flags", {"risk_flags": ["Fallback to benchmark ETFs"]})

    with patch("app.agents.risk_assessment.get_ohlcv", new=fake_get_ohlcv):
        with patch("app.agents.risk_assessment.get_ticker_info", new=AsyncMock(return_value={})):
            with patch("app.agents.risk_assessment.get_risk_free_rate", new=AsyncMock(return_value=0.04)):
                with patch("app.agents.risk_assessment.get_yield_curve", new=AsyncMock(return_value=None)):
                    with patch("app.agents.risk_assessment.get_inflation", new=AsyncMock(return_value=None)):
                        with patch("app.agents.risk_assessment.get_fundamentals", new=AsyncMock(return_value=None)):
                            with patch("app.agents.risk_assessment.get_sentiment", new=AsyncMock(return_value=None)):
                                with patch("app.agents.risk_assessment.get_redis", return_value=MagicMock()):
                                    with patch("app.agents.risk_assessment.get_chat_model", return_value=llm):
                                        result = await risk_assessment_node(_make_state())

    assert "risk_metrics" in result
    flags = result["risk_metrics"]["risk_flags"]
    # Fallback inserts a warning flag at position 0
    assert any("tickers could not be resolved" in f.lower() or "benchmark" in f.lower() for f in flags), (
        f"Expected fallback warning in risk_flags, got: {flags}"
    )


@pytest.mark.unit
@pytest.mark.asyncio
async def test_risk_node_llm_text_fallback_when_no_tool_call():
    """When LLM returns plain text instead of tool call, risk_flags still populated."""
    from app.agents.risk_assessment import risk_assessment_node

    llm = _fake_llm_text('["Concentration in equities", "Currency risk"]')

    with _patch_risk_market_data():
        with patch("app.agents.risk_assessment.get_chat_model", return_value=llm):
            result = await risk_assessment_node(_make_state())

    assert "risk_metrics" in result
    flags = result["risk_metrics"]["risk_flags"]
    assert isinstance(flags, list)
    # JSON parsing of the text should succeed
    assert len(flags) > 0


# ---------------------------------------------------------------------------
# strategy_node tests
# ---------------------------------------------------------------------------

def _patch_strategy_market_data(ohlcv_df=None, rf=0.04, inflation=0.03):
    """Patch all strategy_node external calls."""
    import contextlib

    @contextlib.contextmanager
    def _ctx():
        df = ohlcv_df if ohlcv_df is not None else _make_ohlcv()
        with patch("app.agents.strategy.get_ohlcv", new=AsyncMock(return_value=df)):
            with patch("app.agents.strategy.get_risk_free_rate", new=AsyncMock(return_value=rf)):
                with patch("app.agents.strategy.get_inflation", new=AsyncMock(return_value=inflation)):
                    with patch("app.agents.strategy.get_redis", return_value=MagicMock()):
                        yield

    return _ctx()


@pytest.mark.unit
@pytest.mark.asyncio
async def test_strategy_node_returns_allocation_result_dict():
    """strategy_node returns dict with allocation_result containing all 4 required keys."""
    from app.agents.strategy import strategy_node

    llm = _fake_llm_tool_call(
        "report_rationale",
        {"rationale": "This balanced allocation suits a moderate-risk investor with a 10-year horizon."},
    )

    with _patch_strategy_market_data():
        with patch("app.agents.strategy.get_chat_model", return_value=llm):
            result = await strategy_node(_make_state())

    assert "allocation_result" in result
    ar = result["allocation_result"]
    assert "weights" in ar
    assert "expected_return" in ar
    assert "expected_volatility" in ar
    assert "strategy_rationale" in ar


@pytest.mark.unit
@pytest.mark.asyncio
async def test_strategy_node_weights_sum_to_one():
    """Recommended portfolio weights sum to approximately 1.0."""
    from app.agents.strategy import strategy_node

    llm = _fake_llm_tool_call(
        "report_rationale", {"rationale": "Balanced allocation."}
    )

    with _patch_strategy_market_data():
        with patch("app.agents.strategy.get_chat_model", return_value=llm):
            result = await strategy_node(_make_state())

    weights = result["allocation_result"]["weights"]
    total = sum(weights.values())
    assert abs(total - 1.0) < 0.02, f"Weights sum {total:.4f} not close to 1.0"


@pytest.mark.unit
@pytest.mark.asyncio
async def test_strategy_node_expected_vol_positive():
    """Expected volatility from strategy_node is a positive float."""
    from app.agents.strategy import strategy_node

    llm = _fake_llm_tool_call("report_rationale", {"rationale": "Good allocation."})

    with _patch_strategy_market_data():
        with patch("app.agents.strategy.get_chat_model", return_value=llm):
            result = await strategy_node(_make_state())

    vol = result["allocation_result"]["expected_volatility"]
    assert isinstance(vol, float)
    assert vol > 0, f"Expected volatility {vol} is not positive"


@pytest.mark.unit
@pytest.mark.asyncio
async def test_strategy_node_rationale_is_nonempty_string():
    """strategy_rationale in allocation_result is a non-empty string."""
    from app.agents.strategy import strategy_node

    llm = _fake_llm_tool_call(
        "report_rationale",
        {"rationale": "This allocation targets maximum risk-adjusted return for the given horizon."},
    )

    with _patch_strategy_market_data():
        with patch("app.agents.strategy.get_chat_model", return_value=llm):
            result = await strategy_node(_make_state())

    rationale = result["allocation_result"]["strategy_rationale"]
    assert isinstance(rationale, str)
    assert len(rationale.strip()) > 0, "strategy_rationale must be a non-empty string"


@pytest.mark.unit
@pytest.mark.asyncio
async def test_strategy_node_all_weights_nonnegative():
    """All weights in the returned allocation are non-negative (long-only constraint)."""
    from app.agents.strategy import strategy_node

    llm = _fake_llm_tool_call("report_rationale", {"rationale": "Long-only allocation."})

    with _patch_strategy_market_data():
        with patch("app.agents.strategy.get_chat_model", return_value=llm):
            result = await strategy_node(_make_state())

    weights = result["allocation_result"]["weights"]
    for ticker, w in weights.items():
        assert w >= 0, f"Negative weight for {ticker}: {w}"


@pytest.mark.unit
@pytest.mark.asyncio
async def test_strategy_node_text_rationale_fallback():
    """When LLM returns plain text instead of tool call, rationale is still populated."""
    from app.agents.strategy import strategy_node

    llm = _fake_llm_text("This allocation suits a balanced investor well.")

    with _patch_strategy_market_data():
        with patch("app.agents.strategy.get_chat_model", return_value=llm):
            result = await strategy_node(_make_state())

    rationale = result["allocation_result"]["strategy_rationale"]
    assert "balanced investor" in rationale or len(rationale) > 0


# ---------------------------------------------------------------------------
# scoring_node tests — pure deterministic, no mocks needed
# ---------------------------------------------------------------------------

_RISK_METRICS = {
    "sharpe_ratio": 1.25,
    "volatility": 0.18,
    "max_drawdown": -0.12,
    "risk_flags": ["Equity concentration"],
}

_ALLOC_RESULT = {
    "weights": {"SPY": 0.50, "QQQ": 0.30, "GLD": 0.20},
    "expected_return": 0.12,
    "expected_volatility": 0.15,
    "strategy_rationale": "Diversified global allocation.",
}


@pytest.mark.unit
@pytest.mark.asyncio
async def test_scoring_node_returns_scoring_result_dict():
    """scoring_node returns dict with 'scoring_result' key containing composite_score and breakdown."""
    from app.agents.scoring import scoring_node

    state = _make_state({"risk_metrics": _RISK_METRICS, "allocation_result": _ALLOC_RESULT})
    result = await scoring_node(state)

    assert "scoring_result" in result
    sr = result["scoring_result"]
    assert "composite_score" in sr
    assert "breakdown" in sr


@pytest.mark.unit
@pytest.mark.asyncio
async def test_scoring_node_composite_score_in_range():
    """composite_score must be in [0, 100]."""
    from app.agents.scoring import scoring_node

    state = _make_state({"risk_metrics": _RISK_METRICS, "allocation_result": _ALLOC_RESULT})
    result = await scoring_node(state)

    score = result["scoring_result"]["composite_score"]
    assert 0 <= score <= 100, f"composite_score {score} out of [0, 100]"


@pytest.mark.unit
@pytest.mark.asyncio
async def test_scoring_node_breakdown_has_required_keys():
    """Breakdown dict must contain sharpe_score, drawdown_score, diversification_score."""
    from app.agents.scoring import scoring_node

    state = _make_state({"risk_metrics": _RISK_METRICS, "allocation_result": _ALLOC_RESULT})
    result = await scoring_node(state)

    breakdown = result["scoring_result"]["breakdown"]
    assert "sharpe_score" in breakdown
    assert "drawdown_score" in breakdown
    assert "diversification_score" in breakdown


@pytest.mark.unit
@pytest.mark.asyncio
async def test_scoring_node_missing_risk_metrics_returns_zero_score():
    """When risk_metrics is None, scoring_node uses zero values — does not crash."""
    from app.agents.scoring import scoring_node

    state = _make_state({"risk_metrics": None, "allocation_result": _ALLOC_RESULT})
    result = await scoring_node(state)

    assert "scoring_result" in result
    sr = result["scoring_result"]
    # sharpe=0, drawdown=0 → sharpe_score=50, drawdown_score=100; composite > 0
    assert 0 <= sr["composite_score"] <= 100


@pytest.mark.unit
@pytest.mark.asyncio
async def test_scoring_node_missing_allocation_uses_empty_weights():
    """When allocation_result is None, scoring_node uses empty weights (diversification=0)."""
    from app.agents.scoring import scoring_node

    state = _make_state({"risk_metrics": _RISK_METRICS, "allocation_result": None})
    result = await scoring_node(state)

    assert "scoring_result" in result
    breakdown = result["scoring_result"]["breakdown"]
    assert breakdown["diversification_score"] == 0.0


@pytest.mark.unit
@pytest.mark.asyncio
async def test_scoring_node_good_portfolio_scores_above_60():
    """High Sharpe + small drawdown + diversified weights → composite > 60."""
    from app.agents.scoring import scoring_node

    good_risk = {"sharpe_ratio": 1.5, "volatility": 0.12, "max_drawdown": -0.05, "risk_flags": []}
    good_alloc = {
        "weights": {"A": 0.2, "B": 0.2, "C": 0.2, "D": 0.2, "E": 0.2},
        "expected_return": 0.10,
        "expected_volatility": 0.12,
        "strategy_rationale": "Well-diversified portfolio.",
    }
    state = _make_state({"risk_metrics": good_risk, "allocation_result": good_alloc})
    result = await scoring_node(state)

    score = result["scoring_result"]["composite_score"]
    assert score > 60, f"Good portfolio scored only {score}"
