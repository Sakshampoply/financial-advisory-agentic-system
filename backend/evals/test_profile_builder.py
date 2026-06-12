"""Unit tests for profile_builder logic — no network or DB calls."""
import pytest
from unittest.mock import AsyncMock, MagicMock, patch


def _make_mongo_record(holdings: list[dict], account_value: float | None = None) -> dict:
    extraction = {"holdings": holdings}
    if account_value is not None:
        extraction["account_value"] = account_value
    return {"session_id": "test", "extraction": extraction}


def _make_async_cursor(records: list[dict]):
    """Return an async iterator that yields the given records."""
    async def _iter():
        for r in records:
            yield r
    return _iter()


async def _run_profile_builder(records: list[dict], existing_profile: dict | None = None):
    """Run profile_builder_node with mocked MongoDB returning the given records."""
    from app.agents.profile_builder import profile_builder_node

    state = {
        "session_id": "test",
        "user_profile": existing_profile or {},
        "messages": [],
        "intake_complete": False,
        "documents_uploaded": [],
        "documents_extracted": True,
        "risk_metrics": None,
        "allocation_result": None,
        "scoring_result": None,
        "iteration_count": 0,
        "error": None,
        "intent": None,
        "advisor_report_generated": False,
    }

    mock_collection = MagicMock()
    mock_collection.find.return_value = _make_async_cursor(records)
    mock_mongo = MagicMock()
    mock_mongo.__getitem__ = MagicMock(return_value=mock_collection)

    with patch("app.agents.profile_builder.get_mongo_db", return_value=mock_mongo):
        return await profile_builder_node(state)


# ---------------------------------------------------------------------------
# intake_complete promotion
# ---------------------------------------------------------------------------

@pytest.mark.unit
@pytest.mark.asyncio
async def test_intake_complete_set_when_all_required_fields_present():
    """profile_builder sets intake_complete when risk/horizon/amount already in profile."""
    existing = {
        "risk_tolerance": "high",
        "investment_horizon_years": 10,
        "investment_amount_usd": 50_000,
    }
    records = [_make_mongo_record([{"ticker": "AAPL", "value": 10000}])]
    result = await _run_profile_builder(records, existing_profile=existing)
    assert result.get("intake_complete") is True


@pytest.mark.unit
@pytest.mark.asyncio
async def test_intake_complete_not_set_when_risk_tolerance_missing():
    """Without risk_tolerance, intake_complete is not set."""
    existing = {
        "investment_horizon_years": 10,
        "investment_amount_usd": 50_000,
    }
    records = [_make_mongo_record([{"ticker": "AAPL", "value": 10000}])]
    result = await _run_profile_builder(records, existing_profile=existing)
    assert not result.get("intake_complete")


@pytest.mark.unit
@pytest.mark.asyncio
async def test_intake_complete_not_set_when_horizon_missing():
    """Without investment_horizon_years, intake_complete is not set."""
    existing = {
        "risk_tolerance": "moderate",
        "investment_amount_usd": 50_000,
    }
    records = [_make_mongo_record([{"ticker": "AAPL", "value": 10000}])]
    result = await _run_profile_builder(records, existing_profile=existing)
    assert not result.get("intake_complete")


# ---------------------------------------------------------------------------
# portfolio weight normalisation
# ---------------------------------------------------------------------------

@pytest.mark.unit
@pytest.mark.asyncio
async def test_portfolio_weights_normalise_to_one():
    """Holdings from a doc → portfolio weights sum ≈ 1.0."""
    records = [
        _make_mongo_record([
            {"ticker": "AAPL", "value": 3000},
            {"ticker": "MSFT", "value": 7000},
        ])
    ]
    result = await _run_profile_builder(records)
    portfolio = result["user_profile"]["portfolio"]
    assert abs(sum(portfolio.values()) - 1.0) < 0.01


@pytest.mark.unit
@pytest.mark.asyncio
async def test_overlapping_tickers_across_docs_are_summed():
    """Same ticker appearing in two docs → values added before normalisation."""
    records = [
        _make_mongo_record([{"ticker": "AAPL", "value": 4000}]),
        _make_mongo_record([{"ticker": "AAPL", "value": 6000}]),
    ]
    result = await _run_profile_builder(records)
    portfolio = result["user_profile"]["portfolio"]
    # AAPL has 10000 total — only ticker, so weight should be 1.0
    assert "AAPL" in portfolio
    assert abs(portfolio["AAPL"] - 1.0) < 0.01


@pytest.mark.unit
@pytest.mark.asyncio
async def test_zero_value_holdings_excluded():
    """Holdings with value=0 are silently dropped from portfolio dict."""
    records = [
        _make_mongo_record([
            {"ticker": "AAPL", "value": 5000},
            {"ticker": "ZERO", "value": 0},
        ])
    ]
    result = await _run_profile_builder(records)
    portfolio = result["user_profile"]["portfolio"]
    assert "ZERO" not in portfolio
    assert "AAPL" in portfolio


@pytest.mark.unit
@pytest.mark.asyncio
async def test_tickers_uppercased():
    """Ticker symbols extracted from docs are normalised to uppercase."""
    records = [_make_mongo_record([{"ticker": "aapl", "value": 5000}])]
    result = await _run_profile_builder(records)
    assert "AAPL" in result["user_profile"]["portfolio"]


# ---------------------------------------------------------------------------
# investment_amount_usd handling
# ---------------------------------------------------------------------------

@pytest.mark.unit
@pytest.mark.asyncio
async def test_investment_amount_not_overwritten_if_already_in_profile():
    """If user_profile already has investment_amount_usd, profile_builder does not replace it."""
    existing = {"investment_amount_usd": 99_999}
    records = [_make_mongo_record([], account_value=200_000)]
    result = await _run_profile_builder(records, existing_profile=existing)
    assert result["user_profile"]["investment_amount_usd"] == 99_999


@pytest.mark.unit
@pytest.mark.asyncio
async def test_investment_amount_set_from_document_if_missing():
    """If user_profile has no investment_amount_usd, profile_builder fills it from document."""
    existing: dict = {}
    records = [_make_mongo_record([], account_value=150_000)]
    result = await _run_profile_builder(records, existing_profile=existing)
    assert result["user_profile"].get("investment_amount_usd") == 150_000


@pytest.mark.unit
@pytest.mark.asyncio
async def test_largest_account_value_wins_across_docs():
    """When multiple docs provide account_value, the largest is used."""
    records = [
        _make_mongo_record([], account_value=50_000),
        _make_mongo_record([], account_value=200_000),
        _make_mongo_record([], account_value=100_000),
    ]
    result = await _run_profile_builder(records)
    assert result["user_profile"].get("investment_amount_usd") == 200_000


# ---------------------------------------------------------------------------
# portfolio key always set
# ---------------------------------------------------------------------------

@pytest.mark.unit
@pytest.mark.asyncio
async def test_portfolio_key_always_set_even_when_no_holdings():
    """Even when no holdings found in docs, profile['portfolio'] = {} (empty dict)."""
    records = [_make_mongo_record([])]
    result = await _run_profile_builder(records)
    assert "portfolio" in result["user_profile"]
    assert result["user_profile"]["portfolio"] == {}


@pytest.mark.unit
@pytest.mark.asyncio
async def test_portfolio_key_set_when_no_docs():
    """When no docs exist for session, portfolio is an empty dict (not absent)."""
    result = await _run_profile_builder([])
    assert "portfolio" in result["user_profile"]
    assert result["user_profile"]["portfolio"] == {}
