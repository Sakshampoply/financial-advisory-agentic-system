"""Unit tests for market data clients — all network calls mocked."""
import pickle
import pytest
import pandas as pd
import numpy as np
from unittest.mock import AsyncMock, MagicMock, patch, call


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_ohlcv(n: int = 60) -> pd.DataFrame:
    """Return a minimal valid OHLCV DataFrame."""
    dates = pd.date_range("2024-01-01", periods=n, freq="B")
    close = np.cumprod(1 + np.random.randn(n) * 0.01) * 100
    return pd.DataFrame(
        {"Open": close, "High": close * 1.01, "Low": close * 0.99, "Close": close, "Volume": 1_000_000},
        index=dates,
    )


def _empty_df() -> pd.DataFrame:
    return pd.DataFrame()


# ---------------------------------------------------------------------------
# yfinance_client: .NS fallback logic
# ---------------------------------------------------------------------------

@pytest.mark.unit
def test_ns_fallback_tried_when_bare_ticker_fails(mocker):
    """INFOSYS (empty) → INFOSYS.NS (valid) is attempted automatically."""
    valid_df = _make_ohlcv()

    call_count = {"n": 0}

    def fake_download(ticker, **kwargs):
        call_count["n"] += 1
        if ticker == "INFOSYS":
            return _empty_df()
        if ticker == "INFOSYS.NS":
            return valid_df
        return _empty_df()

    mocker.patch("yfinance.download", side_effect=fake_download)

    from app.market_data.yfinance_client import _download_ohlcv
    result = _download_ohlcv("INFOSYS")

    assert result is not None
    assert len(result) == len(valid_df)
    assert call_count["n"] == 2  # tried original then .NS


@pytest.mark.unit
def test_ns_fallback_not_applied_to_hyphen_tickers(mocker):
    """BRK-B contains a hyphen → .NS suffix must NOT be appended."""
    calls = []

    def fake_download(ticker, **kwargs):
        calls.append(ticker)
        return _empty_df()

    mocker.patch("yfinance.download", side_effect=fake_download)

    from app.market_data.yfinance_client import _download_ohlcv
    _download_ohlcv("BRK-B")

    assert "BRK-B.NS" not in calls
    assert "BRK-B" in calls


@pytest.mark.unit
def test_ns_fallback_not_applied_to_dot_tickers(mocker):
    """VNQ.NS is already suffixed — no further .NS appended."""
    calls = []

    def fake_download(ticker, **kwargs):
        calls.append(ticker)
        return _empty_df()

    mocker.patch("yfinance.download", side_effect=fake_download)

    from app.market_data.yfinance_client import _download_ohlcv
    _download_ohlcv("RELIANCE.NS")

    # RELIANCE.NS is not purely alphabetic — no extra .NS
    assert calls == ["RELIANCE.NS"]


@pytest.mark.unit
def test_pure_alpha_ticker_tries_both_variants(mocker):
    """SPY is alpha-only → tries SPY then SPY.NS before giving up."""
    calls = []

    def fake_download(ticker, **kwargs):
        calls.append(ticker)
        return _empty_df()

    mocker.patch("yfinance.download", side_effect=fake_download)

    from app.market_data.yfinance_client import _download_ohlcv
    result = _download_ohlcv("SPY")

    assert result is None
    assert calls == ["SPY", "SPY.NS"]


@pytest.mark.unit
def test_invalid_ticker_format_returns_none():
    """Ticker with invalid characters bypasses download and returns None."""
    from app.market_data.yfinance_client import _download_ohlcv
    result = _download_ohlcv("BAD TICKER!")
    assert result is None


@pytest.mark.unit
def test_multiindex_columns_flattened(mocker):
    """MultiIndex columns from yf.download are flattened to single-level."""
    import pandas as pd

    close = np.ones(60) * 100
    dates = pd.date_range("2024-01-01", periods=60, freq="B")
    mi_cols = pd.MultiIndex.from_tuples(
        [("Open", "SPY"), ("High", "SPY"), ("Low", "SPY"), ("Close", "SPY"), ("Volume", "SPY")]
    )
    df = pd.DataFrame(
        np.column_stack([close, close * 1.01, close * 0.99, close, np.ones(60) * 1_000_000]),
        index=dates,
        columns=mi_cols,
    )

    mocker.patch("yfinance.download", return_value=df)

    from app.market_data.yfinance_client import _download_ohlcv
    result = _download_ohlcv("SPY")

    assert result is not None
    assert not isinstance(result.columns, pd.MultiIndex)
    assert "Close" in result.columns


# ---------------------------------------------------------------------------
# Cache: get_ohlcv cache hit / miss
# ---------------------------------------------------------------------------

@pytest.mark.unit
@pytest.mark.asyncio
async def test_cache_hit_skips_download(mocker, mock_redis):
    """get_ohlcv returns cached DataFrame without calling yf.download."""
    cached_df = _make_ohlcv()
    mock_redis.get = AsyncMock(return_value=pickle.dumps(cached_df))

    download_spy = mocker.patch("yfinance.download")

    from app.market_data.yfinance_client import get_ohlcv
    result = await get_ohlcv("SPY", mock_redis)

    assert result is not None
    assert len(result) == len(cached_df)
    download_spy.assert_not_called()


@pytest.mark.unit
@pytest.mark.asyncio
async def test_cache_miss_calls_download_and_stores(mocker, mock_redis):
    """Cache miss → yf.download called → result stored in Redis."""
    mock_redis.get = AsyncMock(return_value=None)
    mock_redis.set = AsyncMock(return_value=True)

    valid_df = _make_ohlcv()
    mocker.patch("yfinance.download", return_value=valid_df)

    from app.market_data.yfinance_client import get_ohlcv
    result = await get_ohlcv("SPY", mock_redis)

    assert result is not None
    mock_redis.set.assert_called_once()
    # Verify the key pattern
    key_used = mock_redis.set.call_args[0][0]
    assert key_used == "mkt:ohlcv:SPY"


@pytest.mark.unit
@pytest.mark.asyncio
async def test_cache_miss_with_none_result_not_stored(mocker, mock_redis):
    """If download returns None (invalid ticker), nothing is stored in Redis."""
    mock_redis.get = AsyncMock(return_value=None)
    mock_redis.set = AsyncMock(return_value=True)

    mocker.patch("yfinance.download", return_value=_empty_df())

    from app.market_data.yfinance_client import get_ohlcv
    result = await get_ohlcv("INVALID", mock_redis)

    assert result is None
    mock_redis.set.assert_not_called()


# ---------------------------------------------------------------------------
# FRED client
# ---------------------------------------------------------------------------

@pytest.mark.unit
@pytest.mark.asyncio
async def test_fred_timeout_returns_none(mocker, mock_redis):
    """FRED API timeout → get_risk_free_rate returns None without raising."""
    import httpx
    mock_redis.get = AsyncMock(return_value=None)

    mocker.patch("httpx.AsyncClient.__aenter__", side_effect=httpx.TimeoutException("timeout"))

    with patch("app.market_data.fred_client.settings") as mock_settings:
        mock_settings.FRED_API_KEY = "test-key"
        from app.market_data.fred_client import get_risk_free_rate
        result = await get_risk_free_rate(mock_redis)

    assert result is None


@pytest.mark.unit
@pytest.mark.asyncio
async def test_fred_missing_api_key_returns_none(mock_redis):
    """With no FRED API key configured, get_risk_free_rate returns None."""
    mock_redis.get = AsyncMock(return_value=None)

    with patch("app.market_data.fred_client.settings") as mock_settings:
        mock_settings.FRED_API_KEY = None
        from app.market_data.fred_client import get_risk_free_rate
        result = await get_risk_free_rate(mock_redis)

    assert result is None


@pytest.mark.unit
@pytest.mark.asyncio
async def test_fred_valid_response_returns_decimal(mocker, mock_redis):
    """FRED returns 5.3% → get_risk_free_rate returns 0.053."""
    mock_redis.get = AsyncMock(return_value=None)
    mock_redis.set = AsyncMock(return_value=True)

    mock_resp = MagicMock()
    mock_resp.json.return_value = {
        "observations": [{"value": "5.3"}, {"value": "5.1"}]
    }

    mock_client = AsyncMock()
    mock_client.get = AsyncMock(return_value=mock_resp)

    with patch("app.market_data.fred_client.settings") as mock_settings:
        mock_settings.FRED_API_KEY = "test-key"
        with patch("httpx.AsyncClient") as mock_httpx:
            mock_httpx.return_value.__aenter__ = AsyncMock(return_value=mock_client)
            mock_httpx.return_value.__aexit__ = AsyncMock(return_value=False)
            from app.market_data import fred_client
            # Reload to pick up patched settings
            result = await fred_client._fetch_fred("DGS3MO")

    assert result == 5.3


# ---------------------------------------------------------------------------
# Alpha Vantage client
# ---------------------------------------------------------------------------

@pytest.mark.unit
@pytest.mark.asyncio
async def test_alpha_vantage_missing_api_key_returns_none(mock_redis):
    """With no Alpha Vantage key, get_fundamentals returns None without HTTP call."""
    mock_redis.get = AsyncMock(return_value=None)

    with patch("app.market_data.alpha_vantage_client.settings") as mock_settings:
        mock_settings.ALPHA_VANTAGE_API_KEY = None
        from app.market_data.alpha_vantage_client import get_fundamentals
        result = await get_fundamentals("AAPL", mock_redis)

    assert result is None


@pytest.mark.unit
@pytest.mark.asyncio
async def test_alpha_vantage_no_symbol_returns_none(mocker, mock_redis):
    """Alpha Vantage response without 'Symbol' key → get_fundamentals returns None."""
    mock_redis.get = AsyncMock(return_value=None)

    mock_resp = MagicMock()
    mock_resp.json.return_value = {"Note": "API rate limit exceeded"}

    mock_client = AsyncMock()
    mock_client.get = AsyncMock(return_value=mock_resp)

    with patch("app.market_data.alpha_vantage_client.settings") as mock_settings:
        mock_settings.ALPHA_VANTAGE_API_KEY = "test-key"
        with patch("httpx.AsyncClient") as mock_httpx:
            mock_httpx.return_value.__aenter__ = AsyncMock(return_value=mock_client)
            mock_httpx.return_value.__aexit__ = AsyncMock(return_value=False)
            from app.market_data.alpha_vantage_client import get_fundamentals
            result = await get_fundamentals("AAPL", mock_redis)

    assert result is None


@pytest.mark.unit
@pytest.mark.asyncio
async def test_alpha_vantage_cache_hit_skips_http(mocker, mock_redis):
    """get_fundamentals returns cached result without making HTTP request."""
    cached = {"pe_ratio": 25.0, "eps": 6.1, "dividend_yield": 0.005, "52_week_high": 200.0,
               "52_week_low": 140.0, "beta": 1.2}
    mock_redis.get = AsyncMock(return_value=pickle.dumps(cached))

    http_spy = mocker.patch("httpx.AsyncClient")

    from app.market_data.alpha_vantage_client import get_fundamentals
    result = await get_fundamentals("AAPL", mock_redis)

    assert result == cached
    http_spy.assert_not_called()
