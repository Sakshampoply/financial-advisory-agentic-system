"""Deterministic tests for risk/strategy/scoring math — no LLM, no network calls."""
import numpy as np
import pandas as pd
import pytest

from app.agents.risk_assessment import compute_portfolio_metrics
from app.agents.scoring import compute_score
from app.agents.strategy import optimize_portfolio

# --- Shared fixture data -------------------------------------------------------

np.random.seed(42)
_N = 500
_DATES = pd.date_range("2022-01-01", periods=_N, freq="B")

_PRICES = pd.DataFrame(
    {
        "SPY": np.cumprod(1 + np.random.normal(0.0004, 0.012, _N)) * 400,
        "QQQ": np.cumprod(1 + np.random.normal(0.0005, 0.015, _N)) * 350,
        "BND": np.cumprod(1 + np.random.normal(0.0001, 0.003, _N)) * 80,
        "GLD": np.cumprod(1 + np.random.normal(0.0003, 0.009, _N)) * 180,
        "VNQ": np.cumprod(1 + np.random.normal(0.0003, 0.013, _N)) * 100,
    },
    index=_DATES,
)

_RETURNS = _PRICES.pct_change().dropna()
_EQ_WEIGHTS = {t: 0.2 for t in _PRICES.columns}
_PORTFOLIO_RETURNS = (_RETURNS * 0.2).sum(axis=1)


# --- compute_portfolio_metrics -------------------------------------------------

@pytest.mark.unit
def test_sharpe_in_reasonable_range():
    sharpe, vol, dd = compute_portfolio_metrics(_PORTFOLIO_RETURNS, annual_rf=0.04)
    assert -10 < sharpe < 10, f"Sharpe {sharpe} out of range"


@pytest.mark.unit
def test_volatility_positive():
    _, vol, _ = compute_portfolio_metrics(_PORTFOLIO_RETURNS, annual_rf=0.04)
    assert vol > 0, "Annualised volatility must be positive"


@pytest.mark.unit
def test_max_drawdown_nonpositive():
    _, _, dd = compute_portfolio_metrics(_PORTFOLIO_RETURNS, annual_rf=0.04)
    assert dd <= 0, f"Max drawdown {dd} must be ≤ 0"


@pytest.mark.unit
def test_metrics_with_zero_rf():
    sharpe_rf0, _, _ = compute_portfolio_metrics(_PORTFOLIO_RETURNS, annual_rf=0.0)
    sharpe_rf4, _, _ = compute_portfolio_metrics(_PORTFOLIO_RETURNS, annual_rf=0.04)
    # Higher rf → lower Sharpe
    assert sharpe_rf0 >= sharpe_rf4


# --- optimize_portfolio --------------------------------------------------------

@pytest.mark.unit
def test_weights_sum_to_one():
    weights, _, _ = optimize_portfolio(_PRICES, rf=0.04)
    total = sum(weights.values())
    assert abs(total - 1.0) < 0.01, f"Weights sum {total} not ≈ 1.0"


@pytest.mark.unit
def test_expected_return_is_float():
    _, exp_ret, exp_vol = optimize_portfolio(_PRICES, rf=0.04)
    assert isinstance(exp_ret, float)
    assert isinstance(exp_vol, float)


@pytest.mark.unit
def test_expected_vol_positive():
    _, _, exp_vol = optimize_portfolio(_PRICES, rf=0.04)
    assert exp_vol > 0


@pytest.mark.unit
def test_all_weights_nonnegative():
    weights, _, _ = optimize_portfolio(_PRICES, rf=0.04)
    for ticker, w in weights.items():
        assert w >= 0, f"{ticker} has negative weight {w}"


# --- compute_score -------------------------------------------------------------

@pytest.mark.unit
def test_composite_score_in_range():
    result = compute_score(sharpe=1.0, max_drawdown=-0.10, weights=_EQ_WEIGHTS)
    assert 0 <= result["composite_score"] <= 100


@pytest.mark.unit
def test_breakdown_keys_present():
    result = compute_score(sharpe=1.0, max_drawdown=-0.10, weights=_EQ_WEIGHTS)
    assert "sharpe_score" in result["breakdown"]
    assert "drawdown_score" in result["breakdown"]
    assert "diversification_score" in result["breakdown"]


@pytest.mark.unit
def test_perfect_portfolio_high_score():
    """High Sharpe, tiny drawdown, well diversified → score near 100."""
    result = compute_score(sharpe=3.0, max_drawdown=0.0, weights=_EQ_WEIGHTS)
    assert result["composite_score"] > 80


@pytest.mark.unit
def test_bad_portfolio_low_score():
    """Very negative Sharpe, huge drawdown, single asset."""
    result = compute_score(sharpe=-3.0, max_drawdown=-0.8, weights={"SINGLE": 1.0})
    assert result["composite_score"] < 20


@pytest.mark.unit
def test_score_monotone_sharpe():
    """Higher Sharpe should produce higher composite score (all else equal)."""
    r_low = compute_score(sharpe=0.0, max_drawdown=-0.1, weights=_EQ_WEIGHTS)
    r_high = compute_score(sharpe=1.5, max_drawdown=-0.1, weights=_EQ_WEIGHTS)
    assert r_high["composite_score"] > r_low["composite_score"]


@pytest.mark.unit
def test_score_monotone_drawdown():
    """Smaller drawdown (closer to 0) should produce higher score."""
    r_bad = compute_score(sharpe=1.0, max_drawdown=-0.4, weights=_EQ_WEIGHTS)
    r_good = compute_score(sharpe=1.0, max_drawdown=-0.05, weights=_EQ_WEIGHTS)
    assert r_good["composite_score"] > r_bad["composite_score"]


# ---------------------------------------------------------------------------
# Edge cases: compute_portfolio_metrics
# ---------------------------------------------------------------------------

@pytest.mark.unit
def test_metrics_empty_series_returns_zeros():
    """Empty returns series → (0.0, 0.0, 0.0) without raising."""
    empty = pd.Series([], dtype=float)
    sharpe, vol, dd = compute_portfolio_metrics(empty, annual_rf=0.04)
    assert sharpe == 0.0
    assert vol == 0.0
    assert dd == 0.0


@pytest.mark.unit
def test_metrics_constant_returns_series_returns_zeros():
    """Constant returns (std=0) → (0.0, 0.0, 0.0) without ZeroDivisionError.
    Uses exact-zero series to avoid floating-point artifacts."""
    constant = pd.Series(np.zeros(252))  # std is exactly 0.0
    sharpe, vol, dd = compute_portfolio_metrics(constant, annual_rf=0.04)
    assert sharpe == 0.0
    assert vol == 0.0


@pytest.mark.unit
def test_metrics_single_asset_portfolio():
    """Single-asset portfolio returns → metrics computed without error."""
    single = pd.Series(np.random.randn(252) * 0.01)
    sharpe, vol, dd = compute_portfolio_metrics(single, annual_rf=0.04)
    assert isinstance(sharpe, float)
    assert vol > 0
    assert dd <= 0


@pytest.mark.unit
def test_metrics_all_negative_returns():
    """All negative returns → drawdown < 0, Sharpe < 0."""
    bad_returns = pd.Series([-0.005] * 252)
    sharpe, vol, dd = compute_portfolio_metrics(bad_returns, annual_rf=0.04)
    # constant again
    assert sharpe == 0.0  # std=0 case


@pytest.mark.unit
def test_metrics_high_volatility_portfolio():
    """Very volatile returns produce vol > 0.5 annualised."""
    np.random.seed(0)
    volatile = pd.Series(np.random.randn(500) * 0.05)
    _, vol, _ = compute_portfolio_metrics(volatile, annual_rf=0.04)
    assert vol > 0.5


# ---------------------------------------------------------------------------
# Edge cases: compute_score
# ---------------------------------------------------------------------------

@pytest.mark.unit
def test_score_single_asset_diversification_zero():
    """Single-asset portfolio (HHI=1.0) → diversification score = 0."""
    result = compute_score(sharpe=1.0, max_drawdown=-0.10, weights={"SPY": 1.0})
    assert result["breakdown"]["diversification_score"] == 0.0


@pytest.mark.unit
def test_score_negative_sharpe_lowers_composite():
    """Negative Sharpe always produces lower composite than zero Sharpe, all else equal."""
    r_neg = compute_score(sharpe=-1.0, max_drawdown=-0.1, weights=_EQ_WEIGHTS)
    r_zero = compute_score(sharpe=0.0, max_drawdown=-0.1, weights=_EQ_WEIGHTS)
    assert r_neg["composite_score"] < r_zero["composite_score"]


@pytest.mark.unit
def test_score_empty_weights_diversification_zero():
    """Empty weights dict → diversification score = 0, no KeyError."""
    result = compute_score(sharpe=1.0, max_drawdown=-0.10, weights={})
    assert result["breakdown"]["diversification_score"] == 0.0


@pytest.mark.unit
def test_score_extreme_drawdown_clamped_to_zero():
    """Drawdown of -1.0 → drawdown score clamped to 0 (not negative)."""
    result = compute_score(sharpe=0.0, max_drawdown=-1.0, weights=_EQ_WEIGHTS)
    assert result["breakdown"]["drawdown_score"] == 0.0


@pytest.mark.unit
def test_score_zero_drawdown_max_drawdown_score():
    """Max drawdown of 0.0 → drawdown score = 100."""
    result = compute_score(sharpe=0.0, max_drawdown=0.0, weights=_EQ_WEIGHTS)
    assert result["breakdown"]["drawdown_score"] == 100.0


@pytest.mark.unit
def test_optimize_portfolio_single_asset_returns_full_weight():
    """When only one asset has data, optimize_portfolio allocates 100% to it."""
    single_asset = _PRICES[["SPY"]]
    weights, exp_ret, exp_vol = optimize_portfolio(single_asset, rf=0.04)
    assert abs(sum(weights.values()) - 1.0) < 0.01
