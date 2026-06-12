import logging

import numpy as np
import pandas as pd
from langchain_core.messages import HumanMessage, SystemMessage
from pypfopt import expected_returns, risk_models
from pypfopt.efficient_frontier import EfficientFrontier

from app.agents.state import AllocationResult, GraphState
from app.db.redis_client import get_redis
from app.llm.client import get_chat_model
from app.market_data.fred_client import get_inflation, get_risk_free_rate
from app.market_data.yfinance_client import get_ohlcv
from app.observability.langfuse_setup import traced_node

logger = logging.getLogger(__name__)

DEFAULT_UNIVERSE = ["SPY", "QQQ", "BND", "GLD", "VNQ"]

_RATIONALE_TOOL = {
    "type": "function",
    "function": {
        "name": "report_rationale",
        "description": "Report the strategy rationale for the recommended allocation.",
        "parameters": {
            "type": "object",
            "properties": {
                "rationale": {
                    "type": "string",
                    "description": "2-3 sentence explanation of why this allocation suits the investor.",
                }
            },
            "required": ["rationale"],
        },
    },
}


def optimize_portfolio(
    prices_df: pd.DataFrame, rf: float
) -> tuple[dict[str, float], float, float]:
    """Max-Sharpe optimisation. Returns (weights, expected_return, expected_vol). Pure function."""
    try:
        mu = expected_returns.mean_historical_return(prices_df)
        S = risk_models.sample_cov(prices_df)
        ef = EfficientFrontier(mu, S)
        ef.max_sharpe(risk_free_rate=rf)
        weights = {k: float(v) for k, v in ef.clean_weights().items() if float(v) > 1e-4}
        ret, vol, _ = ef.portfolio_performance(verbose=False, risk_free_rate=rf)
        return weights, float(ret), float(vol)
    except Exception as exc:
        logger.warning("PyPortfolioOpt optimisation failed: %s — using equal weights", exc)
        n = len(prices_df.columns)
        weights = {t: round(1.0 / n, 4) for t in prices_df.columns}
        rets = prices_df.pct_change().dropna()
        port_ret = float(rets.mean(axis=1).mean() * 252)
        port_vol = float(rets.mean(axis=1).std() * np.sqrt(252))
        return weights, port_ret, port_vol


@traced_node("strategy")
async def strategy_node(state: GraphState) -> dict:
    try:
        return await _strategy_impl(state)
    except Exception as exc:
        logger.error("strategy_node failed: %s", exc, exc_info=True)
        return {
            "allocation_result": AllocationResult(
                weights={},
                expected_return=0.0,
                expected_volatility=0.0,
                strategy_rationale=f"Strategy computation failed: {type(exc).__name__}",
            )
        }


async def _strategy_impl(state: GraphState) -> dict:
    redis = get_redis()
    profile = state.get("user_profile") or {}
    portfolio_weights: dict[str, float] = dict(profile.get("portfolio") or {})

    # --- Fetch OHLCV ---
    tickers = list(portfolio_weights.keys()) if portfolio_weights else DEFAULT_UNIVERSE
    prices: dict[str, pd.Series] = {}
    for ticker in tickers:
        df = await get_ohlcv(ticker, redis)
        if df is not None and len(df) >= 30:
            prices[ticker] = df["Close"]

    used_fallback = len(prices) < 2
    if used_fallback:
        logger.warning("strategy_node: falling back to DEFAULT_UNIVERSE")
        for ticker in DEFAULT_UNIVERSE:
            if ticker not in prices:
                df = await get_ohlcv(ticker, redis)
                if df is not None and len(df) >= 30:
                    prices[ticker] = df["Close"]

    prices_df = pd.DataFrame(prices).dropna()

    rf = await get_risk_free_rate(redis) or 0.04
    inflation = await get_inflation(redis)

    # --- Max-Sharpe optimisation ---
    weights, exp_ret, exp_vol = optimize_portfolio(prices_df, rf)

    # --- LLM: strategy_rationale only ---
    alloc_desc = ", ".join(f"{t} {w * 100:.0f}%" for t, w in weights.items())
    inflation_line = f"CPI inflation (YoY): {inflation:.1%}" if inflation is not None else ""
    macro_context = f"Risk-free rate: {rf:.1%}" + (f" | {inflation_line}" if inflation_line else "")
    prompt = (
        f"Investor profile:\n"
        f"  Risk tolerance: {profile.get('risk_tolerance', '?')}\n"
        f"  Horizon: {profile.get('investment_horizon_years', '?')} years\n"
        f"  Amount: ${profile.get('investment_amount_usd', 0):,.0f}\n\n"
        f"Recommended allocation (max-Sharpe optimised):\n  {alloc_desc}\n\n"
        f"Expected annual return: {exp_ret:.1%} | Expected volatility: {exp_vol:.1%}\n"
        f"Macro context: {macro_context}\n\n"
        "In 2-3 sentences, explain why this allocation suits the investor's profile "
        "and what trade-offs it makes."
    )

    llm = get_chat_model().bind_tools([_RATIONALE_TOOL])
    resp = await llm.ainvoke([
        SystemMessage(content="You are a portfolio strategist. Call report_rationale."),
        HumanMessage(content=prompt),
    ])

    rationale = ""
    if resp.tool_calls:
        rationale = resp.tool_calls[0]["args"].get("rationale", "")
    else:
        rationale = resp.content or ""

    return {
        "allocation_result": AllocationResult(
            weights=weights,
            expected_return=round(exp_ret, 4),
            expected_volatility=round(exp_vol, 4),
            strategy_rationale=rationale,
        )
    }
