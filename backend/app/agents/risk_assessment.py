import asyncio
import json
import logging

import numpy as np
import pandas as pd
from langchain_core.messages import HumanMessage, SystemMessage

from app.agents.state import GraphState, RiskMetrics
from app.db.redis_client import get_redis
from app.llm.client import get_chat_model
from app.market_data.alpha_vantage_client import get_fundamentals, get_sentiment
from app.market_data.fred_client import get_inflation, get_risk_free_rate, get_yield_curve
from app.market_data.yfinance_client import get_ohlcv, get_ticker_info
from app.observability.langfuse_setup import traced_node

logger = logging.getLogger(__name__)

DEFAULT_UNIVERSE = ["SPY", "QQQ", "BND", "GLD", "VNQ"]

_RISK_FLAGS_TOOL = {
    "type": "function",
    "function": {
        "name": "report_risk_flags",
        "description": "Report qualitative risk flags for the portfolio.",
        "parameters": {
            "type": "object",
            "properties": {
                "risk_flags": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "3-5 specific qualitative risk flags for the portfolio.",
                }
            },
            "required": ["risk_flags"],
        },
    },
}


def compute_portfolio_metrics(
    portfolio_returns: pd.Series, annual_rf: float
) -> tuple[float, float, float]:
    """Return (annualised_sharpe, annualised_vol, max_drawdown). Pure function."""
    if portfolio_returns.empty or portfolio_returns.std() == 0:
        return 0.0, 0.0, 0.0

    daily_rf = annual_rf / 252
    excess = portfolio_returns - daily_rf
    ann_sharpe = float((excess.mean() / excess.std()) * np.sqrt(252))
    ann_vol = float(portfolio_returns.std() * np.sqrt(252))

    cum = (1 + portfolio_returns).cumprod()
    running_max = cum.cummax()
    drawdown = (cum - running_max) / running_max
    max_dd = float(drawdown.min())

    return ann_sharpe, ann_vol, max_dd


@traced_node("risk_assessment")
async def risk_assessment_node(state: GraphState) -> dict:
    try:
        return await _risk_assessment_impl(state)
    except Exception as exc:
        logger.error("risk_assessment_node failed: %s", exc, exc_info=True)
        return {
            "risk_metrics": RiskMetrics(
                sharpe_ratio=0.0,
                volatility=0.0,
                max_drawdown=0.0,
                risk_flags=[f"Risk analysis unavailable: {type(exc).__name__}"],
            )
        }


async def _risk_assessment_impl(state: GraphState) -> dict:
    redis = get_redis()
    profile = state.get("user_profile") or {}
    portfolio_weights: dict[str, float] = dict(profile.get("portfolio") or {})

    # --- Fetch OHLCV for portfolio tickers, filter invalid ones ---
    valid: dict[str, pd.DataFrame] = {}
    for ticker in portfolio_weights:
        df = await get_ohlcv(ticker, redis)
        if df is not None and len(df) >= 30:
            valid[ticker] = df

    used_fallback = len(valid) < 2
    if used_fallback:
        logger.warning("Falling back to DEFAULT_UNIVERSE — portfolio tickers did not resolve")
        for ticker in DEFAULT_UNIVERSE:
            if ticker not in valid:
                df = await get_ohlcv(ticker, redis)
                if df is not None and len(df) >= 30:
                    valid[ticker] = df

    # --- Sector breakdown via ticker_info (concurrent, best-effort) ---
    info_results = await asyncio.gather(
        *[get_ticker_info(t, redis) for t in valid], return_exceptions=True
    )
    ticker_info_map: dict[str, dict] = {}
    for t, result in zip(valid, info_results):
        if isinstance(result, dict):
            ticker_info_map[t] = result

    # --- Build aligned returns DataFrame ---
    returns_map = {t: df["Close"].pct_change().dropna() for t, df in valid.items()}
    returns_df = pd.DataFrame(returns_map).dropna()

    # --- Portfolio weights (normalise to valid tickers only) ---
    if used_fallback or not portfolio_weights:
        n = len(returns_df.columns)
        weights = {t: 1.0 / n for t in returns_df.columns}
    else:
        sub = {t: portfolio_weights[t] for t in returns_df.columns if t in portfolio_weights}
        total = sum(sub.values())
        weights = {t: v / total for t, v in sub.items()} if total > 0 else {
            t: 1.0 / len(returns_df.columns) for t in returns_df.columns
        }

    portfolio_returns = sum(
        returns_df[t] * w for t, w in weights.items() if t in returns_df.columns
    )

    # --- Market data context (concurrent where independent) ---
    rf_task = get_risk_free_rate(redis)
    yc_task = get_yield_curve(redis)
    inf_task = get_inflation(redis)
    rf_raw, yc, inflation = await asyncio.gather(rf_task, yc_task, inf_task, return_exceptions=False)
    rf = rf_raw or 0.04

    # --- Compute metrics ---
    sharpe, vol, max_dd = compute_portfolio_metrics(portfolio_returns, rf)

    # --- Enrichment: fundamentals (top 5) + sentiment (top 3) ---
    sorted_tickers = sorted(weights, key=lambda t: weights.get(t, 0), reverse=True)
    top5 = sorted_tickers[:5]
    top3 = sorted_tickers[:3]

    fund_results = await asyncio.gather(
        *[get_fundamentals(t, redis) for t in top5], return_exceptions=True
    )
    fundamentals: dict[str, dict] = {}
    for t, res in zip(top5, fund_results):
        if isinstance(res, dict):
            fundamentals[t] = res

    sent_results = await asyncio.gather(
        *[get_sentiment(t, redis) for t in top3], return_exceptions=True
    )
    sentiments: dict[str, float] = {}
    for t, res in zip(top3, sent_results):
        if isinstance(res, (int, float)):
            sentiments[t] = float(res)

    # --- Assemble enriched LLM prompt ---
    portfolio_desc = ", ".join(
        f"{t} ({w * 100:.0f}%)" for t, w in list(weights.items())[:8]
    )

    # Sector breakdown
    sector_parts = [
        f"{t} → {info['sector']}"
        for t, info in ticker_info_map.items()
        if info.get("sector")
    ]
    sector_line = "Sector info: " + ", ".join(sector_parts) if sector_parts else ""

    # Weighted-average beta + per-ticker PE
    beta_num = beta_den = 0.0
    pe_parts: list[str] = []
    for t in top5:
        fd = fundamentals.get(t)
        if not fd:
            continue
        w = weights.get(t, 0)
        if fd.get("beta") is not None:
            beta_num += fd["beta"] * w
            beta_den += w
        if fd.get("pe_ratio") is not None:
            pe_parts.append(f"{t} PE {fd['pe_ratio']:.1f}")
    beta_line = ""
    if beta_den > 0:
        beta_line = f"Portfolio beta: {beta_num / beta_den:.2f} (weighted avg)"
        if pe_parts:
            beta_line += " | " + ", ".join(pe_parts)

    # Macro context
    macro_parts = [f"Risk-free rate: {rf:.1%}"]
    if yc is not None:
        macro_parts.append(f"Yield curve (10Y-2Y): {yc:+.2f}pp")
    if inflation is not None:
        macro_parts.append(f"CPI inflation (YoY): {inflation:.1%}")
    macro_line = " | ".join(macro_parts)

    # News sentiment (only flag notable scores)
    sent_parts = [
        f"{t} {'bearish' if s < -0.15 else 'bullish' if s > 0.15 else 'neutral'} ({s:+.2f})"
        for t, s in sentiments.items()
    ]
    sentiment_line = "News sentiment: " + ", ".join(sent_parts) if sent_parts else ""

    prompt_lines = [f"Portfolio: {portfolio_desc}"]
    if sector_line:
        prompt_lines.append(sector_line)
    prompt_lines.append(
        f"Annualised Sharpe: {sharpe:.2f} | Vol: {vol:.1%} | Max drawdown: {max_dd:.1%}"
    )
    if beta_line:
        prompt_lines.append(beta_line)
    prompt_lines.append(macro_line)
    if sentiment_line:
        prompt_lines.append(sentiment_line)
    prompt_lines += [
        f"Investment horizon: {profile.get('investment_horizon_years', '?')} years | "
        f"Risk tolerance: {profile.get('risk_tolerance', '?')}",
        "",
        "Identify 3-5 specific qualitative risk flags for this portfolio. "
        "Consider concentration, sector/asset-class exposure, liquidity, macro sensitivity, "
        "and alignment with the stated risk tolerance and horizon.",
    ]
    prompt = "\n".join(prompt_lines)

    llm = get_chat_model().bind_tools([_RISK_FLAGS_TOOL])
    resp = await llm.ainvoke([
        SystemMessage(content="You are a quantitative risk analyst. Call report_risk_flags."),
        HumanMessage(content=prompt),
    ])

    risk_flags: list[str] = []
    if resp.tool_calls:
        risk_flags = resp.tool_calls[0]["args"].get("risk_flags") or []
    else:
        # Fallback: try to parse JSON array from plain text
        try:
            risk_flags = json.loads(resp.content)
        except Exception:
            risk_flags = [resp.content] if resp.content else []

    if used_fallback:
        risk_flags.insert(0,
            "Portfolio tickers could not be resolved to market data. "
            "Metrics are based on benchmark ETFs (SPY/QQQ/BND/GLD/VNQ) as proxies."
        )

    return {
        "risk_metrics": RiskMetrics(
            sharpe_ratio=round(sharpe, 4),
            volatility=round(vol, 4),
            max_drawdown=round(max_dd, 4),
            risk_flags=risk_flags,
        )
    }
