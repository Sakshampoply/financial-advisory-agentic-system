import logging

import httpx
import redis.asyncio as aioredis

from app.config import settings
from app.market_data.cache import get_cached, set_cached

logger = logging.getLogger(__name__)

_BASE = "https://www.alphavantage.co/query"
_TTL_FUNDAMENTALS = 43_200   # 12h
_TTL_SENTIMENT = 1_800       # 30min


async def get_fundamentals(ticker: str, redis: aioredis.Redis) -> dict | None:
    key = f"mkt:fundamentals:{ticker}"
    cached = await get_cached(redis, key)
    if cached is not None:
        return cached

    if not settings.ALPHA_VANTAGE_API_KEY:
        return None

    try:
        async with httpx.AsyncClient(timeout=10) as client:
            resp = await client.get(_BASE, params={
                "function": "OVERVIEW",
                "symbol": ticker,
                "apikey": settings.ALPHA_VANTAGE_API_KEY,
            })
            data = resp.json()

        if "Symbol" not in data:
            return None

        result = {
            "pe_ratio": _safe_float(data.get("PERatio")),
            "eps": _safe_float(data.get("EPS")),
            "dividend_yield": _safe_float(data.get("DividendYield")),
            "52_week_high": _safe_float(data.get("52WeekHigh")),
            "52_week_low": _safe_float(data.get("52WeekLow")),
            "beta": _safe_float(data.get("Beta")),
        }
        await set_cached(redis, key, result, _TTL_FUNDAMENTALS)
        return result
    except Exception as exc:
        logger.warning("Alpha Vantage fundamentals failed for %s: %s", ticker, exc)
        return None


async def get_sentiment(ticker: str, redis: aioredis.Redis) -> float | None:
    """Returns a composite sentiment score in [-1, 1] from recent news."""
    key = f"mkt:sentiment:{ticker}"
    cached = await get_cached(redis, key)
    if cached is not None:
        return cached

    if not settings.ALPHA_VANTAGE_API_KEY:
        return None

    try:
        async with httpx.AsyncClient(timeout=10) as client:
            resp = await client.get(_BASE, params={
                "function": "NEWS_SENTIMENT",
                "tickers": ticker,
                "limit": 20,
                "apikey": settings.ALPHA_VANTAGE_API_KEY,
            })
            data = resp.json()

        feed = data.get("feed") or []
        if not feed:
            return None

        scores = []
        for article in feed:
            for ts in article.get("ticker_sentiment") or []:
                if ts.get("ticker") == ticker:
                    s = _safe_float(ts.get("ticker_sentiment_score"))
                    if s is not None:
                        scores.append(s)

        if not scores:
            return None

        score = sum(scores) / len(scores)
        await set_cached(redis, key, score, _TTL_SENTIMENT)
        return score
    except Exception as exc:
        logger.warning("Alpha Vantage sentiment failed for %s: %s", ticker, exc)
        return None


def _safe_float(value) -> float | None:
    try:
        f = float(value)
        return None if f != f else f  # NaN check
    except (TypeError, ValueError):
        return None
