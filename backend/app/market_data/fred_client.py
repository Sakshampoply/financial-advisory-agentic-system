import logging

import httpx
import redis.asyncio as aioredis

from app.config import settings
from app.market_data.cache import get_cached, set_cached

logger = logging.getLogger(__name__)

_BASE = "https://api.stlouisfed.org/fred/series/observations"
_TTL_DAY = 86_400       # 24h
_TTL_MONTH = 2_592_000  # 30d


async def _fetch_fred(series_id: str) -> float | None:
    if not settings.FRED_API_KEY:
        return None
    try:
        async with httpx.AsyncClient(timeout=10) as client:
            resp = await client.get(_BASE, params={
                "series_id": series_id,
                "api_key": settings.FRED_API_KEY,
                "file_type": "json",
                "sort_order": "desc",
                "limit": 10,
            })
            data = resp.json()

        for obs in data.get("observations") or []:
            val = obs.get("value", ".")
            if val != ".":
                return float(val)
        return None
    except Exception as exc:
        logger.warning("FRED fetch failed for %s: %s", series_id, exc)
        return None


async def get_risk_free_rate(redis: aioredis.Redis) -> float | None:
    """3-month T-bill rate (DGS3MO) as a decimal, e.g. 0.053 for 5.3%."""
    key = "mkt:fred:DGS3MO"
    cached = await get_cached(redis, key)
    if cached is not None:
        return cached
    raw = await _fetch_fred("DGS3MO")
    if raw is None:
        return None
    rate = raw / 100.0
    await set_cached(redis, key, rate, _TTL_DAY)
    return rate


async def get_inflation(redis: aioredis.Redis) -> float | None:
    """Most recent CPI YoY change (CPIAUCSL) as a decimal."""
    key = "mkt:fred:CPIAUCSL"
    cached = await get_cached(redis, key)
    if cached is not None:
        return cached

    if not settings.FRED_API_KEY:
        return None

    try:
        async with httpx.AsyncClient(timeout=10) as client:
            resp = await client.get(_BASE, params={
                "series_id": "CPIAUCSL",
                "api_key": settings.FRED_API_KEY,
                "file_type": "json",
                "sort_order": "desc",
                "limit": 13,
            })
            data = resp.json()

        obs = [
            float(o["value"]) for o in (data.get("observations") or [])
            if o.get("value", ".") != "."
        ]
        if len(obs) < 13:
            return None
        # YoY = (current - year_ago) / year_ago
        yoy = (obs[0] - obs[12]) / obs[12]
        await set_cached(redis, key, yoy, _TTL_MONTH)
        return yoy
    except Exception as exc:
        logger.warning("FRED inflation fetch failed: %s", exc)
        return None


async def get_yield_curve(redis: aioredis.Redis) -> float | None:
    """10Y-2Y Treasury spread (T10Y2Y) in percentage points."""
    key = "mkt:fred:T10Y2Y"
    cached = await get_cached(redis, key)
    if cached is not None:
        return cached
    raw = await _fetch_fred("T10Y2Y")
    if raw is None:
        return None
    await set_cached(redis, key, raw, _TTL_DAY)
    return raw
