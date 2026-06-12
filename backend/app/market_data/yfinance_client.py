import asyncio
import re
from typing import Any

import pandas as pd
import redis.asyncio as aioredis
import yfinance as yf

from app.market_data.cache import get_cached, set_cached

# Valid ticker: 1-12 uppercase alphanum chars, optionally with . - ^ (e.g. BRK-B, ^GSPC, RELIANCE.NS)
_TICKER_RE = re.compile(r'^[A-Z0-9.\-\^]{1,12}$')

_TTL_OHLCV = 86_400    # 24h
_TTL_INFO = 3_600      # 1h
_TTL_PRICE = 300       # 5min


def _exchange_candidates(ticker: str) -> list[str]:
    """Return ticker variants to try in order. Adds .NS (NSE India) for bare alpha tickers."""
    candidates = [ticker]
    # Only add .NS for plain alpha tickers that have no exchange suffix yet.
    # Avoids turning SPY → SPY.NS or BRK-B → BRK-B.NS.
    if ticker.isalpha():
        candidates.append(f"{ticker}.NS")
    return candidates


def _download_ohlcv(ticker: str) -> pd.DataFrame | None:
    if not _TICKER_RE.match(ticker):
        return None
    for candidate in _exchange_candidates(ticker):
        try:
            df = yf.download(candidate, period="2y", interval="1d", progress=False, auto_adjust=True)
            if df.empty:
                continue
            # Flatten MultiIndex columns produced when downloading a single ticker
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)
            return df[["Open", "High", "Low", "Close", "Volume"]].dropna()
        except Exception:
            continue
    return None


def _fetch_info(ticker: str) -> dict | None:
    if not _TICKER_RE.match(ticker):
        return None
    for candidate in _exchange_candidates(ticker):
        try:
            info = yf.Ticker(candidate).info
            if not info.get("regularMarketPrice") and not info.get("currentPrice"):
                continue
            return {
                "sector": info.get("sector"),
                "market_cap": info.get("marketCap"),
                "currency": info.get("currency"),
                "long_name": info.get("longName"),
            }
        except Exception:
            continue
    return None


def _fetch_price(ticker: str) -> float | None:
    if not _TICKER_RE.match(ticker):
        return None
    for candidate in _exchange_candidates(ticker):
        try:
            data = yf.download(candidate, period="1d", interval="1m", progress=False, auto_adjust=True)
            if data.empty:
                continue
            if isinstance(data.columns, pd.MultiIndex):
                data.columns = data.columns.get_level_values(0)
            return float(data["Close"].iloc[-1])
        except Exception:
            continue
    return None


async def get_ohlcv(ticker: str, redis: aioredis.Redis) -> pd.DataFrame | None:
    key = f"mkt:ohlcv:{ticker}"
    cached = await get_cached(redis, key)
    if cached is not None:
        return cached
    df = await asyncio.to_thread(_download_ohlcv, ticker)
    if df is not None:
        await set_cached(redis, key, df, _TTL_OHLCV)
    return df


async def get_ticker_info(ticker: str, redis: aioredis.Redis) -> dict | None:
    key = f"mkt:info:{ticker}"
    cached = await get_cached(redis, key)
    if cached is not None:
        return cached
    info = await asyncio.to_thread(_fetch_info, ticker)
    if info is not None:
        await set_cached(redis, key, info, _TTL_INFO)
    return info


async def get_price(ticker: str, redis: aioredis.Redis) -> float | None:
    key = f"mkt:price:{ticker}"
    cached = await get_cached(redis, key)
    if cached is not None:
        return cached
    price = await asyncio.to_thread(_fetch_price, ticker)
    if price is not None:
        await set_cached(redis, key, price, _TTL_PRICE)
    return price
