# Market Data

This document covers the three external data sources (yfinance, Alpha Vantage, FRED), the Redis caching layer, cache key scheme, TTLs, and fallback behavior.

---

## Table of Contents

1. [Overview](#1-overview)
2. [Redis Caching Utility](#2-redis-caching-utility)
3. [yfinance Client](#3-yfinance-client)
4. [Alpha Vantage Client](#4-alpha-vantage-client)
5. [FRED Client](#5-fred-client)
6. [Cache Key Scheme and TTLs](#6-cache-key-scheme-and-ttls)
7. [Fallback Behavior](#7-fallback-behavior)

---

## 1. Overview

Market data is consumed by two pipeline nodes:

- **`risk_assessment`** — needs OHLCV history for Sharpe/volatility/drawdown, ticker fundamentals for beta and P/E, news sentiment for qualitative flags, and FRED macro data for context
- **`strategy`** — needs OHLCV history for mean-variance optimization

All API responses are cached in Redis using pickle serialization. This means:
- Repeated requests for the same ticker in the same session hit the cache
- Multiple sessions analyzing the same portfolio don't redundantly call external APIs
- Free-tier API rate limits (Alpha Vantage: 25 req/day) are preserved across sessions

**Files**: `backend/app/market_data/`
- `cache.py` — Redis get/set utilities
- `yfinance_client.py` — yfinance wrappers
- `alpha_vantage_client.py` — Alpha Vantage REST client
- `fred_client.py` — FRED REST client

---

## 2. Redis Caching Utility

**File**: `market_data/cache.py`

```python
async def get_cached(redis: aioredis.Redis, key: str) -> Any | None:
    try:
        data = await redis.get(key)
        if data is None:
            return None
        return pickle.loads(data)
    except Exception:
        return None  # cache miss on any error

async def set_cached(redis: aioredis.Redis, key: str, value: Any, ttl: int) -> None:
    try:
        await redis.set(key, pickle.dumps(value), ex=ttl)
    except Exception:
        pass  # silently skip cache write on error
```

**Pickle serialization** stores any Python object (pandas DataFrames, dicts, floats) without a schema — convenient for heterogeneous market data types.

**NaN detection**: Before caching a float value, the client checks `math.isnan(value)` and replaces with `None`. Storing NaN in Redis and retrieving it works technically, but downstream code that expects a valid float would fail silently.

**Silent failure**: Both `get_cached` and `set_cached` swallow all exceptions. A Redis connection failure is treated as a cache miss — the system falls back to a live API call. This ensures the application continues working if Redis is temporarily unavailable.

---

## 3. yfinance Client

**File**: `market_data/yfinance_client.py`

### `get_ohlcv(ticker, redis)` — price history

```python
key = f"mkt:ohlcv:{ticker}"
cached = await get_cached(redis, key)
if cached is not None:
    return cached

df = yf.download(ticker, period="2y", interval="1d", progress=False, auto_adjust=True)
await set_cached(redis, key, df, ttl=86400)  # 24h TTL
return df
```

Returns a pandas DataFrame with columns: Open, High, Low, Close, Volume. 2-year daily history provides enough data for stable annualized metrics.

**Indian stock fallback**: yfinance uses `.NS` suffix for NSE-listed stocks. If `yf.download(ticker)` returns an empty DataFrame, the client retries with `f"{ticker}.NS"`. This handles cases where users enter bare tickers like `INFY` instead of `INFY.NS`.

### `get_ticker_info(ticker, redis)` — fundamentals

```python
key = f"mkt:info:{ticker}"
# TTL: 3600 (1h) — company fundamentals change slowly but can shift intraday
info = yf.Ticker(ticker).info
return {
    "sector": info.get("sector"),
    "market_cap": info.get("marketCap"),
    "currency": info.get("currency"),
    "beta": info.get("beta"),
    "pe_ratio": info.get("trailingPE"),
}
```

### `get_price(ticker, redis)` — latest close

```python
key = f"mkt:price:{ticker}"
# TTL: 300 (5min) — used for real-time portfolio valuation
```

Returns the most recent closing price from `yf.Ticker(ticker).fast_info.last_price`.

---

## 4. Alpha Vantage Client

**File**: `market_data/alpha_vantage_client.py`

Base URL: `https://www.alphavantage.co/query`

### `get_fundamentals(ticker, redis)` — company overview

Calls `function=OVERVIEW` endpoint. Extracts:

| Field | Alpha Vantage key | Notes |
|-------|------------------|-------|
| Beta | `Beta` | 5-year monthly beta vs S&P 500 |
| P/E ratio | `TrailingPE` | Trailing 12-month |
| EPS | `EPS` | Diluted earnings per share |
| Dividend yield | `DividendYield` | As decimal |
| 52-week high | `52WeekHigh` | |
| 52-week low | `52WeekLow` | |
| Market cap | `MarketCapitalization` | In USD |

**NaN-safe float parsing**:
```python
def _safe_float(val: str | None) -> float | None:
    if val is None or val in ("None", "-", "N/A", ""):
        return None
    try:
        result = float(val)
        return None if math.isnan(result) else result
    except (ValueError, TypeError):
        return None
```

Alpha Vantage returns string `"None"` for missing values. Plain `float("None")` raises `ValueError`. This wrapper handles all edge cases.

### `get_sentiment(ticker, redis)` — news sentiment

Calls `function=NEWS_SENTIMENT&tickers={ticker}&limit=10` endpoint.

**Relevance-weighted average**:
```python
weighted_sum = 0.0
weight_total = 0.0
for article in feed:
    for ticker_sentiment in article.get("ticker_sentiment", []):
        if ticker_sentiment["ticker"] == ticker:
            relevance = float(ticker_sentiment["relevance_score"])
            score = float(ticker_sentiment["ticker_sentiment_score"])
            weighted_sum += relevance * score
            weight_total += relevance

sentiment = weighted_sum / weight_total if weight_total > 0 else 0.0
```

**Classification thresholds**:
- `score > 0.15` → bullish
- `score < -0.15` → bearish
- otherwise → neutral

---

## 5. FRED Client

**File**: `market_data/fred_client.py`

Base URL: `https://api.stlouisfed.org/fred/series/observations`

All three FRED series return the most recent data point. Responses are JSON with an `observations` array.

### `get_risk_free_rate(redis)` — DGS3MO

3-month US Treasury Bill yield, used as the risk-free rate in Sharpe ratio calculations.

```python
# Returns value as decimal (e.g., 5.25% → 0.0525)
rate = float(latest_observation["value"]) / 100
```

TTL: 24h — Treasury yields are published daily, but 1-day staleness is immaterial for annualized Sharpe calculations.

### `get_inflation(redis)` — CPIAUCSL

Consumer Price Index for All Urban Consumers. Returns year-over-year percentage change:

```python
# Fetch last 13 months, compute YoY change
current = float(observations[-1]["value"])
year_ago = float(observations[-13]["value"])
inflation = (current - year_ago) / year_ago
```

TTL: 30 days — CPI is published monthly. The 30-day cache means tests and development sessions never hit the API more than once per month for this series.

### `get_yield_curve(redis)` — T10Y2Y

10-year minus 2-year Treasury yield spread. A negative value (yield curve inversion) is a recession indicator. Used as context in the risk assessment's qualitative analysis.

TTL: 24h

---

## 6. Cache Key Scheme and TTLs

All keys follow `mkt:{data_type}:{identifier}`:

| Data | Key Pattern | TTL | Rationale |
|------|-------------|:---:|-----------|
| OHLCV price history | `mkt:ohlcv:{ticker}` | 24h | Daily data; intraday changes don't affect 2-year metrics |
| Ticker info | `mkt:info:{ticker}` | 1h | Sector/beta change slowly; 1h balances freshness vs API calls |
| Latest price | `mkt:price:{ticker}` | 5min | Near real-time for portfolio valuation |
| Fundamentals (AV) | `mkt:fundamentals:{ticker}` | 12h | OVERVIEW data is quarterly; 12h is generous |
| News sentiment | `mkt:sentiment:{ticker}` | 30min | News cycle; 30min provides freshness without exhausting free tier |
| Risk-free rate | `mkt:dgs3mo:rate` | 24h | Published daily; 1-day staleness is immaterial |
| Inflation | `mkt:cpiaucsl:inflation` | 30d | Published monthly |
| Yield curve | `mkt:t10y2y:yield_curve` | 24h | Published daily |

---

## 7. Fallback Behavior

The market data pipeline degrades gracefully when data is unavailable:

**OHLCV fetch fails or returns < 2 valid tickers**:
- `risk_assessment` uses `DEFAULT_UNIVERSE = ["SPY", "QQQ", "BND", "GLD", "VNQ"]` as a proxy portfolio
- Risk metrics are computed on this benchmark universe instead
- `risk_flags` LLM call notes that metrics are based on benchmark ETF proxies, not the user's actual holdings

**Alpha Vantage fundamentals fail**:
- Beta defaults to `1.0` (market-neutral)
- P/E ratio is omitted from the fundamentals summary

**FRED call fails or times out**:
- Risk-free rate defaults to `0.05` (5%) — a conservative estimate for current rates
- Yield curve and inflation are omitted from macro context

**Portfolio optimization fails in strategy node**:
- Singular covariance matrix, zero-variance assets, or numerical solver failures trigger the fallback
- Equal weights across all portfolio tickers are used instead
- `strategy_rationale` notes that optimization converged to equal weights due to data constraints
