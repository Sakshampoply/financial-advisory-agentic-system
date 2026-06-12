import pickle
from typing import Any

import redis.asyncio as aioredis


async def get_cached(redis: aioredis.Redis, key: str) -> Any | None:
    raw = await redis.get(key)
    if raw is None:
        return None
    try:
        return pickle.loads(raw)
    except Exception:
        return None


async def set_cached(redis: aioredis.Redis, key: str, data: Any, ttl: int) -> None:
    try:
        await redis.set(key, pickle.dumps(data), ex=ttl)
    except Exception:
        pass
