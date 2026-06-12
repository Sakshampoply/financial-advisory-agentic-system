"""Langfuse observability — thin wrapper around the v4 `observe` decorator.

Usage:
    from app.observability.langfuse_setup import traced_node

    @traced_node("my_node")
    async def my_node(state: GraphState) -> dict:
        ...

No-ops gracefully when LANGFUSE_PUBLIC_KEY / LANGFUSE_SECRET_KEY are not set.
"""
import logging
from functools import wraps
from typing import Callable

logger = logging.getLogger(__name__)

# Initialise the Langfuse client once at import time.
# The observe decorator reads credentials from the client singleton.
try:
    from langfuse import Langfuse, observe as _observe

    from app.config import settings

    if settings.LANGFUSE_PUBLIC_KEY and settings.LANGFUSE_SECRET_KEY:
        Langfuse(
            public_key=settings.LANGFUSE_PUBLIC_KEY,
            secret_key=settings.LANGFUSE_SECRET_KEY,
            host=settings.LANGFUSE_HOST,
        )
        _LANGFUSE_ENABLED = True
        logger.info("Langfuse tracing enabled (host: %s)", settings.LANGFUSE_HOST)
    else:
        _LANGFUSE_ENABLED = False
        logger.debug("Langfuse keys not set — tracing disabled")

except Exception as exc:  # pragma: no cover
    _LANGFUSE_ENABLED = False
    logger.warning("Langfuse initialisation failed (%s) — tracing disabled", exc)


def traced_node(name: str) -> Callable:
    """Return a decorator that wraps an async node function with a Langfuse agent span."""
    def decorator(fn: Callable) -> Callable:
        if not _LANGFUSE_ENABLED:
            return fn

        traced = _observe(fn, name=name, as_type="agent")

        @wraps(fn)
        async def wrapper(*args, **kwargs):
            return await traced(*args, **kwargs)

        return wrapper

    return decorator


def get_langfuse_callbacks() -> list | None:
    """Return a LangChain CallbackHandler list for token/cost tracking, or None if disabled.

    Pass the result to ChatOpenAI(callbacks=...) so every LLM call reports token counts
    and USD cost to Langfuse, nested under the enclosing @traced_node span.
    The import is deferred so this module loads cleanly even if `langchain` is absent.
    """
    if not _LANGFUSE_ENABLED:
        return None
    try:
        from langfuse.langchain import CallbackHandler
        return [CallbackHandler()]
    except Exception as exc:  # pragma: no cover
        logger.warning("Langfuse LangChain callback unavailable (%s) — skipping cost tracking", exc)
        return None
