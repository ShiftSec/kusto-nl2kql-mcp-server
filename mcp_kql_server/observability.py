"""Langfuse observability."""

import functools
import logging
import os

from langfuse import observe, get_client, propagate_attributes
import langfuse.openai as langfuse_openai

logger = logging.getLogger(__name__)

# Map team env var convention to Langfuse SDK convention
_baseurl = os.environ.get("LANGFUSE_BASEURL", "")
if _baseurl:
    os.environ["LANGFUSE_HOST"] = _baseurl


def trace(name: str | None = None, tags: list[str] | None = None, **kwargs):
    """Return @observe() decorator with optional tags propagation."""
    if not tags:
        return observe(name=name, **kwargs)

    def decorator(fn):
        observed = observe(name=name, **kwargs)(fn)

        @functools.wraps(fn)
        async def async_wrapper(*a, **kw):
            with propagate_attributes(tags=tags):
                return await observed(*a, **kw)

        @functools.wraps(fn)
        def sync_wrapper(*a, **kw):
            with propagate_attributes(tags=tags):
                return observed(*a, **kw)

        import asyncio
        if asyncio.iscoroutinefunction(fn):
            return async_wrapper
        return sync_wrapper

    return decorator


def get_openai_module():
    """Return langfuse-instrumented openai module."""
    return langfuse_openai


def update_trace(metadata: dict | None = None, **kwargs):
    """Update current Langfuse span with metadata."""
    client = get_client()
    client.update_current_span(metadata=metadata, **kwargs)


def flush():
    """Flush pending Langfuse events. Call on shutdown."""
    client = get_client()
    client.flush()
    logger.info("Langfuse traces flushed")
