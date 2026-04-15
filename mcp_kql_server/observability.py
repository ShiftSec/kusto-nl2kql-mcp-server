"""Langfuse observability."""

import logging
import os

from langfuse import observe, get_client
import langfuse.openai as langfuse_openai

logger = logging.getLogger(__name__)

# Map team env var convention to Langfuse SDK convention
_baseurl = os.environ.get("LANGFUSE_BASEURL", "")
if _baseurl:
    os.environ["LANGFUSE_HOST"] = _baseurl


def trace(name: str | None = None, **kwargs):
    """Return @observe() decorator."""
    return observe(name=name, **kwargs)


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
