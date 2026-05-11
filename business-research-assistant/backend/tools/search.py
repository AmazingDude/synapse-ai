"""
Tavily web-search tool for the business research assistant.

Lazy-singleton pattern: the Tavily client is constructed on the first call to
``get_search_tool()`` and reused thereafter (``functools.lru_cache``).
Importing this module never touches the network; the first agent that actually
needs a search triggers initialisation and gets a clear error if the key is
missing.

Environment variable required:
    TAVILY_API_KEY  -  obtained from https://app.tavily.com
"""

from __future__ import annotations

import logging
import os
from functools import lru_cache
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from langchain_tavily import TavilySearch

logger = logging.getLogger(__name__)

# Load .env eagerly so callers don't each need to call load_dotenv themselves.
# This file lives under backend/tools/, so three ``parent`` segments reach the project root.
load_dotenv(Path(__file__).resolve().parent.parent.parent / ".env")

@lru_cache(maxsize=1)
def get_search_tool(max_results: int = 5) -> TavilySearch:
    """Return the shared Tavily search tool, creating it on first call.

    Reads ``TAVILY_API_KEY`` from the environment and raises a clear error if
    missing.  Bind directly to an LLM with ``.bind_tools([get_search_tool()])``
    or call imperatively via ``get_search_tool().invoke({"query": "..."})``.

    Args:
        max_results: Result count per query (default 5).

    Raises:
        RuntimeError: If ``TAVILY_API_KEY`` is not set.
    """
    api_key = os.getenv("TAVILY_API_KEY")
    if not api_key:
        raise RuntimeError(
            "TAVILY_API_KEY not found in environment. Check your .env file."
        )

    logger.info("Initialising Tavily search (max_results=%d)", max_results)
    return TavilySearch(max_results=max_results)

# ---------------------------------------------------------------------------
# Imperative helpers used by research_agent.py
# ---------------------------------------------------------------------------

def web_search(query: str, *, max_results: int = 5) -> dict[str, Any]:
    """Run a synchronous Tavily search and return the raw result list as a dict."""
    tool = get_search_tool(max_results)
    raw = tool.invoke({"query": query})
    if isinstance(raw, list):
        return {"results": raw}
    if isinstance(raw, dict) and "results" not in raw:
        return {"results": [raw]}
    return raw  # type: ignore[return-value]

def format_search_results(response: dict[str, Any]) -> str:
    """Render a ``web_search`` response as readable markdown for the LLM."""
    results = response.get("results") or []
    lines: list[str] = []
    for i, hit in enumerate(results, start=1):
        title = hit.get("title") or "(no title)"
        url = hit.get("url") or ""
        content = hit.get("content") or ""
        lines.append(f"{i}. **{title}**\n   {url}\n   {content}\n")
    return "\n".join(lines).strip() or "No results returned."
