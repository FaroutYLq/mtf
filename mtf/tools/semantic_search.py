"""Semantic Scholar search tool for the claude-agent-sdk."""

from __future__ import annotations

import asyncio
import json
import logging
import time

import claude_agent_sdk as sdk
import semanticscholar as sch

logger = logging.getLogger(__name__)

# Retry configuration for Semantic Scholar 429 rate-limit responses.
_MAX_RETRIES = 4
_INITIAL_BACKOFF = 2.0   # seconds
_BACKOFF_FACTOR = 2.0    # exponential multiplier
_MAX_BACKOFF = 30.0      # seconds


def _is_rate_limit_error(exc: BaseException) -> bool:
    """Return True if exc (or its cause) indicates an HTTP 429.

    The semanticscholar library wraps its own retries with tenacity, so by the
    time the exception reaches us it is a ``tenacity.RetryError`` whose string
    representation is ``"RetryError[<Future ... raised ConnectionRefusedError>]"``
    — the actual "429" message is buried in ``exc.last_attempt.exception()``.
    We therefore check both the outer exception and the tenacity inner cause.
    """
    _MARKERS = ("429", "Too Many Requests")

    def _matches(e: BaseException) -> bool:
        return any(m in str(e) for m in _MARKERS)

    if _matches(exc):
        return True
    # Unwrap tenacity.RetryError
    inner = getattr(exc, "last_attempt", None)
    if inner is not None:
        cause = getattr(inner, "exception", None)
        if callable(cause):
            try:
                return _matches(cause())
            except Exception:
                pass
    # Unwrap standard exception chaining (__cause__ / __context__)
    for chained in (exc.__cause__, exc.__context__):
        if chained is not None and _matches(chained):
            return True
    return False


def make_semantic_search_tool() -> sdk.SdkMcpTool:
    """Return a claude-agent-sdk SdkMcpTool that searches Semantic Scholar."""

    def _semantic_scholar_search(query: str, limit: int = 5) -> list[dict[str, str]]:
        limit = min(limit, 20)
        backoff = _INITIAL_BACKOFF
        for attempt in range(_MAX_RETRIES + 1):
            try:
                api = sch.SemanticScholar()
                papers = api.search_paper(query, limit=limit)
                results = []
                for p in papers:
                    results.append(
                        {
                            "title": p.title or "",
                            "authors": ", ".join(a["name"] for a in (p.authors or [])),
                            "abstract": (p.abstract or "")[:500],
                            "year": str(p.year or ""),
                            "url": p.url or "",
                            "citation_count": str(p.citationCount or 0),
                        }
                    )
                return results
            except Exception as exc:
                is_rate_limit = _is_rate_limit_error(exc)
                if is_rate_limit and attempt < _MAX_RETRIES:
                    wait = min(backoff, _MAX_BACKOFF)
                    logger.warning(
                        "Semantic Scholar 429 on attempt %d/%d; retrying in %.1fs",
                        attempt + 1, _MAX_RETRIES, wait,
                    )
                    time.sleep(wait)
                    backoff *= _BACKOFF_FACTOR
                else:
                    raise
        return []  # unreachable, satisfies type checker

    async def _handler(args: dict) -> dict:
        try:
            result = await asyncio.to_thread(
                _semantic_scholar_search,
                query=args.get("query", ""),
                limit=int(args.get("limit", 5)),
            )
            return {"content": [{"type": "text", "text": json.dumps(result)}]}
        except Exception as exc:
            error_msg = f"Semantic Scholar search failed: {exc}"
            logger.error(error_msg)
            return {"content": [{"type": "text", "text": json.dumps({"error": error_msg})}]}

    return sdk.SdkMcpTool(
        name="semantic_scholar_search",
        description=(
            "Search Semantic Scholar for papers matching a query. "
            "Returns title, authors, abstract, year, url, and citation count."
        ),
        input_schema={
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "Natural language or keyword query."},
                "limit": {"type": "integer", "description": "Maximum number of results.", "default": 5, "minimum": 1, "maximum": 20},
            },
            "required": ["query"],
            "additionalProperties": False,
        },
        handler=_handler,
    )
