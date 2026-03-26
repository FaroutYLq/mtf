"""Semantic Scholar search tool for the claude-agent-sdk."""

from __future__ import annotations

import asyncio
import json

import claude_agent_sdk as sdk
import semanticscholar as sch


def make_semantic_search_tool() -> sdk.SdkMcpTool:
    """Return a claude-agent-sdk SdkMcpTool that searches Semantic Scholar."""

    def _semantic_scholar_search(query: str, limit: int = 5) -> list[dict[str, str]]:
        limit = min(limit, 20)
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

    async def _handler(args: dict) -> dict:
        result = await asyncio.to_thread(
            _semantic_scholar_search,
            query=args.get("query", ""),
            limit=int(args.get("limit", 5)),
        )
        return {"content": [{"type": "text", "text": json.dumps(result)}]}

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
                "limit": {"type": "integer", "description": "Maximum number of results (default 5, max 20)."},
            },
            "required": ["query"],
        },
        handler=_handler,
    )
