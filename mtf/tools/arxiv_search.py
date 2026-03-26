"""Arxiv search tool for the claude-agent-sdk."""

from __future__ import annotations

import asyncio
import json

import arxiv
import claude_agent_sdk as sdk


def make_arxiv_search_tool() -> sdk.SdkMcpTool:
    """Return a claude-agent-sdk SdkMcpTool that searches arxiv."""

    def _arxiv_search(query: str, max_results: int = 5) -> list[dict[str, str]]:
        max_results = min(max_results, 20)
        client = arxiv.Client()
        search = arxiv.Search(query=query, max_results=max_results)
        results = []
        for paper in client.results(search):
            results.append(
                {
                    "title": paper.title,
                    "authors": ", ".join(a.name for a in paper.authors),
                    "summary": paper.summary[:500],
                    "url": paper.entry_id,
                    "published": str(paper.published.date()),
                }
            )
        return results

    async def _handler(args: dict) -> dict:
        result = await asyncio.to_thread(
            _arxiv_search,
            query=args.get("query", ""),
            max_results=int(args.get("max_results", 5)),
        )
        return {"content": [{"type": "text", "text": json.dumps(result)}]}

    return sdk.SdkMcpTool(
        name="arxiv_search",
        description=(
            "Search arxiv for papers matching a query. "
            "Returns title, authors, summary, url, and published date."
        ),
        input_schema={
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "Search string (supports arxiv advanced query syntax)."},
                "max_results": {"type": "integer", "description": "Maximum number of results to return (default 5, max 20)."},
            },
            "required": ["query"],
        },
        handler=_handler,
    )
