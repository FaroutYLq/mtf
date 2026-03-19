"""Semantic Scholar search tool for the claude-agent-sdk."""

from __future__ import annotations

import claude_agent_sdk as sdk
import semanticscholar as sch


def make_semantic_search_tool() -> sdk.Tool:
    """Return a claude-agent-sdk Tool that searches Semantic Scholar."""

    def semantic_scholar_search(query: str, limit: int = 5) -> list[dict[str, str]]:
        """Search Semantic Scholar for papers matching *query*.

        Args:
            query: Natural language or keyword query.
            limit: Maximum number of results (default 5, max 20).

        Returns:
            List of dicts with keys: title, authors, abstract, year, url, citation_count.
        """
        limit = min(limit, 20)
        api = sch.SemanticScholar()
        papers = api.search_paper(query, limit=limit)
        results = []
        for p in papers:
            results.append(
                {
                    "title": p.title or "",
                    "authors": ", ".join(
                        a["name"] for a in (p.authors or [])
                    ),
                    "abstract": (p.abstract or "")[:500],
                    "year": str(p.year or ""),
                    "url": p.url or "",
                    "citation_count": str(p.citationCount or 0),
                }
            )
        return results

    return sdk.Tool.from_function(semantic_scholar_search)
