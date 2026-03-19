"""Arxiv search tool for the claude-agent-sdk."""

from __future__ import annotations

import arxiv
import claude_agent_sdk as sdk


def make_arxiv_search_tool() -> sdk.Tool:
    """Return a claude-agent-sdk Tool that searches arxiv."""

    def arxiv_search(query: str, max_results: int = 5) -> list[dict[str, str]]:
        """Search arxiv for papers matching *query*.

        Args:
            query: Search string (supports arxiv advanced query syntax).
            max_results: Maximum number of results to return (default 5, max 20).

        Returns:
            List of dicts with keys: title, authors, summary, url, published.
        """
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

    return sdk.Tool.from_function(arxiv_search)
