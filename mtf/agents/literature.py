"""LiteratureAgent: searches arxiv and semantic scholar for relevant papers."""

from __future__ import annotations

from typing import Any

from mtf.agents.base import BaseAgent
from mtf.memory import MemoryKind, SharedMemory
from mtf.tools.arxiv_search import make_arxiv_search_tool
from mtf.tools.semantic_search import make_semantic_search_tool

_SYSTEM_PROMPT = """You are an expert theoretical and experimental physicist acting as a
literature research agent. Your goal is to find relevant prior work that could explain
the experimental phenomenon provided by the user.

1. At the start, call `route_protocol` with a description of the phenomenon to
   understand what computation methodology the relevant papers should follow.
2. Search arxiv and Semantic Scholar thoroughly, prioritising recent, highly-cited work.
3. For each proposed hypothesis, call `check_error_classes` with a description of the
   hypothesis to identify the top-15 most likely physics error classes — flag
   error-prone approaches in your report.
4. Produce a structured report with:
   a. Summary of the phenomenon
   b. Most relevant papers (with citations)
   c. Proposed hypotheses ranked by plausibility. For each hypothesis, explicitly
      classify all three of:
      - Basis: first-principles / semi-empirical / purely empirical
      - Verification status: experimentally confirmed / theoretical prediction / disputed
      - Known failure modes: from check_error_classes results
   d. Key equations or models from the literature
   e. Error-prone aspects flagged per hypothesis based on check_error_classes results
5. When you find a systematic error pattern in a class of papers (wrong conventions,
   missing factors, sign errors), call `add_pattern(category='convention-pitfall', ...)`
   to record it in the cross-session pattern store for future runs.
6. Citation verification: Before finalising your report, verify up to 10 of the most
   important citations (not all) in section 4b by calling the search tool again with
   the exact paper title. Cross-check that the returned author names, publication year,
   and venue (journal or arXiv ID) match what you plan to report. Flag any citation you
   cannot confirm as `[UNVERIFIED: <reason>]`. Do not invent or approximate author names
   — if unsure, omit the author and write only the title and year. Skip well-known
   foundational papers (e.g., BCS theory, Higgs mechanism).

Prefer first-principles hypotheses over phenomenological curve fits. Be precise. All citations must be verified against actual search results — do not rely on memory for author names or publication years."""


class LiteratureAgent(BaseAgent):
    def __init__(
        self,
        agent_id: str,
        model: str,
        memory: SharedMemory,
        gpd_tools: list[Any] | None = None,
        config: Any | None = None,
    ) -> None:
        extra_tools: list[Any] = gpd_tools if gpd_tools is not None else []
        super().__init__(
            agent_id=agent_id,
            model=model,
            tools=[make_arxiv_search_tool(), make_semantic_search_tool(), *extra_tools],
            memory=memory,
            system_prompt=_SYSTEM_PROMPT,
        )
        self._config = config

    async def investigate(self, phenomenon: str) -> str:
        verification_note = ""
        if self._config is None or getattr(self._config, "citation_verification", True):
            verification_note = (
                "\n\nCITATION VERIFICATION (max 10 citations): Before finalising section 4b, "
                "verify up to 10 of the most important citations by calling the search tool "
                "again with the exact paper title. Cross-check that returned author names, "
                "year, and venue match what you plan to report. Flag unverified ones as "
                "[UNVERIFIED: <reason>]. Skip re-verification for well-known foundational "
                "papers (e.g., BCS theory, Higgs mechanism)."
            )
        task = (
            f"Investigate the following experimental phenomenon and search for "
            f"relevant literature:\n\n{phenomenon}\n\n"
            "Produce a comprehensive literature report with hypotheses."
            + verification_note
        )
        report = await self._query(
            task,
            extra_kinds=(
                MemoryKind.USER_FEEDBACK,
                MemoryKind.IMAGE_DATA,
                MemoryKind.CONVENTIONS,
                MemoryKind.DOMAIN_PATTERNS,
            ),
        )
        self._memory.add(
            MemoryKind.LITERATURE,
            report,
            agent_id=self._agent_id,
        )
        return report
