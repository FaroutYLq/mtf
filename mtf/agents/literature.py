"""LiteratureAgent: searches arxiv and semantic scholar for relevant papers."""

from __future__ import annotations

from mtf.agents.base import BaseAgent
from mtf.memory import MemoryKind, SharedMemory
from mtf.tools.arxiv_search import make_arxiv_search_tool
from mtf.tools.semantic_search import make_semantic_search_tool
from mtf.tools.gpd_mcp import GPDMCPClient

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
   c. Proposed hypotheses ranked by plausibility, each explicitly classified as one of:
      - first-principles (derived from theory)
      - semi-empirical (phenomenological with physical motivation)
      - purely empirical (fitting function)
   d. Key equations or models from the literature
   e. Error-prone aspects flagged per hypothesis based on check_error_classes results

Be precise, cite paper titles and authors."""


class LiteratureAgent(BaseAgent):
    def __init__(
        self,
        agent_id: str,
        model: str,
        memory: SharedMemory,
        gpd: GPDMCPClient | None = None,
    ) -> None:
        gpd_tools = []
        if gpd is not None and gpd.available:
            gpd_tools = [
                t
                for t in [
                    gpd.make_tool(
                        "protocols",
                        "route_protocol",
                        "Find the canonical computation protocol relevant to this phenomenon before searching literature. Input: computation_type (str). Returns matching protocols — use to understand what methodology the relevant papers should follow.",
                    ),
                    gpd.make_tool(
                        "errors",
                        "check_error_classes",
                        "Given a proposed hypothesis, identify the most likely physics error classes to watch for. Input: computation_desc (str describing the hypothesis or phenomenon). Returns top-15 relevant error classes — flag these when evaluating candidate hypotheses from literature.",
                    ),
                ]
                if t is not None
            ]
        super().__init__(
            agent_id=agent_id,
            model=model,
            tools=[make_arxiv_search_tool(), make_semantic_search_tool(), *gpd_tools],
            memory=memory,
            system_prompt=_SYSTEM_PROMPT,
        )

    async def investigate(self, phenomenon: str) -> str:
        task = (
            f"Investigate the following experimental phenomenon and search for "
            f"relevant literature:\n\n{phenomenon}\n\n"
            "Produce a comprehensive literature report with hypotheses."
        )
        report = await self._query(
            task,
            extra_kinds=(
                MemoryKind.USER_FEEDBACK,
                MemoryKind.IMAGE_DATA,
                MemoryKind.CONVENTIONS,
            ),
        )
        self._memory.add(
            MemoryKind.LITERATURE,
            report,
            agent_id=self._agent_id,
        )
        return report
