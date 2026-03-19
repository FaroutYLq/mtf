"""LiteratureAgent: searches arxiv and semantic scholar for relevant papers."""

from __future__ import annotations

from mtf.agents.base import BaseAgent
from mtf.memory import MemoryKind, SharedMemory
from mtf.tools.arxiv_search import make_arxiv_search_tool
from mtf.tools.semantic_search import make_semantic_search_tool

_SYSTEM_PROMPT = """You are an expert theoretical and experimental physicist acting as a
literature research agent. Your goal is to find relevant prior work that could explain
the experimental phenomenon provided by the user. Search arxiv and Semantic Scholar
thoroughly. Prioritize recent, highly-cited work. Produce a structured report with:
1. Summary of the phenomenon
2. Most relevant papers (with citations)
3. Proposed hypotheses ranked by plausibility
4. Key equations or models from the literature
Be precise, cite paper titles and authors."""


class LiteratureAgent(BaseAgent):
    def __init__(self, agent_id: str, model: str, memory: SharedMemory) -> None:
        super().__init__(
            agent_id=agent_id,
            model=model,
            tools=[make_arxiv_search_tool(), make_semantic_search_tool()],
            memory=memory,
            system_prompt=_SYSTEM_PROMPT,
        )

    async def investigate(self, phenomenon: str) -> str:
        task = (
            f"Investigate the following experimental phenomenon and search for "
            f"relevant literature:\n\n{phenomenon}\n\n"
            "Produce a comprehensive literature report with hypotheses."
        )
        report = await self._query(task, extra_kinds=(MemoryKind.USER_FEEDBACK,))
        self._memory.add(
            MemoryKind.LITERATURE,
            report,
            agent_id=self._agent_id,
        )
        return report
