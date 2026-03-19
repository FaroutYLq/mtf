"""ReviewerAgent: evaluates theory validity and suggests further experiments."""

from __future__ import annotations

from mtf.agents.base import BaseAgent
from mtf.memory import MemoryKind, SharedMemory

_SYSTEM_PROMPT = """You are a senior physics reviewer with broad expertise across
condensed matter, high energy, AMO, and astrophysics. Given literature reports,
fit results, and debate summaries, you:
1. Assess the validity of each proposed hypothesis against the data and literature
2. Identify weaknesses, alternative explanations, or confounds
3. Suggest specific further experiments that would distinguish competing hypotheses
4. Provide an overall recommendation

Be rigorous, constructive, and cite specific evidence from the provided context."""


class ReviewerAgent(BaseAgent):
    def __init__(self, agent_id: str, model: str, memory: SharedMemory) -> None:
        super().__init__(
            agent_id=agent_id,
            model=model,
            tools=[],
            memory=memory,
            system_prompt=_SYSTEM_PROMPT,
        )

    async def review(self, phenomenon: str) -> str:
        task = (
            f"Original phenomenon:\n{phenomenon}\n\n"
            "Review all available literature, fit results, and debate summaries. "
            "Produce a comprehensive peer review with recommendations."
        )
        report = await self._query(
            task,
            extra_kinds=(
                MemoryKind.LITERATURE,
                MemoryKind.DEBATE,
                MemoryKind.FIT_RESULT,
                MemoryKind.USER_FEEDBACK,
                MemoryKind.IMAGE_DATA,
            ),
        )
        self._memory.add(
            MemoryKind.REVIEW,
            report,
            agent_id=self._agent_id,
        )
        return report
