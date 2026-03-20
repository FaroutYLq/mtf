"""ReviewerAgent: evaluates theory validity and suggests further experiments."""

from __future__ import annotations

from typing import Any

from mtf.agents.base import BaseAgent
from mtf.memory import MemoryKind, SharedMemory

_SYSTEM_PROMPT = """You are a senior physics reviewer with broad expertise across
condensed matter, high energy, AMO, and astrophysics. Given literature reports,
fit results, and debate summaries, you:

1. First call `check_error_classes` with the phenomenon description to identify the
   top-15 most relevant error classes to watch for.
2. Call `get_checklist` with the physics domain (e.g. condensed_matter, qft, plasma,
   amo) to retrieve the structured verification checklist.
3. Work through each checklist item using `run_check` applied to the fit results and
   hypotheses — cite specific check IDs in your verdict (e.g. "check 5.3 FAIL: model
   does not recover free-particle dispersion at g→0").
4. Call `lookup_pattern` to surface any known recurring errors in this domain and
   category before finalising your verdict.
5. Call `add_pattern` whenever you confirm a genuine physics error that would be useful
   to record for future sessions.
6. Assess the validity of each proposed hypothesis against the data and literature.
7. Identify weaknesses, alternative explanations, or confounds.
8. Suggest specific further experiments that would distinguish competing hypotheses.
9. Provide an overall recommendation.

Rank hypotheses by physics correctness first, chi² last. Produce structured verdicts
that cite specific check IDs. Be rigorous, constructive, and cite specific evidence
from the provided context."""


class ReviewerAgent(BaseAgent):
    def __init__(
        self,
        agent_id: str,
        model: str,
        memory: SharedMemory,
        gpd_tools: list[Any] | None = None,
    ) -> None:
        extra_tools: list[Any] = gpd_tools if gpd_tools is not None else []
        super().__init__(
            agent_id=agent_id,
            model=model,
            tools=[*extra_tools],
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
                MemoryKind.CONVENTIONS,
                MemoryKind.PHYSICS_VERDICT,
            ),
        )
        self._memory.add(
            MemoryKind.REVIEW,
            report,
            agent_id=self._agent_id,
        )
        return report
