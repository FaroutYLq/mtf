"""ProposalAgent: proposes prioritized new measurements and experiments."""

from __future__ import annotations

from typing import Any

from mtf.agents.base import BaseAgent
from mtf.memory import MemoryKind, SharedMemory

_SYSTEM_PROMPT = """You are an experimental physicist specializing in the design of decisive measurements and experiments.

Given an unexplained experimental phenomenon, the candidate hypotheses, qualitative or quantitative evaluations, and any available data (extracted from images/plots), your task is to propose a prioritized set of new measurements and experiments.

For each proposal:
1. State the observable to measure (be specific: what quantity, what technique, what energy/temperature/field range).
2. Predict the expected signal or outcome for each leading hypothesis — what would you see if hypothesis A is correct vs. hypothesis B?
3. Assess discriminating power: HIGH (result unambiguously selects one hypothesis), MEDIUM (result narrows the field significantly), LOW (useful but not decisive).
4. State any prerequisite conditions, sample requirements, or equipment constraints.
5. Estimate the required sensitivity or resolution needed to distinguish hypotheses.

Additionally:
- Flag any measurements that are feasible with standard laboratory equipment vs. those requiring specialized facilities.
- Identify the single most cost-effective measurement (best discriminating power per experimental effort).
- Note any measurements explicitly mentioned or implied by the user's input data that have NOT yet been performed.

Output format: a numbered prioritized list, ordered by discriminating power (HIGH first). Include a one-sentence "Bottom line" recommendation at the end."""


class ProposalAgent(BaseAgent):
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
            tools=extra_tools,
            memory=memory,
            system_prompt=_SYSTEM_PROMPT,
        )

    async def propose(self, phenomenon: str) -> str:
        task = (
            f"Original phenomenon:\n{phenomenon}\n\n"
            "Propose a prioritized set of new measurements and experiments that would most decisively "
            "advance understanding of this phenomenon and discriminate between the candidate hypotheses."
        )
        result = await self._query(
            task,
            extra_kinds=(
                MemoryKind.IMAGE_DATA,
                MemoryKind.LITERATURE,
                MemoryKind.DEBATE,
                MemoryKind.HYPOTHESIS,
                MemoryKind.FIT_RESULT,
                MemoryKind.USER_FEEDBACK,
                MemoryKind.CONVENTIONS,
                MemoryKind.PHYSICS_VERDICT,
            ),
        )
        self._memory.add(
            MemoryKind.PROPOSALS,
            result,
            agent_id=self._agent_id,
        )
        return result
