"""QualitativeEvaluationAgent: evaluates hypotheses qualitatively when fitting is skipped."""

from __future__ import annotations

from typing import Any

from mtf.agents.base import BaseAgent
from mtf.memory import MemoryKind, SharedMemory

_SYSTEM_PROMPT = """You are a senior physicist conducting a qualitative evaluation of hypotheses about an experimental phenomenon, in the absence of numerical fitting data.

For each hypothesis provided:
1. Assess plausibility based on established physical theory and the literature context available.
2. Identify what observational or experimental signatures would be expected if the hypothesis is correct, referencing any image/plot data already extracted.
3. Explicitly state what numerical data (measurements, spectra, time series, etc.) would be needed to upgrade a qualitative assessment to a quantitative fit.
4. Rate each hypothesis: SUPPORTED (well-grounded in theory + consistent with available observations) / PLAUSIBLE (theoretically reasonable, no contradicting observations) / SPECULATIVE (possible but lacking strong theoretical or observational backing) / REJECTED (contradicted by known physics or available observations).
5. For each non-REJECTED hypothesis, propose the single most decisive measurement that would confirm or refute it.

Be rigorous. Cite specific physical arguments, known phenomena, and any quantitative features visible in the image-extracted data."""


class QualitativeEvaluationAgent(BaseAgent):
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

    async def evaluate(self, phenomenon: str, hypotheses: list[str]) -> str:
        task = (
            f"Original phenomenon:\n{phenomenon}\n\n"
            "Hypotheses to evaluate:\n"
            + "\n".join(f"- {h}" for h in hypotheses)
            + "\n\nConduct a rigorous qualitative evaluation of each hypothesis."
        )
        result = await self._query(
            task,
            extra_kinds=(
                MemoryKind.IMAGE_DATA,
                MemoryKind.LITERATURE,
                MemoryKind.DEBATE,
                MemoryKind.USER_FEEDBACK,
                MemoryKind.CONVENTIONS,
                MemoryKind.PHYSICS_VERDICT,
                MemoryKind.FITTING_WARNINGS,
            ),
        )
        self._memory.add(
            MemoryKind.QUALITATIVE_EVAL,
            result,
            agent_id=self._agent_id,
        )
        return result
