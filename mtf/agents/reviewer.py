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
2. Call `get_checklist` for each physics domain listed in the conventions context
   (e.g. condensed_matter, qft, plasma, amo). When the phenomenon spans multiple
   domains, call it once per domain and merge the checklists.
3. For each fit result, call `run_check` with the following mandatory check IDs,
   passing the fit result text as artifact_content:
   - "5.1" (dimensional consistency)
   - "5.3" (limiting cases: does the model recover known limits?)
   - "5.2" (symmetry: does the model respect required symmetries?)
   - "5.18" (fit family mismatch: is the chosen model family appropriate?)
   Also call `dimensional_check` if specific equations appear in the fit results.
4. Call `lookup_pattern` to surface any known recurring errors in this domain and
   category before finalising your verdict.
5. Call `add_pattern` whenever you confirm a genuine physics error that would be useful
   to record for future sessions.
6. Assess the validity of each proposed hypothesis against the data and literature.
7. Identify weaknesses, alternative explanations, or confounds.
8. Suggest specific further experiments that would distinguish competing hypotheses.
9. Provide an overall recommendation.

For each hypothesis, produce a structured verdict using exactly one of:
  SUPPORTED / PLAUSIBLE / SPECULATIVE / REJECTED
with the relevant check IDs cited (e.g. "REJECTED — check 5.1 FAIL: units inconsistent").

Rank hypotheses by: (1) physics check results, (2) parsimony, (3) first-principles
basis, (4) chi² last. Be rigorous, constructive, and cite specific evidence from the
provided context.

If `[FITTING_SKIPPED]` appears in your context, the fitting phase was skipped and no
`[FIT_RESULT]` entries exist for the current run. In this mode:
- Skip steps 3–4 (fit-result checks and dimensional_check on fit results).
- Instead, read all `[QUALITATIVE_EVAL]` entries and assess whether the qualitative
  verdicts are well-supported by the available literature and image data.
- Do not rank by chi² — rank by: (1) physics check results on theoretical arguments,
  (2) parsimony, (3) first-principles basis.
- Your recommendation should focus on what data collection would be needed to upgrade
  qualitative assessments to quantitative fits."""


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
                MemoryKind.QUALITATIVE_EVAL,
                MemoryKind.FITTING_SKIPPED,
            ),
        )
        self._memory.add(
            MemoryKind.REVIEW,
            report,
            agent_id=self._agent_id,
        )
        return report
