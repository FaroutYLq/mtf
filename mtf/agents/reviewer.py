"""ReviewerAgent: evaluates theory validity and suggests further experiments."""

from __future__ import annotations

from mtf.agents.base import BaseAgent
from mtf.memory import MemoryKind, SharedMemory
from mtf.tools.gpd_mcp import GPDMCPClient

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
        gpd: GPDMCPClient | None = None,
    ) -> None:
        gpd_tools = []
        if gpd is not None and gpd.available:
            gpd_tools = [
                t
                for t in [
                    gpd.make_tool(
                        "verification",
                        "get_checklist",
                        "Get the domain-specific physics verification checklist. Input: domain (str, e.g. condensed_matter, qft, plasma, amo). Returns a list of check IDs and descriptions to work through systematically.",
                    ),
                    gpd.make_tool(
                        "verification",
                        "run_check",
                        "Run a specific verification check against a hypothesis or fit result. check_id: '5.1'=Dimensional analysis, '5.2'=Symmetry, '5.3'=Limiting cases, '5.4'=Conservation laws, '5.5'=Numerical convergence, '5.18'=Fit family mismatch, '5.19'=Estimator family mismatch. domain: physics domain string. artifact_content: the text of the fit result or hypothesis to check.",
                    ),
                    gpd.make_tool(
                        "verification",
                        "dimensional_check",
                        "Check dimensional consistency of physics equations. Input: expressions (list of strings, each a dimensional equation like '[M][L]^2[T]^-2 = [M][L]^2[T]^-2'). Returns pass/fail per expression with issues.",
                    ),
                    gpd.make_tool(
                        "verification",
                        "limiting_case_check",
                        "Verify that a model's limiting cases are correctly documented. Input: expression (str, the model equation), limits (dict mapping limit description to expected behavior, e.g. {'g -> 0': 'reduces to free particle'}).",
                    ),
                    gpd.make_tool(
                        "errors",
                        "check_error_classes",
                        "Given a description of a physics computation or hypothesis, return the top-15 most relevant error classes from a catalog of 104 known physics errors (sign errors, dimensional errors, convention pitfalls, convergence issues, etc.) with relevance scores.",
                    ),
                    gpd.make_tool(
                        "errors",
                        "get_detection_strategy",
                        "Get the specific detection strategy for a known physics error class by integer ID (1-104). Use after check_error_classes to get actionable detection methods.",
                    ),
                    gpd.make_tool(
                        "patterns",
                        "lookup_pattern",
                        "Search the persistent cross-session physics error pattern library for known error patterns in this domain. Input: domain (str), category (str, e.g. sign-error, dimensional-error, approximation-failure), keywords (str). Returns patterns found in previous runs.",
                    ),
                    gpd.make_tool(
                        "patterns",
                        "add_pattern",
                        "Record a newly discovered physics error pattern for future reference across sessions. Input: domain, title, category (sign-error/factor-error/convention-pitfall/convergence-issue/approximation-failure/numerical-instability/conceptual-error/dimensional-error), severity (critical/warning/info), description, detection, prevention, example, test_value.",
                    ),
                ]
                if t is not None
            ]
        super().__init__(
            agent_id=agent_id,
            model=model,
            tools=[*gpd_tools],
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
