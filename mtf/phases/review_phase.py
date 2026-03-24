"""Review phase: fan-out reviewer agents, final debate, final report."""

from __future__ import annotations

import asyncio
from typing import Any

from mtf.agents.proposal import ProposalAgent
from mtf.agents.reviewer import ReviewerAgent
from mtf.config import MTFConfig
from mtf.debate import DebateEngine
from mtf.interface import HumanInterface
from mtf.memory import MemoryKind, SharedMemory


async def run_review_phase(
    phenomenon: str,
    config: MTFConfig,
    memory: SharedMemory,
    interface: HumanInterface,
    debate_engine: DebateEngine,
    gpd: Any | None = None,
) -> str:
    """Fan out K reviewer agents and proposal agents, synthesize a final report."""
    # Build GPD tools for ReviewerAgent
    gpd_tools: list[Any] = []
    if gpd is not None:
        gpd_tools = [t for t in [
            gpd.make_tool("verification", "get_checklist",
                "Get the domain-specific physics verification checklist. "
                "Input: domain (str, e.g. condensed_matter, qft, gr, amo). "
                "Returns list of check IDs and descriptions. Call this FIRST, "
                "once per domain if the phenomenon spans multiple domains, "
                "and merge the resulting checklists."),
            gpd.make_tool("verification", "run_check",
                "Run a specific physics verification check against a fit result or hypothesis text. "
                "Input: check_id (str: '5.1'=dimensional, '5.2'=symmetry, '5.3'=limiting_cases, "
                "'5.4'=conservation, '5.5'=convergence, '5.18'=fit_family_mismatch), "
                "domain (str), artifact_content (str of the fit result). "
                "Returns automated issues found."),
            gpd.make_tool("verification", "dimensional_check",
                "Check dimensional consistency of physics equations. "
                "Input: expressions (list of strings like '[M][L]^2[T]^-2 = [M][L]^2[T]^-2'). "
                "Returns pass/fail per expression."),
            gpd.make_tool("verification", "limiting_case_check",
                "Verify that a model's limiting cases are documented and correct. "
                "Input: expression (str), limits (dict mapping limit description to expected result)."),
            gpd.make_tool("errors", "check_error_classes",
                "Given a physics computation description, return the top-15 relevant error classes "
                "from a catalog of 104 known physics errors. Call this early in your review "
                "to know what specific mistakes to hunt for."),
            gpd.make_tool("errors", "get_detection_strategy",
                "Get the detection strategy for a specific physics error class by ID (int 1-104). "
                "Use after check_error_classes identifies relevant error classes."),
            gpd.make_tool("patterns", "lookup_pattern",
                "Search the persistent cross-session physics error pattern library. "
                "Input: domain (str), category (str: sign-error/factor-error/convention-pitfall/"
                "convergence-issue/approximation-failure/dimensional-error), keywords (str). "
                "Returns known patterns from previous runs."),
            gpd.make_tool("patterns", "add_pattern",
                "Record a newly discovered physics error pattern for future reference across sessions. "
                "Input: domain, title, category, severity (low/medium/high/critical), "
                "description, detection, prevention, example, test_value."),
        ] if t is not None]

    # Build GPD tools for ProposalAgent (simpler subset)
    proposal_gpd_tools: list[Any] = []
    if gpd is not None:
        proposal_gpd_tools = [t for t in [
            gpd.make_tool("patterns", "lookup_pattern",
                "Search the persistent cross-session physics error pattern library. "
                "Input: domain (str), category (str: sign-error/factor-error/convention-pitfall/"
                "convergence-issue/approximation-failure/dimensional-error), keywords (str). "
                "Returns known patterns from previous runs that should be addressed by new measurements."),
            gpd.make_tool("errors", "check_error_classes",
                "Given a physics computation description, return the top-15 relevant error classes "
                "from a catalog of 104 known physics errors. Use to identify systematic issues "
                "that new measurements should resolve."),
        ] if t is not None]

    await interface.show(
        f"**Review phase**\n\nDispatching {config.n_reviewer} reviewer agents "
        f"and {config.n_proposal} proposal agents...",
        title="MTF: Review",
    )

    def _reviewer_model(i: int) -> str:
        models = config.reviewer_models
        if models:
            return models[i % len(models)]
        return config.reviewer_model

    agents = [
        ReviewerAgent(
            agent_id=f"rev-{i}",
            model=_reviewer_model(i),
            memory=memory,
            gpd_tools=gpd_tools,
        )
        for i in range(config.n_reviewer)
    ]

    proposal_agents = [
        ProposalAgent(
            agent_id=f"proposal-{i}",
            model=config.proposal_model,
            memory=memory,
            gpd_tools=proposal_gpd_tools,
        )
        for i in range(config.n_proposal)
    ]

    all_results = await asyncio.gather(
        *(a.review(phenomenon) for a in agents),
        *(a.propose(phenomenon) for a in proposal_agents),
    )
    reviewer_reports = list(all_results[:config.n_reviewer])
    proposal_reports = list(all_results[config.n_reviewer:])

    # Second-pass verification loop (vibe-physics: "repeat Check again until nothing new")
    if config.reviewer_verification_passes > 1:
        await interface.show(
            f"Running {config.reviewer_verification_passes - 1} additional verification pass(es)...",
            title="MTF: Review",
        )
        for pass_num in range(config.reviewer_verification_passes - 1):
            second_pass_tasks = [
                (
                    f"Original phenomenon:\n{phenomenon}\n\n"
                    f"Your previous review:\n{report}\n\n"
                    f"Re-read the above review from the beginning. Did you miss anything? "
                    f"Check every claim, equation, parameter range, and citation again. "
                    f"Append additional findings under 'Additional concerns:'. "
                    f"If you found nothing new, state that explicitly."
                )
                for report in reviewer_reports
            ]
            second_pass_reports = await asyncio.gather(
                *(
                    agent._query(
                        task,
                        extra_kinds=(
                            MemoryKind.LITERATURE,
                            MemoryKind.DEBATE,
                            MemoryKind.FIT_RESULT,
                            MemoryKind.USER_FEEDBACK,
                            MemoryKind.IMAGE_DATA,
                            MemoryKind.CONVENTIONS,
                            MemoryKind.PHYSICS_VERDICT,
                            MemoryKind.INTEGRITY_WARNING,
                            MemoryKind.QUALITATIVE_EVAL,
                            MemoryKind.FITTING_SKIPPED,
                        ),
                    )
                    for agent, task in zip(agents, second_pass_tasks)
                )
            )
            reviewer_reports = list(second_pass_reports)

    review_synthesis = await debate_engine.synthesize(
        reviewer_reports,
        phase="review",
        extra_context=f"Original phenomenon: {phenomenon}",
    )

    proposal_synthesis = await debate_engine.synthesize(
        proposal_reports,
        phase="proposals",
        extra_context=f"Original phenomenon: {phenomenon}",
        store_as_debate=False,
    )
    memory.add(MemoryKind.PROPOSALS, proposal_synthesis)

    final_report = f"{review_synthesis}\n\n---\n\n## Proposed Measurements\n\n{proposal_synthesis}"

    await interface.show(final_report, title="Final Report")
    return final_report
