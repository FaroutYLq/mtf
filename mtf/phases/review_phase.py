"""Review phase: fan-out reviewer agents, final debate, final report."""

from __future__ import annotations

import asyncio
from typing import Any

from mtf.agents.reviewer import ReviewerAgent
from mtf.config import MTFConfig
from mtf.debate import DebateEngine
from mtf.interface import HumanInterface
from mtf.memory import SharedMemory


async def run_review_phase(
    phenomenon: str,
    config: MTFConfig,
    memory: SharedMemory,
    interface: HumanInterface,
    debate_engine: DebateEngine,
    gpd: Any | None = None,
) -> str:
    """Fan out K reviewer agents and synthesize a final report."""
    # Build GPD tools for ReviewerAgent
    gpd_tools: list[Any] = []
    if gpd is not None:
        gpd_tools = [t for t in [
            gpd.make_tool("verification", "get_checklist",
                "Get the domain-specific physics verification checklist. "
                "Input: domain (str, e.g. condensed_matter, qft, gr, amo). "
                "Returns list of check IDs and descriptions. Call this FIRST."),
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

    await interface.show(
        f"**Review phase**\n\nDispatching {config.n_reviewer} reviewer agents...",
        title="MTF: Review",
    )

    agents = [
        ReviewerAgent(
            agent_id=f"rev-{i}",
            model=config.reviewer_model,
            memory=memory,
            gpd_tools=gpd_tools,
        )
        for i in range(config.n_reviewer)
    ]
    reports = await asyncio.gather(*(a.review(phenomenon) for a in agents))

    final_report = await debate_engine.synthesize(
        list(reports),
        phase="review",
        extra_context=f"Original phenomenon: {phenomenon}",
    )

    await interface.show(final_report, title="Final Report")
    return final_report
