"""Literature phase: fan-out literature agents, debate loop, user approval gate."""

from __future__ import annotations

import asyncio
from typing import Any

from mtf.agents.literature import LiteratureAgent
from mtf.config import MTFConfig
from mtf.debate import DebateEngine
from mtf.interface import HumanInterface
from mtf.memory import MemoryKind, SharedMemory


async def _prefetch_domain_patterns(
    config: MTFConfig,
    memory: SharedMemory,
    phenomenon: str,
    gpd: Any,
) -> None:
    """Pre-fetch cross-session convention-pitfall patterns per domain (Addition 6a)."""
    async def fetch_one(domain: str) -> None:
        result = await asyncio.to_thread(
            gpd.call, "patterns", "lookup_pattern",
            domain=domain, category="convention-pitfall", context=phenomenon[:200],
        )
        if result:
            memory.add(MemoryKind.DOMAIN_PATTERNS, result, domain=domain, source="literature_prefetch")

    await asyncio.gather(*(fetch_one(d) for d in config.physics_domains))


async def _screen_hypothesis_plausibility(
    hypothesis_candidates: list[str],
    config: MTFConfig,
    memory: SharedMemory,
    interface: HumanInterface,
    gpd: Any,
) -> set[str]:
    """Run limiting_case_check per candidate hypothesis and show plausibility badges (Addition 5).

    Returns the set of CRITICAL-FAIL hypothesis texts.  The caller uses this to
    optionally filter hyp_lines when ``config.auto_reject_physics_failures`` is True.
    """
    if not hypothesis_candidates or not config.literature_plausibility_screen:
        return set()

    async def check_one(hyp: str) -> tuple[str, str]:
        result = await asyncio.to_thread(
            gpd.call, "verification", "limiting_case_check",
            expression=hyp[:500],
            limits=["classical_limit", "zero_coupling", "large_N"],
        )
        return hyp, result or ""

    results = await asyncio.gather(*(check_one(h) for h in hypothesis_candidates))

    lines = ["**Hypothesis Plausibility Screen**\n"]
    rejected: set[str] = set()
    for hyp, result in results:
        if not result:
            badge = "[UNKNOWN]"
        elif "CRITICAL" in result.upper() and "FAIL" in result.upper():
            badge = "[FAIL]"
            rejected.add(hyp)
        elif "FAIL" in result.upper() or "WARN" in result.upper():
            badge = "[WARN]"
        else:
            badge = "[PASS]"
        lines.append(f"{badge} {hyp[:120]}")
        if result:
            memory.add(
                MemoryKind.PHYSICS_VERDICT,
                result,
                source="limiting_case_screen",
                hypothesis=hyp[:200],
            )

    await interface.show("\n".join(lines), title="MTF: Plausibility Screen")
    return rejected


async def run_literature_phase(
    phenomenon: str,
    config: MTFConfig,
    memory: SharedMemory,
    interface: HumanInterface,
    debate_engine: DebateEngine,
    gpd: Any | None = None,
) -> list[str]:
    """Fan out N literature agents, debate, get user approval, loop.

    Returns approved hypotheses as a list of strings.
    """
    # Build GPD tools for LiteratureAgent
    gpd_tools: list[Any] = []
    if gpd is not None:
        gpd_tools = [t for t in [
            gpd.make_tool("errors", "check_error_classes",
                "Given a physics phenomenon or hypothesis description, return the top relevant "
                "physics error classes from a catalog of 104 known errors, with relevance scores "
                "and detection strategies. Call this first before searching literature."),
            gpd.make_tool("protocols", "route_protocol",
                "Find the canonical computation protocol for a given physics calculation type. "
                "Input: computation_type (str). Returns matching protocols with relevance scores. "
                "Use to identify what methodology the literature should follow."),
            gpd.make_tool("patterns", "lookup_pattern",
                "Look up cross-session patterns for a given domain and category (e.g. "
                "'convention-pitfall', 'sign-error'). Call with domain and category to retrieve "
                "known pitfalls recorded from prior runs."),
            gpd.make_tool("patterns", "add_pattern",
                "Record a new systematic error pattern found in a class of papers. Call with "
                "domain, category='convention-pitfall', and a description of the pattern. "
                "This persists across runs so future literature agents benefit."),
        ] if t is not None]

    # Lock physics conventions before the first fan-out (one entry per domain)
    if gpd is not None:
        for domain in config.physics_domains:
            conventions = gpd.call("conventions", "subfield_defaults", domain=domain)
            if conventions:
                memory.add(MemoryKind.CONVENTIONS, conventions, domain=domain)
        locked = config.physics_domains
        if locked:
            domains_str = ", ".join(f"**{d}**" for d in locked)
            await interface.show(
                f"Physics conventions locked for: {domains_str}.",
                title="MTF: GPD Conventions",
            )

        # Pre-fetch cross-session domain patterns (Addition 6a)
        await _prefetch_domain_patterns(config, memory, phenomenon, gpd)

    synthesis = ""
    for round_num in range(1, config.max_debate_rounds + 1):
        await interface.show(
            f"**Literature phase — round {round_num}/{config.max_debate_rounds}**\n\n"
            f"Dispatching {config.n_literature} literature agents...",
            title="MTF: Literature",
        )

        agents = [
            LiteratureAgent(
                agent_id=f"lit-{i}",
                model=config.literature_model,
                memory=memory,
                gpd_tools=gpd_tools,
            )
            for i in range(config.n_literature)
        ]
        reports = await asyncio.gather(*(a.investigate(phenomenon) for a in agents))

        synthesis = await debate_engine.synthesize(
            list(reports),
            phase="literature",
            extra_context=f"Phenomenon: {phenomenon}",
        )

        await interface.show(synthesis, title=f"Literature Synthesis (round {round_num})")

        # Extract candidate hypotheses for plausibility screening (Addition 5).
        # Skip the screen when no keyword-matching lines are found rather than passing
        # raw synthesis text as an expression — that produces misleading verdicts.
        rejected: set[str] = set()
        if gpd is not None:
            hyp_candidates = [
                line.strip()
                for line in synthesis.splitlines()
                if line.strip() and any(
                    kw in line.lower()
                    for kw in ("hypothesis", "proposed", "model", "theory")
                )
            ]
            if hyp_candidates:
                rejected = await _screen_hypothesis_plausibility(
                    hyp_candidates, config, memory, interface, gpd
                )

        approved = await interface.confirm("Do you approve these hypotheses?")
        if approved:
            # Extract hypotheses from synthesis and store them
            hyp_lines = [
                line.strip()
                for line in synthesis.splitlines()
                if line.strip() and any(
                    kw in line.lower() for kw in ("hypothesis", "proposed", "model", "theory")
                )
            ]
            if not hyp_lines:
                # Fall back: store synthesis as one hypothesis
                hyp_lines = [synthesis]
            # Optionally filter CRITICAL-FAIL hypotheses (config.auto_reject_physics_failures)
            if config.auto_reject_physics_failures and rejected:
                filtered = [h for h in hyp_lines if h not in rejected]
                hyp_lines = filtered if filtered else hyp_lines  # never return empty
            for hyp in hyp_lines:
                memory.add(MemoryKind.HYPOTHESIS, hyp)
            return hyp_lines

        feedback = await interface.ask(
            "What guidance do you have for the next round of literature search?"
        )
        if feedback.strip():
            memory.add(MemoryKind.USER_FEEDBACK, feedback)

    # If max rounds reached without explicit approval, use last synthesis
    await interface.show(
        "Max debate rounds reached. Proceeding with last synthesis.",
        title="MTF: Literature",
    )
    hypotheses = [synthesis]
    for hyp in hypotheses:
        memory.add(MemoryKind.HYPOTHESIS, hyp)
    return hypotheses
