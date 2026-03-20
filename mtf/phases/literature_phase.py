"""Literature phase: fan-out literature agents, debate loop, user approval gate."""

from __future__ import annotations

import asyncio
from typing import Any

from mtf.agents.literature import LiteratureAgent
from mtf.config import MTFConfig
from mtf.debate import DebateEngine
from mtf.interface import HumanInterface
from mtf.memory import MemoryKind, SharedMemory


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
