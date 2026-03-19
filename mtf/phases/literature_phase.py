"""Literature phase: fan-out literature agents, debate loop, user approval gate."""

from __future__ import annotations

import asyncio

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
) -> list[str]:
    """Fan out N literature agents, debate, get user approval, loop.

    Returns approved hypotheses as a list of strings.
    """
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
