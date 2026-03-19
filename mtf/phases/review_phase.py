"""Review phase: fan-out reviewer agents, final debate, final report."""

from __future__ import annotations

import asyncio

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
) -> str:
    """Fan out K reviewer agents and synthesize a final report."""
    await interface.show(
        f"**Review phase**\n\nDispatching {config.n_reviewer} reviewer agents...",
        title="MTF: Review",
    )

    agents = [
        ReviewerAgent(
            agent_id=f"rev-{i}",
            model=config.reviewer_model,
            memory=memory,
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
