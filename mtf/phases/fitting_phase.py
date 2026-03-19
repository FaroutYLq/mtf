"""Fitting phase: toolkit resolution, fan-out fitting agents, debate."""

from __future__ import annotations

import asyncio

from mtf.agents.fitting import FittingAgent
from mtf.config import MTFConfig
from mtf.debate import DebateEngine
from mtf.interface import HumanInterface
from mtf.memory import MemoryKind, SharedMemory
from mtf.toolkit.registry import ToolkitRegistry


async def run_fitting_phase(
    hypotheses: list[str],
    config: MTFConfig,
    memory: SharedMemory,
    interface: HumanInterface,
    debate_engine: DebateEngine,
    toolkit: ToolkitRegistry,
) -> str:
    """For each hypothesis, fan out M fitting agents and synthesize results."""
    semaphore = asyncio.Semaphore(config.fitting_semaphore_limit)

    async def fit_with_semaphore(agent: FittingAgent, hypothesis: str) -> dict[str, object]:
        async with semaphore:
            return await agent.fit(hypothesis)

    await interface.show(
        f"**Fitting phase**\n\nFitting {len(hypotheses)} hypotheses "
        f"with {config.n_fitting} agents each...",
        title="MTF: Fitting",
    )

    # Check for missing toolkit items
    probe_agent = FittingAgent(
        agent_id="probe",
        model=config.fitting_model,
        memory=memory,
        toolkit=toolkit,
    )
    for hypothesis in hypotheses:
        needed = await probe_agent.identify_needed_toolkit_items(hypothesis)
        missing = [item for item in needed if item.startswith("MISSING:")]
        if missing:
            await interface.show(
                "The following toolkit items are needed but missing:\n"
                + "\n".join(missing),
                title="MTF: Toolkit Request",
            )
            for item in missing:
                item_name = item.replace("MISSING:", "").strip()
                value = await interface.ask(
                    f"Please provide value for '{item_name}' (Python literal or skip):"
                )
                if value.strip() and value.strip().lower() != "skip":
                    try:
                        toolkit.register_data(item_name, eval(value))  # noqa: S307
                    except Exception:
                        toolkit.register_data(item_name, value)

    # Fan out fitting agents per hypothesis
    all_fit_reports: list[str] = []
    if config.fitting_scope == "per_hypothesis":
        for hyp in hypotheses:
            agents = [
                FittingAgent(
                    agent_id=f"fit-{hyp[:20]}-{i}",
                    model=config.fitting_model,
                    memory=memory,
                    toolkit=toolkit,
                )
                for i in range(config.n_fitting)
            ]
            results = await asyncio.gather(
                *(fit_with_semaphore(a, hyp) for a in agents)
            )
            reports = [str(r) for r in results]
            all_fit_reports.extend(reports)
    else:
        agents = [
            FittingAgent(
                agent_id=f"fit-{i}",
                model=config.fitting_model,
                memory=memory,
                toolkit=toolkit,
            )
            for i in range(config.n_fitting)
        ]
        results = await asyncio.gather(
            *(fit_with_semaphore(a, hyp) for hyp in hypotheses for a in agents)
        )
        all_fit_reports = [str(r) for r in results]

    synthesis = await debate_engine.synthesize(
        all_fit_reports,
        phase="fitting",
        extra_context=f"Hypotheses being evaluated: {hypotheses}",
    )

    await interface.show(synthesis, title="Fitting Synthesis")

    approved = await interface.confirm("Do you approve the fitting results?")
    if not approved:
        feedback = await interface.ask("Guidance for fitting?")
        if feedback.strip():
            memory.add(MemoryKind.USER_FEEDBACK, feedback)

    return synthesis
