"""Fitting phase: toolkit resolution, fan-out fitting agents, debate."""

from __future__ import annotations

import asyncio

from mtf.agents.fitting import FittingAgent
from mtf.agents.tool_builder import ToolBuilderAgent
from mtf.config import MTFConfig
from mtf.debate import DebateEngine
from mtf.interface import HumanInterface
from mtf.memory import MemoryKind, SharedMemory
from mtf.toolkit.registry import ToolkitRegistry


def _looks_complex(value: str) -> bool:
    """Return True when the input is too rich for a plain eval()."""
    triggers = ("\n", "def ", "class ", "import ", "->", "lambda ", "csv", "\t")
    return any(t in value for t in triggers)


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
    seen_missing: set[str] = set()
    for hypothesis in hypotheses:
        needed = await probe_agent.identify_needed_toolkit_items(hypothesis)
        missing = [item for item in needed if item.startswith("MISSING:")]
        new_missing = [item for item in missing if item not in seen_missing]
        seen_missing.update(new_missing)
        if new_missing:
            await interface.show(
                "The following toolkit items are needed but missing:\n"
                + "\n".join(new_missing),
                title="MTF: Toolkit Request",
            )
            for item in new_missing:
                item_name = item.replace("MISSING:", "").strip()
                value = await interface.ask(
                    f"Please provide value for '{item_name}'\n"
                    "(Python literal, function definition, datasheet, or 'skip'):"
                )
                if not value.strip() or value.strip().lower() == "skip":
                    continue

                # --- Fast path: try plain eval for simple literals ----------------
                if not _looks_complex(value):
                    try:
                        toolkit.register_data(item_name, eval(value))  # noqa: S307
                        continue
                    except Exception:
                        pass  # fall through to tool-builder

                # --- Slow path: spawn a ToolBuilderAgent for complex input --------
                await interface.show(
                    f"Input looks complex — spawning tool-builder agent for '{item_name}'...",
                    title="MTF: Tool Builder",
                )
                builder = ToolBuilderAgent(
                    agent_id=f"toolbuild-{item_name}",
                    model=config.fitting_model,
                    memory=memory,
                )
                result = await builder.digest(item_name, value)

                if result.error:
                    await interface.show(
                        f"Tool-builder failed for '{item_name}':\n{result.error}\n\n"
                        "Storing raw input as string fallback.",
                        title="MTF: Tool Builder Error",
                    )
                    toolkit.register_data(item_name, value)
                    continue

                for data_name, data_val in result.data.items():
                    toolkit.register_data(data_name, data_val)
                for model_name, model_fn in result.models.items():
                    toolkit.register_model(model_name, model_fn)

                await interface.show(result.summary, title="MTF: Toolkit Built")

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
