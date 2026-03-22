"""Fitting phase: toolkit resolution, fan-out fitting agents, debate."""

from __future__ import annotations

import asyncio
from typing import Any

from mtf.agents.fitting import FittingAgent
from mtf.agents.tool_builder import ToolBuilderAgent
from mtf.config import MTFConfig
from mtf.debate import DebateEngine
from mtf.interface import HumanInterface
from mtf.memory import MemoryKind, SharedMemory
from mtf.toolkit.registry import ToolkitRegistry


def _looks_complex(value: str) -> bool:
    """Return True when the input is too rich for a plain eval().

    Uses the stdlib compiler to check whether *value* is a valid single
    expression.  If ``compile()`` with mode ``'eval'`` raises a
    ``SyntaxError`` the string cannot be safely evaluated and the
    ToolBuilderAgent slow path is needed.
    """
    try:
        compile(value, "<string>", "eval")
        return False
    except SyntaxError:
        return True


async def _prefetch_fitting_warnings(
    hypotheses: list[str],
    config: MTFConfig,
    memory: SharedMemory,
    gpd: Any,
) -> None:
    """Pre-fetch known pitfall patterns and error classes before the fit fan-out (Addition 4)."""
    async def check_hyp(domain: str, hyp: str) -> list[str]:
        results = await asyncio.gather(
            asyncio.to_thread(
                gpd.call, "patterns", "lookup_pattern",
                domain=domain, category="sign-error", context=hyp[:200],
            ),
            asyncio.to_thread(
                gpd.call, "patterns", "lookup_pattern",
                domain=domain, category="convergence-issue", context=hyp[:200],
            ),
            asyncio.to_thread(
                gpd.call, "errors", "check_error_classes",
                description=hyp[:500],
            ),
        )
        return [r for r in results if r]

    for domain in config.physics_domains:
        domain_results = await asyncio.gather(
            *(check_hyp(domain, hyp) for hyp in hypotheses)
        )
        for hyp, results in zip(hypotheses, domain_results):
            combined = "\n".join(results)
            if combined.strip():
                memory.add(
                    MemoryKind.FITTING_WARNINGS,
                    combined,
                    domain=domain,
                    hypothesis=hyp[:200],
                )


async def _run_phase_physics_checks(
    fit_reports: list[str],
    hypotheses: list[str],
    memory: SharedMemory,
    gpd: Any,
    n_fitting: int = 1,
) -> None:
    """Run dimensional (5.1) and limiting-case (5.3) checks on fit results (Addition 2).

    Writes non-empty results as PHYSICS_VERDICT entries so DebateEngine.synthesize()
    picks them up for the fitting synthesis.

    ``n_fitting`` is the number of fitting agents per hypothesis; it is used to map
    each report back to its originating hypothesis.  Reports are ordered
    [hyp0*n_fitting, hyp1*n_fitting, ...] in both ``per_hypothesis`` and ``all`` modes,
    so ``hypotheses[i // n_fitting]`` gives the correct hypothesis for report ``i``.
    """
    if gpd is None or not hypotheses or not fit_reports:
        return

    n_hyp = len(hypotheses)

    async def run_checks(hyp: str, report: str) -> None:
        results = await asyncio.gather(
            asyncio.to_thread(
                gpd.call, "verification", "run_check",
                check_id="5.1", content=report[:1000],
            ),
            asyncio.to_thread(
                gpd.call, "verification", "run_check",
                check_id="5.3", content=report[:1000],
            ),
        )
        for result in results:
            if result:
                memory.add(
                    MemoryKind.PHYSICS_VERDICT,
                    result,
                    source="fitting_phase_check",
                    hypothesis=hyp[:200],
                )

    await asyncio.gather(*(
        run_checks(hypotheses[min(i // n_fitting, n_hyp - 1)], report)
        for i, report in enumerate(fit_reports)
    ))


async def run_fitting_phase(
    hypotheses: list[str],
    config: MTFConfig,
    memory: SharedMemory,
    interface: HumanInterface,
    debate_engine: DebateEngine,
    toolkit: ToolkitRegistry,
    gpd: Any | None = None,
) -> str:
    """For each hypothesis, fan out M fitting agents and synthesize results."""
    semaphore = asyncio.Semaphore(config.fitting_semaphore_limit)

    # Build GPD tools for FittingAgent
    gpd_tools: list[Any] = []
    if gpd is not None:
        gpd_tools = [t for t in [
            gpd.make_tool("protocols", "route_protocol",
                "Find the canonical computation protocol for this type of physics fitting. "
                "Input: computation_type (str describing what you're fitting). "
                "Returns matching protocols. Call this before writing any fitting code."),
            gpd.make_tool("protocols", "get_protocol",
                "Retrieve the full step-by-step methodology for a named physics protocol, "
                "including checkpoints you must satisfy. Input: name (str from route_protocol). "
                "Follow these steps in your fitting code."),
            gpd.make_tool("conventions", "subfield_defaults",
                "Get canonical physics convention defaults for a subfield "
                "(e.g. condensed_matter, qft, gr, amo). Use to ensure your fitting code "
                "uses correct sign conventions, Fourier transforms, and natural units."),
            gpd.make_tool("conventions", "convention_check",
                "Check a code expression or equation for physics convention violations. "
                "Input: expression (str), domain (str). Returns PASS/FAIL with details. "
                "Call on your fitting code before finalizing it."),
            gpd.make_tool("patterns", "add_pattern",
                "Record a fitting failure pattern (convergence issue, sign error, etc.) "
                "that future agents should avoid. Call with domain, category, and description "
                "when your fitting code fails to converge or produces unphysical parameters."),
        ] if t is not None]

    async def fit_with_semaphore(agent: FittingAgent, hypothesis: str) -> dict[str, object]:
        async with semaphore:
            return await agent.fit(hypothesis)

    await interface.show(
        f"**Fitting phase**\n\nFitting {len(hypotheses)} hypotheses "
        f"with {config.n_fitting} agents each...",
        title="MTF: Fitting",
    )

    # Pre-fetch pitfall warnings before the fan-out (Addition 4)
    if gpd is not None:
        await _prefetch_fitting_warnings(hypotheses, config, memory, gpd)

    # Check for missing toolkit items
    probe_agent = FittingAgent(
        agent_id="probe",
        model=config.fitting_model,
        memory=memory,
        toolkit=toolkit,
        gpd_tools=gpd_tools,
        gpd=gpd,
        config=config,
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
                    gpd_tools=gpd_tools,
                    gpd=gpd,
                    config=config,
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
                gpd_tools=gpd_tools,
                gpd=gpd,
                config=config,
            )
            for i in range(config.n_fitting)
        ]
        results = await asyncio.gather(
            *(fit_with_semaphore(a, hyp) for hyp in hypotheses for a in agents)
        )
        all_fit_reports = [str(r) for r in results]

    # Run phase-level physics checks before synthesis so PHYSICS_VERDICT is populated (Addition 2)
    await _run_phase_physics_checks(
        all_fit_reports, hypotheses, memory, gpd, n_fitting=config.n_fitting
    )

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
