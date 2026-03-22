"""FittingAgent: proposes and runs fits for a given hypothesis."""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING, Any

from mtf.agents.base import BaseAgent
from mtf.memory import MemoryKind, SharedMemory
from mtf.toolkit.registry import ToolkitRegistry
from mtf.tools.fitting_tools import run_fitting_code

if TYPE_CHECKING:
    from mtf.config import MTFConfig
    from mtf.tools.gpd_mcp import GPDMCPClient

_SYSTEM_PROMPT = """You are an expert data analysis and model fitting agent for
experimental physics. Given a hypothesis and experimental data, you:

1. At the start, call `route_protocol` with a description of the hypothesis and what
   is being fit to find the canonical computation protocol.
2. Call `get_protocol` with the protocol name returned by route_protocol to retrieve
   the full step-by-step methodology and mandatory checkpoints — use this as a
   blueprint for your fitting code.
3. Call `subfield_defaults` with the relevant subfield (e.g. condensed_matter, qft,
   gr, plasma) to retrieve canonical sign conventions, Fourier transform conventions,
   natural units, etc. Ensure your fitting code uses these correct conventions.
4. Identify what data and model functions are needed from the toolkit.
5. Write Python code using lmfit/numpy/scipy to fit the data, following the protocol
   retrieved above and incorporating the correct conventions.
6. In the result dict, always include:
   - 'protocol_followed': name of the protocol retrieved via get_protocol
   - 'physical_parameter_ranges': dict mapping each parameter name to whether its
     best-fit value falls within expected physical bounds (True/False + note)
   - 'protocol_checkpoints_satisfied': list of protocol checkpoint names that passed
7. Report fit quality (chi-squared, reduced chi-squared, residuals), best-fit
   parameters with uncertainties, and an assessment of whether the hypothesis is
   supported. Chi-squared is a necessary metric but NOT the sole quality criterion;
   physical correctness (parameter bounds, limiting cases, symmetry) matters more.
8. If your fitting code fails to converge or produces unphysical parameters, call
   `add_pattern(category='convergence-issue', ...)` to record the pattern for future
   runs on this domain.

Always write clean, well-commented fitting code."""


def _strip_fences(code: str) -> str:
    """Strip markdown code fences from a code string."""
    if "```" not in code:
        return code
    lines = code.splitlines()
    code_lines = []
    in_block = False
    for line in lines:
        if line.strip().startswith("```"):
            in_block = not in_block
            continue
        if in_block:
            code_lines.append(line)
    return "\n".join(code_lines)


class FittingAgent(BaseAgent):
    def __init__(
        self,
        agent_id: str,
        model: str,
        memory: SharedMemory,
        toolkit: ToolkitRegistry,
        gpd_tools: list[Any] | None = None,
        gpd: GPDMCPClient | None = None,
        config: MTFConfig | None = None,
    ) -> None:
        extra_tools: list[Any] = gpd_tools if gpd_tools is not None else []
        super().__init__(
            agent_id=agent_id,
            model=model,
            tools=[*extra_tools],
            memory=memory,
            system_prompt=_SYSTEM_PROMPT,
        )
        self._toolkit = toolkit
        self._gpd = gpd
        self._config = config

    async def identify_needed_toolkit_items(self, hypothesis: str) -> list[str]:
        """Ask the agent which toolkit items it needs to fit this hypothesis."""
        task = (
            f"Hypothesis to fit:\n{hypothesis}\n\n"
            f"Available toolkit:\n{self._toolkit.describe()}\n\n"
            "List the names of data items and model functions you need (one per line). "
            "If something is missing, prefix it with 'MISSING: '."
        )
        response = await self._query(
            task,
            extra_kinds=(
                MemoryKind.LITERATURE,
                MemoryKind.DEBATE,
                MemoryKind.IMAGE_DATA,
                MemoryKind.CONVENTIONS,
                MemoryKind.FITTING_WARNINGS,
                MemoryKind.DOMAIN_PATTERNS,
            ),
        )
        lines = [l.strip() for l in response.splitlines() if l.strip()]
        return lines

    async def fit(self, hypothesis: str) -> dict[str, object]:
        """Generate and execute fitting code for this hypothesis."""
        task = (
            f"Hypothesis:\n{hypothesis}\n\n"
            f"Available toolkit:\n{self._toolkit.describe()}\n\n"
            "Write Python fitting code using lmfit. "
            "Assign your final result dict to a variable called 'result'. "
            "The result dict must include: 'parameters', 'uncertainties', "
            "'chi_squared', 'reduced_chi_squared', 'assessment', "
            "'protocol_followed' (name of the protocol retrieved via get_protocol), "
            "'physical_parameter_ranges' (dict mapping each parameter name to whether "
            "its best-fit value falls within expected physical bounds), "
            "'protocol_checkpoints_satisfied' (list of checkpoint names that passed)."
        )
        code = await self._query(
            task,
            extra_kinds=(
                MemoryKind.LITERATURE,
                MemoryKind.DEBATE,
                MemoryKind.USER_FEEDBACK,
                MemoryKind.IMAGE_DATA,
                MemoryKind.CONVENTIONS,
                MemoryKind.FITTING_WARNINGS,
                MemoryKind.DOMAIN_PATTERNS,
            ),
        )
        code = _strip_fences(code)

        # Pre-exec convention check (Addition 3): check generated code for convention
        # violations before running it; retry once if violations are found.
        if (
            self._gpd is not None
            and self._config is not None
            and self._config.fitting_convention_check
        ):
            domain = (
                self._config.physics_domains[0]
                if self._config.physics_domains
                else "condensed_matter"
            )
            for attempt in range(self._config.fitting_max_convention_retries + 1):
                # Use asyncio.to_thread so the blocking MCP call doesn't stall the event loop
                # when multiple fitting agents run concurrently under the semaphore.
                check_result = await asyncio.to_thread(
                    self._gpd.call, "conventions", "convention_check",
                    expression=code[:2000], domain=domain,
                )
                if check_result and "FAIL" in check_result.upper():
                    self._memory.add(
                        MemoryKind.PHYSICS_VERDICT,
                        check_result,
                        source="convention_check_pre_exec",
                        hypothesis=hypothesis[:200],
                    )
                    if attempt < self._config.fitting_max_convention_retries:
                        retry_task = (
                            f"{task}\n\nConvention check FAIL (attempt {attempt + 1}):\n"
                            f"{check_result}\n\nPlease fix all convention violations."
                        )
                        code = await self._query(
                            retry_task,
                            extra_kinds=(
                                MemoryKind.LITERATURE,
                                MemoryKind.DEBATE,
                                MemoryKind.USER_FEEDBACK,
                                MemoryKind.IMAGE_DATA,
                                MemoryKind.CONVENTIONS,
                                MemoryKind.FITTING_WARNINGS,
                                MemoryKind.DOMAIN_PATTERNS,
                            ),
                        )
                        code = _strip_fences(code)
                else:
                    break  # No violation — proceed to exec

        fit_output = run_fitting_code(code, self._toolkit.all_data())
        report = f"Hypothesis: {hypothesis}\n\nFit output: {fit_output}"
        self._memory.add(
            MemoryKind.FIT_RESULT,
            report,
            agent_id=self._agent_id,
            hypothesis=hypothesis,
        )
        return fit_output
