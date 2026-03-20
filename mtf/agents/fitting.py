"""FittingAgent: proposes and runs fits for a given hypothesis."""

from __future__ import annotations

from mtf.agents.base import BaseAgent
from mtf.memory import MemoryKind, SharedMemory
from mtf.toolkit.registry import ToolkitRegistry
from mtf.tools.fitting_tools import run_fitting_code
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
6. In the result dict, include protocol checkpoint verification under the key
   'protocol_checkpoints' (a dict mapping checkpoint name to pass/fail/note).
7. Report fit quality (chi-squared, reduced chi-squared, residuals), best-fit
   parameters with uncertainties, and an assessment of whether the hypothesis is
   supported.

Always write clean, well-commented fitting code."""


class FittingAgent(BaseAgent):
    def __init__(
        self,
        agent_id: str,
        model: str,
        memory: SharedMemory,
        toolkit: ToolkitRegistry,
        gpd: GPDMCPClient | None = None,
    ) -> None:
        gpd_tools = []
        if gpd is not None and gpd.available:
            gpd_tools = [
                t
                for t in [
                    gpd.make_tool(
                        "protocols",
                        "route_protocol",
                        "Find the canonical computation protocol for this type of physics calculation. Input: computation_type (str describing what you are computing, e.g. 'fit Drude model to optical conductivity'). Returns a ranked list of matching protocol names with relevance scores.",
                    ),
                    gpd.make_tool(
                        "protocols",
                        "get_protocol",
                        "Retrieve the full step-by-step methodology for a named physics protocol, including mandatory checkpoints. Input: name (str, protocol name returned by route_protocol). Returns steps, checkpoints, and domain. Use this as a blueprint for your fitting code.",
                    ),
                    gpd.make_tool(
                        "conventions",
                        "subfield_defaults",
                        "Get the canonical physics convention defaults for a given subfield (e.g. condensed_matter, qft, gr, plasma). Returns sign conventions, Fourier transform conventions, natural units, etc. Use these to ensure your fitting code uses correct conventions.",
                    ),
                ]
                if t is not None
            ]
        super().__init__(
            agent_id=agent_id,
            model=model,
            tools=[*gpd_tools],
            memory=memory,
            system_prompt=_SYSTEM_PROMPT,
        )
        self._toolkit = toolkit

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
            "'protocol_checkpoints'."
        )
        code = await self._query(
            task,
            extra_kinds=(
                MemoryKind.LITERATURE,
                MemoryKind.DEBATE,
                MemoryKind.USER_FEEDBACK,
                MemoryKind.IMAGE_DATA,
                MemoryKind.CONVENTIONS,
            ),
        )
        # Strip markdown code fences if present
        if "```" in code:
            lines = code.splitlines()
            code_lines = []
            in_block = False
            for line in lines:
                if line.strip().startswith("```"):
                    in_block = not in_block
                    continue
                if in_block:
                    code_lines.append(line)
            code = "\n".join(code_lines)

        fit_output = run_fitting_code(code, self._toolkit.all_data())
        report = f"Hypothesis: {hypothesis}\n\nFit output: {fit_output}"
        self._memory.add(
            MemoryKind.FIT_RESULT,
            report,
            agent_id=self._agent_id,
            hypothesis=hypothesis,
        )
        return fit_output
