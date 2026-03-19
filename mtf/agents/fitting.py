"""FittingAgent: proposes and runs fits for a given hypothesis."""

from __future__ import annotations

from mtf.agents.base import BaseAgent
from mtf.memory import MemoryKind, SharedMemory
from mtf.toolkit.registry import ToolkitRegistry
from mtf.tools.fitting_tools import run_fitting_code

_SYSTEM_PROMPT = """You are an expert data analysis and model fitting agent for
experimental physics. Given a hypothesis and experimental data, you:
1. Identify what data and model functions are needed from the toolkit
2. Write Python code using lmfit/numpy/scipy to fit the data
3. Report fit quality (chi-squared, reduced chi-squared, residuals), best-fit parameters
   with uncertainties, and an assessment of whether the hypothesis is supported.
Always write clean, well-commented fitting code."""


class FittingAgent(BaseAgent):
    def __init__(
        self,
        agent_id: str,
        model: str,
        memory: SharedMemory,
        toolkit: ToolkitRegistry,
    ) -> None:
        super().__init__(
            agent_id=agent_id,
            model=model,
            tools=[],
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
            extra_kinds=(MemoryKind.LITERATURE, MemoryKind.DEBATE),
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
            "'chi_squared', 'reduced_chi_squared', 'assessment'."
        )
        code = await self._query(
            task,
            extra_kinds=(MemoryKind.LITERATURE, MemoryKind.DEBATE, MemoryKind.USER_FEEDBACK),
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
