"""DebateEngine: synthesizes multiple agent reports into a single coherent summary."""

from __future__ import annotations

import re
from typing import TYPE_CHECKING, Any

import anthropic

from mtf.config import MTFConfig
from mtf.memory import MemoryKind, SharedMemory

if TYPE_CHECKING:
    from mtf.tools.gpd_mcp import GPDMCPClient


class DebateEngine:
    """Single-shot Anthropic API call that synthesizes N agent reports."""

    def __init__(
        self,
        config: MTFConfig,
        memory: SharedMemory,
        gpd: GPDMCPClient | None = None,
    ) -> None:
        self._config = config
        self._memory = memory
        self._client = anthropic.Anthropic()
        self._gpd = gpd

    async def synthesize(
        self,
        reports: list[str],
        phase: str,
        extra_context: str = "",
    ) -> str:
        """Synthesize reports from multiple agents into one summary.

        Not an agentic call — a plain messages.create() for speed.
        For fitting and review phases, appends an objective dimensional check
        postscript using GPD if available (Addition 8).
        """
        import asyncio

        numbered = "\n\n".join(
            f"--- Report {i + 1} ---\n{r}" for i, r in enumerate(reports)
        )
        memory_ctx = self._memory.format_context()

        base_system = (
            f"You are a synthesis engine for the {phase} phase of a multi-agent "
            "physics research assistant. Your job is to produce a single coherent "
            "synthesis of the provided agent reports, resolving contradictions and "
            "highlighting the strongest hypotheses or conclusions. Be concise and precise."
        )

        if phase in ("fitting", "review"):
            physics_criterion = (
                "\n\nIMPORTANT — ranking criterion for hypotheses: physical correctness "
                "takes priority over fit quality. Apply this order:\n"
                "1. Physics checks: hypotheses that pass dimensional analysis (5.1), "
                "limiting cases (5.3), symmetry (5.2), and fit-family validity (5.18) "
                "rank above those that fail, regardless of chi².\n"
                "2. Parsimony: prefer fewer free parameters at comparable fit quality.\n"
                "3. First-principles basis: a derivation from known theory outranks a "
                "phenomenological ansatz at equal fit quality.\n"
                "4. Fit quality (chi², reduced chi²): used as tiebreaker only.\n"
                "Cite specific check IDs (e.g. 'check 5.3 FAIL') from the agent reports "
                "when justifying your ranking. A model with chi²=1.5 and all checks PASS "
                "should rank above chi²=0.9 with check 5.1 FAIL."
            )
            system = base_system + physics_criterion
        else:
            system = base_system

        user_parts = []
        if memory_ctx:
            user_parts.append(memory_ctx)
        if extra_context:
            user_parts.append(extra_context)

        # Inject conventions and physics verdicts if present
        conv_entries = self._memory.filter(MemoryKind.CONVENTIONS)
        verdict_entries = self._memory.filter(MemoryKind.PHYSICS_VERDICT)
        if conv_entries:
            user_parts.append("Physics conventions in use:\n" + "\n".join(e.content for e in conv_entries))
        if verdict_entries:
            user_parts.append("Physics verification verdicts:\n" + "\n".join(e.content for e in verdict_entries))

        user_parts.append(f"Agent reports to synthesize:\n\n{numbered}")
        user_content = "\n\n".join(user_parts)

        response = await asyncio.to_thread(
            self._client.messages.create,
            model=self._config.debate_model,
            max_tokens=4096,
            system=system,
            messages=[{"role": "user", "content": user_content}],
        )
        summary: str = response.content[0].text  # type: ignore[index]

        # Addition 8: dimensional check postscript — objective, no second LLM call.
        if self._gpd is not None and phase in ("fitting", "review"):
            summary = await self._append_dimensional_check(summary, phase)

        self._memory.add(MemoryKind.DEBATE, summary, phase=phase)
        return summary

    async def _append_dimensional_check(self, summary: str, phase: str) -> str:
        """Extract equations from summary and run dimensional_check as a postscript."""
        import asyncio

        # Extract LaTeX inline equations and dimensional expressions (max 5 candidates)
        latex_exprs = re.findall(r'\$([^$]{3,100})\$', summary)
        dim_exprs = re.findall(r'\[[A-Z]\][^\n.]{0,60}', summary)
        candidates = (latex_exprs + dim_exprs)[:5]
        if not candidates:
            return summary

        check_result = await asyncio.to_thread(
            self._gpd.call, "verification", "dimensional_check",
            expressions=candidates,
        )
        if check_result:
            self._memory.add(
                MemoryKind.PHYSICS_VERDICT,
                check_result,
                source="debate_dimensional_check",
                phase=phase,
            )
            summary = summary + "\n\n--- OBJECTIVE DIMENSIONAL CHECK ---\n" + check_result

        return summary
