"""DebateEngine: synthesizes multiple agent reports into a single coherent summary."""

from __future__ import annotations

import anthropic

from mtf.config import MTFConfig
from mtf.memory import MemoryKind, SharedMemory


class DebateEngine:
    """Single-shot Anthropic API call that synthesizes N agent reports."""

    def __init__(self, config: MTFConfig, memory: SharedMemory) -> None:
        self._config = config
        self._memory = memory
        self._client = anthropic.Anthropic()

    async def synthesize(
        self,
        reports: list[str],
        phase: str,
        extra_context: str = "",
    ) -> str:
        """Synthesize reports from multiple agents into one summary.

        Not an agentic call — a plain messages.create() for speed.
        """
        import asyncio

        numbered = "\n\n".join(
            f"--- Report {i + 1} ---\n{r}" for i, r in enumerate(reports)
        )
        memory_ctx = self._memory.format_context()
        system = (
            f"You are a synthesis engine for the {phase} phase of a multi-agent "
            "physics research assistant. Your job is to produce a single coherent "
            "synthesis of the provided agent reports, resolving contradictions and "
            "highlighting the strongest hypotheses or conclusions. Be concise and precise."
        )
        user_parts = []
        if memory_ctx:
            user_parts.append(memory_ctx)
        if extra_context:
            user_parts.append(extra_context)
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
        self._memory.add(MemoryKind.DEBATE, summary, phase=phase)
        return summary
