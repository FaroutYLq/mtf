"""Base agent wrapping claude-agent-sdk query()."""

from __future__ import annotations

from typing import Any

import claude_agent_sdk as sdk

from mtf.memory import MemoryKind, SharedMemory

_HONESTY_REMINDER = (
    "\n\nVERIFICATION STANDARD: Do NOT use phrases like 'this becomes', "
    "'for consistency', or 'as expected' to skip showing your reasoning. "
    "Do not claim to have verified something unless you explicitly performed "
    "the check in this response. If uncertain, say so explicitly."
)

_CONVENTION_REMINDER = (
    "\n\nCONVENTION LOCK: The physics conventions shown in the context above are "
    "LOCKED for this run. Do not revert to textbook defaults (different metric "
    "signature, Fourier convention, unit system, or gauge choice)."
)


class BaseAgent:
    """Wraps sdk.query() and injects shared memory context into every prompt."""

    def __init__(
        self,
        agent_id: str,
        model: str,
        tools: list[Any],
        memory: SharedMemory,
        system_prompt: str,
    ) -> None:
        self._agent_id = agent_id
        self._model = model
        self._tools = tools
        self._memory = memory
        self._system_prompt = system_prompt

    def _build_prompt(self, task: str, extra_kinds: tuple[MemoryKind, ...] = ()) -> str:
        ctx = self._memory.format_context(*extra_kinds)
        has_conventions = bool(self._memory.filter(MemoryKind.CONVENTIONS))
        suffix = _HONESTY_REMINDER
        if has_conventions:
            suffix += _CONVENTION_REMINDER
        if ctx:
            return f"{ctx}\n\nTask: {task}{suffix}"
        return task + suffix

    async def _query(self, task: str, extra_kinds: tuple[MemoryKind, ...] = ()) -> str:
        prompt = self._build_prompt(task, extra_kinds)
        collected: list[str] = []
        async for chunk in sdk.query(
            prompt=prompt,
            model=self._model,
            system_prompt=self._system_prompt,
            tools=self._tools,
        ):
            if hasattr(chunk, "text"):
                collected.append(chunk.text)
        return "".join(collected)
