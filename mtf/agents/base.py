"""Base agent wrapping claude-agent-sdk query()."""

from __future__ import annotations

from typing import Any

import claude_agent_sdk as sdk

from mtf.memory import MemoryKind, SharedMemory


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
        if ctx:
            return f"{ctx}\n\nTask: {task}"
        return task

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
