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

    # Subclasses may override to grant broader tool permissions.
    # Use "bypassPermissions" only for agents whose tools are read-only.
    _permission_mode: str = "default"

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
        mcp_servers: dict = {}
        if self._tools:
            # Use agent_id as server name to avoid collisions in concurrent fan-outs
            server_name = f"mtf-{self._agent_id}"
            mcp_servers[server_name] = sdk.create_sdk_mcp_server(server_name, tools=self._tools)
        options = sdk.ClaudeAgentOptions(
            model=self._model,
            system_prompt=self._system_prompt,
            mcp_servers=mcp_servers,
            permission_mode=self._permission_mode,
        )
        async for chunk in sdk.query(prompt=prompt, options=options):
            if hasattr(chunk, "text"):
                collected.append(chunk.text)
        return "".join(collected)
