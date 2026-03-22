"""Follow-up chat agent for post-report Q&A."""

from __future__ import annotations

from typing import Any

from mtf.agents.base import BaseAgent
from mtf.config import MTFConfig
from mtf.memory import MemoryKind, SharedMemory
from mtf.tools.gpd_mcp import GPDMCPClient

_SYSTEM_PROMPT = """\
You are a theoretical physics expert who has just completed a full multi-agent analysis \
of an experimental phenomenon. You have access to the complete analysis: literature review, \
candidate hypotheses, fitting results, reviewer verdicts, and proposed measurements.

Answer follow-up questions clearly and precisely. Reference specific results, numbers, or \
verdicts from the analysis where relevant. If the user asks about something outside the \
scope of the analysis, say so honestly rather than speculating beyond what was established.
"""

# All memory kinds that are meaningful for follow-up context.
_ALL_KINDS: tuple[MemoryKind, ...] = (
    MemoryKind.LITERATURE,
    MemoryKind.DEBATE,
    MemoryKind.HYPOTHESIS,
    MemoryKind.FIT_RESULT,
    MemoryKind.REVIEW,
    MemoryKind.PROPOSALS,
    MemoryKind.USER_FEEDBACK,
    MemoryKind.IMAGE_DATA,
    MemoryKind.CONVENTIONS,
    MemoryKind.PHYSICS_VERDICT,
    MemoryKind.FITTING_WARNINGS,
    MemoryKind.QUALITATIVE_EVAL,
    MemoryKind.DOMAIN_PATTERNS,
    MemoryKind.FITTING_SKIPPED,
    MemoryKind.TOOLKIT_DIGEST,
)

# Maximum number of past exchanges kept in the rolling history window.
# Prevents unbounded context growth across long sessions.
_MAX_HISTORY = 10


class FollowUpChatAgent(BaseAgent):
    """Single agent for post-report follow-up questions.

    Maintains a rolling conversation history (capped at _MAX_HISTORY exchanges)
    so that each new question is sent together with recent prior exchanges,
    giving the agent multi-turn memory even though sdk.query() is stateless
    per call.
    """

    def __init__(
        self,
        config: MTFConfig,
        memory: SharedMemory,
        gpd_tools: list[Any] | None = None,
        gpd: GPDMCPClient | None = None,
    ) -> None:
        super().__init__(
            agent_id="followup",
            model=config.followup_model,
            tools=[],
            memory=memory,
            system_prompt=_SYSTEM_PROMPT,
        )
        self._history: list[str] = []

    async def chat(self, question: str) -> str:
        """Send a follow-up question and return the agent's answer."""
        # Build a multi-turn dialogue block from the rolling history window.
        if self._history:
            history_block = "\n\n".join(self._history)
            task = f"{history_block}\n\nUser: {question}\nAssistant:"
        else:
            task = f"User: {question}\nAssistant:"

        answer = await self._query(task, extra_kinds=_ALL_KINDS)

        # Append this exchange and enforce the rolling window cap.
        self._history.append(f"User: {question}\nAssistant: {answer}")
        if len(self._history) > _MAX_HISTORY:
            self._history = self._history[-_MAX_HISTORY:]
        return answer
