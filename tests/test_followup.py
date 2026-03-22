"""Tests for FollowUpChatAgent and _run_followup_chat orchestrator method."""

from __future__ import annotations

from unittest.mock import AsyncMock, patch

import pytest

from mtf.agents.followup import FollowUpChatAgent, _MAX_HISTORY
from mtf.config import MTFConfig
from mtf.interface import HumanInterface
from mtf.memory import MemoryKind, SharedMemory
from mtf.orchestrator import MTFOrchestrator


class MockInterface(HumanInterface):
    def __init__(self, confirm_value: bool = True, ask_values: list[str] | None = None):
        self._confirm = confirm_value
        self._ask_values = list(ask_values or [])
        self._ask_index = 0

    async def show(self, content: str, title: str = "") -> None:
        pass

    async def ask(self, prompt: str) -> str:
        if self._ask_index < len(self._ask_values):
            val = self._ask_values[self._ask_index]
            self._ask_index += 1
            return val
        return ""

    async def confirm(self, prompt: str) -> bool:
        return self._confirm


# ---------------------------------------------------------------------------
# FollowUpChatAgent construction
# ---------------------------------------------------------------------------

def test_followup_agent_construction():
    """FollowUpChatAgent should construct without error using config.followup_model."""
    config = MTFConfig()
    memory = SharedMemory()
    agent = FollowUpChatAgent(config, memory)
    assert agent._agent_id == "followup"
    assert agent._model == config.followup_model


def test_followup_agent_accepts_gpd_params():
    """Constructor should accept gpd_tools and gpd without error (forward-compat)."""
    config = MTFConfig()
    memory = SharedMemory()
    agent = FollowUpChatAgent(config, memory, gpd_tools=[], gpd=None)
    assert agent is not None


# ---------------------------------------------------------------------------
# History management
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_history_accumulates():
    """chat() should append each exchange to _history."""
    config = MTFConfig()
    memory = SharedMemory()
    agent = FollowUpChatAgent(config, memory)

    with patch.object(agent, "_query", new=AsyncMock(return_value="answer")):
        await agent.chat("question 1")
        await agent.chat("question 2")

    assert len(agent._history) == 2
    assert "question 1" in agent._history[0]
    assert "question 2" in agent._history[1]


@pytest.mark.asyncio
async def test_history_rolling_window():
    """History should not exceed _MAX_HISTORY entries."""
    config = MTFConfig()
    memory = SharedMemory()
    agent = FollowUpChatAgent(config, memory)

    with patch.object(agent, "_query", new=AsyncMock(return_value="ans")):
        for i in range(_MAX_HISTORY + 5):
            await agent.chat(f"q{i}")

    assert len(agent._history) == _MAX_HISTORY


@pytest.mark.asyncio
async def test_history_prepended_to_subsequent_query():
    """Second call should include the first exchange in the task string."""
    config = MTFConfig()
    memory = SharedMemory()
    agent = FollowUpChatAgent(config, memory)

    calls: list[str] = []

    async def capture_query(task, extra_kinds=()):
        calls.append(task)
        return "answer"

    with patch.object(agent, "_query", new=capture_query):
        await agent.chat("first question")
        await agent.chat("second question")

    # First call: no history prefix
    assert calls[0].startswith("User: first question")
    # Second call: contains the first exchange
    assert "first question" in calls[1]
    assert "second question" in calls[1]


# ---------------------------------------------------------------------------
# _run_followup_chat orchestrator method
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_followup_chat_skipped_when_declined():
    """Declining the confirm prompt should skip the loop entirely."""
    config = MTFConfig()
    interface = MockInterface(confirm_value=False)
    orc = MTFOrchestrator(config=config, interface=interface)

    # Should complete without calling sdk.query
    with patch("mtf.agents.followup.FollowUpChatAgent.chat") as mock_chat:
        await orc._run_followup_chat()
        mock_chat.assert_not_called()


@pytest.mark.asyncio
async def test_followup_chat_exits_on_empty_input():
    """Empty input should exit the loop after one iteration."""
    config = MTFConfig()
    interface = MockInterface(confirm_value=True, ask_values=[""])
    orc = MTFOrchestrator(config=config, interface=interface)

    with patch("mtf.agents.followup.FollowUpChatAgent.chat") as mock_chat:
        await orc._run_followup_chat()
        mock_chat.assert_not_called()


@pytest.mark.asyncio
async def test_followup_chat_exits_on_quit_keyword():
    """'quit' input should exit the loop."""
    config = MTFConfig()
    interface = MockInterface(confirm_value=True, ask_values=["quit"])
    orc = MTFOrchestrator(config=config, interface=interface)

    with patch("mtf.agents.followup.FollowUpChatAgent.chat") as mock_chat:
        await orc._run_followup_chat()
        mock_chat.assert_not_called()


@pytest.mark.asyncio
async def test_followup_chat_handles_api_error_gracefully():
    """An API error on one question should not abort the loop."""
    config = MTFConfig()
    # First ask raises, second ask exits
    interface = MockInterface(confirm_value=True, ask_values=["bad question", ""])
    orc = MTFOrchestrator(config=config, interface=interface)

    with patch(
        "mtf.orchestrator.FollowUpChatAgent.chat",
        new=AsyncMock(side_effect=RuntimeError("API failure")),
    ):
        # Should not raise; loop exits cleanly on the empty second input
        await orc._run_followup_chat()


@pytest.mark.asyncio
async def test_followup_chat_calls_agent_and_shows_answer():
    """A valid question should invoke chat() and display the answer."""
    config = MTFConfig()
    shown: list[str] = []

    class CapturingInterface(MockInterface):
        async def show(self, content: str, title: str = "") -> None:
            shown.append(content)

    interface = CapturingInterface(confirm_value=True, ask_values=["what is the best fit?", ""])
    orc = MTFOrchestrator(config=config, interface=interface)

    with patch(
        "mtf.orchestrator.FollowUpChatAgent.chat",
        new=AsyncMock(return_value="The best fit is model A."),
    ):
        await orc._run_followup_chat()

    assert any("The best fit is model A." in s for s in shown)
