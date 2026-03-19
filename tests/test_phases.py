"""Tests for phase control flow with mocked agents and debate engine."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from mtf.config import MTFConfig
from mtf.debate import DebateEngine
from mtf.interface import HumanInterface
from mtf.memory import MemoryKind, SharedMemory


class MockInterface(HumanInterface):
    def __init__(self, confirm_value: bool = True, ask_value: str = ""):
        self._confirm = confirm_value
        self._ask = ask_value

    async def show(self, content: str, title: str = "") -> None:
        pass

    async def ask(self, prompt: str) -> str:
        return self._ask

    async def confirm(self, prompt: str) -> bool:
        return self._confirm


@pytest.fixture
def memory():
    return SharedMemory()


@pytest.fixture
def config():
    return MTFConfig(n_literature=2, n_fitting=1, n_reviewer=1)


@pytest.fixture
def mock_debate(memory, config):
    engine = DebateEngine(config, memory)
    engine.synthesize = AsyncMock(return_value="Synthesized hypothesis: quantum tunneling")
    return engine


@pytest.mark.asyncio
async def test_literature_phase_approves_on_first_round(memory, config, mock_debate):
    from mtf.phases.literature_phase import run_literature_phase

    with patch("mtf.phases.literature_phase.LiteratureAgent") as MockAgent:
        instance = AsyncMock()
        instance.investigate = AsyncMock(return_value="Literature report")
        MockAgent.return_value = instance

        interface = MockInterface(confirm_value=True)
        hypotheses = await run_literature_phase(
            phenomenon="anomalous Hall effect",
            config=config,
            memory=memory,
            interface=interface,
            debate_engine=mock_debate,
        )

    assert isinstance(hypotheses, list)
    assert len(hypotheses) > 0


@pytest.mark.asyncio
async def test_literature_phase_loops_on_rejection(memory, config, mock_debate):
    from mtf.phases.literature_phase import run_literature_phase

    call_count = 0

    class CountingInterface(HumanInterface):
        async def show(self, content: str, title: str = "") -> None:
            pass

        async def ask(self, prompt: str) -> str:
            return "focus on topological effects"

        async def confirm(self, prompt: str) -> bool:
            nonlocal call_count
            call_count += 1
            # Approve on second round
            return call_count >= 2

    with patch("mtf.phases.literature_phase.LiteratureAgent") as MockAgent:
        instance = AsyncMock()
        instance.investigate = AsyncMock(return_value="Literature report")
        MockAgent.return_value = instance

        hypotheses = await run_literature_phase(
            phenomenon="anomalous Hall effect",
            config=config,
            memory=memory,
            interface=CountingInterface(),
            debate_engine=mock_debate,
        )

    assert call_count == 2
    feedback_entries = memory.filter(MemoryKind.USER_FEEDBACK)
    assert len(feedback_entries) >= 1
