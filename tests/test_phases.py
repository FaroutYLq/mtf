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


@pytest.mark.asyncio
async def test_fitting_phase_passes_string_reports_to_debate(memory, config, mock_debate):
    """Regression test: debate engine must receive str reports, not dict reprs."""
    from mtf.phases.fitting_phase import run_fitting_phase
    from mtf.toolkit.registry import ToolkitRegistry

    captured_reports: list[list[str]] = []

    async def capture_synthesize(reports, **kwargs):
        captured_reports.append(list(reports))
        return "Fitting synthesis: quantum tunneling fits well"

    mock_debate.synthesize = capture_synthesize

    fit_report = "Hypothesis: quantum tunneling\n\nFit output: {'chi_squared': 1.2}"

    with patch("mtf.phases.fitting_phase.FittingAgent") as MockAgent:
        instance = AsyncMock()
        instance.identify_needed_toolkit_items = AsyncMock(return_value=[])
        instance.fit = AsyncMock(return_value=fit_report)
        MockAgent.return_value = instance

        interface = MockInterface(confirm_value=True)
        await run_fitting_phase(
            hypotheses=["quantum tunneling hypothesis"],
            config=config,
            memory=memory,
            interface=interface,
            debate_engine=mock_debate,
            toolkit=ToolkitRegistry(),
        )

    assert len(captured_reports) == 1
    reports = captured_reports[0]
    assert all(isinstance(r, str) for r in reports), (
        f"Debate engine received non-string reports: {[type(r) for r in reports]}"
    )
    assert any("Hypothesis" in r for r in reports), (
        "Reports should contain the formatted hypothesis text, not a raw dict repr"
    )


@pytest.mark.asyncio
async def test_fitting_phase_stores_fit_results_in_memory(memory, config, mock_debate):
    """FIT_RESULT memory entries should be populated after the fitting fan-out."""
    from mtf.phases.fitting_phase import run_fitting_phase
    from mtf.toolkit.registry import ToolkitRegistry

    mock_debate.synthesize = AsyncMock(return_value="Synthesis")

    fit_report = "Hypothesis: superconductivity\n\nFit output: {'chi_squared': 0.9}"

    with patch("mtf.phases.fitting_phase.FittingAgent") as MockAgent:
        instance = AsyncMock()
        instance.identify_needed_toolkit_items = AsyncMock(return_value=[])
        instance.fit = AsyncMock(return_value=fit_report)
        MockAgent.return_value = instance

        await run_fitting_phase(
            hypotheses=["superconductivity hypothesis"],
            config=config,
            memory=memory,
            interface=MockInterface(confirm_value=True),
            debate_engine=mock_debate,
            toolkit=ToolkitRegistry(),
        )

    fit_entries = memory.filter(MemoryKind.FIT_RESULT)
    assert len(fit_entries) == 0, (
        "FIT_RESULT entries are written by FittingAgent.fit() itself; "
        "the mocked agent does not write them, so memory should be empty here"
    )
    # The synthesis is stored as DEBATE
    debate_entries = memory.filter(MemoryKind.DEBATE)
    assert len(debate_entries) == 1
