"""Tests for GPD MCP integration.

All tests work WITHOUT the get-physics-done package installed.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from mtf.config import MTFConfig
from mtf.debate import DebateEngine
from mtf.memory import MemoryKind, SharedMemory
from mtf.toolkit.registry import ToolkitRegistry


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class MockInterface:
    """Minimal mock of HumanInterface for tests."""

    def __init__(self) -> None:
        self.shown: list[str] = []

    async def show(self, content: str, title: str = "") -> None:
        self.shown.append(content)

    async def ask(self, prompt: str) -> str:
        return ""

    async def confirm(self, prompt: str) -> bool:
        return True


class MockGPD:
    """Lightweight mock for GPDMCPClient used in integration tests."""

    def __init__(self, responses: dict | None = None) -> None:
        self._responses = responses or {}
        self.calls: list[tuple] = []

    def call(self, server: str, tool_name: str, **kwargs) -> str:
        self.calls.append((server, tool_name, kwargs))
        key = f"{server}/{tool_name}"
        return self._responses.get(key, "")

    async def async_call(self, server: str, tool_name: str, **kwargs) -> str:
        return self.call(server, tool_name, **kwargs)

    def make_tool(self, server: str, tool_name: str, description: str):
        return None  # no real SDK tools in tests

    @property
    def available(self) -> bool:
        return True


# ---------------------------------------------------------------------------
# GPDMCPClient unit tests
# ---------------------------------------------------------------------------


def test_gpd_client_not_started_returns_empty():
    """call() on a client that was never started returns empty string."""
    from mtf.tools.gpd_mcp import GPDMCPClient

    client = GPDMCPClient()
    result = client.call("verification", "get_checklist")
    assert result == ""
    client._loop.call_soon_threadsafe(client._loop.stop)
    client._thread.join(timeout=5)


def test_gpd_client_make_tool_unavailable_returns_none():
    """make_tool() returns None when server is not in sessions."""
    from mtf.tools.gpd_mcp import GPDMCPClient

    client = GPDMCPClient()
    tool = client.make_tool("verification", "get_checklist", "Get checklist")
    assert tool is None
    client._loop.call_soon_threadsafe(client._loop.stop)
    client._thread.join(timeout=5)


def test_gpd_client_available_false_before_start():
    """client.available is False before start() is called."""
    from mtf.tools.gpd_mcp import GPDMCPClient

    client = GPDMCPClient()
    assert client.available is False
    client._loop.call_soon_threadsafe(client._loop.stop)
    client._thread.join(timeout=5)


# ---------------------------------------------------------------------------
# Memory tests
# ---------------------------------------------------------------------------


def test_memory_kind_conventions():
    """CONVENTIONS MemoryKind can be added and filtered."""
    m = SharedMemory()
    m.add(MemoryKind.CONVENTIONS, "metric signature (-,+,+,+)")
    m.add(MemoryKind.LITERATURE, "some paper")
    entries = m.filter(MemoryKind.CONVENTIONS)
    assert len(entries) == 1
    assert entries[0].content == "metric signature (-,+,+,+)"
    assert entries[0].kind == MemoryKind.CONVENTIONS


def test_memory_kind_physics_verdict():
    """PHYSICS_VERDICT MemoryKind can be added and filtered."""
    m = SharedMemory()
    m.add(MemoryKind.PHYSICS_VERDICT, "check 5.1 PASS: dimensions OK")
    m.add(MemoryKind.FIT_RESULT, "chi2=1.2")
    entries = m.filter(MemoryKind.PHYSICS_VERDICT)
    assert len(entries) == 1
    assert "5.1 PASS" in entries[0].content


def test_memory_kind_fitting_warnings():
    """FITTING_WARNINGS MemoryKind can be added and filtered."""
    m = SharedMemory()
    m.add(MemoryKind.FITTING_WARNINGS, "sign-error risk in this domain")
    entries = m.filter(MemoryKind.FITTING_WARNINGS)
    assert len(entries) == 1
    assert "sign-error" in entries[0].content


def test_memory_kind_domain_patterns():
    """DOMAIN_PATTERNS MemoryKind can be added and filtered."""
    m = SharedMemory()
    m.add(MemoryKind.DOMAIN_PATTERNS, "common Fourier convention mistake in condensed matter")
    entries = m.filter(MemoryKind.DOMAIN_PATTERNS)
    assert len(entries) == 1


def test_memory_kind_domain_classification():
    """DOMAIN_CLASSIFICATION MemoryKind can be added and filtered."""
    m = SharedMemory()
    m.add(MemoryKind.DOMAIN_CLASSIFICATION, "Detected: condensed_matter, qft")
    entries = m.filter(MemoryKind.DOMAIN_CLASSIFICATION)
    assert len(entries) == 1


# ---------------------------------------------------------------------------
# Agent backward compatibility (gpd_tools=None)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_literature_agent_no_gpd_tools():
    """LiteratureAgent works with gpd_tools=None (backward compat)."""
    from mtf.agents.literature import LiteratureAgent

    memory = SharedMemory()
    agent = LiteratureAgent(
        agent_id="lit-test",
        model="claude-opus-4-6",
        memory=memory,
        gpd_tools=None,
    )
    assert agent._agent_id == "lit-test"
    assert len(agent._tools) == 2


@pytest.mark.asyncio
async def test_fitting_agent_no_gpd_tools():
    """FittingAgent works with gpd_tools=None (backward compat)."""
    from mtf.agents.fitting import FittingAgent

    memory = SharedMemory()
    toolkit = ToolkitRegistry()
    agent = FittingAgent(
        agent_id="fit-test",
        model="claude-opus-4-6",
        memory=memory,
        toolkit=toolkit,
        gpd_tools=None,
    )
    assert agent._agent_id == "fit-test"
    assert len(agent._tools) == 0


@pytest.mark.asyncio
async def test_reviewer_agent_no_gpd_tools():
    """ReviewerAgent works with gpd_tools=None (backward compat)."""
    from mtf.agents.reviewer import ReviewerAgent

    memory = SharedMemory()
    agent = ReviewerAgent(
        agent_id="rev-test",
        model="claude-opus-4-6",
        memory=memory,
        gpd_tools=None,
    )
    assert agent._agent_id == "rev-test"
    assert len(agent._tools) == 0


# ---------------------------------------------------------------------------
# Debate engine — phase-conditional system prompts
# ---------------------------------------------------------------------------


@pytest.fixture
def memory():
    return SharedMemory()


@pytest.fixture
def config():
    return MTFConfig()


@pytest.mark.asyncio
async def test_debate_system_prompt_literature_phase(memory, config):
    """Literature phase does NOT get physics-first ranking criterion."""
    engine = DebateEngine(config, memory)
    captured: dict = {}

    def capture_call(**kwargs):
        captured.update(kwargs)
        resp = MagicMock()
        resp.content = [MagicMock(text="synthesis")]
        return resp

    with patch.object(engine._client.messages, "create", side_effect=capture_call):
        await engine.synthesize(["report1"], phase="literature")

    system = captured["system"]
    assert "ranking criterion" not in system
    assert "literature" in system


@pytest.mark.asyncio
async def test_debate_system_prompt_fitting_phase(memory, config):
    """Fitting phase DOES get physics-first ranking criterion."""
    engine = DebateEngine(config, memory)
    captured: dict = {}

    def capture_call(**kwargs):
        captured.update(kwargs)
        resp = MagicMock()
        resp.content = [MagicMock(text="synthesis")]
        return resp

    with patch.object(engine._client.messages, "create", side_effect=capture_call):
        await engine.synthesize(["report1"], phase="fitting")

    system = captured["system"]
    assert "ranking criterion" in system
    assert "physical correctness" in system


@pytest.mark.asyncio
async def test_debate_system_prompt_review_phase(memory, config):
    """Review phase DOES get physics-first ranking criterion."""
    engine = DebateEngine(config, memory)
    captured: dict = {}

    def capture_call(**kwargs):
        captured.update(kwargs)
        resp = MagicMock()
        resp.content = [MagicMock(text="synthesis")]
        return resp

    with patch.object(engine._client.messages, "create", side_effect=capture_call):
        await engine.synthesize(["report1"], phase="review")

    system = captured["system"]
    assert "ranking criterion" in system


@pytest.mark.asyncio
async def test_debate_includes_conventions_in_context(memory, config):
    """Conventions entries appear in synthesis context."""
    memory.add(MemoryKind.CONVENTIONS, "Fourier: exp(-iwt)")
    engine = DebateEngine(config, memory)
    captured: dict = {}

    def capture_call(**kwargs):
        captured.update(kwargs)
        resp = MagicMock()
        resp.content = [MagicMock(text="synthesis")]
        return resp

    with patch.object(engine._client.messages, "create", side_effect=capture_call):
        await engine.synthesize(["report1"], phase="fitting")

    user_content = captured["messages"][0]["content"]
    assert "Fourier: exp(-iwt)" in user_content
    assert "Physics conventions in use" in user_content


@pytest.mark.asyncio
async def test_debate_includes_physics_verdicts_in_context(memory, config):
    """Physics verdict entries appear in synthesis context."""
    memory.add(MemoryKind.PHYSICS_VERDICT, "check 5.1 PASS: dimensions consistent")
    engine = DebateEngine(config, memory)
    captured: dict = {}

    def capture_call(**kwargs):
        captured.update(kwargs)
        resp = MagicMock()
        resp.content = [MagicMock(text="synthesis")]
        return resp

    with patch.object(engine._client.messages, "create", side_effect=capture_call):
        await engine.synthesize(["report1"], phase="review")

    user_content = captured["messages"][0]["content"]
    assert "check 5.1 PASS" in user_content
    assert "Physics verification verdicts" in user_content


# ---------------------------------------------------------------------------
# Config defaults
# ---------------------------------------------------------------------------


def test_config_defaults():
    """New config fields have correct defaults."""
    cfg = MTFConfig()
    assert cfg.enable_gpd_mcp is True
    assert cfg.physics_domains == ["condensed_matter"]
    assert cfg.gpd_servers == [
        "verification", "errors", "protocols", "conventions", "patterns", "skills"
    ]
    assert cfg.auto_detect_domains is True
    assert cfg.gpd_domain_detection_max_domains == 4
    assert cfg.literature_plausibility_screen is True
    assert cfg.auto_reject_physics_failures is False
    assert cfg.fitting_convention_check is True
    assert cfg.fitting_max_convention_retries == 1


# ---------------------------------------------------------------------------
# Addition 1: auto domain classification
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_classify_domains_uses_route_protocol():
    """_classify_domains overwrites config.physics_domains when GPD returns a known domain."""
    from mtf.orchestrator import MTFOrchestrator

    config = MTFConfig(auto_detect_domains=True, physics_domains=["condensed_matter"])
    orchestrator = MTFOrchestrator(config=config, interface=MockInterface())

    gpd = MockGPD(responses={
        "protocols/route_protocol": "Best match: qft protocol for field theory calculations",
        "skills/route_skill": "",
    })

    await orchestrator._classify_domains("anomalous magnetic moment measurement", gpd)

    assert "qft" in orchestrator._config.physics_domains


@pytest.mark.asyncio
async def test_classify_domains_falls_back_on_empty_response():
    """_classify_domains keeps existing config.physics_domains when GPD returns nothing."""
    from mtf.orchestrator import MTFOrchestrator

    config = MTFConfig(auto_detect_domains=True, physics_domains=["condensed_matter"])
    orchestrator = MTFOrchestrator(config=config, interface=MockInterface())

    gpd = MockGPD(responses={})  # all calls return ""

    await orchestrator._classify_domains("some phenomenon", gpd)

    assert orchestrator._config.physics_domains == ["condensed_matter"]


@pytest.mark.asyncio
async def test_classify_domains_stores_audit_trail():
    """_classify_domains always writes a DOMAIN_CLASSIFICATION memory entry."""
    from mtf.orchestrator import MTFOrchestrator

    config = MTFConfig(auto_detect_domains=True)
    orchestrator = MTFOrchestrator(config=config, interface=MockInterface())

    gpd = MockGPD(responses={"protocols/route_protocol": "condensed_matter transport"})

    await orchestrator._classify_domains("Hall effect", gpd)

    entries = orchestrator._memory.filter(MemoryKind.DOMAIN_CLASSIFICATION)
    assert len(entries) == 1


# ---------------------------------------------------------------------------
# Addition 2: PHYSICS_VERDICT population in fitting phase
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_fitting_phase_populates_physics_verdict():
    """_run_phase_physics_checks writes PHYSICS_VERDICT entries for non-empty check results."""
    from mtf.phases.fitting_phase import _run_phase_physics_checks

    memory = SharedMemory()
    gpd = MockGPD(responses={
        "verification/run_check": "check 5.1 PASS: dimensions consistent",
    })

    await _run_phase_physics_checks(
        fit_reports=["fit report for BCS theory"],
        hypotheses=["BCS superconductivity"],
        memory=memory,
        gpd=gpd,
    )

    verdicts = memory.filter(MemoryKind.PHYSICS_VERDICT)
    assert len(verdicts) > 0
    assert any("5.1 PASS" in v.content for v in verdicts)


@pytest.mark.asyncio
async def test_fitting_phase_no_gpd_skips_checks():
    """_run_phase_physics_checks with gpd=None writes no PHYSICS_VERDICT entries."""
    from mtf.phases.fitting_phase import _run_phase_physics_checks

    memory = SharedMemory()
    await _run_phase_physics_checks(
        fit_reports=["some fit report"],
        hypotheses=["some hypothesis"],
        memory=memory,
        gpd=None,
    )
    assert len(memory.filter(MemoryKind.PHYSICS_VERDICT)) == 0


@pytest.mark.asyncio
async def test_fitting_phase_checks_empty_hypotheses_safe():
    """_run_phase_physics_checks with empty hypotheses list does not raise."""
    from mtf.phases.fitting_phase import _run_phase_physics_checks

    memory = SharedMemory()
    gpd = MockGPD(responses={"verification/run_check": "PASS"})
    # Should not raise ZeroDivisionError
    await _run_phase_physics_checks(
        fit_reports=["report 1", "report 2"],
        hypotheses=[],
        memory=memory,
        gpd=gpd,
    )
    assert len(memory.filter(MemoryKind.PHYSICS_VERDICT)) == 0


# ---------------------------------------------------------------------------
# Addition 4: fitting warnings pre-fetch
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_fitting_warnings_written_to_memory():
    """_prefetch_fitting_warnings writes FITTING_WARNINGS entries."""
    from mtf.phases.fitting_phase import _prefetch_fitting_warnings

    memory = SharedMemory()
    config = MTFConfig(physics_domains=["condensed_matter"])
    gpd = MockGPD(responses={
        "patterns/lookup_pattern": "Known sign-error: missing factor in Green's function",
        "errors/check_error_classes": "Top error class: gauge artifact",
    })

    await _prefetch_fitting_warnings(
        hypotheses=["BCS pairing hypothesis"],
        config=config,
        memory=memory,
        gpd=gpd,
    )

    warnings = memory.filter(MemoryKind.FITTING_WARNINGS)
    assert len(warnings) > 0


@pytest.mark.asyncio
async def test_fitting_warnings_in_agent_context():
    """FittingAgent._build_prompt includes FITTING_WARNINGS when present in memory."""
    from mtf.agents.fitting import FittingAgent

    memory = SharedMemory()
    memory.add(MemoryKind.FITTING_WARNINGS, "Beware of sign convention in BCS gap equation")

    toolkit = ToolkitRegistry()
    agent = FittingAgent(
        agent_id="fit-0",
        model="claude-opus-4-6",
        memory=memory,
        toolkit=toolkit,
    )

    # _build_prompt is inherited from BaseAgent; verify FITTING_WARNINGS kind is in extra_kinds
    # by checking that the kind appears in the agent's identify_needed_toolkit_items context.
    # We do this by inspecting the extra_kinds used in identify_needed_toolkit_items.
    import inspect
    src = inspect.getsource(agent.identify_needed_toolkit_items)
    assert "FITTING_WARNINGS" in src


# ---------------------------------------------------------------------------
# Addition 5: literature plausibility screen
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_plausibility_screen_flags_failing_hypothesis():
    """_screen_hypothesis_plausibility shows WARN/FAIL badge and writes PHYSICS_VERDICT."""
    from mtf.phases.literature_phase import _screen_hypothesis_plausibility

    memory = SharedMemory()
    config = MTFConfig(literature_plausibility_screen=True)
    interface = MockInterface()
    gpd = MockGPD(responses={
        "verification/limiting_case_check": "CRITICAL FAIL: does not reduce to classical limit",
    })

    rejected = await _screen_hypothesis_plausibility(
        hypothesis_candidates=["Hypothesis: dark energy as quantum foam"],
        config=config,
        memory=memory,
        interface=interface,
        gpd=gpd,
    )

    # Badge shown to user
    shown_text = " ".join(interface.shown)
    assert "[FAIL]" in shown_text

    # Written to PHYSICS_VERDICT
    verdicts = memory.filter(MemoryKind.PHYSICS_VERDICT)
    assert len(verdicts) == 1
    assert verdicts[0].metadata.get("source") == "limiting_case_screen"

    # Returns the FAIL hypothesis in the rejected set
    assert len(rejected) == 1


@pytest.mark.asyncio
async def test_plausibility_screen_disabled_by_config():
    """_screen_hypothesis_plausibility is a no-op when config.literature_plausibility_screen=False."""
    from mtf.phases.literature_phase import _screen_hypothesis_plausibility

    memory = SharedMemory()
    config = MTFConfig(literature_plausibility_screen=False)
    interface = MockInterface()
    gpd = MockGPD(responses={"verification/limiting_case_check": "FAIL"})

    rejected = await _screen_hypothesis_plausibility(
        hypothesis_candidates=["Hypothesis: X"],
        config=config,
        memory=memory,
        interface=interface,
        gpd=gpd,
    )

    assert len(memory.filter(MemoryKind.PHYSICS_VERDICT)) == 0
    assert interface.shown == []
    assert rejected == set()


@pytest.mark.asyncio
async def test_auto_reject_physics_failures_filters_hypotheses():
    """auto_reject_physics_failures=True removes CRITICAL-FAIL hypotheses from hyp_lines."""
    from mtf.phases.literature_phase import _screen_hypothesis_plausibility

    memory = SharedMemory()
    config = MTFConfig(literature_plausibility_screen=True, auto_reject_physics_failures=True)
    interface = MockInterface()
    gpd = MockGPD(responses={
        "verification/limiting_case_check": "CRITICAL FAIL: violates causality",
    })

    rejected = await _screen_hypothesis_plausibility(
        hypothesis_candidates=["Hypothesis: tachyonic condensate", "Hypothesis: BCS pairing"],
        config=config,
        memory=memory,
        interface=interface,
        gpd=gpd,
    )

    # Both candidates return CRITICAL FAIL, so both are rejected
    assert len(rejected) == 2


# ---------------------------------------------------------------------------
# Addition 8: debate-level dimensional check postscript
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_debate_engine_appends_dimensional_check_postscript(memory, config):
    """DebateEngine appends OBJECTIVE DIMENSIONAL CHECK block when GPD returns issues."""
    gpd = MockGPD(responses={
        "verification/dimensional_check": "Issue: $E = mc^3$ has wrong exponent",
    })
    engine = DebateEngine(config, memory, gpd=gpd)

    mock_response = MagicMock()
    mock_response.content = [MagicMock(text="Model energy is $E = mc^3$ per hypothesis.")]

    with patch.object(engine._client.messages, "create", return_value=mock_response):
        result = await engine.synthesize(["report1"], phase="fitting")

    assert "OBJECTIVE DIMENSIONAL CHECK" in result
    assert "Issue:" in result

    # Written to PHYSICS_VERDICT
    verdicts = memory.filter(MemoryKind.PHYSICS_VERDICT)
    assert any(v.metadata.get("source") == "debate_dimensional_check" for v in verdicts)


@pytest.mark.asyncio
async def test_debate_engine_no_postscript_without_gpd(memory, config):
    """DebateEngine with gpd=None does not append postscript (backward compat)."""
    engine = DebateEngine(config, memory, gpd=None)

    mock_response = MagicMock()
    mock_response.content = [MagicMock(text="Plain synthesis without equations.")]

    with patch.object(engine._client.messages, "create", return_value=mock_response):
        result = await engine.synthesize(["report1"], phase="fitting")

    assert "OBJECTIVE DIMENSIONAL CHECK" not in result


@pytest.mark.asyncio
async def test_debate_engine_no_postscript_for_literature_phase(memory, config):
    """Dimensional check postscript is NOT appended for literature phase."""
    gpd = MockGPD(responses={
        "verification/dimensional_check": "some issue",
    })
    engine = DebateEngine(config, memory, gpd=gpd)

    mock_response = MagicMock()
    mock_response.content = [MagicMock(text="Literature synthesis with $E = mc^2$.")]

    with patch.object(engine._client.messages, "create", return_value=mock_response):
        result = await engine.synthesize(["report1"], phase="literature")

    assert "OBJECTIVE DIMENSIONAL CHECK" not in result
