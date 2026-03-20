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

    async def show(self, content: str, title: str = "") -> None:
        pass

    async def ask(self, prompt: str) -> str:
        return ""

    async def confirm(self, prompt: str) -> bool:
        return True


# ---------------------------------------------------------------------------
# GPDMCPClient tests
# ---------------------------------------------------------------------------


def test_gpd_client_not_started_returns_empty():
    """call() on a client that was never started returns empty string."""
    from mtf.tools.gpd_mcp import GPDMCPClient

    client = GPDMCPClient()
    # Never called client.start(), so no sessions exist
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
    # Agent should be created without errors; tools list should contain
    # only arxiv + semantic search (no GPD tools)
    assert agent._agent_id == "lit-test"
    # The tools list should have exactly 2 (arxiv + semantic scholar)
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
    # No GPD tools, so tools list should be empty
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
    # No GPD tools, so tools list should be empty
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
    assert cfg.physics_domain == "condensed_matter"
    assert cfg.gpd_servers == [
        "verification", "errors", "protocols", "conventions", "patterns"
    ]
