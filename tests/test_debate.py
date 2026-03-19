"""Tests for DebateEngine with mocked Anthropic client."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from mtf.config import MTFConfig
from mtf.debate import DebateEngine
from mtf.memory import MemoryKind, SharedMemory


@pytest.fixture
def memory():
    return SharedMemory()


@pytest.fixture
def config():
    return MTFConfig()


@pytest.mark.asyncio
async def test_synthesize_stores_in_memory(memory, config):
    engine = DebateEngine(config, memory)

    mock_response = MagicMock()
    mock_response.content = [MagicMock(text="Synthesized result")]

    with patch.object(engine._client.messages, "create", return_value=mock_response):
        result = await engine.synthesize(
            reports=["Report A", "Report B"],
            phase="literature",
        )

    assert result == "Synthesized result"
    debate_entries = memory.filter(MemoryKind.DEBATE)
    assert len(debate_entries) == 1
    assert debate_entries[0].metadata["phase"] == "literature"


@pytest.mark.asyncio
async def test_synthesize_includes_memory_context(memory, config):
    memory.add(MemoryKind.USER_FEEDBACK, "Focus on quantum effects")
    engine = DebateEngine(config, memory)

    captured_kwargs: dict = {}

    def capture_call(**kwargs):
        captured_kwargs.update(kwargs)
        mock_response = MagicMock()
        mock_response.content = [MagicMock(text="OK")]
        return mock_response

    with patch.object(engine._client.messages, "create", side_effect=capture_call):
        await engine.synthesize(reports=["R1"], phase="test")

    messages = captured_kwargs.get("messages", [])
    assert messages
    user_content = messages[0]["content"]
    assert "SHARED CONTEXT" in user_content
