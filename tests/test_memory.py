"""Unit tests for SharedMemory."""

import pytest

from mtf.memory import MemoryEntry, MemoryKind, SharedMemory


def test_add_and_len():
    m = SharedMemory()
    assert len(m) == 0
    m.add(MemoryKind.LITERATURE, "paper about superconductors")
    assert len(m) == 1


def test_filter_by_kind():
    m = SharedMemory()
    m.add(MemoryKind.LITERATURE, "lit entry")
    m.add(MemoryKind.DEBATE, "debate entry")
    m.add(MemoryKind.USER_FEEDBACK, "feedback")

    lit = m.filter(MemoryKind.LITERATURE)
    assert len(lit) == 1
    assert lit[0].content == "lit entry"

    two = m.filter(MemoryKind.LITERATURE, MemoryKind.DEBATE)
    assert len(two) == 2


def test_filter_no_args_returns_all():
    m = SharedMemory()
    m.add(MemoryKind.LITERATURE, "a")
    m.add(MemoryKind.DEBATE, "b")
    assert len(m.filter()) == 2


def test_format_context_empty():
    m = SharedMemory()
    assert m.format_context() == ""


def test_format_context_with_entries():
    m = SharedMemory()
    m.add(MemoryKind.HYPOTHESIS, "Hypothesis: Bose-Einstein condensation")
    ctx = m.format_context(MemoryKind.HYPOTHESIS)
    assert "HYPOTHESIS" in ctx
    assert "Bose-Einstein" in ctx


def test_hypotheses():
    m = SharedMemory()
    m.add(MemoryKind.HYPOTHESIS, "H1")
    m.add(MemoryKind.HYPOTHESIS, "H2")
    m.add(MemoryKind.LITERATURE, "not a hypothesis")
    assert m.hypotheses() == ["H1", "H2"]


def test_metadata_stored():
    m = SharedMemory()
    m.add(MemoryKind.LITERATURE, "content", agent_id="lit-0")
    entry = m.filter(MemoryKind.LITERATURE)[0]
    assert entry.metadata["agent_id"] == "lit-0"
