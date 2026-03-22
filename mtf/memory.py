"""Shared in-process memory for all agents and phases."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class MemoryKind(str, Enum):
    LITERATURE = "literature"
    DEBATE = "debate"
    USER_FEEDBACK = "user_feedback"
    HYPOTHESIS = "hypothesis"
    FIT_RESULT = "fit_result"
    REVIEW = "review"
    IMAGE_DATA = "image_data"
    TOOLKIT_DIGEST = "toolkit_digest"
    # GPD MCP integration
    CONVENTIONS = "conventions"        # physics convention lock from gpd-conventions
    PHYSICS_VERDICT = "physics_verdict"  # structured check results from gpd-verification
    FITTING_WARNINGS = "fitting_warnings"  # pre-dispatch pitfall warnings from pattern library + error classes
    DOMAIN_PATTERNS = "domain_patterns"    # cross-session patterns pre-fetched at literature phase start
    DOMAIN_CLASSIFICATION = "domain_classification"  # auto-detected domain classification (audit trail)
    QUALITATIVE_EVAL = "qualitative_eval"  # qualitative hypothesis evaluation (used when fitting is skipped)
    FITTING_SKIPPED = "fitting_skipped"    # marker written when --no-fitting is active


@dataclass
class MemoryEntry:
    kind: MemoryKind
    content: str
    metadata: dict[str, Any] = field(default_factory=dict)


class SharedMemory:
    """Central store passed by reference to all agents and phases.

    Thread-safe by virtue of asyncio single-threaded event loop.
    """

    def __init__(self) -> None:
        self._entries: list[MemoryEntry] = []

    def add(self, kind: MemoryKind, content: str, **metadata: Any) -> None:
        self._entries.append(MemoryEntry(kind=kind, content=content, metadata=metadata))

    def filter(self, *kinds: MemoryKind) -> list[MemoryEntry]:
        if not kinds:
            return list(self._entries)
        return [e for e in self._entries if e.kind in kinds]

    def format_context(self, *kinds: MemoryKind) -> str:
        entries = self.filter(*kinds) if kinds else self._entries
        if not entries:
            return ""
        lines = ["=== SHARED CONTEXT ==="]
        for e in entries:
            lines.append(f"[{e.kind.value.upper()}] {e.content}")
        lines.append("=== END CONTEXT ===")
        return "\n".join(lines)

    def hypotheses(self) -> list[str]:
        return [e.content for e in self.filter(MemoryKind.HYPOTHESIS)]

    def fit_results(self) -> list[MemoryEntry]:
        return self.filter(MemoryKind.FIT_RESULT)

    def __len__(self) -> int:
        return len(self._entries)
