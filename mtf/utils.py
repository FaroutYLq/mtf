"""Shared utility helpers for mtf."""

from __future__ import annotations


def strip_fences(code: str) -> str:
    """Strip markdown code fences from a code string.

    Handles single or multiple fenced blocks; returns only the content
    that was inside the fences.  If no fences are present, returns *code*
    unchanged.
    """
    if "```" not in code:
        return code
    lines = code.splitlines()
    code_lines: list[str] = []
    in_block = False
    for line in lines:
        if line.strip().startswith("```"):
            in_block = not in_block
            continue
        if in_block:
            code_lines.append(line)
    return "\n".join(code_lines)
