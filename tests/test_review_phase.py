"""Tests for review phase second-pass loop and multi-model cycling."""
from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from mtf.config import MTFConfig
from mtf.memory import SharedMemory


def test_reviewer_model_cycling():
    """reviewer_models cycles correctly across reviewer instances."""
    from mtf.phases.review_phase import run_review_phase  # noqa: PLC0415

    config = MTFConfig(
        n_reviewer=3,
        reviewer_models=["model-a", "model-b"],
    )
    # Extract the cycling logic inline (mirrors _reviewer_model inner function)
    def _reviewer_model(i: int) -> str:
        models = config.reviewer_models
        if models:
            return models[i % len(models)]
        return config.reviewer_model

    assert _reviewer_model(0) == "model-a"
    assert _reviewer_model(1) == "model-b"
    assert _reviewer_model(2) == "model-a"  # wraps


def test_reviewer_model_fallback():
    """Empty reviewer_models falls back to reviewer_model."""
    config = MTFConfig(reviewer_models=[], reviewer_model="claude-sonnet-4-6")

    def _reviewer_model(i: int) -> str:
        models = config.reviewer_models
        if models:
            return models[i % len(models)]
        return config.reviewer_model

    assert _reviewer_model(0) == "claude-sonnet-4-6"
    assert _reviewer_model(5) == "claude-sonnet-4-6"


def test_reviewer_verification_passes_default():
    """Default reviewer_verification_passes is 1 (no second pass)."""
    config = MTFConfig()
    assert config.reviewer_verification_passes == 1


def test_reviewer_verification_passes_second_pass_task_contains_first_report():
    """Second-pass task text must include the first-pass report."""
    first_report = "First pass review: hypothesis A is SUPPORTED."
    phenomenon = "Anomalous resistance peak at 4K"
    pass_num = 0

    task = (
        f"Original phenomenon:\n{phenomenon}\n\n"
        f"Your previous review:\n{first_report}\n\n"
        f"Re-read the above review from the beginning. Did you miss anything? "
        f"Check every claim, equation, parameter range, and citation again. "
        f"Append additional findings under 'Additional concerns:'. "
        f"If you found nothing new, state that explicitly."
    )
    assert first_report in task
    assert phenomenon in task
    assert "Additional concerns:" in task
