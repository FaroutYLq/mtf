"""ToolkitRegistry: holds user-provided experimental data and model functions."""

from __future__ import annotations

from typing import Any, Callable


class ToolkitRegistry:
    """Registry for user-supplied data arrays and model callables.

    Users register items before calling MTFOrchestrator.run() so that
    FittingAgents can request them by name.
    """

    def __init__(self) -> None:
        self._data: dict[str, Any] = {}
        self._models: dict[str, Callable[..., Any]] = {}

    # --- data ---

    def register_data(self, name: str, value: Any) -> None:
        """Register a named data array or scalar."""
        self._data[name] = value

    def get_data(self, name: str) -> Any:
        if name not in self._data:
            raise KeyError(f"Data item '{name}' not in toolkit registry.")
        return self._data[name]

    def list_data(self) -> list[str]:
        return list(self._data.keys())

    # --- models ---

    def register_model(self, name: str, fn: Callable[..., Any]) -> None:
        """Register a named model callable (x, *params) → y."""
        self._models[name] = fn

    def get_model(self, name: str) -> Callable[..., Any]:
        if name not in self._models:
            raise KeyError(f"Model '{name}' not in toolkit registry.")
        return self._models[name]

    def list_models(self) -> list[str]:
        return list(self._models.keys())

    def all_data(self) -> dict[str, Any]:
        return dict(self._data)

    def describe(self) -> str:
        lines = ["Available toolkit items:"]
        if self._data:
            lines.append(f"  Data: {', '.join(self._data)}")
        if self._models:
            lines.append(f"  Models: {', '.join(self._models)}")
        if not self._data and not self._models:
            lines.append("  (none)")
        return "\n".join(lines)
