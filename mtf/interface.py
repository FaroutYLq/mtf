"""Human interface abstraction and CLI implementation."""

from __future__ import annotations

import asyncio
from abc import ABC, abstractmethod
from pathlib import Path

from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.prompt import Confirm, Prompt


class HumanInterface(ABC):
    """Abstract interface for human interaction."""

    @abstractmethod
    async def show(self, content: str, title: str = "") -> None: ...

    @abstractmethod
    async def ask(self, prompt: str) -> str: ...

    @abstractmethod
    async def confirm(self, prompt: str) -> bool: ...

    async def ask_for_images(self) -> list[str]:
        """Prompt the user to optionally provide image file paths.

        Returns a (possibly empty) list of valid image path strings.
        Default implementation returns an empty list; override in subclasses
        that support interactive input.
        """
        return []


class CLIInterface(HumanInterface):
    """Rich-based CLI interface."""

    def __init__(self) -> None:
        self._console = Console()

    async def show(self, content: str, title: str = "") -> None:
        panel = Panel(Markdown(content), title=title or "MTF", border_style="blue")
        await asyncio.to_thread(self._console.print, panel)

    async def ask(self, prompt: str) -> str:
        return await asyncio.to_thread(Prompt.ask, f"[cyan]{prompt}[/cyan]")

    async def confirm(self, prompt: str) -> bool:
        return await asyncio.to_thread(Confirm.ask, f"[yellow]{prompt}[/yellow]")

    async def ask_for_images(self) -> list[str]:
        """Interactively ask the user for optional image file paths."""
        has_images = await self.confirm(
            "Do you have experimental images or plots to include?"
        )
        if not has_images:
            return []

        raw = await self.ask(
            "Enter image file path(s) separated by spaces (PNG, JPG, GIF, WebP)"
        )
        paths: list[str] = []
        for token in raw.split():
            p = Path(token.strip())
            if p.exists():
                paths.append(str(p))
            else:
                await asyncio.to_thread(
                    self._console.print,
                    f"[red]Warning: image not found and will be skipped: {p}[/red]",
                )
        return paths
