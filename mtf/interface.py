"""Human interface abstraction and CLI implementation."""

from __future__ import annotations

import asyncio
from abc import ABC, abstractmethod

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
