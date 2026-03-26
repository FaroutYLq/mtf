"""GPD MCP client: bridges GPD MCP servers to sdk.SdkMcpTool objects.

Each GPD server runs as a subprocess communicating over stdio MCP protocol.
A dedicated background event loop handles all async MCP I/O, exposing
SdkMcpTool-compatible callables to mtf agents.
"""

from __future__ import annotations

import asyncio
import logging
import sys
import threading
from contextlib import AsyncExitStack
from typing import Any

import claude_agent_sdk as sdk

logger = logging.getLogger(__name__)

# Maps short server names to their Python module paths in the GPD package.
_SERVER_MODULES: dict[str, str] = {
    "verification": "gpd.mcp.servers.verification_server",
    "errors": "gpd.mcp.servers.errors_mcp",
    "protocols": "gpd.mcp.servers.protocols_server",
    "conventions": "gpd.mcp.servers.conventions_server",
    "patterns": "gpd.mcp.servers.patterns_server",
    "skills": "gpd.mcp.servers.skills_server",
}


class GPDMCPClient:
    """Manages live connections to one or more GPD MCP servers.

    All async MCP I/O runs in a dedicated background thread with its own
    event loop so that sync tool functions can block-call into the async MCP
    session without conflicting with the main asyncio event loop used by the
    rest of mtf.

    Usage::

        client = GPDMCPClient()
        client.start(["verification", "errors", "protocols", "conventions", "patterns"])
        tool = client.make_tool("verification", "get_checklist",
                                "Get domain-specific physics checklist.")
        # ... pass tool to agents ...
        client.close()
    """

    def __init__(self) -> None:
        self._loop = asyncio.new_event_loop()
        self._thread = threading.Thread(
            target=self._loop.run_forever, daemon=True, name="gpd-mcp-loop"
        )
        self._thread.start()
        self._sessions: dict[str, Any] = {}  # server_name -> ClientSession
        self._tool_schemas: dict[str, dict[str, Any]] = {}  # server_name -> {tool_name -> inputSchema}
        self._exit_stack: AsyncExitStack | None = None
        self._available: bool = False

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def start(self, server_names: list[str]) -> None:
        """Start the requested GPD MCP servers.

        Silently no-ops if the ``get-physics-done`` package is not installed.
        """
        try:
            self._run(self._start_async(server_names))
            self._available = bool(self._sessions)
        except Exception as exc:
            logger.warning("GPD MCP servers could not start (%s). Continuing without GPD.", exc)
            self._available = False

    async def _start_async(self, server_names: list[str]) -> None:
        from mcp import ClientSession  # noqa: PLC0415
        from mcp.client.stdio import StdioServerParameters, stdio_client  # noqa: PLC0415

        self._exit_stack = AsyncExitStack()
        await self._exit_stack.__aenter__()
        for name in server_names:
            module = _SERVER_MODULES.get(name)
            if module is None:
                logger.warning("Unknown GPD server '%s', skipping.", name)
                continue
            try:
                params = StdioServerParameters(
                    command=sys.executable, args=["-m", module]
                )
                read, write = await self._exit_stack.enter_async_context(
                    stdio_client(params)
                )
                session = await self._exit_stack.enter_async_context(
                    ClientSession(read, write)
                )
                await session.initialize()
                self._sessions[name] = session
                # Cache input schemas for all tools on this server
                try:
                    tools_result = await session.list_tools()
                    self._tool_schemas[name] = {
                        t.name: t.inputSchema for t in tools_result.tools
                    }
                except Exception as exc:
                    logger.debug("Could not list tools for GPD server '%s': %s", name, exc)
                    self._tool_schemas[name] = {}
                logger.debug("GPD MCP server '%s' started.", name)
            except Exception as exc:
                logger.warning("Failed to start GPD server '%s': %s", name, exc)

    def close(self) -> None:
        """Shut down all MCP server subprocesses and stop the background loop."""
        try:
            if self._exit_stack is not None:
                self._run(self._exit_stack.__aexit__(None, None, None))
        except Exception as exc:
            logger.debug("Error during GPD MCP cleanup: %s", exc)
        finally:
            self._loop.call_soon_threadsafe(self._loop.stop)
            self._thread.join(timeout=5)

    # ------------------------------------------------------------------
    # Calling tools
    # ------------------------------------------------------------------

    def call(self, server: str, tool_name: str, **kwargs: Any) -> str:
        """Synchronously call an MCP tool and return its text result.

        Returns an empty string if the server is unavailable.
        """
        session = self._sessions.get(server)
        if session is None:
            return ""
        try:
            result = self._run(session.call_tool(tool_name, kwargs))
            if hasattr(result, "content") and result.content:
                return result.content[0].text
            return str(result)
        except Exception as exc:
            logger.warning("GPD tool call %s/%s failed: %s", server, tool_name, exc)
            return ""

    async def async_call(self, server: str, tool_name: str, **kwargs: Any) -> str:
        """Async wrapper around call() for use inside asyncio.gather() fan-outs.

        Offloads the blocking wait to a thread-pool thread so it does not block
        the main asyncio event loop.
        """
        return await asyncio.to_thread(self.call, server, tool_name, **kwargs)

    def make_tool(self, server: str, tool_name: str, description: str) -> sdk.SdkMcpTool | None:
        """Return an sdk.SdkMcpTool backed by the given MCP server tool.

        Returns ``None`` if the server was not started (e.g. GPD not installed),
        so callers can filter None values out of their tool lists.
        """
        if server not in self._sessions:
            return None

        client = self  # captured by closure
        input_schema = self._tool_schemas.get(server, {}).get(tool_name, {})

        async def _handler(args: Any) -> dict:
            kwargs = args if isinstance(args, dict) else {}
            result = await asyncio.to_thread(client.call, server, tool_name, **kwargs)
            return {"content": [{"type": "text", "text": result}]}

        return sdk.SdkMcpTool(
            name=tool_name,
            description=description,
            input_schema=input_schema or {"type": "object", "properties": {}},
            handler=_handler,
        )

    @property
    def available(self) -> bool:
        """True if at least one GPD MCP server started successfully."""
        return self._available

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _run(self, coro: Any) -> Any:
        """Submit a coroutine to the background event loop and block for result."""
        future = asyncio.run_coroutine_threadsafe(coro, self._loop)
        return future.result(timeout=60)
