"""ToolBuilderAgent: digests unstructured user input into toolkit-ready items.

When users provide raw Python functions, datasheets, code snippets, or mixed
content instead of clean Python literals, this agent:
  1. Classifies the input (data arrays, model callables, or both).
  2. LLM-generates normalised extraction code.
  3. exec()-runs that code in a safe namespace.
  4. Returns structured data dicts and callable model functions that can be
     registered directly into a ToolkitRegistry.
"""

from __future__ import annotations

import textwrap
import traceback
from dataclasses import dataclass, field
from typing import Any, Callable

from mtf.agents.base import BaseAgent
from mtf.memory import MemoryKind, SharedMemory
from mtf.utils import strip_fences

import numpy as np
from lmfit import Model, Parameters, minimize
from scipy import optimize, stats


_SYSTEM_PROMPT = """You are a data-ingestion specialist for an experimental physics fitting system.

Your job: read raw user-supplied input (which may be a Python function, a CSV/table,
free-form numbers, commented code, or a mixture) and produce a single self-contained
Python script that normalises everything into two plain dicts:

  data_items  : dict[str, Any]          — numpy arrays, scalars, or plain lists
  model_items : dict[str, Callable]     — named callables with signature (x, *params)

Rules:
- Use `import numpy as np` inside your code if needed (numpy is pre-imported as `np`).
- Every function stored in model_items must be a plain `def` (no lambdas as dict values).
- If the user input already contains a function definition, preserve its logic exactly.
- If the user input is tabular data (CSV, whitespace-separated, or Python list literals),
  parse it into numpy arrays and store under sensible names derived from column headers
  or the requested item name.
- Always assign BOTH `data_items` and `model_items` at the end, even if one is empty ({}).
- Output ONLY the Python code block, no prose, no markdown fences."""


@dataclass
class ToolkitDigestResult:
    """Outcome of one ToolBuilderAgent.digest() call."""

    data: dict[str, Any] = field(default_factory=dict)
    models: dict[str, Callable[..., Any]] = field(default_factory=dict)
    summary: str = ""
    error: str = ""


def _exec_digest_code(code: str) -> tuple[dict[str, Any], dict[str, Callable[..., Any]], str]:
    """Execute LLM-generated digest code and extract data_items / model_items."""
    namespace: dict[str, Any] = {
        "np": np,
        "numpy": np,
        "Model": Model,
        "Parameters": Parameters,
        "minimize": minimize,
        "optimize": optimize,
        "stats": stats,
        "data_items": {},
        "model_items": {},
    }
    try:
        exec(textwrap.dedent(code), namespace)  # noqa: S102
    except Exception:
        return {}, {}, traceback.format_exc()
    return (
        namespace.get("data_items", {}),
        namespace.get("model_items", {}),
        "",
    )


class ToolBuilderAgent(BaseAgent):
    """Digests raw user input into ToolkitRegistry-ready data and model callables.

    Spawn one instance per missing-toolkit interaction; it is cheap (one
    sdk.query() call) and stateless beyond the shared memory context.
    """

    def __init__(
        self,
        agent_id: str,
        model: str,
        memory: SharedMemory,
    ) -> None:
        super().__init__(
            agent_id=agent_id,
            model=model,
            tools=[],
            memory=memory,
            system_prompt=_SYSTEM_PROMPT,
        )

    async def digest(
        self,
        item_name: str,
        user_input: str,
    ) -> ToolkitDigestResult:
        """Parse `user_input` and build toolkit items for the requested `item_name`.

        The agent may extract additional items beyond the named one if the input
        contains multiple coherent pieces (e.g. both x-data and a model function).
        """
        task = (
            f"The fitting system needs a toolkit item called: '{item_name}'\n\n"
            f"The user provided the following raw input:\n"
            f"{'=' * 60}\n{user_input}\n{'=' * 60}\n\n"
            "Generate the normalisation code as described in your instructions."
        )
        code = await self._query(
            task,
            extra_kinds=(
                MemoryKind.IMAGE_DATA,
                MemoryKind.DEBATE,
                MemoryKind.LITERATURE,
            ),
        )

        # Strip markdown fences if the model added them anyway
        code = strip_fences(code)

        data, models, err = _exec_digest_code(code)

        if err:
            return ToolkitDigestResult(error=err)

        data_names = list(data.keys())
        model_names = list(models.keys())
        summary_parts = []
        if data_names:
            summary_parts.append(f"data: {', '.join(data_names)}")
        if model_names:
            summary_parts.append(f"models: {', '.join(model_names)}")
        summary = f"ToolBuilder extracted — {'; '.join(summary_parts) or 'nothing'} — from user input for '{item_name}'"

        self._memory.add(
            MemoryKind.TOOLKIT_DIGEST,
            summary,
            agent_id=self._agent_id,
            item_name=item_name,
        )

        return ToolkitDigestResult(data=data, models=models, summary=summary)
