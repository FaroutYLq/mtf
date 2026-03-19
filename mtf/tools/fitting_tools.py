"""Fitting tools: executes agent-generated lmfit code in a sandboxed namespace."""

from __future__ import annotations

import textwrap
import traceback
from typing import Any

import numpy as np
from lmfit import Model, Parameters, minimize
from scipy import optimize, stats


def run_fitting_code(code: str, data: dict[str, Any]) -> dict[str, Any]:
    """Execute agent-generated fitting code in a sandboxed namespace.

    The code has access to: numpy (np), lmfit (Model, Parameters, minimize),
    scipy.optimize, scipy.stats, and the 'data' dict provided by the user toolkit.

    The code MUST assign its results to a variable named 'result' which is
    returned to the caller.

    Args:
        code: Python source code string produced by a FittingAgent.
        data: Dict of user-provided arrays / scalars from the toolkit registry.

    Returns:
        The value of 'result' from the executed code, or an error dict.
    """
    namespace: dict[str, Any] = {
        "np": np,
        "numpy": np,
        "Model": Model,
        "Parameters": Parameters,
        "minimize": minimize,
        "optimize": optimize,
        "stats": stats,
        "data": data,
        "result": None,
    }
    try:
        exec(textwrap.dedent(code), namespace)  # noqa: S102
    except Exception:
        return {"error": traceback.format_exc(), "code": code}
    return {"result": namespace.get("result"), "code": code}
