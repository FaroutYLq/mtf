"""Fitting tools: executes agent-generated lmfit code in a sandboxed namespace."""

from __future__ import annotations

import builtins
import functools
import textwrap
import traceback
from typing import Any

import numpy as np
from lmfit import Model, Parameters, minimize as _real_minimize
from scipy import optimize, stats

# Modules that fitting code is allowed to import.  Anything outside this set
# raises ImportError so the generated code cannot reach os, subprocess, socket, etc.
_ALLOWED_IMPORTS = frozenset({
    "numpy", "np", "lmfit", "scipy", "scipy.optimize", "scipy.stats",
    "math", "cmath", "functools", "itertools", "collections",
})


def _make_restricted_import(real_import: Any) -> Any:
    """Return an __import__ that allows only _ALLOWED_IMPORTS."""
    def _import(name: str, *args: Any, **kwargs: Any) -> Any:
        top = name.split(".")[0]
        if top not in _ALLOWED_IMPORTS and name not in _ALLOWED_IMPORTS:
            raise ImportError(
                f"Import of '{name}' is not permitted in fitting code. "
                f"Allowed: {sorted(_ALLOWED_IMPORTS)}"
            )
        return real_import(name, *args, **kwargs)
    return _import


_SAFE_BUILTINS: dict[str, Any] = {
    k: getattr(builtins, k)
    for k in (
        "abs", "all", "any", "bool", "bytes", "callable", "chr", "complex",
        "dict", "divmod", "enumerate", "filter", "float", "format",
        "frozenset", "getattr", "hasattr", "hash", "int", "isinstance",
        "issubclass", "iter", "len", "list", "map", "max", "min", "next",
        "object", "ord", "pow", "print", "range", "repr", "reversed",
        "round", "set", "slice", "sorted", "str", "sum", "tuple", "type",
        "zip", "NotImplemented", "Ellipsis", "None", "True", "False",
        "ArithmeticError", "Exception", "IndexError", "KeyError",
        "NameError", "RuntimeError", "StopIteration", "TypeError",
        "ValueError", "ZeroDivisionError", "ImportError",
    )
}
_SAFE_BUILTINS["__import__"] = _make_restricted_import(builtins.__import__)


def _make_sentinel_minimize(called_flag: list[bool]):
    """Wrap lmfit.minimize to record whether it was actually invoked."""
    @functools.wraps(_real_minimize)
    def sentinel_minimize(*args, **kwargs):
        called_flag[0] = True
        return _real_minimize(*args, **kwargs)
    return sentinel_minimize


def _make_sentinel_model(called_flag: list[bool]):
    """Wrap lmfit.Model so Model(...).fit(...) records a call."""
    class SentinelModel(Model):
        def fit(self, *args, **kwargs):
            called_flag[0] = True
            return super().fit(*args, **kwargs)
    return SentinelModel


def _validate_result(result: Any, optimizer_called: bool) -> list[str]:
    """Return a list of integrity warning strings (empty = clean)."""
    warnings: list[str] = []

    if not optimizer_called:
        warnings.append(
            "INTEGRITY WARNING: No lmfit optimizer call (minimize() or Model.fit()) "
            "was detected during code execution. Result values may be hardcoded."
        )

    if result is None:
        return warnings  # error handled upstream

    if not isinstance(result, dict):
        return warnings  # unusual but not necessarily wrong

    # Check chi_squared is physically plausible
    chi2 = result.get("chi_squared") or result.get("chisqr")
    red_chi2 = result.get("reduced_chi_squared") or result.get("redchi")
    if chi2 is not None:
        try:
            chi2_val = float(chi2)
            if chi2_val < 0:
                warnings.append(
                    f"INTEGRITY WARNING: chi_squared={chi2_val} is negative, "
                    "which is physically impossible."
                )
        except (TypeError, ValueError):
            pass

    if red_chi2 is not None and chi2 is not None:
        try:
            rchi = float(red_chi2)
            if rchi < 0:
                warnings.append(
                    f"INTEGRITY WARNING: reduced_chi_squared={rchi} is negative."
                )
        except (TypeError, ValueError):
            pass

    # Check parameters is non-empty
    params = result.get("parameters")
    if params is not None and isinstance(params, dict) and len(params) == 0:
        warnings.append(
            "INTEGRITY WARNING: 'parameters' dict is empty — no fitted parameters reported."
        )

    return warnings


def run_fitting_code(code: str, data: dict[str, Any], integrity_check: bool = True) -> dict[str, Any]:
    """Execute agent-generated fitting code in a sandboxed namespace.

    The code has access to: numpy (np), lmfit (Model, Parameters, minimize),
    scipy.optimize, scipy.stats, and the 'data' dict provided by the user toolkit.

    The code MUST assign its results to a variable named 'result' which is
    returned to the caller.

    When integrity_check=True (default), sentinel wrappers detect whether a real
    optimizer was called and _validate_result() checks for obvious fabrications.
    When integrity_check=False, the real lmfit symbols are injected directly and
    no validation is performed.

    Args:
        code: Python source code string produced by a FittingAgent.
        data: Dict of user-provided arrays / scalars from the toolkit registry.
        integrity_check: Whether to run anti-fabrication checks (default True).

    Returns:
        The value of 'result' from the executed code, or an error dict.
        May include an 'integrity_warnings' key with a list of warning strings.
    """
    if integrity_check:
        optimizer_called: list[bool] = [False]
        namespace: dict[str, Any] = {
            "__builtins__": _SAFE_BUILTINS,
            "np": np,
            "numpy": np,
            "Model": _make_sentinel_model(optimizer_called),
            "Parameters": Parameters,
            "minimize": _make_sentinel_minimize(optimizer_called),
            "optimize": optimize,
            "stats": stats,
            "data": data,
            "result": None,
        }
    else:
        namespace = {
            "__builtins__": _SAFE_BUILTINS,
            "np": np,
            "numpy": np,
            "Model": Model,
            "Parameters": Parameters,
            "minimize": _real_minimize,
            "optimize": optimize,
            "stats": stats,
            "data": data,
            "result": None,
        }

    try:
        exec(textwrap.dedent(code), namespace)  # noqa: S102
    except Exception:
        return {"error": traceback.format_exc(), "code": code}

    raw_result = namespace.get("result")

    if not integrity_check:
        return {"result": raw_result, "code": code}

    warnings = _validate_result(raw_result, optimizer_called[0])

    output: dict[str, Any] = {"result": raw_result, "code": code}
    if warnings:
        output["integrity_warnings"] = warnings
    return output
