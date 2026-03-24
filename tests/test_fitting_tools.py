"""Tests for fitting_tools integrity checks."""
import pytest
from mtf.tools.fitting_tools import run_fitting_code, _validate_result


def test_hardcoded_result_triggers_warning():
    """Code that assigns result without calling minimize() gets flagged."""
    code = "result = {'chi_squared': 1.0, 'parameters': {'a': 2.0}}"
    output = run_fitting_code(code, {})
    assert "integrity_warnings" in output
    assert any("INTEGRITY WARNING" in w for w in output["integrity_warnings"])


def test_legitimate_fit_no_warning():
    """Real lmfit minimize() call should not trigger integrity warning."""
    import numpy as np
    code = """
import numpy as np
x = np.array([1.0, 2.0, 3.0, 4.0])
y = np.array([2.1, 4.0, 6.1, 7.9])

def residuals(params, x, y):
    a = params['a']
    return a * x - y

from lmfit import Parameters
params = Parameters()
params.add('a', value=1.0)
out = minimize(residuals, params, args=(x, y))
result = {
    'chi_squared': out.chisqr,
    'parameters': {name: par.value for name, par in out.params.items()},
}
"""
    output = run_fitting_code(code, {})
    assert "error" not in output
    assert "integrity_warnings" not in output or not any(
        "No lmfit optimizer" in w for w in output.get("integrity_warnings", [])
    )


def test_negative_chi_squared_triggers_warning():
    """Negative chi_squared is physically impossible and must be flagged."""
    code = "result = {'chi_squared': -1.5, 'parameters': {'a': 2.0}}"
    output = run_fitting_code(code, {})
    warnings = output.get("integrity_warnings", [])
    assert any("negative" in w.lower() for w in warnings)


def test_integrity_check_disabled():
    """When integrity_check=False, hardcoded result passes without warning."""
    code = "result = {'chi_squared': 1.0, 'parameters': {'a': 2.0}}"
    output = run_fitting_code(code, {}, integrity_check=False)
    assert "integrity_warnings" not in output


def test_validate_result_empty_parameters():
    """Empty parameters dict should trigger a warning."""
    warnings = _validate_result({'chi_squared': 1.0, 'parameters': {}}, optimizer_called=True)
    assert any("empty" in w.lower() for w in warnings)
