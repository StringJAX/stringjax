"""Unit tests for the diagnostics helpers.

These run without any member package installed: ``report`` must degrade
gracefully, returning ``None`` for absent members.
"""

import stringjax
from stringjax.doctor import MEMBERS, hints, report


def test_version_is_string():
    assert isinstance(stringjax.__version__, str)
    assert stringjax.__version__.count(".") >= 2


def test_report_has_expected_keys():
    info = report()
    for key in ("python", "platform", "stringjax", *MEMBERS):
        assert key in info
    assert "optional:gurobipy" in info
    # Either JAX is importable, or the failure was recorded.
    assert ("jax" in info) or ("jax_error" in info)


def test_report_returns_dict():
    assert isinstance(stringjax.report(), dict)


def test_hints_returns_list_of_str():
    suggestions = hints(report())
    assert isinstance(suggestions, list)
    assert all(isinstance(s, str) for s in suggestions)


def _needs_jax():
    import importlib.util

    import pytest

    if importlib.util.find_spec("jax") is None:
        pytest.skip("JAX not installed")


def test_float64_follows_jaxvacua_precision_before_import(monkeypatch):
    """Before jaxvacua is imported, float64_enabled is what jaxvacua will set."""
    _needs_jax()
    import sys

    monkeypatch.delitem(sys.modules, "jaxvacua", raising=False)
    monkeypatch.delenv("JAXVACUA_PRECISION", raising=False)
    assert report()["float64_enabled"] is True
    monkeypatch.setenv("JAXVACUA_PRECISION", "float32")
    info = report()
    assert info["float64_enabled"] is False
    assert "jaxvacua_precision_invalid" not in info
    assert any("JAXVACUA_PRECISION" in h for h in hints(info))
    for invalid in ("Float64", "bogus", ""):  # jaxvacua raises ValueError at import
        monkeypatch.setenv("JAXVACUA_PRECISION", invalid)
        info = report()
        assert info["float64_enabled"] is None
        assert info["jaxvacua_precision_invalid"] == invalid
        assert any("will fail" in h for h in hints(info))


def test_float64_follows_jax_once_jaxvacua_is_imported(monkeypatch):
    """With jaxvacua imported, float64_enabled is JAX's current setting."""
    _needs_jax()
    import sys
    import types

    import jax

    monkeypatch.setitem(sys.modules, "jaxvacua", types.ModuleType("jaxvacua"))
    monkeypatch.setenv("JAXVACUA_PRECISION", "float64")
    previous = bool(jax.config.jax_enable_x64)
    try:
        for value in (False, True):
            jax.config.update("jax_enable_x64", value)
            info = report()
            assert info["jax_enable_x64"] is value
            assert info["float64_enabled"] is value
    finally:
        jax.config.update("jax_enable_x64", previous)


def test_gurobi_hint_only_with_jaxvacua_installed():
    base = {"jax_backend": "gpu", "float64_enabled": True, "optional:gurobipy": False}
    assert not any("gurobipy" in h for h in hints({**base, "jaxvacua": None}))
    assert any("gurobipy" in h for h in hints({**base, "jaxvacua": "0.1.3"}))
