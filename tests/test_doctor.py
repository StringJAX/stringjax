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
