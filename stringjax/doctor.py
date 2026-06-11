"""Environment diagnostics for the StringJAX ecosystem.

The :func:`report` function gathers everything needed to answer "why isn't this
working?" -- installed member-package versions, the active JAX backend and
devices, whether 64-bit precision is enabled, and whether the licence-gated
optional dependencies are present.
"""

from __future__ import annotations

import importlib.metadata as _md
import importlib.util as _util
import platform

#: Member packages of the StringJAX ecosystem.
MEMBERS = ["jaxvacua", "jaxpolylog", "stringforge"]

#: Optional, licence-gated dependencies whose mere presence is worth reporting.
OPTIONAL = ["gurobipy"]


def report() -> dict:
    """Return a dictionary describing the current StringJAX environment.

    Returns
    -------
    dict
        Keys include the Python version and platform, the version of each
        member package (``None`` if not installed), ``optional:<name>`` booleans
        for the optional dependencies, and -- when JAX is importable -- the JAX
        version, default backend, visible devices, and whether ``float64`` is
        enabled.
    """
    info: dict = {
        "stringjax": _safe_version("stringjax"),
        "python": platform.python_version(),
        "platform": platform.platform(),
    }

    for pkg in MEMBERS:
        info[pkg] = _safe_version(pkg)

    for pkg in OPTIONAL:
        info[f"optional:{pkg}"] = _util.find_spec(pkg) is not None

    try:
        import jax
        import jax.numpy as jnp

        info["jax"] = jax.__version__
        info["jax_backend"] = jax.default_backend()
        info["jax_devices"] = [str(d) for d in jax.devices()]
        info["float64_enabled"] = bool(jnp.asarray(1.0).dtype == jnp.float64)
    except Exception as exc:  # pragma: no cover - depends on local install
        info["jax_error"] = repr(exc)

    return info


def hints(info: dict) -> list[str]:
    """Return human-readable suggestions based on a :func:`report` dictionary."""
    out: list[str] = []
    if info.get("jaxvacua") is None:
        out.append(
            "jaxvacua is not installed. Install the ecosystem with "
            '`pip install stringjax` (or `pip install "stringjax[all]"`).'
        )
    if "jax_error" in info:
        out.append("JAX failed to import; check your JAX/jaxlib installation.")
    else:
        if info.get("jax_backend") == "cpu":
            out.append(
                "JAX is using the CPU backend. For GPU/TPU install the matching "
                'wheel, e.g. `pip install -U "jax[cuda12]"`.'
            )
        if info.get("float64_enabled") is False:
            out.append(
                "float64 is disabled. Flux-vacuum searches require it: set "
                "`JAXVACUA_PRECISION=float64` before importing JAX, or call "
                '`jaxvacua.set_precision("float64")`.'
            )
    if info.get("optional:gurobipy") is False:
        out.append(
            "Gurobi is not available. Systematic flux enumeration needs it: "
            '`pip install "jaxvacua[bounding]"` and activate a (free academic) licence.'
        )
    return out


def _safe_version(pkg: str):
    try:
        return _md.version(pkg)
    except _md.PackageNotFoundError:
        return None


if __name__ == "__main__":  # pragma: no cover
    data = report()
    for key, value in data.items():
        print(f"{key:24s} {value}")
    for line in hints(data):
        print(f"  - {line}")
