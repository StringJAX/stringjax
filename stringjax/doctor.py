"""Environment diagnostics for the StringJAX ecosystem.

The :func:`report` function gathers everything needed to answer "why isn't this
working?" -- installed member-package versions, the active JAX backend and
devices, whether 64-bit precision is enabled, and whether the licence-gated
optional dependencies are present.
"""

from __future__ import annotations

import importlib.metadata as _md
import importlib.util as _util
import os
import platform
import sys

#: Member packages of the StringJAX ecosystem (distribution names).
MEMBERS = ["jaxvacua", "jaxpolylog", "stringforge", "stringjax-tools"]

#: Licence-gated dependencies whose presence is worth reporting. gurobipy is a core
#: dependency of jaxvacua; its pip wheel carries a size-limited licence.
OPTIONAL = ["gurobipy"]


def report() -> dict:
    """Return a dictionary describing the current StringJAX environment.

    Returns
    -------
    dict
        Keys include the Python version and platform, the version of each
        member package (``None`` if not installed), ``optional:<name>`` booleans
        for the licence-gated dependencies, and -- when JAX is importable -- the
        JAX version, default backend, visible devices, and ``float64_enabled``.

        ``float64_enabled`` is the precision a jaxvacua session runs in. JAX
        itself starts in float32 and jaxvacua switches to float64 when it is
        imported, unless ``JAXVACUA_PRECISION=float32``; so if jaxvacua is not
        yet imported, the value is read from ``JAXVACUA_PRECISION`` (default
        ``float64``), and otherwise from JAX's current setting. A value of
        ``JAXVACUA_PRECISION`` other than ``float64`` or ``float32``, which
        jaxvacua rejects at import, gives ``None`` and the key
        ``jaxvacua_precision_invalid``.
        ``jax_enable_x64`` records JAX's setting in this process as it is.
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
        info["jax_enable_x64"] = bool(jnp.asarray(1.0).dtype == jnp.float64)
        if "jaxvacua" in sys.modules:
            info["float64_enabled"] = info["jax_enable_x64"]
        else:
            mode = os.environ.get("JAXVACUA_PRECISION", "float64")
            if mode in ("float64", "float32"):
                info["float64_enabled"] = mode == "float64"
            else:  # jaxvacua refuses it at import
                info["float64_enabled"] = None
                info["jaxvacua_precision_invalid"] = mode
    except Exception as exc:  # noqa: BLE001 - report any broken JAX install, never crash  # pragma: no cover
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
        if "jaxvacua_precision_invalid" in info:
            out.append(
                f"JAXVACUA_PRECISION={info['jaxvacua_precision_invalid']!r} is neither "
                "`float64` nor `float32`, so importing jaxvacua will fail: unset it or set "
                "one of the two."
            )
        if info.get("float64_enabled") is False:
            out.append(
                "jaxvacua runs in float32 here. Flux-vacuum searches need float64: "
                "unset `JAXVACUA_PRECISION` (or set it to `float64`) before importing "
                'jaxvacua, or call `jaxvacua.set_precision("float64")`.'
            )
    if info.get("jaxvacua") is not None and info.get("optional:gurobipy") is False:
        out.append(
            "gurobipy is missing although jaxvacua requires it: `pip install gurobipy`. "
            "Its pip wheel carries a size-limited licence; larger models need a full "
            "(free academic) Gurobi licence."
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
