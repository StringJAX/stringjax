"""StringJAX: umbrella metapackage for differentiable, JAX-native string-theory tools.

This package contains no physics code. It exists only to install a
mutually-compatible set of member packages and to provide small convenience
helpers. Import the member packages directly, e.g.::

    import jaxvacua as jvc

To inspect the environment from Python::

    import stringjax
    stringjax.report()        # dict of versions, JAX backend, float64 status, ...

or, equivalently, from a shell::

    stringjax doctor

The diagnostics live in the :mod:`stringjax.doctor` submodule; the ``report`` and
``hints`` functions can also be imported from there directly.
"""

from ._version import __version__
from .doctor import report

__all__ = ["__version__", "report"]
