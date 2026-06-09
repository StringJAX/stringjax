"""Sphinx configuration for the StringJAX documentation hub."""

from __future__ import annotations

from importlib.metadata import version as _v

project = "StringJAX"
author = "Andreas Schachner"
copyright = "2026, Andreas Schachner"

try:
    release = _v("stringjax")
except Exception:  # pragma: no cover
    release = "0.1.0"
version = ".".join(release.split(".")[:2])

extensions = [
    "myst_nb",                 # Markdown + executable notebooks
    "sphinx.ext.intersphinx",  # cross-link to member docs and NumPy/JAX/SciPy
    "sphinx.ext.napoleon",     # NumPy-style docstrings (for any local autodoc)
    "sphinx_copybutton",
    "sphinx_design",
]

myst_enable_extensions = ["colon_fence", "deflist", "linkify"]

# The hub orients; it does not execute heavy notebooks itself.
nb_execution_mode = "off"

templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

html_theme = "sphinx_book_theme"
html_title = "StringJAX"
html_static_path = ["_static"]
html_theme_options = {
    "repository_url": "https://github.com/StringJAX/stringjax",
    "use_repository_button": True,
    "use_issues_button": True,
}

# Cross-references into the member packages' documentation.
intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable", None),
    "scipy": ("https://docs.scipy.org/doc/scipy", None),
    "jax": ("https://jax.readthedocs.io/en/latest", None),
    # Member packages (uncomment once their docs are live):
    # "jaxvacua": ("https://jaxvacua.readthedocs.io/en/latest", None),
    "jaxpolylog": ("https://jaxpolylog.readthedocs.io/en/latest", None),
    # "stringforge": ("https://stringforge.readthedocs.io/en/latest", None),
}

linkcheck_ignore = [r"https://github\.com/StringJAX/.*"]  # repos may be private pre-release
