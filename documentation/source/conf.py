"""Sphinx configuration for the StringJAX documentation hub.

Mirrors the layout used by JAXVacua: ``sphinx_book_theme`` HTML output,
``myst_nb`` for Markdown + notebooks, ``intersphinx`` for cross-links
into member-package documentation, and ``sphinx_design`` for the
grid-card landing page.
"""

from __future__ import annotations

from importlib.metadata import version as _v

# -- Project information -----------------------------------------------------
project = "StringJAX"
author = "Andreas Schachner"
copyright = "2026, Andreas Schachner"

try:
    release = _v("stringjax")
except Exception:  # pragma: no cover
    release = "0.1.0"
version = ".".join(release.split(".")[:2])


# -- General configuration ---------------------------------------------------
extensions = [
    "sphinx.ext.autodoc",  # optional autodoc for any local module
    "sphinx.ext.autosummary",
    "sphinx.ext.intersphinx",  # cross-link to member docs + NumPy / JAX / SciPy
    "sphinx.ext.mathjax",
    "sphinx.ext.napoleon",  # NumPy / Google-style docstrings
    "sphinx_autodoc_typehints",
    "sphinx_copybutton",
    "sphinx_design",
    "sphinx_togglebutton",
    "sphinxcontrib.mermaid",
    "myst_nb",  # Markdown + executable notebooks
]

source_suffix = {
    ".rst": "restructuredtext",
    ".md": "myst-nb",
    ".ipynb": "myst-nb",
}

# The hub orients; it does not execute heavy notebooks itself.  Cell
# outputs that ship in the .ipynb files are rendered as-is.
nb_execution_mode = "off"
nb_execution_allow_errors = False
nb_merge_streams = True
nb_execution_timeout = 120

myst_enable_extensions = ["colon_fence", "deflist", "linkify", "dollarmath"]
myst_heading_anchors = 4

autosummary_generate = True
napoleon_use_rtype = False
add_module_names = False

templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]
suppress_warnings = [
    # member-package autodoc can emit cosmetic warnings about Google-style
    # argument sections; the rendered pages remain correct.
    "docutils",
]


# -- HTML output -------------------------------------------------------------
html_theme = "sphinx_book_theme"
html_title = "StringJAX"
html_static_path = ["_static"]
html_theme_options = {
    "repository_url": "https://github.com/StringJAX/stringjax",
    "repository_branch": "main",
    "path_to_docs": "documentation/source",
    "use_repository_button": True,
    "use_issues_button": True,
    "use_edit_page_button": True,
}


# -- MathJax 3 ---------------------------------------------------------------
mathjax3_config = {
    "tex": {"inlineMath": [["$", "$"], ["\\(", "\\)"]]},
    "svg": {"fontCache": "global"},
}


# -- Intersphinx -------------------------------------------------------------
# Cross-references into the member packages' documentation.  Mappings are
# enabled only when a stable ``objects.inv`` is published.
intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable", None),
    "scipy": ("https://docs.scipy.org/doc/scipy", None),
    "jax": ("https://jax.readthedocs.io/en/latest", None),
    "jaxpolylog": ("https://jaxpolylog.readthedocs.io/en/latest", None),
    # "jaxvacua":   ("https://jaxvacua.readthedocs.io/en/latest",    None),
    # "stringforge":("https://stringforge.readthedocs.io/en/latest", None),
}

linkcheck_ignore = [r"https://github\.com/StringJAX/.*"]  # repos may be private pre-release
