# Changelog

All notable changes to the `stringjax` metapackage are documented here. The format
follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and the project
adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.1.1] - 2026-10-28
### Changed
- Requirements follow the released members: `jaxvacua>=0.1.3,<0.2` (the first release with the
  `all` and `bounding` extras that `stringjax[all]` and the documentation name) and, in the `data`
  and `all` extras, `stringforge>=0.1.3,<0.2`. The compatibility page gains a 0.1.1 row and adds
  `stringjax-tools`; `jaxpolylog` and `stringjax-tools` are marked as coming through `jaxvacua`, which
  sets their ranges (the 0.1.0 page's `jaxpolylog>=0.3,<0.4` was enforced by no package).
- `stringjax doctor` lists `stringjax-tools` among the members. `float64_enabled` now reports the
  precision a jaxvacua session runs in: JAX starts in float32 and jaxvacua switches to float64 on
  import unless `JAXVACUA_PRECISION=float32`, so before jaxvacua is imported the value comes from that
  variable; it used to test a bare JAX array and so reported float64 as disabled in a working
  environment. The new `jax_enable_x64` key keeps JAX's setting as it is. A `JAXVACUA_PRECISION`
  other than `float64` or `float32`, which jaxvacua rejects at import, is reported as such
  (`jaxvacua_precision_invalid`). The precision hint no longer says that JAX reads
  `JAXVACUA_PRECISION`.
- Gurobi: the documentation and the doctor's hint state that `gurobipy` is a core dependency of
  `jaxvacua` (its default moduli sampling uses it) whose pip wheel carries a size-limited licence,
  instead of "everything else works without it"; if it is missing, the hint is `pip install gurobipy`.
- Python 3.13: classifier added; the unit tests and the `test-install` workflow run on 3.12 and 3.13.
  `test-install` (the core install from PyPI) also runs on pull requests, since the members are on
  PyPI, and imports `stringjax_tools`. Read the Docs builds with `fail_on_warning: true` and no longer
  builds a PDF (its LaTeX builder cannot render the ecosystem page's mermaid diagram), and the
  documentation takes its version from `stringjax/_version.py` (it showed 0.1.0 when stringjax was
  not installed).
- The ecosystem page and the README list `stringjax-tools`; the ecosystem page links the jaxvacua and
  stringforge documentation; the landing page, introduction, motivation and tutorials pages no longer
  count "three member packages".

### Added
- Release workflow `.github/workflows/release.yml`: pushing a tag `vX.Y.Z` builds the sdist and wheel,
  runs `twine check --strict`, refuses a tag that differs from the built version, and publishes to PyPI
  through Trusted Publishing (no API token). Run by hand, it builds and checks without publishing.

## [0.1.0] - 2026-06-11
### Added
- Initial release of the `stringjax` umbrella metapackage.
- Dependency aggregation for the ecosystem: `jaxvacua` (core), with `data`
  (`stringforge`) and `all` extras.
- `stringjax doctor` and `stringjax versions` command-line helpers, plus the
  `stringjax.report()` diagnostics API.
- Documentation hub skeleton (Sphinx + MyST).

[Unreleased]: https://github.com/StringJAX/stringjax/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/StringJAX/stringjax/releases/tag/v0.1.0
