# Contributing to StringJAX

Thank you for considering a contribution! This repository hosts the `stringjax`
metapackage (a one-command installer, the `stringjax` command, and the documentation
hub). **Physics contributions belong in the member repositories** — JAXVacua,
JAXPolyLog, StringForge.

## Where should my contribution go?

- A new feature, bug fix, or docs for the flux-vacuum engine → **JAXVacua**.
- Polylogarithm functionality → **JAXPolyLog**.
- Datasets / data access → **StringForge**.
- Installation, the `stringjax` command, the ecosystem hub, or version pins → here.

## Development setup

```bash
git clone https://github.com/StringJAX/stringjax
cd stringjax
python -m pip install -e ".[dev,docs]"
```

If you only need to work on the metapackage itself (the `stringjax` command, the
hub, the version pins) and the member packages are not yet published, install
without dependencies and add the tooling directly:

```bash
python -m pip install -e . --no-deps
python -m pip install ruff mypy pytest
python -m pip install -r documentation/requirements.txt   # only if building the docs
```

## Checks to run before opening a pull request

```bash
ruff check .            # lint
ruff format --check .   # formatting
mypy src/stringjax      # type checks
pytest                  # tests (if present)
sphinx-build -W -b html documentation/source documentation/build/html   # documentation build (warnings are errors)
```

`stringjax doctor` should also run cleanly in your environment.

## Pull-request checklist

- [ ] The change is in the correct repository (member vs umbrella).
- [ ] Lint, format, and type checks pass.
- [ ] Documentation updated where relevant (including the compatibility matrix if
      dependency ranges change).
- [ ] `CHANGELOG.md` updated under *Unreleased*.
- [ ] Commits are focused and descriptively messaged.

## Versioning

The metapackage follows [Semantic Versioning](https://semver.org). Bump the version
only when a member's compatibility window changes (see
`documentation/source/compatibility.md`).
