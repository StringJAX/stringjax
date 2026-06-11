# The StringJAX project: Compiling String Theory

<p align="center">
    <a href="https://stringjax.readthedocs.io"><img src="https://readthedocs.org/projects/stringjax/badge/?version=latest" alt="Doc"/></a>
    <a href="https://pypi.org/project/stringjax/"><img src="https://img.shields.io/pypi/v/stringjax.svg" alt="StringJAX"/></a>
    <a href="https://www.python.org"><img src="https://img.shields.io/badge/python-3.12%2B-blue.svg" alt="Python"/></a>
    <a href="https://www.gnu.org/licenses/gpl-3.0"><img src="https://img.shields.io/badge/License-GPLv3-blue.svg" alt="License: GPL v3"/></a>
</p>

<!-- The Docs/PyPI badges populate once the project is published; the Python and
     License badges are static. -->

**Differentiable tools for string compactifications in JAX.**

StringJAX is the umbrella project for a family of interoperating, separately
versioned and separately citable Python packages. This repository hosts the
lightweight `stringjax` *metapackage* (a one-command installer and an environment
helper) and the documentation hub. It contains **no physics code** — that lives in
the member packages below.

## Why StringJAX?

Constructing and analysing string-theory vacua has traditionally meant stitching
together bespoke scripts, incompatible data formats, and slow, non-differentiable
numerics. StringJAX exists to make that pipeline *fast, differentiable, and
reproducible* from end to end. Every member package is built on JAX, so the same
code runs on CPU, GPU, or TPU and exposes exact gradients through the whole
geometry → effective theory → moduli-stabilisation chain. The packages are small,
single-purpose, and independently versioned, so you install only what you need and
cite exactly what you used.

## Install

```bash
pip install stringjax            # CPU; JAXVacua engine + CYTools + bundled models
pip install "stringjax[all]"     # + databases (StringForge) and the flux-bounding extra
```

GPU/TPU/Apple-Silicon need one extra step for JAX — see the
[installation guide](https://stringjax.readthedocs.io/en/latest/install.html).

> Note: `stringjax` is permissively licensed (MIT), but using the ecosystem pulls
> in `jaxvacua`, which is **GPL-3.0**; your usage is subject to those terms.

## Packages

| Package        | Role                                | Install                     | Docs | GitHub |
|----------------|-------------------------------------|-----------------------------|------|--------|
| **JAXVacua**   | Type IIB flux-vacuum engine         | `pip install jaxvacua`      | [DOC](https://jaxvacua.readthedocs.io) | ([GitHub](https://github.com/AndreasSchachner/jaxvacua)) |
| **JAXPolyLog** | Differentiable polylogarithms       | `pip install jaxpolylog`    | [DOC](https://jaxpolylog.readthedocs.io) | ([GitHub](https://github.com/AndreasSchachner/jaxpolylog)) |
| **StringForge**| Calabi–Yau databases & vacua vault  | `pip install stringforge`   | [DOC](https://stringforge.readthedocs.io) | ([GitHub](https://github.com/AndreasSchachner/stringforge)) |

`StringJAX` itself is the project/brand and is *not* an installable physics package.

## First five minutes

```python
import jaxvacua as jvc

jvc.set_precision("float64")                       # before heavy compute
model  = jvc.FluxEFT(h12=2, model_ID=1, limit="LCS", maximum_degree=5)
finder = jvc.FluxVacuaFinder.from_model(model)
vacua  = finder.sample_SUSY_flux_vacua(N=100, mode="ISD")
print(len(vacua), "candidate vacua")
```

Check your environment at any time:

```bash
stringjax doctor      # versions, JAX backend/devices, float64 status, Gurobi availability
stringjax versions    # just the member-package versions
```

## Documentation

The documentation hub lives at <https://stringjax.readthedocs.io>. Each member
package has its own authoritative documentation; the hub orients newcomers and
links into them.

## Roadmap & long-term goals

StringJAX is released incrementally, and the umbrella is deliberately thin so that
member packages can evolve at their own pace.

- **Near term** — stabilise the JAXVacua flux-vacuum engine and its public API,
  publish archival releases with DOIs, and grow the StringForge data layer.
- **Medium term** — broaden coverage of the low-energy effective theory (the
  Kähler sector and α′/string-loop corrections, axion physics, and uplift
  scenarios) as further member packages that plug into the same differentiable
  interfaces.
- **Long term** — a unified, differentiable environment in which large regions of
  the landscape can be sampled, optimised, and connected to phenomenology and
  machine-learning workflows, with strong interoperability and reproducibility
  guarantees across the ecosystem.

The [about page](https://stringjax.readthedocs.io/en/latest/about.html) sets out the
motivation and strategy in more detail; the
[compatibility matrix](https://stringjax.readthedocs.io/en/latest/compatibility.html)
tracks current version windows.

## Citing

Please cite the **specific package(s)** you used rather than the umbrella; see
[`CITATION.cff`](CITATION.cff) and each member's citation file. BibTeX entries are
collected in the documentation.

## Contributing

See [`CONTRIBUTING.md`](CONTRIBUTING.md) and the
[code of conduct](CODE_OF_CONDUCT.md). Questions are welcome in GitHub Discussions.
