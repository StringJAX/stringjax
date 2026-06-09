# The StringJAX ecosystem

StringJAX is a **project and brand**, not a single physics package.
The code lives in independent, separately versioned, separately
citable repositories that interoperate through well-defined,
version-pinned interfaces.

This page is the orientation map.  Each member package has its own
authoritative documentation; the descriptions below link out to those.


## Public member packages

::::{grid} 1 1 1 1
:gutter: 2

:::{grid-item-card} jaxvacua &mdash; Type IIB flux-vacuum engine
:link: https://jaxvacua.readthedocs.io
:link-type: url

The package described in the StringJAX release paper.  Covers periods
and special geometry; the K&auml;hler potential, K&auml;hler metric,
and gauge-kinetic matrix; the flux superpotential and its covariant
derivatives; vacuum finding through ISD sampling, linearised ISD
optimisation, and Newton-type solvers; flux bounding; conifold and
coni-LCS limits; and reduced (freezer) effective theories.

`pip install jaxvacua` &nbsp;&middot;&nbsp;
[GitHub](https://github.com/AndreasSchachner/jaxvacua) &nbsp;&middot;&nbsp;
docs (when public)
:::

:::{grid-item-card} jaxpolylog &mdash; Differentiable polylogarithms
:link: https://jaxpolylog.readthedocs.io
:link-type: url

JAX-compatible polylogarithm functions with automatic differentiation
and JIT support, used by `jaxvacua` to resum worldsheet-instanton
contributions to the prepotential.  Designed to stay numerically
stable through arbitrary-order automatic differentiation, including
in deep large-complex-structure regimes where $|z|$ may be as small
as $10^{-30}$.

`pip install jaxpolylog` &nbsp;&middot;&nbsp;
[GitHub](https://github.com/AndreasSchachner/jaxpolylog) &nbsp;&middot;&nbsp;
[docs](https://jaxpolylog.readthedocs.io)
:::

:::{grid-item-card} stringforge &mdash; Databases and vacua vault
:link: https://github.com/AndreasSchachner/stringforge
:link-type: url

The data layer: curated Calabi--Yau geometry databases (TDF, CICY,
KKLT) served from HuggingFace and cached locally, together with the
persistent vault of flux vacua.  Owns I/O, caching, mirror-convention
bridging, model loading, and vault layout / validation / curation;
remains solver-light.

`pip install stringforge` &nbsp;&middot;&nbsp;
[GitHub](https://github.com/AndreasSchachner/stringforge) &nbsp;&middot;&nbsp;
docs (when public)
:::

::::


## How they fit together

```{mermaid}
flowchart LR
    subgraph data[Data layer]
        SF[stringforge<br/>CY databases + vacua vault]
    end
    subgraph compute[Compute layer]
        JV[jaxvacua<br/>flux-vacuum engine]
        JP[jaxpolylog<br/>differentiable polylogs]
    end
    DATASETS[(HuggingFace<br/>cy-database)]
    SF --> JV
    JP --> JV
    DATASETS --> SF
    JV --> RESULTS[Vacua + observables]
    RESULTS --> SF
```

In words:

- **`stringforge`** hosts the data the rest of the ecosystem consumes
  -- Calabi--Yau geometries plus a persistent vault of flux vacua --
  and exposes them as cached, lazily-loaded objects.
- **`jaxpolylog`** is a pure-leaf dependency.  It provides the
  polylogarithm primitive that `jaxvacua` calls to evaluate
  worldsheet-instanton corrections.
- **`jaxvacua`** is the physics engine.  It consumes geometries from
  `stringforge` and the polylog primitive from `jaxpolylog`,
  constructs the flux effective theory, solves for vacua, and writes
  the result back to the `stringforge` vault for archival and reuse.

The umbrella `stringjax` package itself does not introduce any
physics or new APIs.  It pins the compatibility window of the three
member packages and ships a thin command-line tool
(`stringjax doctor`, `stringjax versions`) for environment checks.


## Member-package boundaries

::::{grid} 1 1 2 2
:gutter: 2

:::{grid-item-card} Physics computations
Period vectors, prepotentials, K&auml;hler structures, flux $F$-terms,
scalar potentials, Hessians, mass matrices, ISD samplers, conifold
limits, freezer EFTs.

**Owned by `jaxvacua`.**
:::

:::{grid-item-card} Special functions
JAX-compatible polylogarithms with autodiff.

**Owned by `jaxpolylog`.**
:::

:::{grid-item-card} Data movement
HuggingFace download / cache, mirror-convention bridging, model
loading, vault layout, validation, curation.

**Owned by `stringforge`.**
:::

:::{grid-item-card} Compatibility
Version pins, command-line environment checks, getting-started
glue, this documentation hub.

**Owned by `stringjax`.**
:::

::::

The boundaries are deliberate.  Each member package can evolve at its
own pace, be released with its own DOI, and be cited independently.
StringJAX guarantees that a recent combination of versions tested
together is the easy default.


## Planned siblings

Future member packages are planned but not yet public.  They will be
released with their own repositories, papers, and documentation
sites, and added to the ecosystem with their own compatibility
windows.

- **K&auml;hler-sector extensions** -- moduli stabilisation for the
  K&auml;hler sector of general four-dimensional $\mathcal{N}=1$
  effective theories.
- **Axion-sector extensions** -- spectra, decay constants, and
  couplings for multi-axion effective theories.

The shape of the ecosystem (a thin umbrella, multiple physics
packages, a shared data layer, a leaf special-functions package) is
chosen so that adding such siblings is a routine extension rather
than a re-architecture.


## Where to read next

- {doc}`design_principles` -- the technical commitments shared across
  member packages (differentiability, modularity, reproducibility).
- {doc}`../tutorials` -- three executable quickstart notebooks, one
  per public member package.
- {doc}`../install` -- pinned environments and reproducibility
  guidance.
