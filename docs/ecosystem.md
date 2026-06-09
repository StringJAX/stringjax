# The StringJAX ecosystem

StringJAX is a **project and brand**, not a single physics package. The code lives
in independent, separately versioned and separately citable repositories that
interoperate through well-defined, version-pinned interfaces.

::::{grid} 1 1 1 1
:::{grid-item-card} JAXVacua — Type IIB flux-vacuum engine
The package described in the release paper: periods and special geometry, the flux
effective theory, vacuum finding (ISD sampling and Newton-type solvers), flux
bounding, conifold / coni-LCS limits, and reduced (freezer) effective theories.

`pip install jaxvacua` · docs: TODO · DOI: TODO
:::
:::{grid-item-card} JAXPolyLog — differentiable polylogarithms
JAX-compatible polylogarithms with autodiff and JIT support, used to resum the
worldsheet-instanton corrections in the periods. A dependency of JAXVacua.

`pip install jaxpolylog` · docs: TODO · DOI: TODO
:::
:::{grid-item-card} StringForge — databases & vacua vault
The data-infrastructure layer: curated Calabi–Yau geometry databases and a
persistent vault of flux vacua, served from HuggingFace and cached locally.

`pip install stringforge` · docs: TODO · DOI: TODO
:::
::::

## How they fit together

```text
            jaxpolylog                stringforge
                |                          |
                v                          v
  geometry --> JAXVacua (periods -> css -> FluxEFT -> FluxVacuaFinder) --> vacua
```

- **JAXVacua** depends on **JAXPolyLog** (a hard dependency) and, optionally, on
  **StringForge** for large-scale data.
- **StringJAX** (this metapackage) only pins compatible versions of the above and
  provides the `stringjax` command; it is never itself a dependency of a member.

## Versioning and citation

Each package follows semantic versioning and carries its own `CITATION.cff` and
Zenodo DOI. **Cite the specific package(s) you use**, not the umbrella. See
{doc}`compatibility` for the version matrix.
