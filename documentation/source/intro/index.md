# Introduction

**StringJAX** is a JAX-native ecosystem of Python packages for the
systematic computational study of string compactifications.  This
introduction sketches what the project is, why it exists, and how the
member packages divide the work.

The next three chapters fill in the details:

- {doc}`motivation` -- why a *differentiable, scalable* compactification
  pipeline is worth building, and what kinds of scientific questions it
  helps answer.
- {doc}`ecosystem` -- the three publicly released member packages
  (`jaxvacua`, `stringforge`, `jaxpolylog`), how they fit together, and
  where to find each one's authoritative documentation.
- {doc}`design_principles` -- the technical commitments that bind the
  ecosystem together (differentiability, modularity, reproducibility,
  data + code coupling).


## The project at a glance

String theory is one of the best-developed frameworks in which
four-dimensional effective field theories can be derived from
ultraviolet-complete data.  Concretely turning a compactification
geometry into a vacuum solution -- and a vacuum solution into
predictions -- typically requires a long chain of algebraic,
geometric, and numerical computations: periods, prepotentials,
K&auml;hler structures, flux superpotentials, $F$-term equations,
mass matrices, tadpole charges, stability conditions, ensemble scans.
Each step is well-defined.  Strung together, they form a pipeline
that has historically been split across a patchwork of bespoke
scripts and private code.

StringJAX is an attempt to standardise that pipeline.  Each step in
the chain is implemented as a JAX-traceable function, so that
gradients, Hessians, and batched evaluations are available at every
intermediate stage.  Independent member packages own their physics
content but share a common data interface, so that a search performed
in one package can be archived, reproduced, and re-analysed by
another.

```{admonition} Why "JAX"?
:class: tip
Automatic differentiation, just-in-time compilation, vectorisation,
and hardware acceleration are not merely technical conveniences for
compactification studies.  They directly address the dominant
computational cost: evaluating, differentiating, and batching the
same EFT primitives over many flux choices, initial conditions, and
geometries.  A differentiable pipeline turns moduli stabilisation,
stability analysis, ensemble scans, and machine-learning surrogates
into composable operations on the same set of functions.
```


## What is in the first release

The first public member packages are:

::::{grid} 1 1 3 3
:gutter: 2

:::{grid-item-card} `jaxvacua`
The Type IIB flux-vacuum engine: periods and special geometry, the
flux effective theory, vacuum finding, flux bounding, conifold and
coni-LCS limits, and reduced (freezer) effective theories.
:::

:::{grid-item-card} `stringforge`
The data layer: curated Calabi-Yau geometry databases (`TDF`,
`CICY`, `KKLT`) and the persistent vacua vault, served from
HuggingFace and cached locally.
:::

:::{grid-item-card} `jaxpolylog`
Differentiable polylogarithms in JAX, used by `jaxvacua` to resum
worldsheet-instanton contributions to the prepotential.  Designed to
remain numerically stable through arbitrary-order automatic
differentiation.
:::

::::

These three packages can be installed and used independently.
StringJAX is the umbrella that pins their version windows and routes
new users to the right entry point.


## Where to go next

- {doc}`motivation` -- the longer "why".
- {doc}`ecosystem` -- the longer "what" and "how they fit".
- {doc}`design_principles` -- the longer "how" (technical
  commitments).
- {doc}`../tutorials` -- three executable quickstart notebooks, one
  per public member package.
