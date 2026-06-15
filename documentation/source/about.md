# About StringJAX

StringJAX is an umbrella project for differentiable, JAX-native tools that study
string compactifications — from Calabi-Yau geometry through the four-dimensional
effective theory to moduli stabilisation and phenomenology. This page explains why
the project exists, the principles it is built on, and where it is heading.

## Motivation

Mapping and analysing the landscape of string vacua is computationally demanding.
In practice it has long relied on bespoke scripts, ad-hoc data formats, and
numerics that are slow and non-differentiable. That makes results hard to
reproduce, hard to scale, and hard to connect to modern optimisation and
machine-learning methods.

StringJAX takes a different starting point. By building every component on
[JAX](https://jax.readthedocs.io), the whole pipeline becomes:

- **Differentiable** — exact gradients propagate through the geometry → effective
  theory → moduli-stabilisation chain, so vacua can be *solved for* rather than
  only scanned, and sensitivities are available for free.
- **Hardware-portable** — the same code runs on CPU, GPU, and TPU; large scans
  parallelise with `vmap` and `jit` instead of hand-written loops.
- **Reproducible** — pinned environments, archival releases with DOIs, and curated
  datasets mean a published result can be regenerated from a single specification.

## Design principles

1. **Small, single-purpose packages.** Each member package does one thing well — a
   flux-vacuum engine, differentiable special functions, curated data — and is
   independently versioned and citable. You install only what you need.
2. **A thin umbrella.** The `stringjax` metapackage carries *no physics*. It is a
   one-command installer, an environment doctor, and a documentation hub. This
   keeps the brand stable while the members move quickly.
3. **Shared, differentiable interfaces.** Members exchange JAX-native objects
   (arrays and pytrees), so the output of one stage is the differentiable input of
   the next without glue code.
4. **Reproducibility as a feature.** Compatibility windows, lockable environments,
   and DOIs are first-class, not afterthoughts.

## How the pieces fit together

| Layer | Member package | Responsibility |
|-------|----------------|----------------|
| Geometry & data | **StringForge** | Calabi-Yau databases and the vacua vault |
| Special functions | **JAXPolyLog** | Differentiable polylogarithms / instanton sums |
| Effective theory & vacua | **JAXVacua** | Type IIB flux-EFT construction and vacuum search |
| Distribution | **StringJAX** | Installer, environment doctor, documentation hub |

New capabilities enter the ecosystem as additional member packages that consume
and produce the same differentiable interfaces, rather than as monolithic additions
to a single codebase.

## Long-term goals and strategy

StringJAX is released incrementally. The umbrella is deliberately thin so members
can evolve at their own pace without coordinated "big bang" releases.

- **Near term.** Stabilise the JAXVacua engine and its public API, ship archival
  releases with DOIs, and grow the StringForge data layer so that scans start from
  curated, versioned inputs.
- **Medium term.** Extend coverage of the low-energy effective theory — the Kähler
  sector together with α′ and string-loop corrections, axion physics, and uplift
  scenarios — delivered as new member packages that plug into the existing
  interfaces. Introduce a shared `stringjax-core` *only if* genuinely common code
  emerges, to avoid premature coupling.
- **Long term.** A unified differentiable environment in which large regions of the
  landscape can be sampled, optimised, and connected to phenomenology and
  machine-learning workflows, with strong interoperability and reproducibility
  guarantees across the ecosystem.

## Governance and sustainability

- **Versioning.** The umbrella follows [Semantic Versioning](https://semver.org)
  and pins members to compatible ranges; see the {doc}`compatibility` matrix. A
  member's breaking change widens the umbrella's window in a new minor release.
- **Licensing.** The `stringjax` metapackage is released under
  **GPL-3.0-or-later**, matching the licence of `jaxvacua`; downstream use of the
  ecosystem is subject to those terms. See {doc}`install`.
- **Citation.** Users cite the specific member packages they used, not the
  umbrella; see {doc}`citation`.
- **Contributions.** Physics belongs in the member repositories; the umbrella
  accepts packaging, tooling, and documentation changes. See the project's
  contributing guide.
