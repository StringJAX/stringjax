# Design principles

The StringJAX member packages share a small set of technical
commitments.  They are deliberate constraints that make the
ecosystem composable rather than a loose collection of independent
projects.


## Differentiability is a first-class requirement

Every primitive that maps compactification data to four-dimensional
effective-theory data is implemented as a JAX-traceable function:
period vectors, prepotentials, K&auml;hler potentials, gauge-kinetic
matrices, the flux superpotential, $F$-terms, scalar potentials,
Hessians, and mass matrices.  Higher-level routines such as ISD
sampling, Newton-type vacuum finding, and the reduced (freezer)
effective theories compose these primitives.

The consequence is that **derivatives are available everywhere by
construction**: gradients for Newton refinement and gradient flow,
Jacobians for $F$-term solving, Hessians for stability analysis,
and higher-order derivatives for surrogate construction and
sensitivity analyses.

Concretely this means:

- Numerical stability is a property of the *autodiff trace*, not
  just of the primal evaluation.  Helper utilities such as
  `jaxpolylog` are designed so that arbitrary-order differentiation
  remains numerically stable in regimes where the closed-form
  expressions would lose precision.
- Performance bottlenecks shift from per-step finite differences to
  batched JIT-compiled traces.
- Surrogate construction (Taylor expansions around vacua, learned
  emulators, etc.) is a routine downstream operation rather than a
  separate codebase.


## Modular, layered architecture

The ecosystem is layered to keep the boundary between **data**,
**special functions**, **physics**, and **glue** clear.

```text
                    user code
                        |
                stringjax (glue)
                /       |       \
     jaxvacua  ----  jaxpolylog  ----  stringforge
       (physics)     (special funcs)      (data)
                        |
                       jax
```

Each package owns one layer.  This is what makes it possible to
update one layer (a new database, a new polylog method, a new vacuum
sampler) without rebuilding the others.

Specifically:

- **`stringforge`** is intentionally *solver-light*.  It does not
  evaluate physics; it serves curated geometry data and persists
  vacua.
- **`jaxvacua`** consumes data from `stringforge` and the polylog
  primitive from `jaxpolylog` but does not own data movement or
  special-function implementations.
- **`jaxpolylog`** is a pure leaf with no string-theory content.

This split has a practical pay-off: the same `stringforge` data layer
will support future member packages without modification.


## Reproducibility is a feature, not an after-thought

Compactification calculations have many degrees of freedom (basis
choices, normalisations, sign conventions, sampling seeds, version
windows of intermediate libraries).  Reproducing a published result
years later is hard if those choices are buried in private scripts.
StringJAX makes them explicit.

Concrete commitments:

- **Pinned versions.**  The umbrella `stringjax` pins compatibility
  windows for every member package; `stringjax doctor` reports the
  actual installed versions and flags drift.
- **Conventions.**  The mirror-symmetry convention used by
  `jaxvacua`, the `stringforge` catalogue convention, and the
  conifold basis-change conventions are documented and version-tagged.
- **Persistent storage.**  Vacuum solutions live in a vault
  (`stringforge.VacuaWriter`) with provenance fields: solver,
  configuration hash, seed, git SHA, wall clock, and primary-key
  geometry identifiers.
- **Citeable snapshots.**  Curated dataset and solver snapshots are
  archived with DOIs.

A search performed today on a flux-vacuum problem should be runnable,
byte-for-byte, by another researcher with no shared infrastructure.


## Data and code are coupled, not separated

A unified computational framework is also a vehicle for coupling code
to compactification data.  Curated databases of geometries and vacua
are first-class consumers of the same APIs that user code uses.

In practice:

- `stringforge.LCSDatabase(...)` returns objects in exactly the form
  `jaxvacua.FluxVacuaFinder` expects.
- Vacuum records produced by `jaxvacua` are written back into the
  same vault that the input geometry came from, so that follow-up
  studies can locate them by the same primary key.
- Notebooks and scripts that build the catalogue use the same
  conventions and primitives as user code, so that the "build" and
  "use" workflows stay aligned.


## Boundaries with `jax`, `mpmath`, `scipy`, and `cytools`

StringJAX does not re-implement what general-purpose ecosystems
already do well.  It uses them and bridges them.

- **`jax`** -- autodiff, JIT, `vmap`, hardware acceleration.
  StringJAX consumes the public JAX API and adds primitives where
  none exist (`jaxpolylog`).
- **`mpmath`** -- arbitrary-precision reference for testing and
  benchmarking.  `jaxpolylog`'s accuracy tests pin its output
  against `mpmath` to twelve digits.
- **`scipy`** -- general numerical primitives such as
  `scipy.optimize.root` are used inside `jaxvacua` as drop-in
  solvers when JAX-native solvers are not the best fit.
- **`cytools`** -- toric Calabi--Yau geometry.  StringJAX provides
  the bridges that turn `cytools` objects into pull-once-cache-forever
  ecosystem datasets.


## What this rules out

The principles above also identify things StringJAX deliberately is
*not*:

- It is not a single monolithic package.  Member packages are
  independently versioned, independently released, and independently
  citeable.
- It is not a replacement for `mpmath` or `scipy` -- those are first-class
  dependencies.
- It is not the public release vehicle for every planned sibling
  package.  Future siblings (K&auml;hler-sector, axion-sector) will
  arrive with their own DOIs and authoritative documentation.
- It is not coupled to a particular execution environment.  The same
  code paths run on CPU, GPU, and TPU, and (where supported) under
  Apple-Silicon JAX with the documented complex-arithmetic fallback.

These commitments are what make the umbrella thin enough to remain
useful as the ecosystem grows.
