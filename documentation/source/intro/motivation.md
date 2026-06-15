# Motivation

This chapter explains why a differentiable, JAX-native infrastructure
for compactification studies is worth building.  It covers the
*scientific* motivation -- the kinds of questions the project is meant
to make accessible -- and the *computational* motivation -- why the
underlying calculations need a unified, scalable framework.


## From single examples to model spaces

Many of the most interesting questions in string phenomenology are
not questions about a single construction.  They are questions about
**model spaces**: which four-dimensional effective theories arise
from consistent compactifications, how frequently certain structures
occur, which hierarchies are typical or rare, how low-energy
quantities are correlated with geometric data, and which apparently
admissible effective theories fail to arise in controlled
ultraviolet completions.

These are intrinsically statistical questions.  They require not
isolated examples but **large and reproducible ensembles** of
explicit models.  An infrastructure that makes the construction and
analysis of explicit compactifications systematic, modular, and
scalable is therefore not just a convenience: it changes the natural
unit of analysis.

```{admonition} A change in the mode of inquiry
:class: important
The value of a differentiable compactification framework is not only
computational speed.  It is the ability to test broad ideas against
explicit ensembles rather than a small number of canonical examples,
to make compactification studies more reproducible, more directly
comparable, and more interoperable with modern data-driven methods.
```


## Three scientific drivers

### Particle physics

String compactifications provide an enormous but highly constrained
space of four-dimensional effective theories.  Bottom-up model
building can write down many gauge sectors, spectra, couplings,
symmetry structures, and supersymmetry-breaking patterns that look
consistent from a low-energy point of view.  What is much harder to
determine is which of those effective theories actually arise from
explicit ultraviolet completions, which arise only in tuned corners,
and which are excluded altogether.  Explicit compactification studies
allow one to map the boundary between *apparently consistent*
effective theories and the subset realised in string theory.

A unified computational framework also preserves the correlations
that connect quantities often treated independently in effective
field theory: tadpole constraints, period vectors, couplings, scalar
potentials, and mass matrices are not independent inputs but
different manifestations of the same underlying ultraviolet
construction.  Asking which *combinations* of low-energy properties
are naturally compatible -- and which require substantial tuning --
becomes feasible only when the full map from compactification data
to physical output is automated.


### Cosmology

Many cosmological questions are unusually sensitive to ultraviolet
structure: axion physics, dark-matter candidates, vacuum energy, and
moduli dynamics all depend on data that string compactifications fix.
Rather than postulating an axion or a hidden sector with chosen
parameters, one can ask what distributions of axion masses, decay
constants, couplings, moduli masses, and supersymmetry-breaking
scales arise in explicit ultraviolet completions; how those
distributions depend on the compactification data; and how often
they fall in parameter ranges relevant for misalignment dark matter,
inflationary dynamics, dark radiation, or late moduli decays.

Cosmological viability is rarely about one sector in isolation.  The
same data that fix axions also influence saxion masses, heavier
moduli, reheating channels, hidden sectors, and the vacuum energy.
A computational framework that *preserves* the connections between
these sectors makes it possible to study cosmological scenarios in
settings where they are all tied together by the same ultraviolet
data.


### Machine learning and AI

The interaction between string theory and modern data-driven methods
runs in both directions.

**Machine learning for compactification studies.** String
compactifications generate high-dimensional optimisation problems,
constrained search spaces, mixed discrete-continuous data, expensive
maps from ultraviolet input to low-energy observables, and
classification problems involving stability and consistency.
Surrogate models can approximate expensive intermediate computations,
active learning can guide sampling towards promising regions, and
inverse-design methods can search for vacua with prescribed
properties.  A differentiable framework is especially well suited to
these applications because it can generate gradients, Hessians, and
exact physical labels at scale.

**Compactifications as benchmarks for AI.** String compactifications
provide structured, scientifically meaningful benchmark problems for
machine learning itself: exact symmetries, hard constraints, sparse
combinatorial input data, continuous moduli, singular limits, and
objective functions with rich geometric structure.  Because many
predictions can be checked against exact consistency conditions,
they offer a setting in which to assess whether learned systems
discover meaningful scientific structure rather than just correlations.


## The computational bottleneck

Even when the formal construction is understood, the practical route
from compactification data to vacua and observables is involved.
For a representative Type IIB flux vacuum study one must

- evaluate period vectors and prepotentials,
- construct the K&auml;hler potential and K&auml;hler metric,
- assemble the flux superpotential and its covariant derivatives,
- impose tadpole constraints,
- solve non-linear equations for the moduli,
- compute mass matrices to assess stability,

and repeat these steps over many flux choices and initial conditions.
In large examples the number of fields is substantial and the scalar
potential has a complicated landscape of critical points.  This is
the computational bottleneck that StringJAX is designed to address.

The chosen response is **JAX**.  The same primitives that evaluate a
prepotential, a $F$-term, or a mass matrix become traceable,
batched, JIT-compiled, hardware-accelerated, and -- crucially --
differentiable.  Gradients, Jacobians, and Hessians needed for
vacuum searches, Newton-type refinement, stability analysis, and
ensemble sweeps are obtained by automatic differentiation rather than
finite differences.  This is important both for numerical stability
and for performance.


## Coupling code to data

A unified computational framework is also a vehicle for **coupling
code to compactification data**.  Large datasets of compactification
geometries contain an enormous amount of information, but their
systematic use in phenomenology is difficult if every calculation has
to be reconstructed by hand.  A framework that wraps both the data
layer and the EFT-construction layer can turn such datasets into
*executable* model spaces: given a geometry, flux choice, and
effective-theory prescription, the code constructs the corresponding
quantities, searches for vacua, and produces physical data in a
reproducible way.

This is the role of the [`stringforge`](https://github.com/StringJAX/stringforge)
data layer in the ecosystem.


## Where this leads

Concretely, an ecosystem of this shape makes the following workflow
ordinary rather than exceptional:

1. Query a curated database of compactification geometries.
2. Construct the corresponding flux effective theory and solve for
   vacua, with derivatives available at every step.
3. Run an ensemble scan over many flux choices or geometries.
4. Persist the resulting vacua to a shared, citeable vault.
5. Re-load those vacua later -- in the same or a different package --
   for stability checks, ML training, or follow-up analysis.

Each of those steps is currently distributed across the three public
member packages.  StringJAX's role as the umbrella is to ensure the
steps fit together and that doing all of them in sequence is the
easy default.
