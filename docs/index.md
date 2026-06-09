# StringJAX

**Differentiable, JAX-native tools for string compactifications.**

StringJAX is the umbrella project for a family of interoperating, independently
versioned and independently citable Python packages. This site is the *hub*: it
helps you install the ecosystem, points you to the right package, and links into
each package's own documentation. It deliberately does **not** duplicate the
member APIs. For the motivation, design principles, and long-term strategy, see
{doc}`about`.

```{admonition} New here?
Start with the {doc}`quickstart` — it runs after a single `pip install stringjax`,
with no GPU and no extra dependencies.
```

## Install

```bash
pip install stringjax            # CPU; JAXVacua engine + CYTools + bundled models
pip install "stringjax[all]"     # + databases (StringForge) and the flux-bounding extra
```

See {doc}`install` for GPU/TPU/Apple-Silicon and troubleshooting, and run
`stringjax doctor` to check your environment.

## Where do I go next?

::::{grid} 1 1 2 2
:::{grid-item-card} Find flux vacua
Build effective theories and search for vacua → **JAXVacua**.
:::
:::{grid-item-card} Differentiable polylogarithms
Instanton sums with autodiff/JIT → **JAXPolyLog**.
:::
:::{grid-item-card} Geometry & vacua datasets
Curated Calabi–Yau data and the vacua vault → **StringForge**.
:::
:::{grid-item-card} Reproduce results
Pinned environments and DOIs → {doc}`compatibility`.
:::
::::

```{toctree}
:hidden:
:caption: About the project
about
```

```{toctree}
:hidden:
:caption: Getting started
quickstart
install
ecosystem
```

```{toctree}
:hidden:
:caption: Reference
compatibility
faq
troubleshooting
citation
contributing
```
