# Compatibility matrix

The `stringjax` metapackage installs a mutually-tested set of member packages. It
uses **compatible version ranges** (e.g. `>=0.1,<0.2`) so that *patch* releases of a
member require no new `stringjax` release; a new `stringjax` is published only when a
member's compatibility window changes.

| `stringjax` | `jaxvacua`   | `jaxpolylog` | `stringforge` | `stringjax-tools` | Python |
|-------------|--------------|--------------|---------------|-------------------|--------|
| 0.1.1       | >=0.1.3,<0.2 | >=0.4 (a)    | >=0.1.3,<0.2  | >=0.1 (a)         | >=3.12 |
| 0.1.0       | >=0.1,<0.2   | (a)          | >=0.1,<0.2    | (a)               | >=3.12 |

```{note}
(a) `jaxpolylog`, `stringjax-tools` and `cytools` are installed transitively through
`jaxvacua`; `stringjax` itself does not constrain them, so their ranges are those of
the installed `jaxvacua` (from `jaxvacua` 0.1.3 on: `jaxpolylog>=0.4.0`,
`stringjax-tools>=0.1.0`). The 0.1.0 documentation listed `jaxpolylog>=0.3,<0.4`, a
bound that no package enforced.
`stringforge` is installed via the `data` or `all` extras.
```

## Reproducibility

For bit-stable reproduction of a specific study, install from the pinned files in the
repository:

```bash
pip install -r requirements-cpu.txt -c constraints.txt
```

and record the output of `stringjax versions` (or `stringjax doctor`) alongside your
results.
