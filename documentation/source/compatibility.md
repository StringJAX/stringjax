# Compatibility matrix

The `stringjax` metapackage installs a mutually-tested set of member packages. It
uses **compatible version ranges** (e.g. `>=0.1,<0.2`) so that *patch* releases of a
member require no new `stringjax` release; a new `stringjax` is published only when a
member's compatibility window changes.

| `stringjax` | `jaxvacua` | `jaxpolylog` | `stringforge` | Python |
|-------------|------------|--------------|---------------|--------|
| 0.1.x       | >=0.1,<0.2 | >=0.3,<0.4   | >=0.1,<0.2    | >=3.12 |

```{note}
`jaxpolylog` and `cytools` are installed transitively through `jaxvacua`.
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
