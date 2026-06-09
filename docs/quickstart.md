# Quickstart

This example runs after a single `pip install stringjax` — **no GPU, no Gurobi, no
manual data download** — using a model bundled with JAXVacua.

```python
import jaxvacua as jvc

jvc.set_precision("float64")                       # enable 64-bit before heavy compute

# A bundled two-parameter model at large complex structure.
model  = jvc.FluxEFT(h12=2, model_ID=1)

# Set up the vacuum finder and sample supersymmetric flux vacua.
finder = jvc.FluxVacuaFinder.from_model(model)
vacua  = finder.sample_SUSY_flux_vacua(N=100, mode="ISD")

print(len(vacua), "candidate vacua")
```

Check that JAX, precision, and the member packages are set up as you expect:

```bash
stringjax doctor
```

Next steps:

- {doc}`install` — GPU/TPU/Apple-Silicon and the optional extras.
- The **JAXVacua** tutorials — building models from CYTools, the coni-LCS limit,
  ISD sampling, flux bounding, and landscape statistics.
