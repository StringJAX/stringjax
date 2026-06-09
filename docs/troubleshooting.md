# Troubleshooting

Run `stringjax doctor` first — it reports versions, the JAX backend/devices, the
`float64` status, and Gurobi availability, with targeted suggestions.

## JAX uses the CPU even though I have a GPU
The default install is CPU-only. Install the accelerator wheel:
`pip install -U "jax[cuda12]"` (match your CUDA/cuDNN). Confirm with
`stringjax doctor` (`jax_backend` should report `gpu`).

## `float64` is disabled / results look low-precision
Enable 64-bit **before** importing JAX: `export JAXVACUA_PRECISION=float64`, or call
`jaxvacua.set_precision("float64")` at the top of your script. On Apple Metal,
`float64` may be unavailable and computations can fall back to the CPU.

## `ImportError` mentioning Gurobi
Systematic flux enumeration needs Gurobi: `pip install "jaxvacua[bounding]"` and
activate a (free academic) licence. The rest of JAXVacua works without it.

## CYTools-related errors
CYTools is installed as part of the default `stringjax` install. If imports fail,
reinstall it (`pip install -U cytools`) and consult the CYTools documentation.

## Dependency-resolution conflicts after upgrading a member
The metapackage declares compatible ranges; if `pip` reports a conflict, check
{doc}`compatibility` and upgrade `stringjax` to the version that brackets the member
release you want.

## Datasets are huge / downloading unexpectedly
StringForge fetches datasets on demand and caches them. Set `STRINGFORGE_DATA_DIR`
to control the cache location, use offline mode where available, and pin a dataset
version for reproducibility.
