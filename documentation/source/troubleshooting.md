# Troubleshooting

Run `stringjax doctor` first — it reports versions, the JAX backend/devices, the
`float64` status, and Gurobi presence, with targeted suggestions.

## JAX uses the CPU even though I have a GPU
The default install is CPU-only. Install the accelerator wheel:
`pip install -U "jax[cuda12]"` (match your CUDA/cuDNN). Confirm with
`stringjax doctor` (`jax_backend` should report `gpu`).

## `float64` is disabled / results look low-precision
`jaxvacua` switches JAX to `float64` when it is imported, unless the environment
variable `JAXVACUA_PRECISION` is set to `float32`. Unset it (or set it to `float64`)
before importing `jaxvacua`, or call `jaxvacua.set_precision("float64")` at the top of
your script. On Apple Metal, `float64` may be unavailable and computations can fall
back to the CPU.

## `ImportError` or licence errors mentioning Gurobi
`gurobipy` is a core dependency of `jaxvacua`; if it is missing, reinstall `jaxvacua`.
Its pip wheel carries a size-limited licence: a model-size error means you need a full
(free, for academics) licence.

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
