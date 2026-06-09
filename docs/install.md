# Installation

```bash
pip install stringjax            # CPU; JAXVacua engine + CYTools + bundled models
pip install "stringjax[all]"     # + databases (StringForge) and the flux-bounding extra
```

`stringjax` is an install-only metapackage: it pulls in a mutually-compatible set of
member packages and provides the `stringjax` command. Import the member packages
directly (`import jaxvacua as jvc`).

## Choosing the right JAX build

JAX is the dependency most often misconfigured. The default install gives the **CPU**
build, which works everywhere. For accelerators, install the matching JAX wheel
*after* `stringjax`:

````{tab-set}
```{tab-item} CPU (default)
pip install stringjax
```
```{tab-item} NVIDIA GPU (CUDA 12)
pip install stringjax
pip install -U "jax[cuda12]"
```
```{tab-item} Google TPU
pip install stringjax
pip install -U "jax[tpu]" -f https://storage.googleapis.com/jax-releases/libtpu_releases.html
```
```{tab-item} Apple Silicon (Metal)
pip install stringjax
pip install jax-metal   # experimental; see the float64 caveat below
```
````

## 64-bit precision

Flux-vacuum searches require 64-bit arithmetic. Enable it **before** JAX is imported,
either programmatically

```python
import jaxvacua as jvc
jvc.set_precision("float64")
```

or via the environment:

```bash
export JAXVACUA_PRECISION=float64
```

```{warning}
Some accelerators — notably Apple Metal — have limited or slow `float64` support and
may fall back to the CPU. Run `stringjax doctor` to confirm whether `float64` is
actually enabled on your device.
```

## Optional features

- **Flux bounding (Gurobi).** Systematic flux enumeration uses Gurobi, which needs a
  (free, for academics) licence. Install with `pip install "jaxvacua[bounding]"` and
  activate the licence; everything else works without it.
- **Databases & vacua vault (StringForge).** Included via `stringjax[all]`, or
  `pip install stringforge`. Large datasets are fetched on demand and cached locally;
  set `STRINGFORGE_DATA_DIR` to relocate the cache and pin dataset versions for
  reproducibility.

## Reproducible environments

Pinned specifications are shipped in the repository:

```bash
pip install -r requirements-cpu.txt          # or requirements-gpu.txt
pip install -r requirements-cpu.txt -c constraints.txt   # exact, pinned versions
# or, with conda/mamba:
mamba env create -f environment-cpu.yml
```

## Verify

```bash
stringjax doctor
```

prints the member versions, the active JAX backend and devices, whether `float64` is
enabled, and whether Gurobi is available, with suggestions when something is off.
