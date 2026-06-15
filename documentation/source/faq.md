# FAQ

**Is `stringjax` a physics package?**
No. It is an install-only umbrella that pulls in the member packages and provides the
`stringjax` command. Do your science by importing the members directly, e.g.
`import jaxvacua as jvc`.

**Do I have to install `stringjax` to use JAXVacua?**
No. Every member package installs and runs independently
(`pip install jaxvacua`). `stringjax` is a convenience for installing the whole
ecosystem at compatible versions in one step.

**What licence applies to the ecosystem?**
The `stringjax` metapackage is released under **GPL-3.0-or-later**, matching the
licence of `jaxvacua`. Using the ecosystem subjects your work to those terms.

**How do I get GPU/TPU acceleration?**
Install the matching JAX wheel after `stringjax`; see {doc}`install`.

**My results aren't reproducible across machines.**
JIT and accelerator reductions are not guaranteed bit-identical across hardware. Fix
the backend and precision, pin versions via `constraints.txt`, and record
`stringjax doctor` output. See {doc}`compatibility`.

**Which package do I cite?**
Cite the specific package(s) you used (JAXVacua, JAXPolyLog, StringForge), not the
umbrella. BibTeX is provided in each member's documentation.
