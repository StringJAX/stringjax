Tutorials
=========

This page collects executable quickstart notebooks for the three
public member packages.  Each is short, self-contained, and runs
after a single ``pip install stringjax`` (no GPU, no external data).
They are entry points -- each member package's own documentation
covers its full surface in depth.


Choosing a path
---------------

.. grid:: 1 1 3 3
   :gutter: 2

   .. grid-item-card:: jaxvacua quickstart
      :link: notebooks/quickstart_jaxvacua
      :link-type: doc

      Build a Type IIB flux effective theory from a bundled model,
      compute the flux superpotential and its derivatives, and find
      a vacuum.

   .. grid-item-card:: stringforge quickstart
      :link: notebooks/quickstart_stringforge
      :link-type: doc

      Query the curated CY-database catalogue, load a model in
      mirror convention, and persist a small vacuum record to the
      vault.

   .. grid-item-card:: jaxpolylog quickstart
      :link: notebooks/quickstart_jaxpolylog
      :link-type: doc

      Evaluate :math:`\mathrm{Li}_s(z)` with each of the four
      ``approx`` strategies, benchmark against ``mpmath``, and
      differentiate to second order.


Notebook catalogue
------------------

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Notebook
     - Use it for
   * - :doc:`jaxvacua quickstart <notebooks/quickstart_jaxvacua>`
     - The shortest end-to-end run through the Type IIB flux-vacuum
       engine: model construction, superpotential and F-terms,
       Newton refinement.
   * - :doc:`stringforge quickstart <notebooks/quickstart_stringforge>`
     - Database catalogue queries, lazy model loading, and a vault
       round-trip.
   * - :doc:`jaxpolylog quickstart <notebooks/quickstart_jaxpolylog>`
     - The four approximation strategies, accuracy vs ``mpmath``,
       and arbitrary-order automatic differentiation.

For each member package, the full tutorial library lives in that
package's own documentation:

- jaxvacua: see the JAXVacua tutorials index (link added when the
  jaxvacua docs site is public).
- stringforge: see the stringforge tutorials index (link added when
  the stringforge docs site is public).
- jaxpolylog: see the
  `jaxpolylog documentation <https://jaxpolylog.readthedocs.io>`_,
  in particular the introduction page and the quickstart notebook.


.. toctree::
   :hidden:
   :caption: Quickstart notebooks

   notebooks/quickstart_jaxvacua
   notebooks/quickstart_stringforge
   notebooks/quickstart_jaxpolylog
