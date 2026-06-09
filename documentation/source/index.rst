StringJAX -- Differentiable string compactifications in JAX
============================================================

**StringJAX** is the umbrella project for a family of interoperating,
JAX-native Python packages for the systematic study of string
compactifications.  It is not itself a physics package: it is the
common entry point that installs the member packages, fixes their
pinned compatibility windows, and routes you to the right module.

The guiding principle is that the map from compactification data to
four-dimensional effective field theory should be implemented as a
**differentiable, vectorisable, just-in-time-compilable, and
reproducible** computational pipeline.  Modern automatic
differentiation and accelerated linear algebra turn what used to be a
chain of bespoke, hand-coded calculations into composable primitives
that can be re-used across vacuum searches, stability analyses,
ensemble scans, and machine-learning workflows.

The documentation is organised by how users typically approach the
project: first the *motivation and ecosystem*, then *executable
quickstart notebooks*, then the *member-package documentation* for
detailed APIs.


How to navigate
---------------

.. grid:: 1 1 2 2
   :gutter: 2

   .. grid-item-card:: New to the project
      :link: intro/index
      :link-type: doc

      Start with the introduction.  It explains why a differentiable
      compactification pipeline is worth building, how the member
      packages divide the work, and the design principles they share.

   .. grid-item-card:: New to the code
      :link: tutorials
      :link-type: doc

      Use the tutorial catalogue.  Three minimal quickstart notebooks
      cover the three public member packages -- one each for
      ``jaxvacua``, ``stringforge``, and ``jaxpolylog``.

   .. grid-item-card:: Looking for a member package
      :link: intro/ecosystem
      :link-type: doc

      The ecosystem page links into each member package's authoritative
      documentation and explains how they fit together.

   .. grid-item-card:: Looking for installation help
      :link: install
      :link-type: doc

      The install guide covers CPU / GPU / TPU / Apple-Silicon
      configurations, the ``stringjax[all]`` extras, and the
      ``stringjax doctor`` self-check.


Recommended first path
----------------------

For a first pass through the documentation, read:

1. :doc:`Introduction <intro/index>` for the conceptual map and the
   motivation for a differentiable compactification framework.
2. :doc:`Tutorials <tutorials>` for executable per-package
   quickstart notebooks.
3. The :doc:`ecosystem page <intro/ecosystem>` once you need precise
   per-package APIs and want to link out to the member docs.

The shortest route to a working example is the
:doc:`jaxvacua quickstart notebook <notebooks/quickstart_jaxvacua>`.


Install
-------

.. code-block:: bash

   pip install stringjax            # CPU; JAXVacua engine + CYTools + bundled models
   pip install "stringjax[all]"     # + StringForge databases + the flux-bounding extra

See :doc:`install` for GPU/TPU/Apple-Silicon configurations and run
``stringjax doctor`` to check your environment.


Citing StringJAX
----------------

If you find this project useful, please cite the **member package(s)
you used** rather than the umbrella.  See :doc:`citation` for BibTeX
entries.


.. toctree::
   :hidden:
   :caption: About the project

   intro/index
   intro/motivation
   intro/ecosystem
   intro/design_principles
   about

.. toctree::
   :hidden:
   :caption: Getting started

   install
   tutorials
   compatibility

.. toctree::
   :hidden:
   :caption: Reference

   faq
   troubleshooting
   citation
   contributing
