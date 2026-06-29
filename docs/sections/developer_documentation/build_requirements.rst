Build system requirements
=========================

This section covers compilation checks required so that SPECFEM++ is able to run
across all architectures.

.. note::

    This section is still under development.

Optional dependencies
----------------------

* **Trilinos** -- enables implicit (matrix-assembled) solvers. See
  :ref:`trilinos_configuration` for how to build SPECFEM++ against a shared
  Trilinos install. Trilinos must share the same Kokkos as SPECFEM++.
