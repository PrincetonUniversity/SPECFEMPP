.. _specfem_cache_line:

``specfem::cache_line``
=======================

Cache prefetch hints and utilities for performance optimization.

.. doxygennamespace:: specfem::cache_line
    :members:
    :content-only:

Prefetch Hint Types
^^^^^^^^^^^^^^^^^^^

.. doxygenstruct:: specfem::cache_line::Nta
    :members:

.. doxygenstruct:: specfem::cache_line::Low
    :members:

.. doxygenstruct:: specfem::cache_line::Moderate
    :members:

.. doxygenstruct:: specfem::cache_line::High
    :members:

Prefetch Function
^^^^^^^^^^^^^^^^^

.. doxygenfunction:: specfem::cache_line::prefetch
