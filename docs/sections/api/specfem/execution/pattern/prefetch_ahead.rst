.. _specfem_api_execution_pattern_prefetch_ahead:

``specfem::execution::prefetch_ahead``
======================================

Lookahead prefetch decorator for cache-efficient iteration.

Overview
--------

The prefetch_ahead function decorates chunk indices with prefetch operations
that fire K iterations ahead of the current computation. This enables
cache-resident data by the time it's needed.

Type Traits
^^^^^^^^^^^

.. doxygenstruct:: specfem::execution::has_base_lookup
    :members:

Iterator Decorator
^^^^^^^^^^^^^^^^^^

.. doxygenclass:: specfem::execution::PrefetchAheadIterator
    :members:

Chunk Index Wrapper
^^^^^^^^^^^^^^^^^^^

.. doxygenclass:: specfem::execution::PrefetchChunkIndex
    :members:

Main Function
^^^^^^^^^^^^^

.. doxygenfunction:: specfem::execution::prefetch_ahead
