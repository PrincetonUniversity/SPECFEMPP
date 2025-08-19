.. _specfem_chunk_element:


``specfem::chunk_element``
==========================

.. doxygennamespace:: specfem::chunk_element
    :desc-only:

The namespace :cpp:any:`specfem::chunk_element` contains the data structures and
functions used to represent and manipulate chunks of spectral elements in the
SPECFEM++ framework. This namespace provides high-performance computing primitives
for processing multiple elements simultaneously, enabling improved cache locality,
vectorization, and computational throughput in spectral element simulations.

Chunk-based processing is essential for modern high-performance computing on both
CPU and GPU architectures, where processing elements in groups maximizes memory
bandwidth utilization and enables efficient parallel execution patterns.


.. toctree::
    :maxdepth: 1

    index/index
    field/acceleration
    field/displacement
    field/velocity
    stress_integrand/index
