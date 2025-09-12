.. _assembly_sources_index:

``specfem::assembly::sources``
==============================

The assembly sources module manages source information within assembled finite element
meshes for spectral element simulations. Sources are organized by medium type and
support time-dependent computations with efficient device/host memory management.

Common Templates
----------------

.. toctree::
   :maxdepth: 1

   common/sources
   common/source_medium
   common/locate_sources

Dimension-Specific Implementations
----------------------------------

.. toctree::
   :maxdepth: 1

   dim2/index
   dim3/index
