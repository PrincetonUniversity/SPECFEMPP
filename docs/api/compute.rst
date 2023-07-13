.. _compute::

Data interface to compute forces
=================================

The interfaces provided here stores data required to compute mass and stiffness terms at elemental level. Compute struct enables easy transfer of data between host and device. Organizing compute struct into smaller structs allows us to a pass these structs to host and device functions and eliminate the need for global arrays. This improves readability and maintainability.

.. doxygenfile:: compute.hpp
    :project: SPECFEM KOKKOS IMPLEMENTATION

.. doxygenfile:: compute_partial_derivatives.hpp
    :project: SPECFEM KOKKOS IMPLEMENTATION

.. doxygenfile:: compute_properties.hpp
    :project: SPECFEM KOKKOS IMPLEMENTATION

.. doxygenfile:: compute_sources.hpp
    :project: SPECFEM KOKKOS IMPLEMENTATION

.. doxygenfile:: compute_receivers.hpp
    :project: SPECFEM KOKKOS IMPLEMENTATION
