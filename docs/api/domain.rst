.. _domain::

Domain interface
=================

This module contains the interface to the domain class. This is specialized templated class that is utilized to store fields and derivatives of fields on the mesh and to compute the interaction of sources and stiffness matrix on those fields.

.. doxygenfile:: include/domain/domain.hpp
    :project: SPECFEM KOKKOS IMPLEMENTATION

.. doxygenfile:: include/domain/elastic/elastic_domain.hpp
    :project: SPECFEM KOKKOS IMPLEMENTATION

Elements
========

.. doxygenfile:: include/domain/impl/elements/element.hpp
    :project: SPECFEM KOKKOS IMPLEMENTATION

.. doxygenfile:: include/domain/impl/elements/elastic/elastic2d.hpp
    :project: SPECFEM KOKKOS IMPLEMENTATION

.. doxygenfile:: include/domain/impl/elements/elastic/elastic2d_isotropic.hpp
    :project: SPECFEM KOKKOS IMPLEMENTATION

.. doxygenfile:: include/domain/impl/elements/container.hpp
    :project: SPECFEM KOKKOS IMPLEMENTATION

Elemental sources
=================

.. doxygenfile:: include/domain/impl/sources/source.hpp
    :project: SPECFEM KOKKOS IMPLEMENTATION

.. doxygenfile:: include/domain/impl/sources/elastic/elastic2d.hpp
    :project: SPECFEM KOKKOS IMPLEMENTATION

.. doxygenfile:: include/domain/impl/sources/elastic/elastic2d_isotropic.hpp
    :project: SPECFEM KOKKOS IMPLEMENTATION

.. doxygenfile:: include/domain/impl/sources/container.hpp
    :project: SPECFEM KOKKOS IMPLEMENTATION
