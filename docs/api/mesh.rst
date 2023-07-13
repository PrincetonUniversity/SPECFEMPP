.. _mesh_interface:

Mesh interface
==============

The mesh interface is used to store the mesh information created using meshfem. The mesh struct consists of several logical smaller structs shown below. Having smaller struct allows us to write intuitive databases and database readers while keeping backward compatibility with the Fortran code.

.. note::
    Currently, we only support internal mesher inside `SPECFEM2D <https://specfem2d.readthedocs.io/en/latest/>`_ code. The database should be a binary file created using the internal mesher ``MESHFEM2D``.

.. doxygenfile:: mesh.hpp
   :project: SPECFEM KOKKOS IMPLEMENTATION

.. doxygenfile:: include/mesh/properties/properties.hpp
    :project: SPECFEM KOKKOS IMPLEMENTATION

.. doxygenfile:: material_indic.hpp
    :project: SPECFEM KOKKOS IMPLEMENTATION

.. doxygenfile:: boundaries/boundaries.hpp
    :project: SPECFEM KOKKOS IMPLEMENTATION

.. doxygenfile:: boundaries/absorbing_boundaries.hpp
    :project: SPECFEM KOKKOS IMPLEMENTATION

.. doxygenfile:: boundaries/forcing_boundaries.hpp
    :project: SPECFEM KOKKOS IMPLEMENTATION

.. doxygenfile:: elements/elements.hpp
    :project: SPECFEM KOKKOS IMPLEMENTATION

.. doxygenfile:: elements/axial_elements.hpp
    :project: SPECFEM KOKKOS IMPLEMENTATION

.. doxygenfile:: elements/tangential_elements.hpp
    :project: SPECFEM KOKKOS IMPLEMENTATION

.. doxygenfile:: surfaces.hpp
    :project: SPECFEM KOKKOS IMPLEMENTATION

.. doxygenfile:: acoustic_free_surface.hpp
    :project: SPECFEM KOKKOS IMPLEMENTATION

.. doxygenfile:: read_mesh_database.hpp
   :project: SPECFEM KOKKOS IMPLEMENTATION

.. doxygenfile:: read_material_properties.hpp
   :project: SPECFEM KOKKOS IMPLEMENTATION
