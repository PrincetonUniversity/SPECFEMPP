
.. _point_index:

Point Index
===========

Datatype used to store index of a quadrature point within the mesh.

.. doxygenstruct:: specfem::point::index
   :members:

Implementation details
----------------------

.. doxygenstruct:: specfem::point::index< specfem::dimension::type::dim2 >
    :members:


.. _point_assembly_index:

Point Assembly Index
====================

Datatype used to store index of a quadrature point within an assembled mesh.

.. doxygenstruct:: specfem::point::assembly_index
   :members:


.. _point_simd_index:

Point SIMD Index
================

Datatype used to store index of a quadrature point within the mesh when using SIMD (Single Instruction, Multiple Data) operations.

.. doxygenstruct:: specfem::point::simd_index
   :members:

Implementation details
----------------------

.. doxygenstruct:: specfem::point::simd_index< specfem::dimension::type::dim2 >
    :members:


.. _point_simd_assembly_index:

Point SIMD Assembly Index
=========================

Datatype used to store index of a quadrature point within a assembled mesh. Useful when operating on SIMD datatypes.

.. doxygenstruct:: specfem::point::simd_assembly_index
   :members:
