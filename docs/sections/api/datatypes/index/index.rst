
.. _point_index:

Point Index
===========

Datatype used to store index of a quadrature point within the mesh.

.. doxygenstruct:: specfem::point::index
   :members:

Point Index: Implementation details
-----------------------------------

.. doxygenstruct:: specfem::point::index< specfem::dimension::type::dim2, false >
    :members:


.. _point_simd_index:

Point SIMD Index
================

Datatype used to store index of a quadrature point within the mesh when using SIMD (Single Instruction, Multiple Data) operations.

.. doxygenstruct:: specfem::point::index< specfem::dimension::type::dim2, true >
    :members:


Point SIMD Index: Implementation details
----------------------------------------

.. doxygentypedef:: specfem::point::simd_index


.. _point_simd_assembly_index:


.. _point_assembly_index:


Point Assembly Index
====================

Datatype used to store index of a quadrature point within an assembled mesh.

.. doxygenstruct:: specfem::point::assembly_index
   :members:

Point Assembly Index: Implementation details
--------------------------------------------

.. doxygenstruct:: specfem::point::assembly_index< false >
    :members:


Point SIMD Assembly Index: Implementation details
-------------------------------------------------

Datatype used to store index of a quadrature point within a assembled mesh.
Useful when operating on SIMD datatypes.


.. doxygenstruct:: specfem::point::assembly_index< true >
    :members:


And note that the SIMD assembly index has the alias:

.. doxygentypedef:: specfem::point::simd_assembly_index
