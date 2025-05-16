
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
