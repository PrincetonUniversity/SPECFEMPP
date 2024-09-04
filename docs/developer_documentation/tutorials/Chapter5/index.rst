
.. _Chapter5:

Chapter 5: SPECFEM++ data-types
===============================

We briefly discussed the concept of data-types in :ref:`Chapter4` - where we defined a point datatype to store spatial derivates at the quadrature point ``point_partial_derivatives``. This chapter lists all the data-types that are used in SPECFEM++.

Index Types
-----------

1. :ref:`Index <point_index>`
2. :ref:`SIMD index <point_simd_index>`
3. :ref:`Assembly index <point_assembly_index>`
4. :ref:`SIMD assembly index <point_simd_assembly_index>`

Elementary Data Types
---------------------

We begin with defining elementary datatype. Since SPECFEM++ is an extension of a finite element method, we would naturally want to define basic data types that resemble the mesh heirarchy i.e. quadrature points, elements and chunks of elements.

1. :ref:`Point datatype <datatype_base_point>`
2. :ref:`Element datatype <datatype_base_element>`
3. :ref:`Chunk Element datatype <datatype_base_chunk_element>`

Derived Data Types
------------------

In addition to the elementary data types, we define derived data types that are used to store specific information at quadrature points, elements and chunks of elements.

Point Data Types
^^^^^^^^^^^^^^^^

1. :ref:`Spatial Derivatives <datatype_point_partial_derivatives>`
2. :ref:`Material Properties <datatype_point_material_properties>`
3. :ref:`Wavefield <point_field>`
4. :ref:`Misfit Kernels <point_kernel>`
5. :ref:`Derivatives of wavefield <point_field_derivatives>`
6. :ref:`Boundary Conditions <point_boundary>`
7. :ref:`Global Coordinates <point_global_coordinates>`
8. :ref:`Local Coordinates <point_local_coordinates>`
9. :ref:`Stress Integrands <point_stress_integrands>`

Element Data Types
^^^^^^^^^^^^^^^^^^

1. :ref:`Integration Quadrature <element_quadrature>`

Chunk Element Data Types
^^^^^^^^^^^^^^^^^^^^^^^^

1. :ref:`Wavefield <chunk_element_field>`
2. :ref:`Stress Integrands <chunk_element_stress_integrands>`
