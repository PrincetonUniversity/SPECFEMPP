
.. _Chapter5:

Chapter 5: SPECFEM++ data-types
===============================

We briefly discussed the concept of data-types in :ref:`Chapter4` - where we defined a point datatype to store spatial derivates at the quadrature point ``point_partial_derivatives``. This chapter lists all the data-types that are used in SPECFEM++.

Index Types
-----------

Indices are used to reference a quadrature point within the mesh. Naturally, a quadrature point can be referenced in 2 ways - 1. by referencing its through its element ``(ispec, iz, ix)`` or 2. by referencing it directly within an assembled mesh ``(iglob)``. For each of these cases, we also define their SIMD (single instructions multiple data) counterparts that refer to multiple quadrature points at once - 1. SIMD index and 2. SIMD assembly index. SIMD indices enable us to perform vectorized operations on multiple quadrature points at once.

1. Index :cpp:class:`specfem::point::index`
2. SIMD index :cpp:type:`specfem::point::simd_index`
3. Assembly index :cpp:class:`specfem::point::assembly_index`
4. SIMD assembly index :cpp:type:`specfem::point::simd_assembly_index`

Elementary Data Types
---------------------

We begin with defining elementary datatype. Since SPECFEM++ is an extension of a finite element method, we would naturally want to define basic data types that resemble the mesh heirarchy i.e. quadrature points, elements and chunks of elements.

.. admonition:: Feature request
    :class: hint

    We need the following elementary data types to be defined in SPECFEM++:

    1. Edge element (datatype for storing values at quadrature points on a 1D edge)
    2. Chunk edge element (datatype for storing values at quadrature points on a chunk of 1D edges)

    If you'd like to work on this, please see `issue tracker <https://github.com/PrincetonUniversity/SPECFEMPP/issues/111>`_ for more details.

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
4. :ref:`Misfit Kernels <point_kernels>`
5. :ref:`Derivatives of wavefield <point_field_derivatives>`
6. :ref:`Boundary Conditions <specfem_point_boundary>`
7. Global Coordinates :cpp:class:`specfem::point::global_coordinates`
8. Local Coordinates :cpp:class:`specfem::point::local_coordinates`
9. :ref:`Stress Integrands <point_stress_integrands>`

Element Data Types
^^^^^^^^^^^^^^^^^^

1. :ref:`Integration Quadrature <element_quadrature>`

Chunk Element Data Types
^^^^^^^^^^^^^^^^^^^^^^^^

1. :ref:`Wavefield <chunk_element_field>`
2. :ref:`Stress Integrand <chunk_element_stress_integrand>`
