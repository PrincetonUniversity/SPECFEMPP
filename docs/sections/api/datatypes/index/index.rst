Point Index
===========

Datatype used to store the global coordinates of a point within the mesh.

.. doxygenstruct:: specfem::point::index
   :members:


.. doxygentypedef:: specfem::point::simd_index


2D Specialization
-----------------

.. _point_index_2d_non_simd:

Non-SIMD
~~~~~~~~

.. doxygenstruct:: specfem::point::index< specfem::dimension::type::dim2, false >
   :members:
   :private-members:

.. _point_index_2d_simd:

SIMD
~~~~

.. doxygenstruct:: specfem::point::index< specfem::dimension::type::dim2, true >
   :members:
   :private-members:


3D Specialization
-----------------

.. _point_index_3d_non_simd:

Non-SIMD
~~~~~~~~

.. doxygenstruct:: specfem::point::index< specfem::dimension::type::dim3, false >
   :members:
   :private-members:

.. _point_index_3d_simd:

SIMD
~~~~

.. doxygenstruct:: specfem::point::index< specfem::dimension::type::dim3, true >
   :members:
   :private-members:
