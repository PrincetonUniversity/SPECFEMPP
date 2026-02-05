.. _specfem_point_coordinates_local_coordinates:


``specfem::point::local_coordinates``
=====================================

.. doxygenstruct:: specfem::point::local_coordinates
   :members:

Dimension-specific Specializations
----------------------------------

.. doxygenstruct:: specfem::point::local_coordinates< specfem::element::dimension_tag::dim2 >
   :members:
   :private-members:

.. doxygenstruct:: specfem::point::local_coordinates< specfem::element::dimension_tag::dim3 >
   :members:
   :private-members:


.. _specfem_point_coordinates_global_coordinates:

``specfem::point::global_coordinates``
======================================

.. doxygenstruct:: specfem::point::global_coordinates
   :members:

Dimension-specific Specializations
----------------------------------

.. doxygenstruct:: specfem::point::global_coordinates< specfem::element::dimension_tag::dim2 >
   :members:
   :private-members:

.. doxygenstruct:: specfem::point::global_coordinates< specfem::element::dimension_tag::dim3 >
   :members:
   :private-members:

Related Functions
-----------------

.. doxygenfunction:: specfem::point::distance


The associated indices are stored in the :cpp:class:`specfem::point::index`
struct.
