
.. _point_boundary:

Point Boundary
==============

Datatype used to store boundary conditions at a quadrature point.

.. doxygenstruct:: specfem::point::boundary
   :members:

Implementation Details
----------------------

.. doxygenstruct:: specfem::point::boundary< specfem::element::boundary_tag::none, DimensionTag, UseSIMD >
    :members:
    :private-members:

.. doxygenstruct:: specfem::point::boundary< specfem::element::boundary_tag::acoustic_free_surface, DimensionTag, UseSIMD >
    :members:
    :private-members:

.. doxygenstruct:: specfem::point::boundary< specfem::element::boundary_tag::stacey, DimensionTag, UseSIMD >
    :members:
    :private-members:

.. doxygenstruct:: specfem::point::boundary< specfem::element::boundary_tag::composite_stacey_dirichlet, DimensionTag, UseSIMD >
    :members:
    :private-members:
