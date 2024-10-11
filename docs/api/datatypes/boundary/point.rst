
.. _point_boundary:

Point Boundary
==============

Datatype used to store boundary conditions at a quadrature point.

.. doxygenstruct:: specfem::point::boundary
   :members:

Implementation Details
----------------------

.. doxygenstruct:: specfem::point::boundary< specfem::element::boundary_tag::none, DimensionType, UseSIMD >
    :members:
    :private-members:

.. doxygenstruct:: specfem::point::boundary< specfem::element::boundary_tag::acoustic_free_surface, DimensionType, UseSIMD >
    :members:
    :private-members:

.. doxygenstruct:: specfem::point::boundary< specfem::element::boundary_tag::stacey, DimensionType, UseSIMD >
    :members:
    :private-members:

.. doxygenstruct:: specfem::point::boundary< specfem::element::boundary_tag::composite_stacey_dirichlet, DimensionType, UseSIMD >
    :members:
    :private-members:
