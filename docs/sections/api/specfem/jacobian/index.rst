.. _specfem_jacobian:


``specfem::jacobian``
=====================

.. doxygennamespace:: specfem::jacobian
    :desc-only:

``specfem::jacobian::compute_locations``
----------------------------------------

2D Overload
^^^^^^^^^^^

.. doxygenfunction:: specfem::jacobian::compute_locations(const Kokkos::View<specfem::point::global_coordinates<specfem::element::dimension_tag::dim2> *, Kokkos::HostSpace> &coorg, const int ngnod, const type_real xi, const type_real gamma)

3D Overload
^^^^^^^^^^^

.. doxygenfunction:: specfem::jacobian::compute_locations(const Kokkos::View<specfem::point::global_coordinates<specfem::element::dimension_tag::dim3> *, Kokkos::HostSpace> &coorg, const int ngnod, const type_real xi, const type_real eta, const type_real gamma)


``specfem::jacobian::compute_jacobian``
---------------------------------------

2D Overloads
^^^^^^^^^^^^

.. doxygenfunction:: specfem::jacobian::compute_jacobian(const Kokkos::View<specfem::point::global_coordinates<specfem::element::dimension_tag::dim2> *, Kokkos::HostSpace> &coorg, const int ngnod, const type_real xi, const type_real gamma)

.. doxygenfunction:: specfem::jacobian::compute_jacobian(const Kokkos::View<specfem::point::global_coordinates<specfem::element::dimension_tag::dim2> *, Kokkos::HostSpace> &coorg, const int ngnod, const std::vector<std::vector<type_real> > &dershape2D)

3D Overloads
^^^^^^^^^^^^

.. doxygenfunction:: specfem::jacobian::compute_jacobian(const Kokkos::View<specfem::point::global_coordinates<specfem::element::dimension_tag::dim3> *, Kokkos::HostSpace> &coorg, const int ngnod, const type_real xi, const type_real eta, const type_real gamma)

.. doxygenfunction:: specfem::jacobian::compute_jacobian(const CoordinateView coordinates, const ShapeDerivativesView shape_derivatives)
