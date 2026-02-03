.. _assembly_compute_source_array_tensor:

Tensor Source Implementation
============================

2D Tensor Sources
-----------------

.. doxygenfunction:: specfem::assembly::compute_source_array_impl::from_tensor(const specfem::sources::tensor_source< specfem::element::dimension_tag::dim2 > &source, const specfem::assembly::mesh< specfem::element::dimension_tag::dim2 > &mesh, const specfem::assembly::jacobian_matrix< specfem::element::dimension_tag::dim2 > &jacobian_matrix, Kokkos::View< type_real ***, Kokkos::LayoutRight, Kokkos::HostSpace > source_array)

.. doxygenfunction:: specfem::assembly::compute_source_array_impl::compute_source_array_from_tensor_and_element_jacobian(const specfem::sources::tensor_source< specfem::element::dimension_tag::dim2 > &tensor_source, const JacobianViewType2D &element_jacobian_matrix, const specfem::assembly::mesh_impl::quadrature< specfem::element::dimension_tag::dim2 > &quadrature, Kokkos::View< type_real ***, Kokkos::LayoutRight, Kokkos::HostSpace > source_array)

3D Tensor Sources
-----------------

.. doxygenfunction:: specfem::assembly::compute_source_array_impl::from_tensor(const specfem::sources::tensor_source< specfem::element::dimension_tag::dim3 > &source, const specfem::assembly::mesh< specfem::element::dimension_tag::dim3 > &mesh, const specfem::assembly::jacobian_matrix< specfem::element::dimension_tag::dim3 > &jacobian_matrix, Kokkos::View< type_real ****, Kokkos::LayoutRight, Kokkos::HostSpace > source_array)

.. doxygenfunction:: specfem::assembly::compute_source_array_impl::compute_source_array_from_tensor_and_element_jacobian(const specfem::sources::tensor_source< specfem::element::dimension_tag::dim3 > &tensor_source, const JacobianViewType3D &element_jacobian_matrix, const specfem::assembly::mesh_impl::quadrature< specfem::element::dimension_tag::dim3 > &quadrature, Kokkos::View< type_real ****, Kokkos::LayoutRight, Kokkos::HostSpace > source_array)
