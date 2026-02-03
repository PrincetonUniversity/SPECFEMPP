.. _assembly_compute_source_array_vector:

Vector Source Implementation
=============================

2D Vector Sources
-----------------

.. doxygenfunction:: specfem::assembly::compute_source_array_impl::from_vector(const specfem::sources::vector_source< specfem::element::dimension_tag::dim2 > &source, Kokkos::View< type_real ***, Kokkos::LayoutRight, Kokkos::HostSpace > source_array)

3D Vector Sources
-----------------

.. doxygenfunction:: specfem::assembly::compute_source_array_impl::from_vector(const specfem::sources::vector_source< specfem::element::dimension_tag::dim3 > &source, Kokkos::View< type_real ****, Kokkos::LayoutRight, Kokkos::HostSpace > source_array)
