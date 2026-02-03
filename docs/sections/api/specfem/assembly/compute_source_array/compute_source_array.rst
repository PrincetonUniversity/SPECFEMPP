.. _assembly_compute_source_array:

``specfem::assembly::compute_source_array``
===========================================

2D Source Array Computation
---------------------------

.. doxygenfunction:: specfem::assembly::compute_source_array(const std::shared_ptr< specfem::sources::source< specfem::element::dimension_tag::dim2 > > &source, const specfem::assembly::mesh< specfem::element::dimension_tag::dim2 > &mesh, const specfem::assembly::jacobian_matrix< specfem::element::dimension_tag::dim2 > &jacobian_matrix, SourceArrayViewType &source_array)

3D Source Array Computation
---------------------------

.. doxygenfunction:: specfem::assembly::compute_source_array(const std::shared_ptr< specfem::sources::source< specfem::element::dimension_tag::dim3 > > &source, const specfem::assembly::mesh< specfem::element::dimension_tag::dim3 > &mesh, const specfem::assembly::jacobian_matrix< specfem::element::dimension_tag::dim3 > &jacobian_matrix, SourceArrayViewType &source_array)

Implementation Details
----------------------

.. toctree::
   :maxdepth: 1

   vector_sources
   tensor_sources
