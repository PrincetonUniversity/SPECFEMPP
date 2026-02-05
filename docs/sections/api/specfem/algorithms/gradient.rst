.. _algorithms_gradient:

``specfem::algorithms::gradient``
=================================

.. doxygengroup:: AlgorithmsGradient
    :members:
    :content-only:

Implementation Details
----------------------

.. doxygenfunction:: specfem::algorithms::impl::element_gradient(const VectorFieldType &f, const specfem::point::index<specfem::element::dimension_tag::dim2, VectorFieldType::using_simd> &local_index, const specfem::point::jacobian_matrix<specfem::element::dimension_tag::dim2, false, VectorFieldType::using_simd> &point_jacobian_matrix, const QuadratureType &lagrange_derivative, typename VectorFieldType::simd::datatype (&df_dxi)[VectorFieldType::components], typename VectorFieldType::simd::datatype (&df_dgamma)[VectorFieldType::components])

.. doxygenfunction:: specfem::algorithms::impl::element_gradient(const VectorFieldType &f, const specfem::point::index<specfem::element::dimension_tag::dim3, VectorFieldType::using_simd> &local_index, const specfem::point::jacobian_matrix<specfem::element::dimension_tag::dim3, false, VectorFieldType::using_simd> &point_jacobian_matrix, const QuadratureType &lagrange_derivative, typename VectorFieldType::simd::datatype (&df_dxi)[VectorFieldType::components], typename VectorFieldType::simd::datatype (&df_deta)[VectorFieldType::components], typename VectorFieldType::simd::datatype (&df_dgamma)[VectorFieldType::components])
