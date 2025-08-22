#pragma once
#include "kokkos_abstractions.h"
#include "specfem/assembly/jacobian_matrix.hpp"
#include "specfem/assembly/mesh.hpp"
#include "specfem/source.hpp"

namespace specfem::assembly::compute_source_array_impl {

/*
using PointJacobianMatrix =
  specfem::point::jacobian_matrix<specfem::dimension::type::dim3, false,
                                  false>;
using JacobianViewType = specfem::kokkos::HostView2d<PointJacobianMatrix>;

void from_tensor(
  const specfem::sources::tensor_source<specfem::dimension::type::dim3>
      &source,
  const specfem::assembly::mesh<specfem::dimension::type::dim3> &mesh,
  const specfem::assembly::jacobian_matrix<specfem::dimension::type::dim3>
      &jacobian_matrix,
  specfem::kokkos::HostView4d<type_real> source_array);

// Implementation details for computing source array from tensor and element
// jacobian
void compute_source_array_from_tensor_and_element_jacobian(
  const specfem::sources::tensor_source<specfem::dimension::type::dim3>
      &tensor_source,
  const JacobianViewType &element_jacobian_matrix,
  const specfem::assembly::mesh_impl::quadrature<
      specfem::dimension::type::dim3> &quadrature,
  specfem::kokkos::HostView4d<type_real> source_array);
*/
} // namespace specfem::assembly::compute_source_array_impl
