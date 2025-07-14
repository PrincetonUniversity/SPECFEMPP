
#pragma once

#include "../dim2/impl/adjoint_source_array.hpp"
#include "../dim2/impl/cosserat_force_source_array.hpp"
#include "../dim2/impl/external_source_array.hpp"
#include "../dim2/impl/force_source_array.hpp"
#include "../dim2/impl/moment_tensor_source_array.hpp"
#include "kokkos_abstractions.h"
#include "specfem/assembly/element_types.hpp"
#include "specfem/assembly/jacobian_matrix.hpp"
#include "specfem/assembly/mesh.hpp"
#include <functional>
#include <vector>

namespace specfem::assembly::compute_source_array_impl {

using SourceArrayFunctionVector = std::vector<std::function<bool(
    const std::shared_ptr<specfem::sources::source>,
    const specfem::assembly::mesh<specfem::dimension::type::dim2>,
    const specfem::assembly::jacobian_matrix<specfem::dimension::type::dim2>,
    const specfem::assembly::element_types<specfem::dimension::type::dim2>,
    specfem::kokkos::HostView3d<type_real>)> >;

inline const SourceArrayFunctionVector &get_compute_source_array_functions() {
  static const SourceArrayFunctionVector functions = {
    adjoint_source_array, cosserat_force_source_array, external_source_array,
    force_source_array, moment_tensor_source_array
  };
  return functions;
}
} // namespace specfem::assembly::compute_source_array_impl
