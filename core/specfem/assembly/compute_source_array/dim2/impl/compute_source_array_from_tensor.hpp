#pragma once
#include "kokkos_abstractions.h"
#include "specfem/assembly/jacobian_matrix.hpp"
#include "specfem/assembly/mesh.hpp"
#include "specfem/source.hpp"

namespace specfem::assembly::compute_source_array_impl {

template <>
void from_tensor<specfem::dimension::type::dim2>(
    const specfem::sources::tensor_source<specfem::dimension::type::dim2>
        &source,
    const specfem::assembly::mesh<specfem::dimension::type::dim2> &mesh,
    const specfem::assembly::jacobian_matrix<specfem::dimension::type::dim2>
        &jacobian_matrix,
    specfem::kokkos::HostView3d<type_real> source_array);

} // namespace specfem::assembly::compute_source_array_impl
