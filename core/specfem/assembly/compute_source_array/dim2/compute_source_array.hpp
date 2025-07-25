#pragma once
#include "enumerations/dimension.hpp"
#include "kokkos_abstractions.h"
#include "specfem/assembly/mesh.hpp"
#include "specfem_setup.hpp"

namespace specfem::assembly {
template <>
void compute_source_array<specfem::dimension::type::dim2>(
    const std::shared_ptr<
        specfem::sources::source<specfem::dimension::type::dim2> > &source,
    const specfem::assembly::mesh<specfem::dimension::type::dim2> &mesh,
    const specfem::assembly::jacobian_matrix<specfem::dimension::type::dim2>
        &jacobian_matrix,
    specfem::kokkos::HostView3d<type_real> source_array);

} // namespace specfem::assembly
