#pragma once

#include "kokkos_abstractions.h"
#include "source/cosserat_force_source.hpp"
#include "specfem/assembly/element_types.hpp"
#include "specfem/assembly/jacobian_matrix.hpp"
#include "specfem/assembly/mesh.hpp"
#include "specfem_setup.hpp"

namespace specfem::assembly {

void compute_source_array(
    const std::shared_ptr<specfem::sources::cosserat_force> &source,
    const specfem::assembly::mesh<specfem::dimension::type::dim2> &mesh,
    const specfem::assembly::jacobian_matrix &jacobian_matrix,
    const specfem::assembly::element_types &element_types,
    specfem::kokkos::HostView3d<type_real> source_array);

} // namespace specfem::assembly
