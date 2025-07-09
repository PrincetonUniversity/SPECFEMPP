#pragma once
#include "kokkos_abstractions.h"
#include "source/interface.hpp"
#include "specfem/assembly/element_types.hpp"
#include "specfem/assembly/jacobian_matrix.hpp"
#include "specfem/assembly/mesh.hpp"

namespace specfem::assembly::sources_impl {

/**
 * @brief Compute the source array for a given source
 *
 * This is a helper function that computes the source array for a given
 * source. We implement it here instead of the source to remove dependency of
 * source on the assembly module.
 */
template <specfem::dimension::type DimensionTag>
void compute_source_array(
    const std::shared_ptr<specfem::sources::source> &source,
    const specfem::assembly::mesh<DimensionTag> &mesh,
    const specfem::assembly::jacobian_matrix &jacobian_matrix,
    const specfem::assembly::element_types &element_types,
    specfem::kokkos::HostView3d<type_real> source_array) {
  source->compute_source_array(mesh, jacobian_matrix, element_types,
                               source_array);
}

} // namespace specfem::assembly::sources_impl
