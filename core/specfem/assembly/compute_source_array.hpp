#pragma once

#include "enumerations/macros.hpp"
#include "kokkos_abstractions.h"
#include "specfem/assembly/element_types.hpp"
#include "specfem/assembly/jacobian_matrix.hpp"
#include "specfem/assembly/mesh.hpp"
#include "specfem/source.hpp"
#include "specfem_setup.hpp"

namespace specfem::assembly {

/**
 * @brief Compute the lagrange interpolants for a specific source location in
 * an element.
 *
 * This is a helper function that computes the source array for a given
 * source. We implement it here instead of the source to remove dependency of
 * source on the assembly module.
 *
 * @param source The source for which the source array is computed.
 * @param mesh The mesh on which the source is defined.
 * @param jacobian_matrix The Jacobian matrix for the mesh.
 * @param element_types The element types for the mesh.
 * @param source_array The output source array to be filled.
 *
 */
template <specfem::dimension::type DimensionTag>
void compute_source_array(
    const std::shared_ptr<specfem::sources::source> &source,
    const specfem::assembly::mesh<DimensionTag> &mesh,
    const specfem::assembly::jacobian_matrix<DimensionTag> &jacobian_matrix,
    specfem::kokkos::HostView3d<type_real> source_array);

} // namespace specfem::assembly

#include "compute_source_array/dim2/compute_source_array.hpp"
