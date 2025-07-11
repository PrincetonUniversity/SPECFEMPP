#pragma once

#include "compute_source_array/impl/compute_source_array.hpp"
#include "enumerations/macros.hpp"
#include "kokkos_abstractions.h"
#include "source/interface.hpp"
#include "specfem/assembly/element_types.hpp"
#include "specfem/assembly/jacobian_matrix.hpp"
#include "specfem/assembly/mesh.hpp"
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
    const specfem::assembly::jacobian_matrix &jacobian_matrix,
    const specfem::assembly::element_types &element_types,
    specfem::kokkos::HostView3d<type_real> source_array) {

  // Marker to indicate that the source array has been computed
  bool computed_source = false;

  // Loop through the function that compute the source.
  for (const auto &compute_source_array :
       compute_source_array_impl::compute_source_array_functions) {
    computed_source = compute_source_array(source, mesh, jacobian_matrix,
                                           element_types, source_array);
    if (computed_source) {
      return;
    }
  }

  KOKKOS_ABORT_WITH_LOCATION(
      "For this source type, the compute_source_array "
      "function is not implemented. Please implement "
      "it in the source class or provide a specialization "
      "for this function.");
}

} // namespace specfem::assembly
