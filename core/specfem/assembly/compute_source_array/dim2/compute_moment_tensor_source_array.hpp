#pragma once

#include "kokkos_abstractions.h"
#include "source/moment_tensor_source.hpp"
#include "specfem/assembly/element_types.hpp"
#include "specfem/assembly/jacobian_matrix.hpp"
#include "specfem/assembly/mesh.hpp"
#include "specfem_setup.hpp"

namespace specfem::assembly {

/**
 * @brief Compute the lagrange interpolants for a specific source location in
 * an element.
 *
 * This is a helper function that computes the source array for a given moment
 * tensor source. We implement it here instead of the source to remove
 * dependency of source on the assembly module.
 *
 * @param source The moment tensor source for which the source array is
 * computed.
 * @param mesh The mesh on which the source is defined.
 * @param jacobian_matrix The Jacobian matrix for the mesh.
 * @param element_types The element types for the mesh.
 * @param source_array The output source array to be filled.
 */
void compute_source_array(
    const std::shared_ptr<specfem::sources::moment_tensor> &source,
    const specfem::assembly::mesh<specfem::dimension::type::dim2> &mesh,
    const specfem::assembly::jacobian_matrix &jacobian_matrix,
    const specfem::assembly::element_types &element_types,
    specfem::kokkos::HostView3d<type_real> source_array);

} // namespace specfem::assembly
