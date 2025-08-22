#pragma once

#include "enumerations/macros.hpp"
#include "kokkos_abstractions.h"
#include "specfem/assembly/element_types.hpp"
#include "specfem/assembly/jacobian_matrix.hpp"
#include "specfem/assembly/mesh.hpp"
#include "specfem/source.hpp"
#include "specfem_setup.hpp"
#include <Kokkos_Core.hpp>

namespace specfem::assembly {

/**
 * @brief Compute the lagrange interpolants for a specific 2d source location in
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
template <typename SourceArrayViewType>
void compute_source_array(
    const std::shared_ptr<
        specfem::sources::source<specfem::dimension::type::dim2> > &source,
    const specfem::assembly::mesh<specfem::dimension::type::dim2> &mesh,
    const specfem::assembly::jacobian_matrix<specfem::dimension::type::dim2>
        &jacobian_matrix,
    SourceArrayViewType &source_array);

/**
 * @brief Compute the lagrange interpolants for a specific 3d source location
 * in a 3d element.
 *
 * This is a helper function that dispatches the computation to the appropriate
 * implementation based on the source type.
 *
 * @param source The source for which the source array is computed.
 * @param mesh The mesh on which the source is defined.
 * @param jacobian_matrix The Jacobian matrix for the mesh.
 * @param source_array The output source array to be filled.
 */
template <typename SourceArrayViewType>
void compute_source_array(
    const std::shared_ptr<
        specfem::sources::source<specfem::dimension::type::dim3> > &source,
    const specfem::assembly::mesh<specfem::dimension::type::dim3> &mesh,
    const specfem::assembly::jacobian_matrix<specfem::dimension::type::dim3>
        &jacobian_matrix,
    SourceArrayViewType &source_array);

} // namespace specfem::assembly

#include "specfem/assembly/compute_source_array/dim2/compute_source_array.tpp"
#include "specfem/assembly/compute_source_array/dim3/compute_source_array.tpp"
