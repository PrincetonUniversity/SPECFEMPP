
#pragma once

#include "kokkos_abstractions.h"
#include "specfem/assembly/jacobian_matrix.hpp"
#include "specfem/assembly/mesh.hpp"
#include "specfem/point.hpp"
#include "specfem/source.hpp"

namespace specfem::assembly::compute_source_array_impl {

/**
 * @brief Compute the source array from a vector source.
 *
 * @tparam DimensionTag
 * @param source
 * @param source_array
 */
template <specfem::dimension::type DimensionTag>
void from_vector(const specfem::sources::vector_source<DimensionTag> &source,
                 specfem::kokkos::HostView3d<type_real> source_array);

/** * @brief Compute the source array from a tensor source.
 * @tparam DimensionTag
 * @param source
 * @param source_array
 */
template <specfem::dimension::type DimensionTag>
void from_tensor(
    const specfem::sources::tensor_source<DimensionTag> &source,
    const specfem::assembly::mesh<DimensionTag> &mesh,
    const specfem::assembly::jacobian_matrix<DimensionTag> &jacobian_matrix,
    specfem::kokkos::HostView3d<type_real> source_array);

// Testable helper function for tensor sources with pre-computed jacobian matrix
void compute_source_array_from_tensor_and_element_jacobian(
    const specfem::sources::tensor_source<specfem::dimension::type::dim2>
        &tensor_source,
    const specfem::kokkos::HostView2d<specfem::point::jacobian_matrix<
        specfem::dimension::type::dim2, false, false> >
        &element_jacobian_matrix,
    specfem::kokkos::HostView3d<type_real> source_array);

} // namespace specfem::assembly::compute_source_array_impl

// Dim2 implementations
#include "specfem/assembly/compute_source_array/dim2/impl/compute_source_array_from_tensor.hpp"
#include "specfem/assembly/compute_source_array/dim2/impl/compute_source_array_from_vector.hpp"
