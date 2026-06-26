#pragma once

#include "specfem/assembly/element_types.hpp"
#include "specfem/enums.hpp"
#include <Kokkos_Core.hpp>

namespace specfem::assembly::fields_impl {

/**
 * @brief Remap dim2 GLL DOF indices to per-medium compact indices in-place.
 *
 * Iterates over all spectral elements of medium @p MediumTag (in chunk order)
 * and writes back @p base_dof + compact_count into @p h_index_mapping for
 * every GLL point belonging to that medium. Shared boundary DOFs (same
 * original value) receive the same compact index via @p dedup_table.
 *
 * @param h_index_mapping Host view (nspec x ngllz x ngllx) to remap in-place
 * @param element_types Element classification for medium lookup
 * @param dedup_table Scratch table of size >= max(h_index_mapping), init to -1
 * @param nglob Output: number of unique DOFs found for this medium
 * @param MediumTag Medium to process
 * @param base_dof Offset added to each compact index before writing back
 */
void assign_assembly_index_mapping(
    Kokkos::View<int ***, Kokkos::LayoutLeft, Kokkos::HostSpace>
        h_index_mapping,
    const specfem::assembly::element_types<
        specfem::element::dimension_tag::dim2> &element_types,
    Kokkos::View<int *, Kokkos::LayoutLeft, Kokkos::HostSpace> dedup_table,
    int &nglob, const specfem::element::medium_tag MediumTag,
    const int base_dof);

/**
 * @brief Remap dim3 GLL DOF indices to per-medium compact indices in-place.
 *
 * @param h_index_mapping Host view (nspec x ngllz x nglly x ngllx)
 * @param element_types Element classification for medium lookup
 * @param dedup_table Scratch table of size >= max(h_index_mapping), init to -1
 * @param nglob Output: number of unique DOFs found for this medium
 * @param MediumTag Medium to process
 * @param base_dof Offset added to each compact index before writing back
 */
void assign_assembly_index_mapping(
    Kokkos::View<int ****, Kokkos::LayoutLeft, Kokkos::HostSpace>
        h_index_mapping,
    const specfem::assembly::element_types<
        specfem::element::dimension_tag::dim3> &element_types,
    Kokkos::View<int *, Kokkos::LayoutLeft, Kokkos::HostSpace> dedup_table,
    int &nglob, const specfem::element::medium_tag MediumTag,
    const int base_dof);

} // namespace specfem::assembly::fields_impl
