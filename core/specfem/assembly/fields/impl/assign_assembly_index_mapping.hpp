#pragma once

#include "specfem/assembly/element_types.hpp"
#include "specfem/datatype/element_index_range.hpp"
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
 * @brief Number the dim3 GLL points of one medium with compact local indices.
 *
 * Points of the medium's elements that share a mesh point receive the same
 * index, via @p dedup_table. Indices are assigned in the chunk order the
 * solver iterates elements in.
 *
 * @param h_mesh_index_mapping Host view (nspec x ngllz x nglly x ngllx) of
 * mesh point numbers; read only
 * @param h_local_index_mapping Host view (elements.size() x ngllz x nglly x
 * ngllx) of the medium's elements only; element `ispec` is row
 * `ispec - elements.begin_index()`. Receives the row of the point in the
 * medium's field arrays, in [0, nglob)
 * @param elements The medium's contiguous element range, from
 * element_types::get_elements_on_host(medium)
 * @param dedup_table Scratch table of size >= max(h_mesh_index_mapping) + 1,
 * init to -1
 * @param nglob Output: number of unique points of the medium
 */
void assign_assembly_index_mapping(
    Kokkos::View<const int ****, Kokkos::LayoutLeft, Kokkos::HostSpace>
        h_mesh_index_mapping,
    Kokkos::View<int ****, Kokkos::LayoutLeft, Kokkos::HostSpace>
        h_local_index_mapping,
    const specfem::datatype::ElementIndexRange &elements,
    Kokkos::View<int *, Kokkos::LayoutLeft, Kokkos::HostSpace> dedup_table,
    int &nglob);

} // namespace specfem::assembly::fields_impl
