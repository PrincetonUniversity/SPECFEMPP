#pragma once

#include "enumerations/interface.hpp"
#include "specfem/mesh.hpp"
#include <Kokkos_Core.hpp>

namespace specfem::assembly::mesh_impl {

/**
 * @brief Mapping between spectral element indexing within @ref
 * specfem::mesh::mesh and @ref specfem::assembly::mesh
 *
 * We reorder the mesh to enable better memory access patterns when computing
 * forces.
 *
 * To access the mapping, use the following:
 * @code{.cpp}
 * // Mapping from compute ordering to mesh ordering
 * const int compute_index = ...;
 * int mesh_index = mapping.compute_to_mesh(compute_index);
 * assert(mapping.mesh_to_compute(mesh_index) == compute_index);
 * @endcode
 *
 */
template <> struct mesh_to_compute_mapping<specfem::dimension::type::dim2> {
  constexpr static auto dimension_tag =
      specfem::dimension::type::dim2; ///< Dimension
  int nspec;                          ///< Number of spectral elements

  using ViewType = Kokkos::View<int *, Kokkos::LayoutLeft,
                                Kokkos::DefaultHostExecutionSpace>;
  ViewType compute_to_mesh; ///< Mapping from compute
                            ///< ordering to mesh
                            ///< ordering
  ViewType mesh_to_compute; ///< Mapping from mesh
                            ///< ordering to compute
                            ///< ordering

  /**
   * @brief Construct a new mesh to compute mapping object
   *
   */
  mesh_to_compute_mapping() = default;

  /**
   * @brief Construct a new mesh to compute mapping object
   *
   * @param tags Tags for every spectral element within the mesh
   */
  mesh_to_compute_mapping(const specfem::mesh::tags<dimension_tag> &tags);
};
} // namespace specfem::assembly::mesh_impl
