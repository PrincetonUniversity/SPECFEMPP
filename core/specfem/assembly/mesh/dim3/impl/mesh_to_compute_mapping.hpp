#pragma once

#include "specfem/enums.hpp"
#include "specfem/mesh.hpp"
#include <Kokkos_Core.hpp>

namespace specfem::assembly::mesh_impl {

/**
 * @brief Mapping between spectral element indexing within @ref
 * specfem::mesh::mesh and @ref specfem::assembly::mesh
 *
 * We reorder the mesh to enable better memory access patterns when computing
 * forces. Beyond grouping elements by type, within each
 * (medium, property, boundary) group the inner (partition-interior) elements
 * precede the outer (MPI-boundary) elements so each sublist is a contiguous
 * compute-index range (required by the SIMD stiffness kernels). See the
 * constructor documentation for details.
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
template <>
struct mesh_to_compute_mapping<specfem::element::dimension_tag::dim3> {
  constexpr static auto dimension_tag =
      specfem::element::dimension_tag::dim3; ///< Dimension
  int nspec;                                 ///< Number of spectral elements

  using ViewType =
      Kokkos::View<int *, Kokkos::LayoutLeft, Kokkos::DefaultExecutionSpace>;
  ViewType compute_to_mesh;                     ///< Mapping from compute
                                                ///< ordering to mesh
                                                ///< ordering
  ViewType mesh_to_compute;                     ///< Mapping from mesh
                                                ///< ordering to compute
                                                ///< ordering
  ViewType::host_mirror_type h_compute_to_mesh; ///< Host mirror for
                                                ///< compute_to_mesh
  ViewType::host_mirror_type h_mesh_to_compute; ///< Host mirror for
                                                ///< mesh_to_compute

  /**
   * @brief Construct a new mesh to compute mapping object
   *
   */
  mesh_to_compute_mapping() = default;

  /**
   * @brief Construct a new mesh to compute mapping object
   *
   * Elements are reordered so that elements sharing the same
   * (medium, property, boundary) tag combination are contiguous, and within
   * each such group the inner (partition-interior) elements precede the outer
   * (MPI-boundary) elements. This keeps each
   * (medium, property, boundary, mpi_tag) sublist a contiguous element range,
   * which is required for the contiguous SIMD lane access (`ispec + lane`) used
   * by the stiffness kernels.
   *
   * @param tags Tags for every spectral element within the mesh
   * @param adjacency_graph Mesh adjacency graph used to identify the
   *        outer (MPI-boundary) elements
   */
  mesh_to_compute_mapping(
      const specfem::mesh::tags<dimension_tag> &tags,
      const specfem::mesh::adjacency_graph<dimension_tag> &adjacency_graph);
};
} // namespace specfem::assembly::mesh_impl
