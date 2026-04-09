
#pragma once

#include "specfem/element.hpp"
#include "specfem/mesh.hpp"
#include "specfem/mesh_entity.hpp"
#include "specfem/setup.hpp"
#include <Kokkos_Core.hpp>
#include <vector>

namespace specfem::assembly {

template <specfem::element::dimension_tag DimensionTag> class mpi;

namespace mpi_impl {

/// \class communication_group
/// \brief Stores face communication metadata between two neighboring MPI
/// processes.
///
/// This class manages the interface between two neighboring subdomains (MPI
/// processes) by tracking:
/// - Which faces belong to this inter-process boundary
/// - The orientation of each face in both local and neighbor elements
/// - The discrete rotation index required to align face coordinates across the
///   MPI boundary (computed from anchor points to avoid ambiguity)
///
/// Memory is organized as 1D `Kokkos::View` arrays indexed by face ID within
/// this communication group, enabling efficient GPU-accelerated synchronization
/// kernels.
///
/// \see mpi – Collects one communication_group per neighbor MPI rank
class communication_group {
public:
  constexpr static auto dimension_tag = specfem::element::dimension_tag::dim3;

  unsigned int my_rank;       ///< MPI rank of the current process
  unsigned int neighbor_rank; ///< MPI rank of the neighboring process
  unsigned int nfaces; ///< Number of faces within this MPI communication group
  unsigned int ngll;   ///< Number of GLL points per face dimension

  using FaceNormalViewType = Kokkos::View<specfem::mesh_entity::dim3::type *,
                                          Kokkos::DefaultExecutionSpace>;
  using ThetaViewType =
      Kokkos::View<unsigned char *, Kokkos::DefaultExecutionSpace>;

  FaceNormalViewType my_normal; ///< Normal orientation of each face in the
                                ///< local element (nfaces)
  FaceNormalViewType neighbor_normal; ///< Normal orientation of each face in
                                      ///< the neighboring element (nfaces)
  FaceNormalViewType::HostMirror h_my_normal; ///< Host mirror for my_normal
  FaceNormalViewType::HostMirror h_neighbor_normal; ///< Host mirror for
                                                    ///< neighbor_normal
  ThetaViewType theta; ///< Rotation index r in [0,3] for face-to-face
                       ///< connections (nfaces)
  ThetaViewType::HostMirror h_theta; ///< Host mirror for theta

  communication_group() = default;

  /// \brief Constructs a communication group for face exchange with a neighbor.
  ///
  /// Extracts interface metadata from the mesh adjacency graph and organizes it
  /// into GPU-friendly Kokkos views. For each face in the interface:
  /// - Records the face orientation in both local and neighboring elements
  /// - Computes the discrete rotation index (0–3) using anchor points to
  ///   uniquely determine the face-to-face coordinate alignment
  ///
  /// \param my_rank          MPI rank of the current subdomain
  /// \param neighbor_rank    MPI rank of the neighboring subdomain
  /// \param edges            Face connectivity data from
  /// adjacency_graph.mpi_connections()
  ///                         (already filtered to contain only face
  ///                         adjacencies)
  /// \param ngllz, nglly, ngllx  GLL points per element dimension; used to
  /// construct
  ///                         the element for extracting face corner information
  ///
  /// \throws std::runtime_error if an anchor point is not found among face
  /// corners
  ///
  /// \note The element is constructed internally from GLL parameters to extract
  ///       corner information; it is local to this constructor.
  communication_group(
      const unsigned int my_rank, const unsigned int neighbor_rank,
      const std::vector<specfem::mesh::adjacency_graph<
          specfem::element::dimension_tag::dim3>::MPIEdgeProperties> &edges,
      const int ngllz, const int nglly, const int ngllx);
};

} // namespace mpi_impl

/// \class mpi<dimension_tag::dim3>
/// \brief Manages MPI face communication patterns for 3D distributed
/// simulations.
///
/// This class analyzes the mesh adjacency graph to extract and organize all
/// face-level inter-process interfaces. It builds one `communication_group` per
/// neighboring MPI rank, each containing GPU-friendly arrays for:
/// - Face orientations in local and neighbor elements
/// - Discrete rotation indices for coordinate alignment
/// - GLL metadata for synchronization kernel dispatch
///
/// Construction filters out edge and corner adjacencies, keeping only faces
/// needed for GLL point data exchange.
///
/// The design prioritizes:
/// - **Compact storage**: Rotation indices as `unsigned char` (1 byte) instead
///   of floating-point angles (8 bytes)
/// - **GPU efficiency**: Data organized in `Kokkos::View` arrays for direct
///   kernel access
/// - **Clear semantics**: Discrete rotation values (0–3) are explicit; actual
///   angle = rotation_index × π/2
///
/// \see communication_group – Individual interface between two ranks
template <> class mpi<specfem::element::dimension_tag::dim3> {
public:
  constexpr static auto dimension_tag = specfem::element::dimension_tag::dim3;

  std::vector<mpi_impl::communication_group> communication_groups;

  mpi() = default;

  /// \brief Constructs the complete MPI face communication schedule.
  ///
  /// Performs the following steps:
  /// 1. Extracts all MPI connections from the adjacency graph
  /// 2. Filters to keep only face-level interactions (excludes edges/corners)
  /// 3. Groups connections by neighboring MPI rank
  /// 4. Creates one `communication_group` per unique neighbor
  ///
  /// \param adjacency_graph  Mesh adjacency graph containing MPI connection
  /// metadata
  ///                         (via \c mpi_connections() method)
  /// \param ngllz, nglly, ngllx  GLL points per element dimension
  ///
  /// \note The element is constructed internally from GLL parameters only to
  ///       extract mesh connectivity; it is not stored.
  mpi(const specfem::mesh::adjacency_graph<dimension_tag> &adjacency_graph,
      const int ngllz, const int nglly, const int ngllx);
};

} // namespace specfem::assembly
