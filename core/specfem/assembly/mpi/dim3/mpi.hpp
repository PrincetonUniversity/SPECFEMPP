
#pragma once

#include "specfem/assembly/mesh.hpp"
#include "specfem/element.hpp"
#include "specfem/mesh.hpp"
#include "specfem/mesh_entity.hpp"
#include "specfem/mpi.hpp"
#include "specfem/setup.hpp"
#include <Kokkos_Core.hpp>
#include <tuple>
#include <vector>

namespace specfem::assembly {

template <specfem::element::dimension_tag DimensionTag> class mpi;

namespace mpi_impl {

/**
 * @brief Base class storing common interface metadata shared by all MPI
 * communication group types (face, edge, corner).
 *
 * Captures per-connection data that is independent of the connection geometry:
 * - The mesh entity type (orientation) of each shared interface in both the
 *   local and neighboring element
 * - The spectral element indices for the local and neighboring elements
 *
 * Subclasses extend this with geometry-specific transformation metadata:
 * - `face_communication_group`: adds `theta` (discrete 2D rotation index)
 * - `edge_communication_group`: adds `reflect` (1D direction flag)
 * - `corner_communication_group`: no additional data (single GLL point)
 *
 * Memory is organized as 1D `Kokkos::View` arrays indexed by interface ID
 * within the group, enabling efficient GPU-accelerated synchronization kernels.
 *
 * @note `nfaces` counts shared interfaces regardless of geometry type (face,
 *       edge, or corner); the name is kept from the original face-only design.
 *
 * @see face_communication_group, edge_communication_group,
 *      corner_communication_group
 * @see mpi – Collects one group per connection type per neighbor MPI rank
 */
class communication_group {
public:
  constexpr static auto dimension_tag =
      specfem::element::dimension_tag::dim3; ///< Dimension tag

  unsigned int my_rank;       /**< MPI rank of the current process */
  unsigned int neighbor_rank; /**< MPI rank of the neighboring process */
  unsigned int n; /**< Number of shared interfaces in this communication group
                    (faces, edges, or corners) */
  unsigned int ngll; /**< Number of GLL points per face dimension */

  using FaceNormalViewType = Kokkos::View<specfem::mesh_entity::dim3::type *,
                                          Kokkos::DefaultExecutionSpace>;

  using ElementIndexViewType =
      Kokkos::View<int *, Kokkos::DefaultExecutionSpace>;

  FaceNormalViewType my_orientation; /**< Mesh entity type of each face in the
                                     local element (nfaces) */
  FaceNormalViewType neighbor_orientation; /**< Mesh entity type of each face in
                                           the neighboring element (nfaces) */
  FaceNormalViewType::HostMirror h_my_orientation;       /**< Host mirror for
                                                         my_orientation */
  FaceNormalViewType::HostMirror h_neighbor_orientation; /**< Host mirror for
                                                         neighbor_orientation */
  ElementIndexViewType my_element; /**< Spectral element index of the local
                                   element for each face (nfaces) */
  ElementIndexViewType neighbor_element; /**< Spectral element index of the
                                         neighboring element for each face
                                         (nfaces) */
  ElementIndexViewType::HostMirror h_my_element;       /**< Host mirror for
                                                       my_element */
  ElementIndexViewType::HostMirror h_neighbor_element; /**< Host mirror for
                                                       neighbor_element */

  communication_group() : n(0), ngll(0) {}

  /**
   * @brief Constructs the common interface metadata for a neighbor pair.
   *
   * Populates the orientation and element index arrays from the provided
   * connectivity data. Does **not** compute any transformation metadata
   * (theta or reflect); that is delegated to the derived-class constructors.
   *
   * @param my_rank          MPI rank of the current subdomain
   * @param neighbor_rank    MPI rank of the neighboring subdomain
   * @param edges            Interface connectivity data from
   *                         adjacency_graph.mpi_connections(), already
   *                         filtered to a single connection type (face, edge,
   *                         or corner) for one neighbor rank
   * @param ngllz, nglly, ngllx  GLL points per element dimension; used to
   *                         determine `ngll` via the element helper
   */
  communication_group(
      const unsigned int my_rank, const unsigned int neighbor_rank,
      const std::vector<specfem::mesh::adjacency_graph<
          specfem::element::dimension_tag::dim3>::MPIEdgeProperties> &edges,
      const int ngllz, const int nglly, const int ngllx);
};

/**
 * @brief Communication group for corner-shared MPI interfaces.
 *
 * A corner interface consists of a single shared GLL point. No transformation
 * metadata is required because a point has no orientation; GLL point identity
 * is fully determined by the common corner location alone.
 *
 * Inherits all common interface metadata from `communication_group`.
 *
 * @see communication_group
 */
class corner_communication_group : public communication_group {
public:
  corner_communication_group() = default;
  corner_communication_group(
      const unsigned int my_rank, const unsigned int neighbor_rank,
      const std::vector<specfem::mesh::adjacency_graph<
          specfem::element::dimension_tag::dim3>::MPIEdgeProperties> &edges,
      const int ngllz, const int nglly, const int ngllx);
};

/**
 * @brief Communication group for edge-shared MPI interfaces.
 *
 * An edge interface is a 1D line of GLL points. Because both partitions
 * traverse the edge from a local coordinate origin, their traversal directions
 * may be identical or reversed. The `reflect` flag encodes this: `false` means
 * same direction, `true` means reversed.
 *
 * The flag is computed from anchor points: each edge has two endpoint corners;
 * the anchor identifies which endpoint is the "start" in each partition. If
 * both partitions share the same start endpoint, directions agree
 * (reflect=false); if they disagree, the 1D sequence must be reversed
 * (reflect=true).
 *
 * Inherits all common interface metadata from `communication_group`.
 *
 * @see communication_group, corners_of_edge
 */
class edge_communication_group : public communication_group {
public:
  using ReflectViewType = Kokkos::View<bool *, Kokkos::DefaultExecutionSpace>;
  ReflectViewType reflect; /**< Per-edge reflection flag: true → reversed
                              traversal direction relative to neighbor (nfaces)
                            */
  ReflectViewType::HostMirror h_reflect; /**< Host mirror for reflect */

  edge_communication_group() = default;
  edge_communication_group(
      const unsigned int my_rank, const unsigned int neighbor_rank,
      const std::vector<specfem::mesh::adjacency_graph<
          specfem::element::dimension_tag::dim3>::MPIEdgeProperties> &edges,
      const int ngllz, const int nglly, const int ngllx);
};

/**
 * @brief Communication group for face-shared MPI interfaces.
 *
 * A face interface is a 2D ngll × ngll grid of GLL points. Because both
 * partitions define local (i, j) coordinates from a local corner, the two
 * grids may be rotated relative to each other by a multiple of 90°. The
 * `theta` index encodes this discrete rotation:
 *   - theta=0: identity (0°)
 *   - theta=1: 90° clockwise
 *   - theta=2: 180°
 *   - theta=3: 270° clockwise
 *
 * Theta is computed from anchor points using:
 *   theta = (neigh_anchor_pos − my_anchor_pos + 4) mod 4
 * where anchor positions are indices into the canonical corner order of each
 * face returned by `corners_of_face`.
 *
 * Stored as `unsigned char` (1 byte) instead of a floating-point angle to
 * minimize memory footprint in large-scale simulations.
 *
 * Inherits all common interface metadata from `communication_group`.
 *
 * @see communication_group, corners_of_face, find_anchor_position
 */
class face_communication_group : public communication_group {
public:
  using ThetaViewType =
      Kokkos::View<unsigned char *, Kokkos::DefaultExecutionSpace>;
  ThetaViewType theta; /**< Discrete rotation index ∈ [0,3] aligning face
                          coordinate systems across the MPI boundary (nfaces) */
  ThetaViewType::HostMirror h_theta; /**< Host mirror for theta */

  face_communication_group() = default;
  face_communication_group(
      const unsigned int my_rank, const unsigned int neighbor_rank,
      const std::vector<specfem::mesh::adjacency_graph<
          specfem::element::dimension_tag::dim3>::MPIEdgeProperties> &edges,
      const int ngllz, const int nglly, const int ngllx);
};

struct packer {
public:
  constexpr static auto dimension_tag =
      specfem::element::dimension_tag::dim3; ///< Dimension tag
  unsigned int my_rank;       /**< MPI rank of the current process */
  unsigned int neighbor_rank; /**< MPI rank of the neighboring process */
  unsigned int ncorners;      /**< Number of corners in communication group */
  unsigned int nedges;        /**< Number of edges in communication group */
  unsigned int nfaces;        /**< Number of faces in communication group */
  unsigned int ngll;          /**< GLL points per face dimension */

  unsigned int nglob; /**< Number of unique GLL points on the MPI surface for
                    this communication group */

  /**
   * @brief Maps the unique local mapping index the local partition to the
   * unique global mapping index for each unique GLL point
   *
   */
  using IndexMappingView = Kokkos::View<int *, Kokkos::DefaultExecutionSpace>;

  IndexMappingView mapping; /**< Maps local GLL point indices to global indices
                               for packing/unpacking */
  IndexMappingView::HostMirror h_mapping; /**< Host mirror for mapping */

  packer() : nfaces(0), nedges(0), ncorners(0), ngll(0), nglob(0) {}

  /**
   * @brief Constructs a packer that maps unique local MPI-surface GLL point
   * indices to unique global assembled indices.
   *
   * Iterates through all GLL points on all faces in the communication group,
   * deduplicates them (GLL points shared between adjacent faces at edges or
   * corners are stored only once), and builds a compact mapping from a unique
   * local index to the corresponding global assembled index (iglob).
   *
   * The mapping size equals the number of unique GLL points on the MPI
   * surface for this communication group, which is generally less than
   * nfaces * ngll² due to shared edge/corner points.
   *
   * Algorithm:
   * 1. For each face in communication_group::my_element:
   *    - Get the spectral element index and face orientation
   *    - For each GLL point (ipoint_i, ipoint_j) on the face:
   *      * Transform face 2D coordinates to element 3D coordinates
   *      * Look up the global index in the points index mapping
   *      * If this global index has not been seen before, append it
   * 2. Allocate device and host views sized to the unique point count
   * 3. Populate host view from the collected unique global indices
   * 4. Deep copy host array to device
   *
   * @param comm_group Communication group providing face elements,
   * orientations, and GLL metadata
   * @param element Element object for mapping face coordinates to element
   *                coordinates
   * @param points Points object containing the element-to-global index mapping
   *               (via h_index_mapping)
   */
  packer(const corner_communication_group &corner_group,
         const edge_communication_group &edge_group,
         const face_communication_group &face_group,
         const specfem::mesh_entity::element<dimension_tag> &element,
         const specfem::assembly::mesh<dimension_tag> &mesh);

private:
  /**
   * @brief Sends transformed GLL indices for all interface types to the
   * neighboring MPI process.
   *
   * Applies rotation/reflection transformations and sends 7 blocking MPI
   * messages (base_tag + offset, offset 0..6):
   *   +0  Metadata: {nfaces, nedges, ncorners, ngll, nglob}
   *   +1  Rotated face indices [nfaces][ngll][ngll] (theta permutation applied)
   *   +2  Face neighbor element indices [nfaces]
   *   +3  Reflected edge indices [nedges][ngll] (reflect flag applied)
   *   +4  Edge neighbor element indices [nedges]
   *   +5  Corner neighbor element indices [ncorners]
   *   +6  Corner nglob indices [ncorners]
   *
   * Uses blocking MPI_Send calls wrapped in SPECFEM_MPI_SAFECALL for
   * portability and error handling.
   *
   * @note Message tag base is computed via compute_message_tag_base(my_rank,
   *       neighbor_rank), reserving 7 consecutive tag slots per rank-pair.
   * @note Rotation index theta ∈ [0,3]: 0°, 90° CW, 180°, 270° CW
   */
  void send_unpacking_indices(
      Kokkos::View<int ***, Kokkos::HostSpace> face_index_mapping,
      Kokkos::View<unsigned char *, Kokkos::HostSpace> theta,
      Kokkos::View<int *, Kokkos::HostSpace> neighbor_element,
      Kokkos::View<int **, Kokkos::HostSpace> edge_index_mapping,
      Kokkos::View<bool *, Kokkos::HostSpace> edge_reflect,
      Kokkos::View<int *, Kokkos::HostSpace> edge_neighbor_element,
      Kokkos::View<int *, Kokkos::HostSpace> corner_neighbor_element,
      Kokkos::View<int *, Kokkos::HostSpace> corner_index_mapping);
};

struct unpacker {
public:
  constexpr static auto dimension_tag =
      specfem::element::dimension_tag::dim3; ///< Dimension tag

  unsigned int my_rank;       /**< MPI rank of the current process */
  unsigned int neighbor_rank; /**< MPI rank of the neighboring process */
  unsigned int nfaces;        /**< Number of faces in communication group */
  unsigned int nedges;        /**< Number of edges in communication group */
  unsigned int ncorners;      /**< Number of corners in communication group */
  unsigned int ngll;          /**< GLL points per face dimension */

  using IndexMappingView = Kokkos::View<int *, Kokkos::DefaultExecutionSpace>;

  IndexMappingView mapping; /**< Maps neighbor nglob indices to local iglobs */
  IndexMappingView::HostMirror h_mapping;

  unpacker() = default;

  unpacker(const corner_communication_group &corner_group,
           const edge_communication_group &edge_group,
           const face_communication_group &face_group,
           const specfem::mesh_entity::element<dimension_tag> &element,
           const specfem::assembly::mesh<dimension_tag> &mesh);

  /**
   * @brief Posts all 7 non-blocking MPI_Irecv calls and returns the buffers.
   *
   * All receives are posted before any MPI_Wait to avoid deadlock with the
   * sender's blocking MPI_Send calls.
   *
   * @return Tuple of (requests[7], metadata[5], face_indices, face_elements,
   *         edge_indices, edge_elements, corner_nglob_idx, corner_elements)
   */
  std::tuple<std::array<MPI_Request, 7>,
             Kokkos::View<unsigned int[5], Kokkos::HostSpace>,
             Kokkos::View<int ***, Kokkos::HostSpace>,
             Kokkos::View<int *, Kokkos::HostSpace>,
             Kokkos::View<int **, Kokkos::HostSpace>,
             Kokkos::View<int *, Kokkos::HostSpace>,
             Kokkos::View<int *, Kokkos::HostSpace>,
             Kokkos::View<int *, Kokkos::HostSpace> >
  receive_unpacking_buffers();

  /**
   * @brief Waits for all receives, validates metadata, and builds h_mapping.
   */
  void assemble_unpacking_mapping(
      const std::array<MPI_Request, 7> &requests,
      const Kokkos::View<unsigned int[5], Kokkos::HostSpace> metadata_buf,
      const Kokkos::View<int ***, Kokkos::HostSpace> recv_face_indices,
      const Kokkos::View<int *, Kokkos::HostSpace> face_elements,
      const Kokkos::View<int **, Kokkos::HostSpace> recv_edge_indices,
      const Kokkos::View<int *, Kokkos::HostSpace> edge_elements,
      const Kokkos::View<int *, Kokkos::HostSpace> corner_nglob_idx,
      const Kokkos::View<int *, Kokkos::HostSpace> corner_elements,
      const Kokkos::View<specfem::mesh_entity::dim3::type *, Kokkos::HostSpace>
          my_face_orientations,
      const Kokkos::View<specfem::mesh_entity::dim3::type *, Kokkos::HostSpace>
          my_edge_orientations,
      const Kokkos::View<specfem::mesh_entity::dim3::type *, Kokkos::HostSpace>
          my_corner_orientations,
      const specfem::mesh_entity::element<dimension_tag> &element,
      const specfem::assembly::mesh<dimension_tag> &mesh);
};

struct communication_pattern {
public:
  constexpr static auto dimension_tag =
      specfem::element::dimension_tag::dim3; ///< Dimension tag
  unsigned int my_rank;       /**< MPI rank of the current process */
  unsigned int neighbor_rank; /**< MPI rank of the neighboring process */
  unpacker unpack; /**< Unpacker for receiving data from the neighbor */
  packer pack;     /**< Packer for sending data to the neighbor */

  communication_pattern() = default;

  communication_pattern(
      const corner_communication_group &corner_group,
      const edge_communication_group &edge_group,
      const face_communication_group &face_group,
      const specfem::mesh_entity::element<dimension_tag> &element,
      const specfem::assembly::mesh<dimension_tag> &mesh);
};

} // namespace mpi_impl

/**
 * @brief Manages MPI communication groups and patterns for 3D distributed
 * simulations.
 *
 * Analyzes the mesh adjacency graph to classify all inter-process interfaces by
 * geometry and organize them into typed communication groups:
 *
 * | Member          | Type                        | Geometry             |
 * |-----------------|-----------------------------|----------------------|
 * | `face_groups`   | `face_communication_group`  | 2D ngll×ngll grid    |
 * | `edge_groups`   | `edge_communication_group`  | 1D ngll line         |
 * | `corner_groups` | `corner_communication_group`| single GLL point     |
 *
 * One group is created per unique neighbor rank for each geometry type. Groups
 * hold GPU-friendly `Kokkos::View` arrays of per-interface metadata:
 * - Mesh entity orientations in local and neighbor elements
 * - Spectral element indices for both partitions
 * - Geometry-specific transformation data (theta for faces, reflect for edges)
 *
 * `communication_patterns` (packer/unpacker pairs) are currently built only
 * for face interfaces; edge and corner patterns are deferred.
 *
 * @see face_communication_group, edge_communication_group,
 *      corner_communication_group
 */
template <> class mpi<specfem::element::dimension_tag::dim3> {
public:
  constexpr static auto dimension_tag = specfem::element::dimension_tag::dim3;

  std::vector<mpi_impl::corner_communication_group> corner_groups; /**< MPI
                                                   groups for corner connections
                                                 */
  std::vector<mpi_impl::edge_communication_group> edge_groups; /**< MPI groups
                                                                  for edge
                                                                  connections */
  std::vector<mpi_impl::face_communication_group> face_groups; /**< MPI groups
                                                                  for face
                                                                  connections */
  std::vector<mpi_impl::communication_pattern> communication_patterns;

  mpi() = default;

  /**
   * @brief Constructs all MPI communication groups and face communication
   * patterns.
   *
   * Steps:
   * 1. Extracts all MPI connections from the adjacency graph
   * 2. Classifies each connection as face, edge, or corner by checking against
   *    `dim3::faces`, `dim3::edges`, and `dim3::corners` entity sets
   * 3. Buckets connections by neighbor rank within each geometry class
   * 4. Constructs one typed group per unique (neighbor_rank, geometry) pair
   * 5. Constructs `communication_pattern` objects for face groups only
   *    (edge and corner patterns are deferred)
   *
   * @param adjacency_graph  Mesh adjacency graph containing MPI connection
   *                         metadata (via mpi_connections() method)
   * @param mesh             Assembly mesh providing the global index mapping
   * @param ngllz, nglly, ngllx  GLL points per element dimension
   *
   * @note The current MPI rank is retrieved internally via
   *       `specfem::MPI::get_rank()` at construction time.
   */
  mpi(const specfem::mesh::adjacency_graph<dimension_tag> &adjacency_graph,
      const specfem::assembly::mesh<dimension_tag> &mesh, const int ngllz,
      const int nglly, const int ngllx);
};

} // namespace specfem::assembly
