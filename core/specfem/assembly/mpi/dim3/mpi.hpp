
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

/**
 * @brief Stores face communication metadata between two neighboring MPI
 * processes.
 *
 * This class manages the interface between two neighboring subdomains (MPI
 * processes) by tracking:
 * - Which faces belong to this inter-process boundary
 * - The orientation of each face in both local and neighbor elements
 * - The discrete rotation index required to align face coordinates across the
 *   MPI boundary (computed from anchor points to avoid ambiguity)
 *
 * Memory is organized as 1D `Kokkos::View` arrays indexed by face ID within
 * this communication group, enabling efficient GPU-accelerated synchronization
 * kernels.
 *
 * @see mpi – Collects one communication_group per neighbor MPI rank
 */
class communication_group {
public:
  constexpr static auto dimension_tag =
      specfem::element::dimension_tag::dim3; ///< Dimension tag

  unsigned int my_rank;       /**< MPI rank of the current process */
  unsigned int neighbor_rank; /**< MPI rank of the neighboring process */
  unsigned int nfaces; /**< Number of faces within this MPI communication group
                        */
  unsigned int ngll;   /**< Number of GLL points per face dimension */

  using FaceNormalViewType = Kokkos::View<specfem::mesh_entity::dim3::type *,
                                          Kokkos::DefaultExecutionSpace>;
  using ThetaViewType =
      Kokkos::View<unsigned char *, Kokkos::DefaultExecutionSpace>;
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
  ThetaViewType theta; /**< Rotation index r in [0,3] for face-to-face
                       connections (nfaces) */
  ThetaViewType::HostMirror h_theta; /**< Host mirror for theta */
  ElementIndexViewType my_element;   /**< Spectral element index of the local
                                     element for each face (nfaces) */
  ElementIndexViewType neighbor_element; /**< Spectral element index of the
                                         neighboring element for each face
                                         (nfaces) */
  ElementIndexViewType::HostMirror h_my_element;       /**< Host mirror for
                                                       my_element */
  ElementIndexViewType::HostMirror h_neighbor_element; /**< Host mirror for
                                                       neighbor_element */

  communication_group() = default;

  /**
   * @brief Constructs a communication group for face exchange with a neighbor.
   *
   * Extracts interface metadata from the mesh adjacency graph and organizes it
   * into GPU-friendly Kokkos views. For each face in the interface:
   * - Records the face orientation in both local and neighboring elements
   * - Stores the spectral element indices for both local and neighbor elements
   * - Computes the discrete rotation index (0–3) using anchor points to
   *   uniquely determine the face-to-face coordinate alignment
   *
   * @param my_rank          MPI rank of the current subdomain
   * @param neighbor_rank    MPI rank of the neighboring subdomain
   * @param edges            Face connectivity data from
   *                         adjacency_graph.mpi_connections()
   *                         (already filtered to contain only face
   *                         adjacencies)
   * @param ngllz, nglly, ngllx  GLL points per element dimension; used to
   *                         construct the element for extracting face corner
   *                         information
   *
   * @throws std::runtime_error if an anchor point is not found among face
   *         corners
   *
   * @note The element is constructed internally from GLL parameters to extract
   *       corner information; it is local to this constructor.
   */
  communication_group(
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

  packer() : nfaces(0), ngll(0) {}

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
  packer(const communication_group &comm_group,
         const specfem::mesh_entity::element<dimension_tag> &element,
         const specfem::assembly::mesh_impl::points<dimension_tag> &points);

  /**
   * @brief Sends rotated face GLL indices and element indices to the
   * neighboring MPI process.
   *
   * Applies rotation transformations (via theta value) to each face's GLL
   * grid and sends three distinct MPI messages (Option A):
   * 1. Metadata: nfaces and ngll (for receiver to allocate buffers)
   * 2. Rotated face indices: [nfaces][ngll][ngll] with rotation permutation
   *    applied per face
   * 3. Neighbor element indices: [nfaces] (for receiver to map faces to
   *    correct elements, accounting for potential face reordering)
   *
   * Uses blocking MPI_Send calls wrapped in SPECFEM_MPI_SAFECALL for
   * portability and error handling.
   *
   * @param face_index_mapping [nfaces][ngll][ngll] global indices for each
   * GLL point on each face
   * @param theta [nfaces] rotation index per face, theta ∈ [0,3]
   * @param neighbor_element [nfaces] neighbor element index for each face
   *
   * @note Uses member variables: nfaces, ngll for buffer dimensions
   * @note Rotation index theta ∈ [0,3] represents discrete rotations:
   *       theta=0: 0°, theta=1: 90° CW, theta=2: 180°, theta=3: 270° CW
   * @note Message tags: (my_rank << 16) | neighbor_rank + offset
   */
  void send_unpacking_indices(
      Kokkos::View<int ***, Kokkos::HostSpace> face_index_mapping,
      Kokkos::View<unsigned char *, Kokkos::HostSpace> theta,
      Kokkos::View<int *, Kokkos::HostSpace> neighbor_element);
};

struct unpacker {
public:
  constexpr static auto dimension_tag =
      specfem::element::dimension_tag::dim3; ///< Dimension tag

  unsigned int my_rank;       /**< MPI rank of the current process */
  unsigned int neighbor_rank; /**< MPI rank of the neighboring process */
  unsigned int nfaces;        /**< Number of faces in communication group */
  unsigned int ngll;          /**< GLL points per face dimension */

  /**
   * @brief Maps the unique received GLL point indices from the neighbor
   * partition to the unique global mapping index for each unique GLL point
   *
   */
  using IndexMappingView = Kokkos::View<int *, Kokkos::DefaultExecutionSpace>;

  IndexMappingView mapping; /**< Maps neighbor GLL point indices to global
                               indices for packing/unpacking */
  IndexMappingView::HostMirror h_mapping; /**< Host mirror for mapping */

  unpacker() = default;

  /**
   * @brief Constructs an unpacker that receives and deduplicates GLL point
   * indices from a neighboring MPI process.
   *
   * Posts non-blocking MPI_Irecv calls for three messages (metadata, rotated
   * face indices, element indices), waits for their completion, and builds a
   * compact mapping from unique received GLL points to global indices.
   *
   * Uses the same deduplication strategy as packer: faces are traversed in
   * order, GLL points are deduplicated via an unordered_set, and the mapping
   * size reflects the actual number of unique points on the interface.
   *
   * @param comm_group Communication group providing face metadata, GLL count,
   *                   and rank information
   * @param element    Element object for mapping face coordinates to element
   *                   3D coordinates
   * @param points     Points object containing the element-to-global index
   *                   mapping
   */
  unpacker(const communication_group &comm_group,
           const specfem::mesh_entity::element<dimension_tag> &element,
           const specfem::assembly::mesh_impl::points<dimension_tag> &points);

  /**
   * @brief Receives MPI buffers for rotated indices and element indices.
   *
   * Posts all three MPI_Irecv calls upfront (before any MPI_Wait) to avoid
   * deadlock with blocking MPI_Send on the sender side. Calls MPI_Waitall to
   * block until all receives complete.
   *
   * @param metadata_buf Output: metadata[2] = {nfaces, ngll}
   * @param recv_indices Output: [nfaces][ngll][ngll] rotated face indices
   * @param element_indices Output: [nfaces] neighbor element indices
   *
   * @throws std::runtime_error if metadata validation fails
   */
  void receive_unpacking_buffers(
      unsigned int metadata_buf[2],
      Kokkos::View<int ***, Kokkos::HostSpace> recv_indices,
      Kokkos::View<int *, Kokkos::HostSpace> element_indices);

  /**
   * @brief Assembles the mapping from received buffers.
   *
   * Builds a compact mapping from unique GLL points by traversing received
   * element indices and deduplicating with local iglob lookup. Uses an
   * element-to-orientation lookup map to handle potential face ordering
   * differences between communicating ranks.
   *
   * @param element_indices [nfaces] received neighbor element indices
   * @param my_orientation [nfaces] local face orientation types
   * @param my_element [nfaces] local spectral element indices
   * @param element Element object for map_coordinates
   * @param points Points object for iglob lookup
   *
   * @throws std::runtime_error if neighbor element is not found in lookup map
   */
  void assemble_unpacking_mapping(
      const Kokkos::View<int *, Kokkos::HostSpace> element_indices,
      Kokkos::View<specfem::mesh_entity::dim3::type *, Kokkos::HostSpace>
          my_orientation,
      Kokkos::View<int *, Kokkos::HostSpace> my_element,
      const specfem::mesh_entity::element<dimension_tag> &element,
      const specfem::assembly::mesh_impl::points<dimension_tag> &points);
};

} // namespace mpi_impl

/**
 * @brief Manages MPI face communication patterns for 3D distributed
 * simulations.
 *
 * This class analyzes the mesh adjacency graph to extract and organize all
 * face-level inter-process interfaces. It builds one `communication_group` per
 * neighboring MPI rank, each containing GPU-friendly arrays for:
 * - Face orientations in local and neighbor elements
 * - Discrete rotation indices for coordinate alignment
 * - GLL metadata for synchronization kernel dispatch
 *
 * Construction filters out edge and corner adjacencies, keeping only faces
 * needed for GLL point data exchange.
 *
 * The design prioritizes:
 * - **Compact storage**: Rotation indices as `unsigned char` (1 byte) instead
 *   of floating-point angles (8 bytes)
 * - **GPU efficiency**: Data organized in `Kokkos::View` arrays for direct
 *   kernel access
 * - **Clear semantics**: Discrete rotation values (0–3) are explicit; actual
 *   angle = rotation_index × π/2
 *
 * @see communication_group – Individual interface between two ranks
 */
template <> class mpi<specfem::element::dimension_tag::dim3> {
public:
  constexpr static auto dimension_tag = specfem::element::dimension_tag::dim3;

  std::vector<mpi_impl::communication_group> communication_groups;
  std::vector<mpi_impl::packer> packers;
  std::vector<mpi_impl::unpacker> unpackers;

  mpi() = default;

  /**
   * @brief Constructs the complete MPI face communication schedule.
   *
   * Performs the following steps:
   * 1. Extracts all MPI connections from the adjacency graph
   * 2. Filters to keep only face-level interactions (excludes edges/corners)
   * 3. Groups connections by neighboring MPI rank
   * 4. Creates one `communication_group` per unique neighbor
   *
   * @param adjacency_graph  Mesh adjacency graph containing MPI connection
   *                         metadata (via mpi_connections() method)
   * @param ngllz, nglly, ngllx  GLL points per element dimension
   *
   * @note The current MPI rank is retrieved internally via
   *       `specfem::MPI::get_rank()` at construction time.
   * @note The element is constructed internally from GLL parameters only to
   *       extract mesh connectivity; it is not stored.
   */
  mpi(const specfem::mesh::adjacency_graph<dimension_tag> &adjacency_graph,
      const specfem::assembly::mesh<dimension_tag> &mesh, const int ngllz,
      const int nglly, const int ngllx);
};

} // namespace specfem::assembly
