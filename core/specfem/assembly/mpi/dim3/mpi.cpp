#include "specfem/assembly/mpi/dim3/mpi.hpp"
#include "specfem/assembly/mesh.hpp"
#include "specfem/logger.hpp"
#include "specfem/mesh_entity.hpp"
#include "specfem/mpi.hpp"
#include "specfem/setup.hpp"
#include <array>
#include <cmath>
#include <stdexcept>
#include <unordered_map>
#include <unordered_set>

namespace {

/// Find position of anchor in face corner array
int find_anchor_position(
    const std::array<specfem::mesh_entity::dim3::type, 4> &corners,
    const specfem::mesh_entity::dim3::type anchor) {
  for (int i = 0; i < 4; i++) {
    if (corners[i] == anchor)
      return i;
  }
  specfem::Logger::error("Anchor point not found among face corners");
  return -1;
}

/**
 * @brief Compute MPI message tag base from rank pair.
 *
 * Uses modular arithmetic to stay within the MPI standard minimum tag
 * upper bound (MPI_TAG_UB >= 32767). Reserves 4 tag slots per rank-pair
 * for message-specific offsets (+0..+3).
 *
 * @param my_rank Current MPI rank
 * @param neighbor_rank Target MPI rank
 * @return Base tag for message discrimination (message-specific offsets added
 * later)
 */
int compute_message_tag_base(unsigned int my_rank, unsigned int neighbor_rank) {
  // Combine ranks using modular arithmetic to stay within the MPI standard
  // minimum tag upper bound (MPI_TAG_UB >= 32767). Reserve 4 tag slots
  // per rank-pair for message-specific offsets (+0..+3).
  return static_cast<int>(
      (static_cast<unsigned long>(my_rank) * 1000003UL + neighbor_rank) %
      32764);
}

/**
 * @brief Apply rotation permutation to a 2D GLL grid based on theta value.
 *
 * Theta encodes discrete 90° rotations as [0,3]:
 *   theta=0: identity [i][j] → [i][j]
 *   theta=1: 90° CW [i][j] → [ngll-1-j][i]
 *   theta=2: 180° [i][j] → [ngll-1-i][ngll-1-j]
 *   theta=3: 270° CW [i][j] → [j][ngll-1-i]
 *
 * @param theta Rotation index [0,3]
 * @param ngll Grid dimension (ngll × ngll)
 * @return Pair of rotated (i, j) indices
 */
std::pair<unsigned int, unsigned int> apply_rotation(unsigned char theta,
                                                     unsigned int i,
                                                     unsigned int j,
                                                     unsigned int ngll) {
  switch (theta) {
  case 0: // Identity
    return { i, j };
  case 1: // 90° CW
    return { ngll - 1 - j, i };
  case 2: // 180°
    return { ngll - 1 - i, ngll - 1 - j };
  case 3: // 270° CW
    return { j, ngll - 1 - i };
  default: // Should not reach here; silently use identity
    specfem::Logger::error("Invalid rotation index theta: " +
                           std::to_string(theta));
    return { i, j };
  }
}

} // anonymous namespace

// ---------------------------------------------------------------------------
// communication_group constructor
//
// Organizes face connectivity data into GPU-friendly Kokkos arrays. For each
// face in the interface, computes the discrete rotation index using anchor
// points to disambiguate face-to-face orientation across the MPI boundary.
//
// Algorithm:
// 1. Construct element from GLL parameters (for corner lookup only)
// 2. Allocate device and host views for face orientations and rotation indices
// 3. For each face adjacency:
//    - Record face orientation (mesh entity type) in local and neighbor
//    elements
//    - Find anchor point positions among face corners in both elements
//    - Compute rotation index: r = (neigh_anchor_pos + my_anchor_pos) % 4
//    - Store r directly (reduces memory vs. storing angle)
// 4. Deep copy host data to device
// ---------------------------------------------------------------------------
specfem::assembly::mpi_impl::communication_group::communication_group(
    const unsigned int my_rank, const unsigned int neighbor_rank,
    const std::vector<specfem::mesh::adjacency_graph<
        specfem::element::dimension_tag::dim3>::MPIEdgeProperties> &edges,
    const int ngllz, const int nglly, const int ngllx)
    : my_rank(my_rank), neighbor_rank(neighbor_rank), nfaces(edges.size()) {
  // Construct element from GLL parameters
  const specfem::mesh_entity::element<specfem::element::dimension_tag::dim3>
      element(ngllz, nglly, ngllx);
  ngll = element.ngll;

  // Early exit if no faces in this communication group
  if (nfaces == 0)
    return;

  // Allocate device views for face metadata
  my_orientation =
      FaceNormalViewType("communication_group::my_orientation", nfaces);
  neighbor_orientation =
      FaceNormalViewType("communication_group::neighbor_orientation", nfaces);
  theta = ThetaViewType("communication_group::theta", nfaces);
  my_element = ElementIndexViewType("communication_group::my_element", nfaces);
  neighbor_element =
      ElementIndexViewType("communication_group::neighbor_element", nfaces);

  // Create host mirrors for initialization
  h_my_orientation = Kokkos::create_mirror_view(my_orientation);
  h_neighbor_orientation = Kokkos::create_mirror_view(neighbor_orientation);
  h_theta = Kokkos::create_mirror_view(theta);
  h_my_element = Kokkos::create_mirror_view(my_element);
  h_neighbor_element = Kokkos::create_mirror_view(neighbor_element);

  // Populate host mirrors with connectivity data
  for (unsigned int iface = 0; iface < nfaces; iface++) {
    const auto &edge = edges[iface];

    // --- Face orientations (mesh entity types) ---
    h_my_orientation(iface) = edge.orientation;
    h_neighbor_orientation(iface) = edge.neighbor_orientation;

    // --- Spectral element indices ---
    h_my_element(iface) = static_cast<int>(edge.local_index);
    h_neighbor_element(iface) = static_cast<int>(edge.neighbor_local_index);

    // --- Anchor-based rotation ---
    const auto my_corners =
        specfem::mesh_entity::corners_of_face(edge.orientation);
    const auto neigh_corners =
        specfem::mesh_entity::corners_of_face(edge.neighbor_orientation);

    const int my_anchor_pos =
        find_anchor_position(my_corners, edge.local_anchor_point);
    const int neigh_anchor_pos =
        find_anchor_position(neigh_corners, edge.neighbor_anchor_point);

    // Theta stores the discrete rotation index r in [0,3]
    // r = (neigh_anchor_pos - my_anchor_pos) mod 4
    // Actual rotation angle (if needed) = r * π/2
    // Storing as unsigned char saves memory compared to type_real (1 vs 8
    // bytes)
    const unsigned char r =
        static_cast<unsigned char>((neigh_anchor_pos - my_anchor_pos + 4) % 4);
    h_theta(iface) = r;
  }

  // Deep copy host → device
  Kokkos::deep_copy(my_orientation, h_my_orientation);
  Kokkos::deep_copy(neighbor_orientation, h_neighbor_orientation);
  Kokkos::deep_copy(theta, h_theta);
  Kokkos::deep_copy(my_element, h_my_element);
  Kokkos::deep_copy(neighbor_element, h_neighbor_element);
}

// ---------------------------------------------------------------------------
// packer constructor
//
// Computes a compact mapping from unique local MPI-surface GLL point indices
// to partition-global assembled indices (iglob). GLL points shared between
// adjacent faces (at edges or corners of the MPI interface) are stored only
// once, so the mapping size equals the number of unique assembled points on
// the interface rather than nfaces * ngll².
//
// Algorithm:
// 1. Traverse all faces and their GLL points, collecting unique global
//    indices using an unordered_set for O(1) duplicate detection
// 2. Allocate Kokkos views sized to the unique point count
// 3. Populate host view and deep copy to device
// ---------------------------------------------------------------------------
specfem::assembly::mpi_impl::packer::packer(
    const specfem::assembly::mpi_impl::communication_group &comm_group,
    const specfem::mesh_entity::element<dimension_tag> &element,
    const specfem::assembly::mesh<dimension_tag> &mesh) {

  this->my_rank = comm_group.my_rank;
  this->neighbor_rank = comm_group.neighbor_rank;
  this->nfaces = comm_group.nfaces;
  this->ngll = comm_group.ngll;

  // Early exit if no faces in this communication group
  if (this->nfaces == 0)
    return;

  Kokkos::View<int ***, Kokkos::HostSpace> face_index_mapping(
      "packer::face_index_mapping", this->nfaces, this->ngll, this->ngll);

  // First pass: collect unique global indices across all face GLL points.
  // Points shared between adjacent faces (at shared edges/corners) are
  // inserted only once; insertion order is preserved via the vector.
  std::vector<int> unique_iglobs;
  std::unordered_map<int, int> seen; // Maps global iglob to local mapping index
  unique_iglobs.reserve(this->nfaces * this->ngll * this->ngll); // upper-bound
                                                                 // reserve

  for (unsigned int iface = 0; iface < this->nfaces; iface++) {
    const int ielem = comm_group.h_my_element(iface);
    const auto face_type = comm_group.h_my_orientation(iface);

    for (unsigned int ipoint_j = 0; ipoint_j < this->ngll; ipoint_j++) {
      for (unsigned int ipoint_i = 0; ipoint_i < this->ngll; ipoint_i++) {
        const int point_linear = ipoint_j * this->ngll + ipoint_i;

        // Map face 2D coordinates to element 3D coordinates
        const auto [iz, iy, ix] =
            element.map_coordinates(face_type, point_linear);

        // Look up global assembled index
        const int iglob = mesh.h_index_mapping(ielem, iz, iy, ix);

        // Insert only if not already seen
        if (seen.find(iglob) == seen.end()) {
          seen[iglob] = unique_iglobs.size(); // local mapping index
          unique_iglobs.push_back(iglob);
        }

        face_index_mapping(iface, ipoint_j, ipoint_i) = seen[iglob];
      }
    }
  }

  this->nglob = unique_iglobs.size();

  this->send_unpacking_indices(face_index_mapping, comm_group.h_theta,
                               comm_group.h_neighbor_element,
                               comm_group.h_neighbor_orientation);

  // Allocate views sized to the number of unique points
  const unsigned int n_unique = unique_iglobs.size();
  mapping = IndexMappingView("packer::mapping", n_unique);
  h_mapping = Kokkos::create_mirror_view(mapping);

  // Populate host mirror from collected unique global indices
  for (unsigned int i = 0; i < n_unique; i++) {
    h_mapping(i) = unique_iglobs[i];
  }

  // Deep copy host → device
  Kokkos::deep_copy(mapping, h_mapping);
}

// ---------------------------------------------------------------------------
// send_unpacking_indices implementation
//
// Applies rotation transformations to face GLL indices and sends them to the
// neighboring MPI process using three separate blocking MPI messages
// (Option A):
//   1. Metadata: nfaces and ngll (for receiver buffer allocation)
//   2. Rotated face indices: [nfaces][ngll][ngll] with theta-based rotation
//   3. Neighbor element indices: [nfaces] (for receiver face-to-element
//   mapping)
//
// Rotation is applied per face using the apply_rotation helper function,
// which maps discrete theta values [0,3] to 90° rotation increments.
//
// Uses blocking MPI_Send (wrapped in SPECFEM_MPI_SAFECALL) to ensure data
// is transmitted before local buffers are deallocated. All calls protected
// for non-MPI builds via macro.
// ---------------------------------------------------------------------------
void specfem::assembly::mpi_impl::packer::send_unpacking_indices(
    Kokkos::View<int ***, Kokkos::HostSpace> face_index_mapping,
    Kokkos::View<unsigned char *, Kokkos::HostSpace> theta,
    Kokkos::View<int *, Kokkos::HostSpace> neighbor_element,
    Kokkos::View<specfem::mesh_entity::dim3::type *, Kokkos::HostSpace>
        neighbor_orientation) {

  // Early exit if no faces to send
  if (this->nfaces == 0)
    return;

  // -----------------------------------------------------------------------
  // Step 1: Allocate host-space buffer for rotated indices
  // -----------------------------------------------------------------------
  Kokkos::View<int ***, Kokkos::HostSpace> rotated_indices(
      "packer::send::rotated_indices", this->nfaces, this->ngll, this->ngll);

  // -----------------------------------------------------------------------
  // Step 2: Apply rotation permutation per face
  // -----------------------------------------------------------------------
  for (unsigned int iface = 0; iface < this->nfaces; iface++) {
    const unsigned char theta_val = theta(iface);

    // Apply rotation permutation based on theta value
    for (unsigned int i = 0; i < this->ngll; i++) {
      for (unsigned int j = 0; j < this->ngll; j++) {
        const auto [rotated_i, rotated_j] =
            apply_rotation(theta_val, i, j, this->ngll);

        // Copy from original position to rotated position
        rotated_indices(iface, rotated_j, rotated_i) =
            face_index_mapping(iface, j, i);
      }
    }
  }

  // -----------------------------------------------------------------------
  // Step 3: Send three MPI messages (Option A: Separate sends)
  // -----------------------------------------------------------------------
  MPI_Comm comm = specfem::MPI::communicator();

  // Derive message tag base from (my_rank, neighbor_rank) pair to ensure
  // global uniqueness (shared with unpacker)
  const int base_tag =
      compute_message_tag_base(this->my_rank, this->neighbor_rank);

  // Message 1: Send metadata (nfaces, ngll, nglob)
  unsigned int metadata[3] = { this->nfaces, this->ngll, this->nglob };
  SPECFEM_MPI_SAFECALL(MPI_Send(metadata, 3, MPI_UNSIGNED, this->neighbor_rank,
                                base_tag + 0, comm));

  // Message 2: Send rotated face indices [nfaces][ngll][ngll]
  SPECFEM_MPI_SAFECALL(
      MPI_Send(rotated_indices.data(),
               static_cast<int>(this->nfaces * this->ngll * this->ngll),
               MPI_INT, this->neighbor_rank, base_tag + 1, comm));

  // Message 3: Send neighbor element indices [nfaces]
  SPECFEM_MPI_SAFECALL(MPI_Send(neighbor_element.data(),
                                static_cast<int>(this->nfaces), MPI_INT,
                                this->neighbor_rank, base_tag + 2, comm));

  // Message 4: Send neighbor face orientation [nfaces]
  SPECFEM_MPI_SAFECALL(MPI_Send(neighbor_orientation.data(),
                                static_cast<int>(this->nfaces), MPI_INT,
                                this->neighbor_rank, base_tag + 3, comm));
}

// ---------------------------------------------------------------------------
// unpacker constructor
//
// Receives metadata, rotated face indices, and element indices from the
// neighboring MPI process using non-blocking MPI_Irecv calls (all posted
// upfront before any MPI_Wait), then deduplicates received GLL points to build
// a compact mapping.
//
// Algorithm:
// 1. Store my_rank, neighbor_rank, nfaces, ngll from communication_group
// 2. Allocate host-space receive buffers
// 3. Call receive_unpacking_buffers to receive MPI messages
// 4. Call assemble_unpacking_mapping to build the mapping
//
// Non-blocking receives avoid deadlock with blocking MPI_Send on sender side
// because all Irecv are posted before Waitall.
// ---------------------------------------------------------------------------
specfem::assembly::mpi_impl::unpacker::unpacker(
    const specfem::assembly::mpi_impl::communication_group &comm_group,
    const specfem::mesh_entity::element<dimension_tag> &element,
    const specfem::assembly::mesh<dimension_tag> &mesh) {

  this->my_rank = comm_group.my_rank;
  this->neighbor_rank = comm_group.neighbor_rank;
  this->nfaces = comm_group.nfaces;
  this->ngll = comm_group.ngll;

  // Early exit if no faces in this communication group
  if (this->nfaces == 0)
    return;
}

// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// receive_unpacking_buffers implementation (private helper)
//
// Posts three non-blocking MPI_Irecv calls (all before any MPI_Wait):
// 1. Metadata: nfaces and ngll (for validation)
// 2. Rotated face indices: [nfaces][ngll][ngll] (for validation or future use)
// 3. Neighbor element indices: [nfaces] (for face-to-element mapping)
//
// Calls MPI_Waitall to block until all receives complete. Validates that
// received metadata matches expected nfaces and ngll.
// ---------------------------------------------------------------------------
std::tuple<std::array<MPI_Request, 4>,
           Kokkos::View<unsigned int[3], Kokkos::HostSpace>,
           Kokkos::View<int ***, Kokkos::HostSpace>,
           Kokkos::View<int *, Kokkos::HostSpace>,
           Kokkos::View<int *, Kokkos::HostSpace> >
specfem::assembly::mpi_impl::unpacker::receive_unpacking_buffers() {

  // Early exit if no faces to receive
  if (this->nfaces == 0) {
    // Return empty views and uninitialized requests (safe since
    // assemble_unpacking_mapping will also early-exit on nfaces==0)
    return { std::array<MPI_Request, 4>{},
             Kokkos::View<unsigned int[3], Kokkos::HostSpace>(
                 "unpacker::empty_metadata"),
             Kokkos::View<int ***, Kokkos::HostSpace>("unpacker::empty_indices",
                                                      0, 0, 0),
             Kokkos::View<int *, Kokkos::HostSpace>(
                 "unpacker::empty_element_indices", 0),
             Kokkos::View<int *, Kokkos::HostSpace>(
                 "unpacker::empty_orientations", 0) };
  }

  // -----------------------------------------------------------------------
  // Step 1: Compute receive tag (reversed from sender's perspective)
  // -----------------------------------------------------------------------
  const int recv_tag =
      compute_message_tag_base(this->neighbor_rank, this->my_rank);

  Kokkos::View<unsigned int[3], Kokkos::HostSpace> metadata_buf(
      "unpacker::metadata_buf");
  Kokkos::View<int ***, Kokkos::HostSpace> recv_indices(
      Kokkos::view_alloc(Kokkos::WithoutInitializing, "unpacker::recv_indices"),
      this->nfaces, this->ngll, this->ngll);
  Kokkos::View<int *, Kokkos::HostSpace> element_indices(
      Kokkos::view_alloc(Kokkos::WithoutInitializing,
                         "unpacker::element_indices"),
      this->nfaces);
  Kokkos::View<int *, Kokkos::HostSpace> neighbor_orientations(
      Kokkos::view_alloc(Kokkos::WithoutInitializing,
                         "unpacker::neighbor_orientations"),
      this->nfaces);

  // -----------------------------------------------------------------------
  // Step 2: Post all three MPI_Irecv before any MPI_Wait
  // -----------------------------------------------------------------------
  std::array<MPI_Request, 4> requests{};
  MPI_Comm comm = specfem::MPI::communicator();

  // Message 1: Receive metadata (nfaces, ngll)
  SPECFEM_MPI_SAFECALL(MPI_Irecv(metadata_buf.data(), 3, MPI_UNSIGNED,
                                 this->neighbor_rank, recv_tag + 0, comm,
                                 &requests[0]));

  // Message 2: Receive rotated face indices [nfaces][ngll][ngll]
  SPECFEM_MPI_SAFECALL(MPI_Irecv(
      recv_indices.data(),
      static_cast<int>(this->nfaces * this->ngll * this->ngll), MPI_INT,
      this->neighbor_rank, recv_tag + 1, comm, &requests[1]));

  // Message 3: Receive neighbor element indices [nfaces]
  SPECFEM_MPI_SAFECALL(
      MPI_Irecv(element_indices.data(), static_cast<int>(this->nfaces), MPI_INT,
                this->neighbor_rank, recv_tag + 2, comm, &requests[2]));

  // Message 4: Receive neighbor orientations [nfaces]
  SPECFEM_MPI_SAFECALL(MPI_Irecv(
      neighbor_orientations.data(), static_cast<int>(this->nfaces), MPI_INT,
      this->neighbor_rank, recv_tag + 3, comm, &requests[3]));

  return { requests, metadata_buf, recv_indices, element_indices,
           neighbor_orientations };
}

// ---------------------------------------------------------------------------
// assemble_unpacking_mapping implementation (private helper)
//
// Builds the mapping from unique GLL points received from the neighbor by
// traversing received element indices and deduplicating with local iglob
// lookup. Uses an element-to-orientation lookup map to handle potential
// face ordering differences between communicating ranks.
//
// The deduplication strategy mirrors the packer: traverse faces in order,
// GLL points are deduplicated via unordered_set, and the mapping size reflects
// the actual number of unique points on the interface.
// ---------------------------------------------------------------------------
void specfem::assembly::mpi_impl::unpacker::assemble_unpacking_mapping(
    const std::array<MPI_Request, 4> &requests,
    const Kokkos::View<unsigned int[3], Kokkos::HostSpace> metadata_buf,
    const Kokkos::View<int ***, Kokkos::HostSpace> recv_indices,
    const Kokkos::View<int *, Kokkos::HostSpace> element_indices,
    const Kokkos::View<specfem::mesh_entity::dim3::type *, Kokkos::HostSpace>
        my_orientations,
    const specfem::mesh_entity::element<dimension_tag> &element,
    const specfem::assembly::mesh<dimension_tag> &mesh) {

  // Early exit if no faces to assemble
  if (this->nfaces == 0)
    return;

  // -----------------------------------------------------------------------
  // Step 1: Wait for all receives to complete
  // -----------------------------------------------------------------------
  SPECFEM_MPI_SAFECALL(MPI_Waitall(
      4, const_cast<MPI_Request *>(requests.data()), MPI_STATUSES_IGNORE));

  // -----------------------------------------------------------------------
  // Step 2: Validate received metadata
  // -----------------------------------------------------------------------
  if (metadata_buf[0] != this->nfaces || metadata_buf[1] != this->ngll) {
    specfem::Logger::error(
        "unpacker::receive_unpacking_buffers: metadata mismatch with sender");
  }
  // -----------------------------------------------------------------------

  // Step 3: Build mapping from received buffers
  const int nglob = metadata_buf[2];

  mapping = IndexMappingView("unpacker::mapping", nglob);
  h_mapping = Kokkos::create_mirror_view(mapping);

  std::unordered_map<int, int> seen; // Maps global iglob to local mapping index

  for (unsigned int iface = 0; iface < this->nfaces; iface++) {
    const int ielem = element_indices(iface);
    const auto face_type = my_orientations(iface);
    for (unsigned int ipoint_j = 0; ipoint_j < this->ngll; ipoint_j++) {
      for (unsigned int ipoint_i = 0; ipoint_i < this->ngll; ipoint_i++) {
        // Get the rotated face indices from the received buffer
        const int point_linear = ipoint_j * this->ngll + ipoint_i;
        const int face_iglob = recv_indices(iface, ipoint_j, ipoint_i);

        // Map face 2D coordinates to element 3D coordinates
        const auto [iz, iy, ix] =
            element.map_coordinates(face_type, point_linear);

        // Look up global assembled index using local element and face point
        const int iglob = mesh.h_index_mapping(ielem, iz, iy, ix);

        if (seen.find(iglob) == seen.end()) {
          seen[iglob] = face_iglob; // Store the mapping from local iglob to
                                    // received face iglob
        } else {
          // Validate that the same iglob maps to the same face_iglob
          //  if seen again
          if (seen[iglob] != face_iglob) {
            specfem::Logger::error("Inconsistent mapping for iglob " +
                                   std::to_string(iglob));
          }
        }

        if (face_iglob < 0 || face_iglob >= nglob) {
          specfem::Logger::error("Received face iglob out of bounds: " +
                                 std::to_string(face_iglob));
        }

        h_mapping(face_iglob) =
            iglob; // Store the mapping from received face iglob to local iglob
      }
    }
  }

  // Check if all received face iglobs have been mapped
  if (seen.size() != static_cast<size_t>(nglob)) {
    specfem::Logger::error("Not all received face iglobs were mapped");
  }

  Kokkos::deep_copy(mapping, h_mapping);
}

specfem::assembly::mpi_impl::communication_pattern::communication_pattern(
    const communication_group &comm_group,
    const specfem::mesh_entity::element<dimension_tag> &element,
    const specfem::assembly::mesh<dimension_tag> &mesh)
    : my_rank(comm_group.my_rank), neighbor_rank(comm_group.neighbor_rank) {

  // Issue non-blocking receives for unpacking buffers (metadata, indices, etc.)
  unpack = { comm_group, element, mesh };
  const auto [requests, metadata_buf, recv_indices, element_indices,
              neighbor_orientations] = unpack.receive_unpacking_buffers();
  // Create packer and send unpacking indices to neighbor
  pack = { comm_group, element, mesh };

  // Finalize unpacker by assembling the unpacking mapping after receives
  // complete
  unpack.assemble_unpacking_mapping(requests, metadata_buf, recv_indices,
                                    element_indices,
                                    comm_group.h_my_orientation, element, mesh);
}

// ---------------------------------------------------------------------------
// mpi<dim3> constructor
//
// Analyzes the mesh adjacency graph to extract all face-level MPI
// interfaces and groups them by neighboring process. This creates a
// communication schedule that filters out edge and corner adjacencies.
//
// Algorithm:
// 1. Extract all MPI connections from adjacency_graph.mpi_connections()
// 2. Retrieve the current MPI rank via specfem::MPI::get_rank()
// 3. Iterate and group connections by neighbor partition (MPI rank)
// 4. For each group, filter to keep only face connections
//    (skip connections involving edges or corners)
// 5. Create one communication_group per unique neighbor
//
// Result: communication_groups vector is indexed by neighbor rank,
// enabling efficient kernel dispatch for data exchange.
// ---------------------------------------------------------------------------
specfem::assembly::mpi<specfem::element::dimension_tag::dim3>::mpi(
    const specfem::mesh::adjacency_graph<dimension_tag> &adjacency_graph,
    const specfem::assembly::mesh<dimension_tag> &mesh, const int ngllz,
    const int nglly, const int ngllx) {

  const unsigned int my_rank =
      static_cast<unsigned int>(specfem::MPI::get_rank());

  const auto &mpi_conns = adjacency_graph.mpi_connections();

  // Group MPI connections by neighbor partition
  std::unordered_map<unsigned int, std::vector<specfem::mesh::adjacency_graph<
                                       dimension_tag>::MPIEdgeProperties> >
      grouped;

  for (const auto &conn : mpi_conns) {
    // Only include face connections (skip edge/corner MPI adjacencies)
    // Faces are defined in mesh_entity::dim3::faces enumeration
    if (!specfem::mesh_entity::contains(specfem::mesh_entity::dim3::faces,
                                        conn.orientation)) {
      continue;
    }
    grouped[conn.neighbor_partition].push_back(conn);
  }

  // Build one communication_group per unique neighbor
  // Each group manages all faces shared with a single neighboring process
  communication_groups.reserve(grouped.size());
  for (auto &[neighbor_rank, edges] : grouped) {
    communication_groups.emplace_back(my_rank, neighbor_rank, edges, ngllz,
                                      nglly, ngllx);
  }

  // Construct element from GLL parameters for coordinate mapping
  const specfem::mesh_entity::element<dimension_tag> element(ngllz, nglly,
                                                             ngllx);

  for (const auto &comm_group : communication_groups) {

    communication_patterns.emplace_back(comm_group, element, mesh);
  }
}
