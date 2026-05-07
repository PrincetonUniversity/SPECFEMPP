#include "specfem/assembly/mpi/dim3/mpi.hpp"
#include "specfem/logger.hpp"
#include "specfem/mesh_entity.hpp"
#include "specfem/mpi.hpp"
#include "specfem/setup.hpp"
#include "specfem/tag_dispatch.hpp"
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

/// Filter communication group indices by medium tag
/// Returns tuple of (face_indices, edge_indices, corner_indices)
template <specfem::element::medium_tag MediumTag>
std::tuple<std::vector<unsigned int>, std::vector<unsigned int>,
           std::vector<unsigned int> >
filter_indices_by_medium_tag(
    const specfem::assembly::mpi_impl::face_communication_group &face_group,
    const specfem::assembly::mpi_impl::edge_communication_group &edge_group,
    const specfem::assembly::mpi_impl::corner_communication_group
        &corner_group) {
  std::vector<unsigned int> face_indices;
  std::vector<unsigned int> edge_indices;
  std::vector<unsigned int> corner_indices;

  for (unsigned int i = 0; i < face_group.n; i++) {
    if (face_group.h_connection_medium_tag(i) == MediumTag) {
      face_indices.push_back(i);
    }
  }

  for (unsigned int i = 0; i < edge_group.n; i++) {
    if (edge_group.h_connection_medium_tag(i) == MediumTag) {
      edge_indices.push_back(i);
    }
  }

  for (unsigned int i = 0; i < corner_group.n; i++) {
    if (corner_group.h_connection_medium_tag(i) == MediumTag) {
      corner_indices.push_back(i);
    }
  }

  return std::make_tuple(face_indices, edge_indices, corner_indices);
}

} // anonymous namespace

// ---------------------------------------------------------------------------
// compute_message_tag_base
//
// Uses modular arithmetic to stay within the MPI standard minimum tag
// upper bound (MPI_TAG_UB >= 32767). Reserves 8 tag slots per rank-pair
// for the handshake protocol (+0..+7):
//   +0  metadata (nfaces, nedges, ncorners, ngll, nglob)
//   +1  rotated face indices [nfaces][ngll][ngll]
//   +2  face neighbor element indices [nfaces]
//   +3  reflected edge indices [nedges][ngll]
//   +4  edge neighbor element indices [nedges]
//   +5  corner neighbor element indices [ncorners]
//   +6  corner nglob indices [ncorners]
//   +7  (reserved for field-data exchange in mpi_buffer)
// ---------------------------------------------------------------------------
int specfem::assembly::mpi_impl::compute_message_tag_base(
    unsigned int sender_rank, unsigned int receiver_rank) {
  // Modulo chosen so that 8 * modulo < 2^18 and modulo < 32767/8
  return static_cast<int>(
      (static_cast<unsigned long>(sender_rank) * 1000003UL + receiver_rank) %
      4093); // 4093 * 8 = 32744 < 32767 (MPI_TAG_UB minimum)
}

namespace {

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

unsigned int apply_reflection(unsigned int ipoint, unsigned int ngll,
                              bool do_reflect) {
  return do_reflect ? (ngll - 1 - ipoint) : ipoint;
}

} // anonymous namespace

// ---------------------------------------------------------------------------
// communication_group constructor (base)
//
// Populates the common fields shared by face, edge, and corner groups:
// orientations (local and neighbor) and spectral element indices. Does not
// compute any transformation metadata (theta/reflect) — that is left to the
// derived-class constructors.
// ---------------------------------------------------------------------------
specfem::assembly::mpi_impl::communication_group::communication_group(
    const unsigned int my_rank, const unsigned int neighbor_rank,
    const std::vector<specfem::mesh::adjacency_graph<
        specfem::element::dimension_tag::dim3>::MPIEdgeProperties> &edges,
    const specfem::assembly::element_types<dimension_tag> element_types,
    const int ngllz, const int nglly, const int ngllx)
    : my_rank(my_rank), neighbor_rank(neighbor_rank), n(edges.size()) {
  const specfem::mesh_entity::element<specfem::element::dimension_tag::dim3>
      element(ngllz, nglly, ngllx);
  ngll = element.ngll;

  if (n == 0)
    return;

  my_orientation = FaceNormalViewType("communication_group::my_orientation", n);
  neighbor_orientation =
      FaceNormalViewType("communication_group::neighbor_orientation", n);
  my_element = ElementIndexViewType("communication_group::my_element", n);
  neighbor_element =
      ElementIndexViewType("communication_group::neighbor_element", n);
  connection_medium_tag =
      MediumTagViewType("communication_group::connection_medium_tag", n);

  h_my_orientation = Kokkos::create_mirror_view(my_orientation);
  h_neighbor_orientation = Kokkos::create_mirror_view(neighbor_orientation);
  h_my_element = Kokkos::create_mirror_view(my_element);
  h_neighbor_element = Kokkos::create_mirror_view(neighbor_element);
  h_connection_medium_tag = Kokkos::create_mirror_view(connection_medium_tag);

  for (unsigned int iface = 0; iface < n; iface++) {
    const auto &edge = edges[iface];
    h_my_orientation(iface) = edge.orientation;
    h_neighbor_orientation(iface) = edge.neighbor_orientation;
    h_my_element(iface) = static_cast<int>(edge.local_index);
    h_neighbor_element(iface) = static_cast<int>(edge.neighbor_local_index);

    h_connection_medium_tag(iface) = element_types.get_medium_tag(
        edge.local_index); // Assuming medium tag is determined by local element
  }

  Kokkos::deep_copy(my_orientation, h_my_orientation);
  Kokkos::deep_copy(neighbor_orientation, h_neighbor_orientation);
  Kokkos::deep_copy(my_element, h_my_element);
  Kokkos::deep_copy(neighbor_element, h_neighbor_element);
  Kokkos::deep_copy(connection_medium_tag, h_connection_medium_tag);
}

// ---------------------------------------------------------------------------
// face_communication_group constructor
//
// Calls the base constructor for common fields, then computes the discrete
// rotation index theta ∈ [0,3] for each face using anchor points. The anchor
// point uniquely identifies one corner of the shared face in both elements,
// allowing the relative 90°-rotation step to be computed as:
//   r = (neigh_anchor_pos - my_anchor_pos) mod 4
// Stored as unsigned char (1 byte vs 8 bytes for a floating-point angle).
// ---------------------------------------------------------------------------
specfem::assembly::mpi_impl::face_communication_group::face_communication_group(
    const unsigned int my_rank, const unsigned int neighbor_rank,
    const std::vector<specfem::mesh::adjacency_graph<
        specfem::element::dimension_tag::dim3>::MPIEdgeProperties> &edges,
    const specfem::assembly::element_types<dimension_tag> element_types,
    const int ngllz, const int nglly, const int ngllx)
    : communication_group(my_rank, neighbor_rank, edges, element_types, ngllz,
                          nglly, ngllx) {

  if (n == 0)
    return;

  theta = ThetaViewType("face_communication_group::theta", n);
  h_theta = Kokkos::create_mirror_view(theta);

  for (unsigned int iface = 0; iface < n; iface++) {
    const auto &edge = edges[iface];

    const auto my_corners =
        specfem::mesh_entity::corners_of_face(edge.orientation);
    const auto neigh_corners =
        specfem::mesh_entity::corners_of_face(edge.neighbor_orientation);

    const int my_anchor_pos =
        find_anchor_position(my_corners, edge.local_anchor_point);
    const int neigh_anchor_pos =
        find_anchor_position(neigh_corners, edge.neighbor_anchor_point);

    h_theta(iface) =
        static_cast<unsigned char>((neigh_anchor_pos - my_anchor_pos + 4) % 4);
  }

  Kokkos::deep_copy(theta, h_theta);
}

// ---------------------------------------------------------------------------
// edge_communication_group constructor
//
// Calls the base constructor for common fields, then computes the reflection
// flag for each shared edge. An edge has two endpoint corners; the anchor
// point identifies which corner is at index 0 in each partition. If the
// anchor is at the same endpoint in both partitions, the 1D traversal
// directions match (reflect=false); if at opposite endpoints, they are
// reversed (reflect=true).
// ---------------------------------------------------------------------------
specfem::assembly::mpi_impl::edge_communication_group::edge_communication_group(
    const unsigned int my_rank, const unsigned int neighbor_rank,
    const std::vector<specfem::mesh::adjacency_graph<
        specfem::element::dimension_tag::dim3>::MPIEdgeProperties> &edges,
    const specfem::assembly::element_types<dimension_tag> element_types,
    const int ngllz, const int nglly, const int ngllx)
    : communication_group(my_rank, neighbor_rank, edges, element_types, ngllz,
                          nglly, ngllx) {

  if (n == 0)
    return;

  reflect = ReflectViewType("edge_communication_group::reflect", n);
  h_reflect = Kokkos::create_mirror_view(reflect);

  for (unsigned int iedge = 0; iedge < n; iedge++) {
    const auto &edge = edges[iedge];

    const auto my_corners =
        specfem::mesh_entity::dim3::corners_of_edge(edge.orientation);
    const auto neigh_corners =
        specfem::mesh_entity::dim3::corners_of_edge(edge.neighbor_orientation);

    // Position of the anchor in the local endpoint array (0 or 1)
    int pos_local = -1;
    for (int k = 0; k < 2; k++) {
      if (my_corners[k] == edge.local_anchor_point) {
        pos_local = k;
        break;
      }
    }
    // Position of the anchor in the neighbor endpoint array (0 or 1)
    int pos_neigh = -1;
    for (int k = 0; k < 2; k++) {
      if (neigh_corners[k] == edge.neighbor_anchor_point) {
        pos_neigh = k;
        break;
      }
    }

    if (pos_local == -1 || pos_neigh == -1) {
      specfem::Logger::error("edge_communication_group: anchor point not found "
                             "among edge endpoints");
    }

    // Anchors at the same endpoint → same traversal direction (no reflect)
    // Anchors at opposite endpoints → traversal is reversed (reflect)
    h_reflect(iedge) = (pos_local != pos_neigh);
  }

  Kokkos::deep_copy(reflect, h_reflect);
}

// ---------------------------------------------------------------------------
// corner_communication_group constructor
//
// Corners are single GLL points; no transformation metadata is needed.
// Delegates entirely to the base constructor.
// ---------------------------------------------------------------------------
specfem::assembly::mpi_impl::corner_communication_group::
    corner_communication_group(
        const unsigned int my_rank, const unsigned int neighbor_rank,
        const std::vector<specfem::mesh::adjacency_graph<
            specfem::element::dimension_tag::dim3>::MPIEdgeProperties> &edges,
        const specfem::assembly::element_types<dimension_tag> element_types,
        const int ngllz, const int nglly, const int ngllx)
    : communication_group(my_rank, neighbor_rank, edges, element_types, ngllz,
                          nglly, ngllx) {}

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
template <specfem::simulation::field_type FieldType,
          specfem::element::dimension_tag DimensionTag,
          specfem::element::medium_tag MediumTag>
specfem::assembly::mpi_impl::packer<FieldType, DimensionTag, MediumTag>::packer(
    const specfem::assembly::mpi_impl::corner_communication_group &corner_group,
    const specfem::assembly::mpi_impl::edge_communication_group &edge_group,
    const specfem::assembly::mpi_impl::face_communication_group &face_group,
    const specfem::mesh_entity::element<dimension_tag> &element,
    const specfem::assembly::fields<dimension_tag> &fields) {

  // Derive ranks from first non-empty group
  if (face_group.n > 0) {
    this->my_rank = face_group.my_rank;
    this->neighbor_rank = face_group.neighbor_rank;
  } else if (edge_group.n > 0) {
    this->my_rank = edge_group.my_rank;
    this->neighbor_rank = edge_group.neighbor_rank;
  } else if (corner_group.n > 0) {
    this->my_rank = corner_group.my_rank;
    this->neighbor_rank = corner_group.neighbor_rank;
  } else {
    this->my_rank = 0;
    this->neighbor_rank = 0;
  }

  // Validate that all non-empty groups agree on rank pair
  auto check_ranks = [&](const auto &group, const char *label) {
    if (group.n > 0 && (group.my_rank != this->my_rank ||
                        group.neighbor_rank != this->neighbor_rank)) {
      specfem::Logger::error(
          std::string(label) +
          ": communication groups do not share the same ranks");
    }
  };
  check_ranks(face_group, "packer::face_group");
  check_ranks(edge_group, "packer::edge_group");
  check_ranks(corner_group, "packer::corner_group");

  // Filter out only faces/edges/corners that match the specified medium tag
  // (MediumTag)
  auto [face_indices, edge_indices, corner_indices] =
      filter_indices_by_medium_tag<MediumTag>(face_group, edge_group,
                                              corner_group);

  this->nfaces = face_indices.size();
  this->nedges = edge_indices.size();
  this->ncorners = corner_indices.size();

  // Pick ngll from first non-empty group (all share the same value)
  if (face_group.n > 0)
    this->ngll = face_group.ngll;
  else if (edge_group.n > 0)
    this->ngll = edge_group.ngll;
  else if (corner_group.n > 0)
    this->ngll = corner_group.ngll;
  else
    this->ngll = 0;

  // Early exit if no faces in this communication group
  if ((this->nfaces == 0) && (this->nedges == 0) && (this->ncorners == 0))
    return;

  const auto &simulation_field =
      fields.template get_simulation_field<FieldType>();

  Kokkos::View<unsigned char *, Kokkos::HostSpace> filtered_theta(
      "packer::face_theta", this->nfaces);
  Kokkos::View<int *, Kokkos::HostSpace> filtered_face_neighbors(
      "packer::face_neighbors", this->nfaces);

  // First pass: collect unique global indices across all face GLL points.
  // Points shared between adjacent faces (at shared edges/corners) are
  // inserted only once; insertion order is preserved via the vector.
  std::vector<int> unique_iglobs;
  std::unordered_map<int, int> seen; // Maps global iglob to local mapping index
  unique_iglobs.reserve(
      this->nfaces * this->ngll * this->ngll + this->nedges * this->ngll +
      this->ncorners); // Reserve upper bound to avoid rehashing

  Kokkos::View<int ***, Kokkos::HostSpace> face_index_mapping(
      "packer::face_index_mapping", this->nfaces, this->ngll, this->ngll);

  for (unsigned int iface = 0; iface < this->nfaces; iface++) {
    const int ielem = face_group.h_my_element(face_indices[iface]);
    const auto face_type = face_group.h_my_orientation(face_indices[iface]);
    filtered_theta(iface) = face_group.h_theta(face_indices[iface]);
    filtered_face_neighbors(iface) =
        face_group.h_neighbor_element(face_indices[iface]);

    for (unsigned int ipoint_j = 0; ipoint_j < this->ngll; ipoint_j++) {
      for (unsigned int ipoint_i = 0; ipoint_i < this->ngll; ipoint_i++) {
        const int point_linear = ipoint_j * this->ngll + ipoint_i;

        // Map face 2D coordinates to element 3D coordinates
        const auto [iz, iy, ix] =
            element.map_coordinates(face_type, point_linear);

        // Look up global assembled index
        const int iglob = simulation_field.template get_iglob<false, MediumTag>(
            ielem, iz, iy, ix);

        // Insert only if not already seen
        if (seen.find(iglob) == seen.end()) {
          seen[iglob] = unique_iglobs.size(); // local mapping index
          unique_iglobs.push_back(iglob);
        }

        face_index_mapping(iface, ipoint_j, ipoint_i) = seen[iglob];
      }
    }
  }

  Kokkos::View<int **, Kokkos::HostSpace> edge_index_mapping(
      "packer::edge_index_mapping", this->nedges, this->ngll);
  Kokkos::View<bool *, Kokkos::HostSpace> filtered_reflect(
      "packer::edge_reflect", this->nedges);
  Kokkos::View<int *, Kokkos::HostSpace> filtered_edge_neighbors(
      "packer::edge_neighbors", this->nedges);

  for (unsigned int iedge = 0; iedge < this->nedges; iedge++) {
    const int ielem = edge_group.h_my_element(edge_indices[iedge]);
    const auto edge_type = edge_group.h_my_orientation(edge_indices[iedge]);
    filtered_reflect(iedge) = edge_group.h_reflect(edge_indices[iedge]);
    filtered_edge_neighbors(iedge) =
        edge_group.h_neighbor_element(edge_indices[iedge]);

    for (unsigned int ipoint = 0; ipoint < this->ngll; ipoint++) {
      // Map edge 1D coordinate to element 3D coordinates
      const auto [iz, iy, ix] = element.map_coordinates(edge_type, ipoint);

      // Look up global assembled index
      const int iglob = simulation_field.template get_iglob<false, MediumTag>(
          ielem, iz, iy, ix);

      // Insert only if not already seen
      if (seen.find(iglob) == seen.end()) {
        seen[iglob] = unique_iglobs.size(); // local mapping index
        unique_iglobs.push_back(iglob);
      }

      edge_index_mapping(iedge, ipoint) = seen[iglob];
    }
  }

  Kokkos::View<int *, Kokkos::HostSpace> corner_index_mapping(
      "packer::corner_index_mapping", this->ncorners);
  Kokkos::View<int *, Kokkos::HostSpace> filtered_corner_neighbors(
      "packer::corner_neighbors", this->ncorners);

  for (unsigned int icorner = 0; icorner < this->ncorners; icorner++) {
    const int ielem = corner_group.h_my_element(corner_indices[icorner]);
    const auto corner_type =
        corner_group.h_my_orientation(corner_indices[icorner]);
    filtered_corner_neighbors(icorner) =
        corner_group.h_neighbor_element(corner_indices[icorner]);

    // Map corner coordinate to element 3D coordinates
    const auto [iz, iy, ix] = element.map_coordinates(corner_type);
    // Look up global assembled index
    const int iglob = simulation_field.template get_iglob<false, MediumTag>(
        ielem, iz, iy, ix);
    // Insert only if not already seen
    if (seen.find(iglob) == seen.end()) {
      seen[iglob] = unique_iglobs.size(); // local mapping index
      unique_iglobs.push_back(iglob);
    }
    corner_index_mapping(icorner) = seen.at(iglob);
  }

  this->nglob = unique_iglobs.size();

  this->send_unpacking_indices(face_index_mapping, filtered_theta,
                               filtered_face_neighbors, edge_index_mapping,
                               filtered_reflect, filtered_edge_neighbors,
                               filtered_corner_neighbors, corner_index_mapping);

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
// Applies rotation/reflection transformations to GLL indices and sends them
// to the neighboring MPI process using 7 separate blocking MPI messages:
//   +0  Metadata: {nfaces, nedges, ncorners, ngll, nglob}
//   +1  Rotated face indices [nfaces][ngll][ngll] (theta-based rotation)
//   +2  Face neighbor element indices [nfaces]
//   +3  Reflected edge indices [nedges][ngll] (reflect flag applied)
//   +4  Edge neighbor element indices [nedges]
//   +5  Corner neighbor element indices [ncorners]
//   +6  Corner nglob indices [ncorners]
//
// Uses blocking MPI_Send (wrapped in SPECFEM_MPI_SAFECALL) to ensure data
// is transmitted before local buffers are deallocated. All calls protected
// for non-MPI builds via macro.
// ---------------------------------------------------------------------------
template <specfem::simulation::field_type FieldType,
          specfem::element::dimension_tag DimensionTag,
          specfem::element::medium_tag MediumTag>
void specfem::assembly::mpi_impl::packer<FieldType, DimensionTag, MediumTag>::
    send_unpacking_indices(
        Kokkos::View<int ***, Kokkos::HostSpace> face_index_mapping,
        Kokkos::View<unsigned char *, Kokkos::HostSpace> theta,
        Kokkos::View<int *, Kokkos::HostSpace> neighbor_element,
        Kokkos::View<int **, Kokkos::HostSpace> edge_index_mapping,
        Kokkos::View<bool *, Kokkos::HostSpace> edge_reflect,
        Kokkos::View<int *, Kokkos::HostSpace> edge_neighbor_element,
        Kokkos::View<int *, Kokkos::HostSpace> corner_neighbor_element,
        Kokkos::View<int *, Kokkos::HostSpace> corner_index_mapping) {

  // Early exit if no faces to send
  if (this->nfaces == 0 && this->nedges == 0 && this->ncorners == 0)
    return;

  // -----------------------------------------------------------------------
  // Step 1: Allocate host-space buffer for rotated indices
  // -----------------------------------------------------------------------
  Kokkos::View<int ***, Kokkos::HostSpace> rotated_indices(
      "packer::send::rotated_indices", this->nfaces, this->ngll, this->ngll);

  Kokkos::View<int **, Kokkos::HostSpace> reflect_indices(
      "packer::send::reflect_indices", this->nedges, this->ngll);

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

  for (unsigned int iedge = 0; iedge < this->nedges; iedge++) {
    const bool do_reflect = edge_reflect(iedge);
    for (unsigned int ipoint = 0; ipoint < this->ngll; ipoint++) {
      const unsigned int idx = apply_reflection(ipoint, this->ngll, do_reflect);

      reflect_indices(iedge, idx) = edge_index_mapping(iedge, ipoint);
    }
  }
  // -----------------------------------------------------------------------
  // Step 3: Send 7 MPI messages (base_tag + offset, see tag layout above)
  // -----------------------------------------------------------------------
  MPI_Comm comm = specfem::MPI::communicator();

  // Message 1: Send metadata (nfaces, ngll, nglob)
  unsigned int metadata[5] = { this->nfaces, this->nedges, this->ncorners,
                               this->ngll, this->nglob };
  SPECFEM_MPI_SAFECALL(
      MPI_Send(metadata, 5, MPI_UNSIGNED, this->neighbor_rank, 101, comm));

  // Message 2: Send rotated face indices [nfaces][ngll][ngll]
  SPECFEM_MPI_SAFECALL(
      MPI_Send(rotated_indices.data(),
               static_cast<int>(this->nfaces * this->ngll * this->ngll),
               MPI_INT, this->neighbor_rank, 102, comm));

  // Message 3: Send neighbor element indices [nfaces]
  SPECFEM_MPI_SAFECALL(MPI_Send(neighbor_element.data(),
                                static_cast<int>(this->nfaces), MPI_INT,
                                this->neighbor_rank, 103, comm));

  // Message 4: Send reflected edge indices [nedges][ngll]
  SPECFEM_MPI_SAFECALL(MPI_Send(reflect_indices.data(),
                                static_cast<int>(this->nedges * this->ngll),
                                MPI_INT, this->neighbor_rank, 104, comm));

  // Message 5: Send edge neighbor element indices [nedges]
  SPECFEM_MPI_SAFECALL(MPI_Send(edge_neighbor_element.data(),
                                static_cast<int>(this->nedges), MPI_INT,
                                this->neighbor_rank, 105, comm));

  // Message 6: Send corner neighbor element indices [ncorners]
  SPECFEM_MPI_SAFECALL(MPI_Send(corner_neighbor_element.data(),
                                static_cast<int>(this->ncorners), MPI_INT,
                                this->neighbor_rank, 106, comm));

  // Message 7: Send corner nglob indices [ncorners]
  SPECFEM_MPI_SAFECALL(MPI_Send(corner_index_mapping.data(),
                                static_cast<int>(this->ncorners), MPI_INT,
                                this->neighbor_rank, 107, comm));
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
template <specfem::simulation::field_type FieldType,
          specfem::element::dimension_tag DimensionTag,
          specfem::element::medium_tag MediumTag>
specfem::assembly::mpi_impl::unpacker<FieldType, DimensionTag, MediumTag>::
    unpacker(
        const specfem::assembly::mpi_impl::corner_communication_group
            &corner_group,
        const specfem::assembly::mpi_impl::edge_communication_group &edge_group,
        const specfem::assembly::mpi_impl::face_communication_group &face_group,
        const specfem::mesh_entity::element<dimension_tag> &element) {

  // Filter interfaces by medium tag using helper function
  auto [face_indices, edge_indices, corner_indices] =
      filter_indices_by_medium_tag<MediumTag>(face_group, edge_group,
                                              corner_group);

  this->nfaces = face_indices.size();
  this->nedges = edge_indices.size();
  this->ncorners = corner_indices.size();

  // Build filtered local orientation views for use in
  // assemble_unpacking_mapping
  h_face_orientations =
      OrientationHostView("unpacker::h_face_orientations", this->nfaces);
  for (unsigned int i = 0; i < this->nfaces; i++)
    h_face_orientations(i) = face_group.h_my_orientation(face_indices[i]);

  h_edge_orientations =
      OrientationHostView("unpacker::h_edge_orientations", this->nedges);
  for (unsigned int i = 0; i < this->nedges; i++)
    h_edge_orientations(i) = edge_group.h_my_orientation(edge_indices[i]);

  h_corner_orientations =
      OrientationHostView("unpacker::h_corner_orientations", this->ncorners);
  for (unsigned int i = 0; i < this->ncorners; i++)
    h_corner_orientations(i) = corner_group.h_my_orientation(corner_indices[i]);

  // Derive ranks and ngll from first non-empty group
  if (face_group.n > 0) {
    this->my_rank = face_group.my_rank;
    this->neighbor_rank = face_group.neighbor_rank;
    this->ngll = face_group.ngll;
  } else if (edge_group.n > 0) {
    this->my_rank = edge_group.my_rank;
    this->neighbor_rank = edge_group.neighbor_rank;
    this->ngll = edge_group.ngll;
  } else if (corner_group.n > 0) {
    this->my_rank = corner_group.my_rank;
    this->neighbor_rank = corner_group.neighbor_rank;
    this->ngll = corner_group.ngll;
  } else {
    this->my_rank = 0;
    this->neighbor_rank = 0;
    this->ngll = 0;
  }

  // Validate that all non-empty groups agree on rank pair
  auto check_ranks = [&](const auto &group, const char *label) {
    if (group.n > 0 && (group.my_rank != this->my_rank ||
                        group.neighbor_rank != this->neighbor_rank)) {
      specfem::Logger::error(
          std::string(label) +
          ": communication groups do not share the same ranks");
    }
  };
  check_ranks(face_group, "unpacker::face_group");
  check_ranks(edge_group, "unpacker::edge_group");
  check_ranks(corner_group, "unpacker::corner_group");
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
template <specfem::simulation::field_type FieldType,
          specfem::element::dimension_tag DimensionTag,
          specfem::element::medium_tag MediumTag>
std::tuple<std::array<MPI_Request, 7>,
           Kokkos::View<unsigned int[5], Kokkos::HostSpace>,
           Kokkos::View<int ***, Kokkos::HostSpace>,
           Kokkos::View<int *, Kokkos::HostSpace>,
           Kokkos::View<int **, Kokkos::HostSpace>,
           Kokkos::View<int *, Kokkos::HostSpace>,
           Kokkos::View<int *, Kokkos::HostSpace>,
           Kokkos::View<int *, Kokkos::HostSpace> >
specfem::assembly::mpi_impl::unpacker<FieldType, DimensionTag,
                                      MediumTag>::receive_unpacking_buffers() {

  // Early exit: if there are no interfaces of any type, return zero-sized
  // views and null requests so that no MPI_Irecv calls are posted and
  // assemble_unpacking_mapping can skip MPI_Waitall safely.
  if (this->nfaces == 0 && this->nedges == 0 && this->ncorners == 0) {
    std::array<MPI_Request, 7> null_requests{};
    null_requests.fill(MPI_REQUEST_NULL);
    return {
      null_requests,
      Kokkos::View<unsigned int[5], Kokkos::HostSpace>(
          "unpacker::metadata_buf"),
      Kokkos::View<int ***, Kokkos::HostSpace>("unpacker::recv_face_indices", 0,
                                               0, 0),
      Kokkos::View<int *, Kokkos::HostSpace>("unpacker::face_elements", 0),
      Kokkos::View<int **, Kokkos::HostSpace>("unpacker::recv_edge_indices", 0,
                                              0),
      Kokkos::View<int *, Kokkos::HostSpace>("unpacker::edge_elements", 0),
      Kokkos::View<int *, Kokkos::HostSpace>("unpacker::corner_nglob_idx", 0),
      Kokkos::View<int *, Kokkos::HostSpace>("unpacker::corner_elements", 0)
    };
  }

  // Allocate all receive buffers before posting any Irecv
  Kokkos::View<unsigned int[5], Kokkos::HostSpace> metadata_buf(
      "unpacker::metadata_buf");

  Kokkos::View<int ***, Kokkos::HostSpace> recv_face_indices(
      Kokkos::view_alloc(Kokkos::WithoutInitializing,
                         "unpacker::recv_face_indices"),
      this->nfaces, this->ngll, this->ngll);
  Kokkos::View<int *, Kokkos::HostSpace> face_elements(
      Kokkos::view_alloc(Kokkos::WithoutInitializing,
                         "unpacker::face_elements"),
      this->nfaces);

  Kokkos::View<int **, Kokkos::HostSpace> recv_edge_indices(
      Kokkos::view_alloc(Kokkos::WithoutInitializing,
                         "unpacker::recv_edge_indices"),
      this->nedges, this->ngll);
  Kokkos::View<int *, Kokkos::HostSpace> edge_elements(
      Kokkos::view_alloc(Kokkos::WithoutInitializing,
                         "unpacker::edge_elements"),
      this->nedges);

  Kokkos::View<int *, Kokkos::HostSpace> corner_elements(
      Kokkos::view_alloc(Kokkos::WithoutInitializing,
                         "unpacker::corner_elements"),
      this->ncorners);
  Kokkos::View<int *, Kokkos::HostSpace> corner_nglob_idx(
      Kokkos::view_alloc(Kokkos::WithoutInitializing,
                         "unpacker::corner_nglob_idx"),
      this->ncorners);

  // Post all 7 Irecv calls before any MPI_Wait
  std::array<MPI_Request, 7> requests{};
  MPI_Comm comm = specfem::MPI::communicator();

  // Message 1: metadata {nfaces, nedges, ncorners, ngll, nglob}
  SPECFEM_MPI_SAFECALL(MPI_Irecv(metadata_buf.data(), 5, MPI_UNSIGNED,
                                 this->neighbor_rank, 101, comm, &requests[0]));

  // Message 2: rotated face indices [nfaces][ngll][ngll]
  SPECFEM_MPI_SAFECALL(
      MPI_Irecv(recv_face_indices.data(),
                static_cast<int>(this->nfaces * this->ngll * this->ngll),
                MPI_INT, this->neighbor_rank, 102, comm, &requests[1]));

  // Message 3: face neighbor element indices [nfaces]
  SPECFEM_MPI_SAFECALL(MPI_Irecv(face_elements.data(),
                                 static_cast<int>(this->nfaces), MPI_INT,
                                 this->neighbor_rank, 103, comm, &requests[2]));

  // Message 4: reflected edge indices [nedges][ngll]
  SPECFEM_MPI_SAFECALL(MPI_Irecv(
      recv_edge_indices.data(), static_cast<int>(this->nedges * this->ngll),
      MPI_INT, this->neighbor_rank, 104, comm, &requests[3]));

  // Message 5: edge neighbor element indices [nedges]
  SPECFEM_MPI_SAFECALL(MPI_Irecv(edge_elements.data(),
                                 static_cast<int>(this->nedges), MPI_INT,
                                 this->neighbor_rank, 105, comm, &requests[4]));

  // Message 6: corner neighbor element indices [ncorners]
  SPECFEM_MPI_SAFECALL(MPI_Irecv(corner_elements.data(),
                                 static_cast<int>(this->ncorners), MPI_INT,
                                 this->neighbor_rank, 106, comm, &requests[5]));
  // Message 7: corner nglob indices [ncorners]
  SPECFEM_MPI_SAFECALL(MPI_Irecv(corner_nglob_idx.data(),
                                 static_cast<int>(this->ncorners), MPI_INT,
                                 this->neighbor_rank, 107, comm, &requests[6]));

  return {
    requests,          metadata_buf,  recv_face_indices, face_elements,
    recv_edge_indices, edge_elements, corner_nglob_idx,  corner_elements
  };
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
template <specfem::simulation::field_type FieldType,
          specfem::element::dimension_tag DimensionTag,
          specfem::element::medium_tag MediumTag>
void specfem::assembly::mpi_impl::unpacker<FieldType, DimensionTag, MediumTag>::
    assemble_unpacking_mapping(
        const std::array<MPI_Request, 7> &requests,
        const Kokkos::View<unsigned int[5], Kokkos::HostSpace> metadata_buf,
        const Kokkos::View<int ***, Kokkos::HostSpace> recv_face_indices,
        const Kokkos::View<int *, Kokkos::HostSpace> face_elements,
        const Kokkos::View<int **, Kokkos::HostSpace> recv_edge_indices,
        const Kokkos::View<int *, Kokkos::HostSpace> edge_elements,
        const Kokkos::View<int *, Kokkos::HostSpace> corner_nglob_idx,
        const Kokkos::View<int *, Kokkos::HostSpace> corner_elements,
        const Kokkos::View<specfem::mesh_entity::dim3::type *,
                           Kokkos::HostSpace>
            my_face_orientations,
        const Kokkos::View<specfem::mesh_entity::dim3::type *,
                           Kokkos::HostSpace>
            my_edge_orientations,
        const Kokkos::View<specfem::mesh_entity::dim3::type *,
                           Kokkos::HostSpace>
            my_corner_orientations,
        const specfem::mesh_entity::element<dimension_tag> &element,
        const specfem::assembly::fields<dimension_tag> &fields) {

  if (this->nfaces == 0 && this->nedges == 0 && this->ncorners == 0)
    return; // No Irecv calls were posted; nothing to wait on.

  const auto &simulation_field =
      fields.template get_simulation_field<FieldType>();

  // Wait for all 7 receives to complete
  SPECFEM_MPI_SAFECALL(MPI_Waitall(
      7, const_cast<MPI_Request *>(requests.data()), MPI_STATUSES_IGNORE));

  // Validate received metadata {nfaces, nedges, ncorners, ngll, nglob}
  if (metadata_buf[0] != this->nfaces || metadata_buf[1] != this->nedges ||
      metadata_buf[2] != this->ncorners || metadata_buf[3] != this->ngll) {
    specfem::Logger::error(
        "unpacker::assemble_unpacking_mapping: metadata mismatch with sender");
  }

  const int nglob = static_cast<int>(metadata_buf[4]);
  this->nglob = static_cast<unsigned int>(nglob);
  mapping = IndexMappingView("unpacker::mapping", nglob);
  h_mapping = Kokkos::create_mirror_view(mapping);

  // Fill mapping from faces: recv_face_indices[iface][j][i] → nglob position
  for (unsigned int iface = 0; iface < this->nfaces; iface++) {
    const int ielem = face_elements(iface);
    const auto face_type = my_face_orientations(iface);
    for (unsigned int j = 0; j < this->ngll; j++) {
      for (unsigned int i = 0; i < this->ngll; i++) {
        const int nglob_idx = recv_face_indices(iface, j, i);
        const int point_linear = j * this->ngll + i;
        const auto [iz, iy, ix] =
            element.map_coordinates(face_type, point_linear);
        h_mapping(nglob_idx) =
            simulation_field.template get_iglob<false, MediumTag>(ielem, iz, iy,
                                                                  ix);
      }
    }
  }

  // Fill mapping from edges: recv_edge_indices[iedge][ipoint] → nglob position
  for (unsigned int iedge = 0; iedge < this->nedges; iedge++) {
    const int ielem = edge_elements(iedge);
    const auto edge_type = my_edge_orientations(iedge);
    for (unsigned int ipoint = 0; ipoint < this->ngll; ipoint++) {
      const int nglob_idx = recv_edge_indices(iedge, ipoint);
      const auto [iz, iy, ix] = element.map_coordinates(edge_type, ipoint);
      h_mapping(nglob_idx) =
          simulation_field.template get_iglob<false, MediumTag>(ielem, iz, iy,
                                                                ix);
    }
  }

  // Fill mapping from corners: corner_nglob_idx[icorner] → nglob position
  for (unsigned int icorner = 0; icorner < this->ncorners; icorner++) {
    const int ielem = corner_elements(icorner);
    const auto corner_type = my_corner_orientations(icorner);
    const int nglob_idx = corner_nglob_idx(icorner);
    const auto [iz, iy, ix] = element.map_coordinates(corner_type);
    h_mapping(nglob_idx) =
        simulation_field.template get_iglob<false, MediumTag>(ielem, iz, iy,
                                                              ix);
  }

  Kokkos::deep_copy(mapping, h_mapping);
}

template <specfem::simulation::field_type FieldType,
          specfem::element::dimension_tag DimensionTag,
          specfem::element::medium_tag MediumTag>
specfem::assembly::mpi_impl::communication_pattern<
    FieldType, DimensionTag, MediumTag>::PendingReceiveState
specfem::assembly::mpi_impl::
    communication_pattern<FieldType, DimensionTag, MediumTag>::post_receives(
        const corner_communication_group &corner_group,
        const edge_communication_group &edge_group,
        const face_communication_group &face_group,
        const specfem::mesh_entity::element<dimension_tag> &element) {

  // Derive ranks from first non-empty group
  if (face_group.n > 0) {
    this->my_rank = face_group.my_rank;
    this->neighbor_rank = face_group.neighbor_rank;
  } else if (edge_group.n > 0) {
    this->my_rank = edge_group.my_rank;
    this->neighbor_rank = edge_group.neighbor_rank;
  } else {
    this->my_rank = corner_group.my_rank;
    this->neighbor_rank = corner_group.neighbor_rank;
  }

  // Validate that all non-empty groups agree on rank pair
  auto check_ranks = [&](const auto &group, const char *label) {
    if (group.n > 0 && (group.my_rank != this->my_rank ||
                        group.neighbor_rank != this->neighbor_rank)) {
      specfem::Logger::error(
          std::string(label) +
          ": communication groups do not share the same ranks");
    }
  };
  check_ranks(face_group, "communication_pattern::face_group");
  check_ranks(edge_group, "communication_pattern::edge_group");
  check_ranks(corner_group, "communication_pattern::corner_group");

  // Initialise the unpacker and post all non-blocking receives
  unpack = { corner_group, edge_group, face_group, element };
  return unpack.receive_unpacking_buffers();
}

template <specfem::simulation::field_type FieldType,
          specfem::element::dimension_tag DimensionTag,
          specfem::element::medium_tag MediumTag>
void specfem::assembly::mpi_impl::communication_pattern<FieldType, DimensionTag,
                                                        MediumTag>::
    complete_construction(
        const corner_communication_group &corner_group,
        const edge_communication_group &edge_group,
        const face_communication_group &face_group,
        const specfem::mesh_entity::element<dimension_tag> &element,
        const specfem::assembly::fields<dimension_tag> &fields,
        PendingReceiveState pending) {

  auto &[requests, metadata_buf, recv_face_indices, face_elements,
         recv_edge_indices, edge_elements, corner_nglob_idx, corner_elements] =
      pending;

  // Send unpacking indices to neighbor (blocking sends)
  pack = { corner_group, edge_group, face_group, element, fields };

  // Wait for receives and assemble the unpacking mapping
  unpack.assemble_unpacking_mapping(
      requests, metadata_buf, recv_face_indices, face_elements,
      recv_edge_indices, edge_elements, corner_nglob_idx, corner_elements,
      unpack.h_face_orientations, unpack.h_edge_orientations,
      unpack.h_corner_orientations, element, fields);
}

// ---------------------------------------------------------------------------
// mpi<dim3> constructor
//
// Analyzes the mesh adjacency graph to extract all MPI interfaces, groups
// them by neighboring process, and separates them by connection type
// (face, edge, corner). Creates one group object per unique neighbor for
// each connection type.
//
// Algorithm:
// 1. Extract all MPI connections from adjacency_graph.mpi_connections()
// 2. Classify each connection as face, edge, or corner and bucket by neighbor
// 3. Build face_groups, edge_groups, corner_groups from the bucketed maps
// 4. Build communication_patterns from face_groups (edge/corner patterns
//    deferred to a later implementation step)
// ---------------------------------------------------------------------------
specfem::assembly::mpi<specfem::element::dimension_tag::dim3>::mpi(
    const specfem::mesh::adjacency_graph<dimension_tag> &adjacency_graph,
    const specfem::assembly::element_types<dimension_tag> &element_types,
    const specfem::simulation::type simulation,
    const specfem::assembly::fields<dimension_tag> &fields, const int ngllz,
    const int nglly, const int ngllx) {

  const unsigned int my_rank =
      static_cast<unsigned int>(specfem::MPI::get_rank());

  const auto &mpi_conns = adjacency_graph.mpi_connections();

  this->simulation = simulation;

  using MPIEdgeProperties =
      specfem::mesh::adjacency_graph<dimension_tag>::MPIEdgeProperties;
  std::unordered_map<unsigned int, std::vector<MPIEdgeProperties> >
      face_grouped, edge_grouped, corner_grouped;

  for (const auto &conn : mpi_conns) {
    if (specfem::mesh_entity::contains(specfem::mesh_entity::dim3::faces,
                                       conn.orientation)) {
      face_grouped[conn.neighbor_partition].push_back(conn);
    } else if (specfem::mesh_entity::contains(specfem::mesh_entity::dim3::edges,
                                              conn.orientation)) {
      edge_grouped[conn.neighbor_partition].push_back(conn);
    } else {
      corner_grouped[conn.neighbor_partition].push_back(conn);
    }
  }

  face_groups.reserve(face_grouped.size());
  for (auto &[neighbor_rank, edges] : face_grouped) {
    face_groups.emplace_back(my_rank, neighbor_rank, edges, element_types,
                             ngllz, nglly, ngllx);
  }

  edge_groups.reserve(edge_grouped.size());
  for (auto &[neighbor_rank, edges] : edge_grouped) {
    edge_groups.emplace_back(my_rank, neighbor_rank, edges, element_types,
                             ngllz, nglly, ngllx);
  }

  corner_groups.reserve(corner_grouped.size());
  for (auto &[neighbor_rank, edges] : corner_grouped) {
    corner_groups.emplace_back(my_rank, neighbor_rank, edges, element_types,
                               ngllz, nglly, ngllx);
  }

  const specfem::mesh_entity::element<dimension_tag> element(ngllz, nglly,
                                                             ngllx);

  // Build rank → group-index lookup maps
  std::unordered_map<unsigned int, int> face_idx, edge_idx, corner_idx;
  for (int i = 0; i < static_cast<int>(face_groups.size()); i++)
    face_idx[face_groups[i].neighbor_rank] = i;
  for (int i = 0; i < static_cast<int>(edge_groups.size()); i++)
    edge_idx[edge_groups[i].neighbor_rank] = i;
  for (int i = 0; i < static_cast<int>(corner_groups.size()); i++)
    corner_idx[corner_groups[i].neighbor_rank] = i;

  // Collect union of all neighbor ranks across all connection types
  std::unordered_set<unsigned int> all_neighbors;
  for (const auto &g : face_groups)
    all_neighbors.insert(g.neighbor_rank);
  for (const auto &g : edge_groups)
    all_neighbors.insert(g.neighbor_rank);
  for (const auto &g : corner_groups)
    all_neighbors.insert(g.neighbor_rank);

  static const mpi_impl::face_communication_group empty_face{};
  static const mpi_impl::edge_communication_group empty_edge{};
  static const mpi_impl::corner_communication_group empty_corner{};

  specfem::tag_dispatch::for_each(
      medium_combinations, [&]<typename TagsType>() {
        auto build_map = [&]<specfem::simulation::field_type FT>(auto &map) {
          // Build communication patterns using three-phase approach:
          // Phase 1 (per neighbor): Construct unpacker and post all MPI_Irecv
          // calls Phase 2 (per neighbor): Construct packer and send blocking
          // MPI_Send messages Phase 3 (per neighbor): Wait for receives and
          // assemble unpacking mapping
          //
          // This ordering prevents deadlock under circular connectivity: all
          // Irecv calls are posted globally before any MPI_Send, ensuring
          // receiver buffers exist before sender data arrives.
          using PatternType =
              mpi_impl::communication_pattern<FT, dimension_tag,
                                              TagsType::medium_tag>;
          using PendingState = typename PatternType::PendingReceiveState;

          map = std::unordered_map<unsigned int, PatternType>();
          map.reserve(all_neighbors.size());

          // Phase 1: construct all patterns and post all MPI_Irecv calls
          std::unordered_map<unsigned int, PendingState> pending;
          pending.reserve(all_neighbors.size());
          for (unsigned int nr : all_neighbors) {
            const auto &fg =
                face_idx.count(nr) ? face_groups[face_idx.at(nr)] : empty_face;
            const auto &eg =
                edge_idx.count(nr) ? edge_groups[edge_idx.at(nr)] : empty_edge;
            const auto &cg = corner_idx.count(nr)
                                 ? corner_groups[corner_idx.at(nr)]
                                 : empty_corner;
            PatternType pattern;
            pending.emplace(nr, pattern.post_receives(cg, eg, fg, element));
            map.emplace(nr, std::move(pattern));
          }

          // Phase 2+3: send then wait+assemble for each neighbor
          for (unsigned int nr : all_neighbors) {
            const auto &fg =
                face_idx.count(nr) ? face_groups[face_idx.at(nr)] : empty_face;
            const auto &eg =
                edge_idx.count(nr) ? edge_groups[edge_idx.at(nr)] : empty_edge;
            const auto &cg = corner_idx.count(nr)
                                 ? corner_groups[corner_idx.at(nr)]
                                 : empty_corner;
            map.at(nr).complete_construction(cg, eg, fg, element, fields,
                                             std::move(pending.at(nr)));
          }
        };

        if (simulation == specfem::simulation::type::forward) {
          build_map
              .template operator()<specfem::simulation::field_type::forward>(
                  forward_communication_patterns.template get<TagsType>());
        } else if (simulation == specfem::simulation::type::combined) {
          build_map
              .template operator()<specfem::simulation::field_type::backward>(
                  backward_communication_patterns.template get<TagsType>());
          build_map
              .template operator()<specfem::simulation::field_type::adjoint>(
                  adjoint_communication_patterns.template get<TagsType>());
        }
      });
}
