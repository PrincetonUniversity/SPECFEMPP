#include "specfem/assembly/mpi/dim3/mpi.hpp"
#include "specfem/mesh_entity.hpp"
#include "specfem/setup.hpp"
#include <array>
#include <cmath>
#include <stdexcept>
#include <unordered_map>

namespace {

/// Find position of anchor in face corner array
int find_anchor_position(
    const std::array<specfem::mesh_entity::dim3::type, 4> &corners,
    const specfem::mesh_entity::dim3::type anchor) {
  for (int i = 0; i < 4; i++) {
    if (corners[i] == anchor)
      return i;
  }
  throw std::runtime_error("Anchor point not found among face corners");
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

  // Allocate device views for face metadata
  my_orientation =
      FaceNormalViewType("communication_group::my_orientation", nfaces);
  neighbor_orientation =
      FaceNormalViewType("communication_group::neighbor_orientation", nfaces);
  theta = ThetaViewType("communication_group::theta", nfaces);

  // Create host mirrors for initialization
  h_my_orientation = Kokkos::create_mirror_view(my_orientation);
  h_neighbor_orientation = Kokkos::create_mirror_view(neighbor_orientation);
  h_theta = Kokkos::create_mirror_view(theta);

  // Populate host mirrors with connectivity data
  for (unsigned int iface = 0; iface < nfaces; iface++) {
    const auto &edge = edges[iface];

    // --- Face orientations (mesh entity types) ---
    h_my_orientation(iface) = edge.orientation;
    h_neighbor_orientation(iface) = edge.neighbor_orientation;

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
    // r = (neigh_anchor_pos + my_anchor_pos) mod 4
    // Actual rotation angle (if needed) = r * π/2
    // Storing as unsigned char saves memory compared to type_real (1 vs 8
    // bytes)
    const unsigned char r = (neigh_anchor_pos + my_anchor_pos) % 4;
    h_theta(iface) = r;
  }

  // Deep copy host → device
  Kokkos::deep_copy(my_orientation, h_my_orientation);
  Kokkos::deep_copy(neighbor_orientation, h_neighbor_orientation);
  Kokkos::deep_copy(theta, h_theta);
}

// ---------------------------------------------------------------------------
// mpi<dim3> constructor
//
// Analyzes the mesh adjacency graph to extract all face-level MPI interfaces
// and groups them by neighboring process. This creates a communication schedule
// that filters out edge and corner adjacencies.
//
// Algorithm:
// 1. Extract all MPI connections from adjacency_graph.mpi_connections()
// 2. Retrieve the current MPI rank via specfem::MPI::get_rank()
// 3. Iterate and group connections by neighbor partition (MPI rank)
// 4. For each group, filter to keep only face connections
//    (skip connections involving edges or corners)
// 5. Create one communication_group per unique neighbor
//
// Result: communication_groups vector is indexed by neighbor rank, enabling
// efficient kernel dispatch for data exchange.
// ---------------------------------------------------------------------------
specfem::assembly::mpi<specfem::element::dimension_tag::dim3>::mpi(
    const specfem::mesh::adjacency_graph<dimension_tag> &adjacency_graph,
    const int ngllz, const int nglly, const int ngllx) {

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
}
