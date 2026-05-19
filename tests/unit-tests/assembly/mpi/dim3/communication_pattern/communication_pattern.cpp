/**
 * @file communication_pattern.cpp
 * @brief Unit tests for MPI communication pattern round-trip coordinate
 * consistency in 3D spectral element assembly.
 *
 * Validates that the GLL points packed by rank A and sent to rank B physically
 * coincide with the GLL points unpacked by rank B. Correctness is verified via
 * Cartesian (x,y,z) coordinates rather than global indices, which are not
 * unique across partitions.
 *
 * For each communication_pattern, the test:
 * 1. Collects physical coordinates for all unique packed GLL points (packer)
 * 2. Exchanges them with the neighbor rank via MPI
 * 3. Compares the received packer coordinates against the local unpacker
 *    coordinates
 *
 * @see specfem::assembly::mpi_impl::communication_pattern
 */

#include "../fixture.hpp"
#include "specfem/assembly/assembly.hpp"
#include "specfem/enums.hpp"
#include "specfem/mpi.hpp"
#include "specfem/tags.hpp"
#include <algorithm>
#include <array>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace {

using CoordTriple = std::array<double, 3>;

std::vector<CoordTriple> coords_for_mapping(
    const Kokkos::View<int *, Kokkos::DefaultExecutionSpace>::HostMirror
        &h_mapping,
    const std::unordered_map<int, CoordTriple> &iglob_to_coord) {
  const int n = static_cast<int>(h_mapping.extent(0));
  std::vector<CoordTriple> coords(n);
  for (int i = 0; i < n; i++) {
    coords[i] = iglob_to_coord.at(h_mapping(i));
  }
  return coords;
}

} // namespace

/**
 * @brief Parameterized test for communication_pattern round-trip coordinate
 * consistency.
 *
 * After assembly construction (which internally performs the packer/unpacker
 * MPI exchange), this test performs an independent coordinate exchange to
 * verify that the physical GLL points packed on each rank match those unpacked
 * on the neighboring rank.
 */
TEST_P(AssemblyMPI3DTest, CommunicationPattern) {
  const auto &mpi_interfaces = getMPIInterfaces();
  const auto &assembly_mesh = getAssemblyMesh();
  const auto &fields = getFields();
  const auto &element_types = getElementTypes();

  // Count unique neighbors across all connection types
  std::unordered_set<unsigned int> expected_neighbors;
  for (const auto &g : mpi_interfaces.face_groups)
    expected_neighbors.insert(g.neighbor_rank);
  for (const auto &g : mpi_interfaces.edge_groups)
    expected_neighbors.insert(g.neighbor_rank);
  for (const auto &g : mpi_interfaces.corner_groups)
    expected_neighbors.insert(g.neighbor_rank);
  using ElasticTags =
      specfem::tags::Tags<specfem::element::dimension_tag::dim3,
                          specfem::element::medium_tag::elastic>;
  const auto &patterns_map =
      mpi_interfaces.forward_communication_patterns.template get<ElasticTags>();
  ASSERT_EQ(patterns_map.size(), expected_neighbors.size())
      << "communication_patterns count must match unique neighbor count";

  // Build iglob → (x,y,z) map by scanning all elements on this rank
  std::unordered_map<int, CoordTriple> iglob_to_coord;
  {
    const auto forward_field = fields.template get_simulation_field<
        specfem::simulation::field_type::forward>();
    const int nspec = assembly_mesh.nspec;
    const int ngllz = assembly_mesh.element_grid.ngllz;
    const int nglly = assembly_mesh.element_grid.nglly;
    const int ngllx = assembly_mesh.element_grid.ngllx;
    for (int ispec = 0; ispec < nspec; ispec++) {
      if (element_types.get_medium_tag(ispec) !=
          specfem::element::medium_tag::elastic) {
        continue;
      }
      for (int iz = 0; iz < ngllz; iz++) {
        for (int iy = 0; iy < nglly; iy++) {
          for (int ix = 0; ix < ngllx; ix++) {
            const int iglob = forward_field.template get_iglob<
                false, specfem::element::medium_tag::elastic>(ispec, iz, iy,
                                                              ix);
            iglob_to_coord[iglob] = {
              static_cast<double>(assembly_mesh.h_coord(ispec, iz, iy, ix, 0)),
              static_cast<double>(assembly_mesh.h_coord(ispec, iz, iy, ix, 1)),
              static_cast<double>(assembly_mesh.h_coord(ispec, iz, iy, ix, 2))
            };
          }
        }
      }
    }
  }

  const auto &patterns = patterns_map;
  const int n_patterns = static_cast<int>(patterns.size());

  // Verify nglob consistency per pattern
  for (const auto &[neighbor, pattern] : patterns) {
    ASSERT_EQ(static_cast<size_t>(pattern.pack.nglob),
              pattern.unpack.h_mapping.extent(0))
        << "packer.nglob != unpacker mapping size for neighbor_rank="
        << neighbor;
  }

  // Collect packer coordinates and post all Irecvs before any sends
  std::unordered_map<unsigned int, std::vector<double> > packer_coord_bufs;
  std::unordered_map<unsigned int, std::vector<double> > recv_coord_bufs;
  std::vector<MPI_Request> recv_reqs;
  recv_reqs.reserve(n_patterns);

  const MPI_Comm comm = specfem::MPI::communicator();
  const int my_rank = specfem::MPI::get_rank();

  for (const auto &[neighbor, pattern] : patterns) {
    const int nglob = static_cast<int>(pattern.pack.nglob);
    const int nglob_recv = static_cast<int>(pattern.unpack.h_mapping.extent(0));

    // Pack coordinates for sending
    const auto packer_coords =
        coords_for_mapping(pattern.pack.h_mapping, iglob_to_coord);
    auto &send_buf = packer_coord_bufs[neighbor];
    send_buf.resize(3 * nglob);
    for (int i = 0; i < nglob; i++) {
      send_buf[3 * i + 0] = packer_coords[i][0];
      send_buf[3 * i + 1] = packer_coords[i][1];
      send_buf[3 * i + 2] = packer_coords[i][2];
    }

    // Post receive for neighbor's packer coordinates
    auto &recv_buf = recv_coord_bufs[neighbor];
    recv_buf.resize(3 * nglob_recv);
    const int neighbor_rank = static_cast<int>(neighbor);
    const int recv_tag = neighbor_rank * 10000 + my_rank;
    MPI_Request recv_req;
    SPECFEM_MPI_SAFECALL(MPI_Irecv(recv_buf.data(), 3 * nglob_recv, MPI_DOUBLE,
                                   neighbor_rank, recv_tag, comm, &recv_req));
    recv_reqs.push_back(recv_req);
  }

  // Send packer coordinates to each neighbor
  for (const auto &[neighbor, pattern] : patterns) {
    const int nglob = static_cast<int>(pattern.pack.nglob);
    const int neighbor_rank = static_cast<int>(neighbor);
    const int send_tag = my_rank * 10000 + neighbor_rank;
    SPECFEM_MPI_SAFECALL(MPI_Send(packer_coord_bufs.at(neighbor).data(),
                                  3 * nglob, MPI_DOUBLE, neighbor_rank,
                                  send_tag, comm));
  }

  // Wait for all receives
  SPECFEM_MPI_SAFECALL(
      MPI_Waitall(n_patterns, recv_reqs.data(), MPI_STATUSES_IGNORE));

  // For each pattern: sort both sets of coordinates and compare element-wise
  for (const auto &[neighbor, pattern] : patterns) {
    const int nglob = static_cast<int>(pattern.unpack.h_mapping.extent(0));

    // Collect unpacker coordinates (local side)
    auto unpacker_coords =
        coords_for_mapping(pattern.unpack.h_mapping, iglob_to_coord);

    // Reconstruct received coordinates as CoordTriple vector
    std::vector<CoordTriple> received_coords(nglob);
    const auto &recv_buf = recv_coord_bufs.at(neighbor);
    for (int i = 0; i < nglob; i++) {
      received_coords[i] = { recv_buf[3 * i + 0], recv_buf[3 * i + 1],
                             recv_buf[3 * i + 2] };
    }

    const double tol = 1e-6; // Tolerance for coordinate comparison
    const auto &xsize = assembly_mesh.xmax - assembly_mesh.xmin;
    const auto &ysize = assembly_mesh.ymax - assembly_mesh.ymin;
    const auto &zsize = assembly_mesh.zmax - assembly_mesh.zmin;

    const double scaled_tol = tol * std::max({ xsize, ysize, zsize });

    for (int i = 0; i < nglob; i++) {
      EXPECT_NEAR(received_coords[i][0], unpacker_coords[i][0], scaled_tol)
          << "x mismatch at point " << i << " for pattern my_rank=" << my_rank
          << " & neighbor_rank=" << pattern.neighbor_rank;
      EXPECT_NEAR(received_coords[i][1], unpacker_coords[i][1], scaled_tol)
          << "y mismatch at point " << i << " for pattern my_rank=" << my_rank
          << " & neighbor_rank=" << pattern.neighbor_rank;
      EXPECT_NEAR(received_coords[i][2], unpacker_coords[i][2], scaled_tol)
          << "z mismatch at point " << i << " for pattern my_rank=" << my_rank
          << " & neighbor_rank=" << pattern.neighbor_rank;
    }
  }
}
