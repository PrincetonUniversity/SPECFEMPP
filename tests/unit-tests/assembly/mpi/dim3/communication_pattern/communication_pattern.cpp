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

  // Count unique neighbors across all connection types
  std::unordered_set<unsigned int> expected_neighbors;
  for (const auto &g : mpi_interfaces.face_groups)
    expected_neighbors.insert(g.neighbor_rank);
  for (const auto &g : mpi_interfaces.edge_groups)
    expected_neighbors.insert(g.neighbor_rank);
  for (const auto &g : mpi_interfaces.corner_groups)
    expected_neighbors.insert(g.neighbor_rank);
  ASSERT_EQ(mpi_interfaces.communication_patterns.size(),
            expected_neighbors.size())
      << "communication_patterns count must match unique neighbor count";

  // Build iglob → (x,y,z) map by scanning all elements on this rank
  std::unordered_map<int, CoordTriple> iglob_to_coord;
  {
    const int nspec = assembly_mesh.nspec;
    const int ngllz = assembly_mesh.element_grid.ngllz;
    const int nglly = assembly_mesh.element_grid.nglly;
    const int ngllx = assembly_mesh.element_grid.ngllx;
    for (int ispec = 0; ispec < nspec; ispec++) {
      for (int iz = 0; iz < ngllz; iz++) {
        for (int iy = 0; iy < nglly; iy++) {
          for (int ix = 0; ix < ngllx; ix++) {
            const int iglob = assembly_mesh.h_index_mapping(ispec, iz, iy, ix);
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

  const auto &patterns = mpi_interfaces.communication_patterns;
  const int n_patterns = static_cast<int>(patterns.size());

  // Verify nglob consistency per pattern
  for (int p = 0; p < n_patterns; p++) {
    const auto &pattern = patterns[p];
    ASSERT_EQ(static_cast<size_t>(pattern.pack.nglob),
              pattern.unpack.h_mapping.extent(0))
        << "packer.nglob != unpacker mapping size for pattern " << p
        << " (neighbor_rank=" << pattern.neighbor_rank << ")";
  }

  // Collect packer coordinates and post all Irecvs before any sends
  std::vector<std::vector<double> > packer_coord_bufs(n_patterns);
  std::vector<std::vector<double> > recv_coord_bufs(n_patterns);
  std::vector<MPI_Request> recv_reqs(n_patterns);

  const MPI_Comm comm = specfem::MPI::communicator();
  const int my_rank = specfem::MPI::get_rank();

  for (int p = 0; p < n_patterns; p++) {
    const auto &pattern = patterns[p];
    const int nglob = static_cast<int>(pattern.pack.nglob);
    const int nglob_recv = static_cast<int>(pattern.unpack.h_mapping.extent(0));
    const int neighbor = static_cast<int>(pattern.neighbor_rank);

    // Pack coordinates for sending
    const auto packer_coords =
        coords_for_mapping(pattern.pack.h_mapping, iglob_to_coord);
    packer_coord_bufs[p].resize(3 * nglob);
    for (int i = 0; i < nglob; i++) {
      packer_coord_bufs[p][3 * i + 0] = packer_coords[i][0];
      packer_coord_bufs[p][3 * i + 1] = packer_coords[i][1];
      packer_coord_bufs[p][3 * i + 2] = packer_coords[i][2];
    }

    // Post receive for neighbor's packer coordinates
    recv_coord_bufs[p].resize(3 * nglob_recv);
    const int recv_tag = neighbor * 10000 + my_rank;
    SPECFEM_MPI_SAFECALL(MPI_Irecv(recv_coord_bufs[p].data(), 3 * nglob_recv,
                                   MPI_DOUBLE, neighbor, recv_tag, comm,
                                   &recv_reqs[p]));
  }

  // Send packer coordinates to each neighbor
  for (int p = 0; p < n_patterns; p++) {
    const auto &pattern = patterns[p];
    const int nglob = static_cast<int>(pattern.pack.nglob);
    const int neighbor = static_cast<int>(pattern.neighbor_rank);
    const int send_tag = my_rank * 10000 + neighbor;
    SPECFEM_MPI_SAFECALL(MPI_Send(packer_coord_bufs[p].data(), 3 * nglob,
                                  MPI_DOUBLE, neighbor, send_tag, comm));
  }

  // Wait for all receives
  SPECFEM_MPI_SAFECALL(
      MPI_Waitall(n_patterns, recv_reqs.data(), MPI_STATUSES_IGNORE));

  // For each pattern: sort both sets of coordinates and compare element-wise
  for (int p = 0; p < n_patterns; p++) {
    const auto &pattern = patterns[p];
    const int nglob = static_cast<int>(pattern.unpack.h_mapping.extent(0));

    // Collect unpacker coordinates (local side)
    auto unpacker_coords =
        coords_for_mapping(pattern.unpack.h_mapping, iglob_to_coord);

    // Reconstruct received coordinates as CoordTriple vector
    std::vector<CoordTriple> received_coords(nglob);
    for (int i = 0; i < nglob; i++) {
      received_coords[i] = { recv_coord_bufs[p][3 * i + 0],
                             recv_coord_bufs[p][3 * i + 1],
                             recv_coord_bufs[p][3 * i + 2] };
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
