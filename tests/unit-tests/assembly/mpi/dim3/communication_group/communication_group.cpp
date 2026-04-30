/**
 * @file communication_group.cpp
 * @brief Unit tests for MPI communication group validation in 3D spectral
 * element assembly.
 *
 * This file validates the construction of
 * specfem::assembly::mpi_impl::communication_group, which stores the
 * face-level inter-process communication metadata between pairs of MPI ranks.
 *
 * Each communication group encodes, per shared face:
 * - The mesh entity type (face orientation) in both the local and neighbor
 *   element
 * - A discrete rotation index theta in [0,3] aligning face coordinate systems
 * - The spectral element index in both the local and neighbor partitions
 *
 * @see specfem::assembly::mpi_impl::communication_group
 * @see ExpectedCommunicationGroup for test data structures
 */

#include "../fixture.hpp"
#include "expected_groups.hpp"
#include "specfem/assembly/assembly.hpp"
#include "specfem/enums.hpp"
#include "specfem/mpi.hpp"
#include <unordered_map>
#include <unordered_set>

using namespace specfem::assembly_test;

// -----------------------------------------------------------------------------
// Test data and test case definitions
// -----------------------------------------------------------------------------

#include "HomogeneousMediumMPI4x4.hpp"

// -----------------------------------------------------------------------------

/**
 * @brief Ground truth test data for MPI communication group validation.
 *
 * Add entries here keyed by the test case folder name (matching the parameter
 * passed via INSTANTIATE_TEST_SUITE_P). Tests skip gracefully when no entry
 * exists for a given test case name.
 *
 * @note When adding entries, the neighbor_rank values must correspond to the
 *       actual MPI ranks that share faces with the current rank in the mesh
 *       partition described by the test data folder.
 */
static const std::unordered_map<
    std::string, std::tuple<ExpectedMPIFaceCommunicationGroups,
                            ExpectedMPIEdgeCommunicationGroups,
                            ExpectedMPICornerCommunicationGroups> >
    expected_communication_groups = {
      { "HomogeneousMediumMPI4x4",
        TestData::CommunicationGroup::HomogeneousMediumMPI4x4::expected }
    };

/**
 * @brief Parameterized test for MPI communication group validation.
 *
 * Validates the communication group structure built during assembly against
 * ground truth data in expected_communication_groups. Tests are skipped when
 * no expected data is registered for the current test case name.
 *
 * @param param_name Test case folder name (e.g., "HomogeneousMediumMPI4x4")
 *
 * @see AssemblyMPI3DTest for the test fixture
 * @see expected_communication_groups for available test cases
 */
TEST_P(AssemblyMPI3DTest, CommunicationGroup) {
  const auto &param_name = GetParam();

  if (expected_communication_groups.find(param_name) ==
      expected_communication_groups.end()) {
    GTEST_SKIP() << "No expected communication group data available for test "
                    "case '"
                 << param_name << "'.";
    return;
  }

  const unsigned int current_rank = specfem::MPI::get_rank();
  const auto &[face_groups, edge_groups, corner_groups] =
      expected_communication_groups.at(param_name);

  // Skip if current rank is neither my_rank nor neighbor_rank in any group
  const bool is_relevant =
      std::any_of(face_groups.groups.begin(), face_groups.groups.end(),
                  [current_rank](const ExpectedFaceCommunicationGroup &g) {
                    return g.my_rank == current_rank ||
                           g.neighbor_rank == current_rank;
                  }) ||
      std::any_of(edge_groups.groups.begin(), edge_groups.groups.end(),
                  [current_rank](const ExpectedEdgeCommunicationGroup &g) {
                    return g.my_rank == current_rank ||
                           g.neighbor_rank == current_rank;
                  }) ||
      std::any_of(corner_groups.groups.begin(), corner_groups.groups.end(),
                  [current_rank](const ExpectedCornerCommunicationGroup &g) {
                    return g.my_rank == current_rank ||
                           g.neighbor_rank == current_rank;
                  });

  if (!is_relevant) {
    GTEST_SKIP() << "Rank " << current_rank
                 << " is not involved in any expected communication group "
                    "for test case '"
                 << param_name << "'.";
    return;
  }

  const auto &mpi_interfaces = getMPIInterfaces();

  // Programmatic group count check: single pass over MPI connections to count
  // unique neighbor ranks per connection type, then assert each group vector
  // size matches.
  {
    const auto &mesh = getMesh();
    std::unordered_set<size_t> face_neighbors, edge_neighbors, corner_neighbors;
    for (const auto &conn : mesh.adjacency_graph.mpi_connections()) {
      if (specfem::mesh_entity::contains(specfem::mesh_entity::dim3::faces,
                                         conn.orientation)) {
        face_neighbors.insert(conn.neighbor_partition);
      } else if (specfem::mesh_entity::contains(
                     specfem::mesh_entity::dim3::edges, conn.orientation)) {
        edge_neighbors.insert(conn.neighbor_partition);
      } else {
        corner_neighbors.insert(conn.neighbor_partition);
      }
    }
    ASSERT_EQ(mpi_interfaces.face_groups.size(), face_neighbors.size())
        << "face_groups count mismatch for rank " << current_rank
        << ". Expected (from adjacency graph): " << face_neighbors.size()
        << ", Got: " << mpi_interfaces.face_groups.size();
    ASSERT_EQ(mpi_interfaces.edge_groups.size(), edge_neighbors.size())
        << "edge_groups count mismatch for rank " << current_rank
        << ". Expected (from adjacency graph): " << edge_neighbors.size()
        << ", Got: " << mpi_interfaces.edge_groups.size();
    ASSERT_EQ(mpi_interfaces.corner_groups.size(), corner_neighbors.size())
        << "corner_groups count mismatch for rank " << current_rank
        << ". Expected (from adjacency graph): " << corner_neighbors.size()
        << ", Got: " << mpi_interfaces.corner_groups.size();
  }

  face_groups.check(mpi_interfaces, current_rank);
  edge_groups.check(mpi_interfaces, current_rank);
  corner_groups.check(mpi_interfaces, current_rank);
}
