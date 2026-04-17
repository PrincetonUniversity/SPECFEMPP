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
static const std::unordered_map<std::string, ExpectedMPICommunicationGroups>
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
  const auto &expected = expected_communication_groups.at(param_name);

  // Skip if current rank is neither my_rank nor neighbor_rank in any group
  const bool is_relevant = std::any_of(
      expected.groups.begin(), expected.groups.end(),
      [current_rank](const ExpectedCommunicationGroup &g) {
        return g.my_rank == current_rank || g.neighbor_rank == current_rank;
      });

  if (!is_relevant) {
    GTEST_SKIP() << "Rank " << current_rank
                 << " is not involved in any expected communication group "
                    "for test case '"
                 << param_name << "'.";
    return;
  }

  const auto &mpi_interfaces = getMPIInterfaces();

  // Programmatic group count check: compute expected number of neighbor ranks
  // from the mesh adjacency graph (face connections only) and verify it matches
  // the actual number of communication groups.
  {
    const auto &mesh = getMesh();
    std::unordered_set<size_t> neighbor_ranks;
    for (const auto &conn : mesh.adjacency_graph.mpi_connections()) {
      if (specfem::mesh_entity::contains(specfem::mesh_entity::dim3::faces,
                                         conn.orientation)) {
        neighbor_ranks.insert(conn.neighbor_partition);
      }
    }
    ASSERT_EQ(mpi_interfaces.communication_groups.size(), neighbor_ranks.size())
        << "Communication group count mismatch for rank " << current_rank
        << ". Expected (from adjacency graph): " << neighbor_ranks.size()
        << ", Got: " << mpi_interfaces.communication_groups.size();
  }

  expected.check(mpi_interfaces, current_rank);
}
