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

#include "fixture.hpp"
#include "specfem/assembly/assembly.hpp"
#include "specfem/enums.hpp"
#include "specfem/mesh_entity.hpp"
#include "gtest/gtest.h"
#include <algorithm>
#include <unordered_map>
#include <vector>

namespace specfem::assembly_test {

/**
 * @brief Expected per-face data for a single face in a communication group.
 *
 * Each field corresponds directly to the per-face arrays stored in
 * specfem::assembly::mpi_impl::communication_group, indexed by face ID within
 * the group.
 */
struct FaceData {
  specfem::mesh_entity::dim3::type my_orientation; ///< Face type in the local
                                                   ///< element
  specfem::mesh_entity::dim3::type neighbor_orientation; ///< Face type in the
                                                         ///< neighboring
                                                         ///< element
  unsigned char theta;  ///< Discrete rotation index in [0,3] aligning the two
                        ///< face coordinate systems
  int my_element;       ///< Spectral element index in the local partition
  int neighbor_element; ///< Spectral element index in the neighboring partition

  /**
   * @brief Constructs per-face expected data.
   *
   * @param my_orientation      Face type (orientation) in the local element
   * @param neighbor_orientation Face type (orientation) in the neighbor element
   * @param theta               Rotation index in [0,3]
   * @param my_element          Local spectral element index
   * @param neighbor_element    Neighbor spectral element index
   */
  FaceData(specfem::mesh_entity::dim3::type my_orientation,
           specfem::mesh_entity::dim3::type neighbor_orientation,
           unsigned char theta, int my_element, int neighbor_element)
      : my_orientation(my_orientation),
        neighbor_orientation(neighbor_orientation), theta(theta),
        my_element(my_element), neighbor_element(neighbor_element) {}
};

/**
 * @brief Expected data container for a single MPI communication group.
 *
 * Mirrors the scalar metadata and per-face arrays of
 * specfem::assembly::mpi_impl::communication_group. The check() method
 * performs a comprehensive validation:
 * - Scalar metadata equality (my_rank, neighbor_rank, nfaces, ngll)
 * - Host-mirror view extents for all five per-face arrays
 * - Per-face field equality for every face in the group
 * - Invariant: theta ∈ [0,3] for every face
 */
struct ExpectedCommunicationGroup {
  unsigned int my_rank;        ///< Expected MPI rank of the local process
  unsigned int neighbor_rank;  ///< Expected MPI rank of the neighboring process
  unsigned int nfaces;         ///< Expected number of shared faces
  unsigned int ngll;           ///< Expected GLL points per face dimension
  std::vector<FaceData> faces; ///< Expected per-face data (size == nfaces)

  /**
   * @brief Constructs expected communication group data.
   *
   * @param my_rank       Local MPI rank
   * @param neighbor_rank Neighbor MPI rank
   * @param nfaces        Number of shared faces
   * @param ngll          GLL points per face dimension
   * @param faces         Per-face expected values
   */
  ExpectedCommunicationGroup(unsigned int my_rank, unsigned int neighbor_rank,
                             unsigned int nfaces, unsigned int ngll,
                             std::initializer_list<FaceData> faces)
      : my_rank(my_rank), neighbor_rank(neighbor_rank), nfaces(nfaces),
        ngll(ngll), faces(faces) {}

  /**
   * @brief Validates a computed communication group against expected data.
   *
   * @param group The communication_group to validate
   *
   * @details Checks:
   * 1. Scalar metadata: my_rank, neighbor_rank, nfaces, ngll
   * 2. Host-mirror extents for h_my_orientation, h_neighbor_orientation,
   *    h_theta, h_my_element, h_neighbor_element
   * 3. Per-face values: orientation types, theta, element indices
   * 4. Invariant: h_theta[i] ∈ [0,3] for all faces
   */
  void
  check(const specfem::assembly::mpi_impl::communication_group &group) const {

    // Validate scalar metadata
    ASSERT_EQ(group.my_rank, my_rank)
        << "my_rank mismatch. Expected: " << my_rank
        << ", Got: " << group.my_rank;
    ASSERT_EQ(group.neighbor_rank, neighbor_rank)
        << "neighbor_rank mismatch. Expected: " << neighbor_rank
        << ", Got: " << group.neighbor_rank;
    ASSERT_EQ(group.nfaces, nfaces)
        << "nfaces mismatch. Expected: " << nfaces << ", Got: " << group.nfaces;
    ASSERT_EQ(group.ngll, ngll)
        << "ngll mismatch. Expected: " << ngll << ", Got: " << group.ngll;

    // Validate host-mirror view extents
    ASSERT_EQ(group.h_my_orientation.extent(0), nfaces)
        << "h_my_orientation extent(0) mismatch. Expected: " << nfaces
        << ", Got: " << group.h_my_orientation.extent(0);
    ASSERT_EQ(group.h_neighbor_orientation.extent(0), nfaces)
        << "h_neighbor_orientation extent(0) mismatch. Expected: " << nfaces
        << ", Got: " << group.h_neighbor_orientation.extent(0);
    ASSERT_EQ(group.h_theta.extent(0), nfaces)
        << "h_theta extent(0) mismatch. Expected: " << nfaces
        << ", Got: " << group.h_theta.extent(0);
    ASSERT_EQ(group.h_my_element.extent(0), nfaces)
        << "h_my_element extent(0) mismatch. Expected: " << nfaces
        << ", Got: " << group.h_my_element.extent(0);
    ASSERT_EQ(group.h_neighbor_element.extent(0), nfaces)
        << "h_neighbor_element extent(0) mismatch. Expected: " << nfaces
        << ", Got: " << group.h_neighbor_element.extent(0);

    // Validate per-face data
    for (unsigned int i = 0; i < nfaces; ++i) {
      EXPECT_EQ(group.h_my_orientation(i), faces[i].my_orientation)
          << "h_my_orientation mismatch at face " << i;
      EXPECT_EQ(group.h_neighbor_orientation(i), faces[i].neighbor_orientation)
          << "h_neighbor_orientation mismatch at face " << i;
      EXPECT_EQ(group.h_theta(i), faces[i].theta)
          << "h_theta mismatch at face " << i;
      EXPECT_EQ(group.h_my_element(i), faces[i].my_element)
          << "h_my_element mismatch at face " << i;
      EXPECT_EQ(group.h_neighbor_element(i), faces[i].neighbor_element)
          << "h_neighbor_element mismatch at face " << i;

      // Invariant: theta must always be in [0,3]
      EXPECT_LE(group.h_theta(i), static_cast<unsigned char>(3))
          << "theta out of range [0,3] at face " << i
          << ". Got: " << static_cast<int>(group.h_theta(i));
    }

    SUCCEED() << "CommunicationGroup check passed for neighbor_rank "
              << group.neighbor_rank;
  }
};

/**
 * @brief Expected data container for all communication groups held by one MPI
 * process.
 *
 * Wraps a collection of ExpectedCommunicationGroup objects and validates them
 * against the full specfem::assembly::mpi communication interface.  Groups are
 * matched by neighbor_rank, making the test independent of the order in which
 * communication_groups are stored in the mpi object.
 */
struct ExpectedMPICommunicationGroups {
  std::vector<ExpectedCommunicationGroup> groups; ///< One entry per expected
                                                  ///< neighbor rank

  /**
   * @brief Constructs expected MPI communication groups.
   *
   * @param groups Initializer list of per-neighbor expected data
   */
  ExpectedMPICommunicationGroups(
      std::initializer_list<ExpectedCommunicationGroup> groups)
      : groups(groups) {}

  /**
   * @brief Validates all communication groups against the assembled MPI
   * interface.
   *
   * @param mpi_interfaces The assembled MPI communication object to validate
   *
   * @details
   * 1. Asserts total group count matches
   * 2. For each expected group, finds the matching actual group by
   * neighbor_rank
   * 3. Delegates per-group validation to ExpectedCommunicationGroup::check()
   */
  void check(const specfem::assembly::mpi<specfem::element::dimension_tag::dim3>
                 &mpi_interfaces) const {

    ASSERT_EQ(mpi_interfaces.communication_groups.size(), groups.size())
        << "Number of communication groups mismatch. Expected: "
        << groups.size()
        << ", Got: " << mpi_interfaces.communication_groups.size();

    for (const auto &expected_group : groups) {
      const auto it = std::find_if(
          mpi_interfaces.communication_groups.begin(),
          mpi_interfaces.communication_groups.end(),
          [&](const specfem::assembly::mpi_impl::communication_group &g) {
            return g.neighbor_rank == expected_group.neighbor_rank;
          });

      if (it == mpi_interfaces.communication_groups.end()) {
        ADD_FAILURE() << "No communication group found for neighbor_rank "
                      << expected_group.neighbor_rank;
        continue;
      }

      expected_group.check(*it);
    }
  }
};

} // namespace specfem::assembly_test

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
      TestData::CommunicationGroup::HomogeneousMediumMPI4x4::expected
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

  const auto &mpi_interfaces = getMPIInterfaces();
  const auto &expected = expected_communication_groups.at(param_name);
  expected.check(mpi_interfaces);
}
