#pragma once

#include "specfem/assembly/mpi/dim3/mpi.hpp"
#include "specfem/mesh_entity.hpp"
#include "gtest/gtest.h"
#include <algorithm>
#include <initializer_list>
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
  std::vector<FaceData> faces; ///< Expected per-face data (subset of nfaces)

  ExpectedCommunicationGroup(unsigned int my_rank, unsigned int neighbor_rank,
                             unsigned int nfaces, unsigned int ngll,
                             std::initializer_list<FaceData> faces)
      : my_rank(my_rank), neighbor_rank(neighbor_rank), nfaces(nfaces),
        ngll(ngll), faces(faces) {}

  void
  check(const specfem::assembly::mpi_impl::communication_group &group) const {

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

    for (unsigned int i = 0; i < nfaces; ++i) {
      EXPECT_LE(group.h_theta(i), static_cast<unsigned char>(3))
          << "theta out of range [0,3] at face " << i
          << ". Got: " << static_cast<int>(group.h_theta(i));
    }

    for (size_t ei = 0; ei < faces.size(); ++ei) {
      const auto &expected_face = faces[ei];
      bool found = false;
      for (unsigned int ai = 0; ai < nfaces; ++ai) {
        if (group.h_my_orientation(ai) == expected_face.my_orientation &&
            group.h_neighbor_orientation(ai) ==
                expected_face.neighbor_orientation &&
            group.h_theta(ai) == expected_face.theta &&
            group.h_my_element(ai) == expected_face.my_element &&
            group.h_neighbor_element(ai) == expected_face.neighbor_element) {
          found = true;
          break;
        }
      }
      EXPECT_TRUE(found) << "Expected face " << ei
                         << " not found in communication group "
                         << "(neighbor_rank=" << neighbor_rank << "): "
                         << "my_orientation="
                         << static_cast<int>(expected_face.my_orientation)
                         << ", neighbor_orientation="
                         << static_cast<int>(expected_face.neighbor_orientation)
                         << ", theta=" << static_cast<int>(expected_face.theta)
                         << ", my_element=" << expected_face.my_element
                         << ", neighbor_element="
                         << expected_face.neighbor_element;
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
 * against the full specfem::assembly::mpi communication interface. Groups are
 * matched by neighbor_rank, making the test independent of the order in which
 * communication_groups are stored in the mpi object.
 */
struct ExpectedMPICommunicationGroups {
  std::vector<ExpectedCommunicationGroup> groups;

  ExpectedMPICommunicationGroups(
      std::initializer_list<ExpectedCommunicationGroup> groups)
      : groups(groups) {}

  void check(const specfem::assembly::mpi<specfem::element::dimension_tag::dim3>
                 &mpi_interfaces,
             unsigned int current_rank) const {

    std::vector<const ExpectedCommunicationGroup *> relevant_groups;
    for (const auto &g : groups) {
      if (g.my_rank == current_rank) {
        relevant_groups.push_back(&g);
      }
    }

    ASSERT_GE(mpi_interfaces.communication_groups.size(),
              relevant_groups.size())
        << "Fewer actual communication groups than expected for rank "
        << current_rank << ". Expected at least: " << relevant_groups.size()
        << ", Got: " << mpi_interfaces.communication_groups.size();

    for (const auto *expected_group : relevant_groups) {
      const auto it = std::find_if(
          mpi_interfaces.communication_groups.begin(),
          mpi_interfaces.communication_groups.end(),
          [&](const specfem::assembly::mpi_impl::communication_group &g) {
            return g.neighbor_rank == expected_group->neighbor_rank;
          });

      if (it == mpi_interfaces.communication_groups.end()) {
        ADD_FAILURE() << "No communication group found for neighbor_rank "
                      << expected_group->neighbor_rank;
        continue;
      }

      expected_group->check(*it);
    }
  }
};

} // namespace specfem::assembly_test
