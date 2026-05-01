#pragma once

#include "specfem/assembly/mpi/dim3/mpi.hpp"
#include "specfem/mesh_entity.hpp"
#include "gtest/gtest.h"
#include <algorithm>
#include <initializer_list>
#include <vector>

namespace specfem::assembly_test {

// =============================================================================
// Face communication group test data
// =============================================================================

/**
 * @brief Expected per-face data for a single face in a face communication
 * group.
 *
 * Each field corresponds directly to the per-face arrays stored in
 * specfem::assembly::mpi_impl::face_communication_group.
 */
struct FaceData {
  specfem::mesh_entity::dim3::type my_orientation;       ///< Face type in local
                                                         ///< element
  specfem::mesh_entity::dim3::type neighbor_orientation; ///< Face type in
                                                         ///< neighboring
                                                         ///< element
  unsigned char theta;  ///< Discrete rotation index ∈ [0,3] aligning face
                        ///< coordinate systems
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
 * @brief Expected data container for a single MPI face communication group.
 *
 * Mirrors the scalar metadata and per-face arrays of
 * specfem::assembly::mpi_impl::face_communication_group. The check() method
 * performs a comprehensive validation:
 * - Scalar metadata equality (my_rank, neighbor_rank, n, ngll)
 * - Host-mirror view extents for all per-face arrays
 * - Per-face field equality for every face in the group (unordered search)
 * - Invariant: theta ∈ [0,3] for every face
 */
struct ExpectedFaceCommunicationGroup {
  unsigned int my_rank;        ///< Expected MPI rank of the local process
  unsigned int neighbor_rank;  ///< Expected MPI rank of the neighboring process
  unsigned int n;              ///< Expected number of shared faces
  unsigned int ngll;           ///< Expected GLL points per face dimension
  std::vector<FaceData> faces; ///< Expected per-face data (may be a subset of
                               ///< n)

  ExpectedFaceCommunicationGroup(unsigned int my_rank,
                                 unsigned int neighbor_rank, unsigned int n,
                                 unsigned int ngll,
                                 std::initializer_list<FaceData> faces)
      : my_rank(my_rank), neighbor_rank(neighbor_rank), n(n), ngll(ngll),
        faces(faces) {}

  void check(const specfem::assembly::mpi_impl::face_communication_group &group)
      const {

    ASSERT_EQ(group.my_rank, my_rank)
        << "my_rank mismatch. Expected: " << my_rank
        << ", Got: " << group.my_rank;
    ASSERT_EQ(group.neighbor_rank, neighbor_rank)
        << "neighbor_rank mismatch. Expected: " << neighbor_rank
        << ", Got: " << group.neighbor_rank;
    ASSERT_EQ(group.n, n) << "n mismatch. Expected: " << n
                          << ", Got: " << group.n;
    ASSERT_EQ(group.ngll, ngll)
        << "ngll mismatch. Expected: " << ngll << ", Got: " << group.ngll;

    ASSERT_EQ(group.h_my_orientation.extent(0), n)
        << "h_my_orientation extent(0) mismatch. Expected: " << n
        << ", Got: " << group.h_my_orientation.extent(0);
    ASSERT_EQ(group.h_neighbor_orientation.extent(0), n)
        << "h_neighbor_orientation extent(0) mismatch. Expected: " << n
        << ", Got: " << group.h_neighbor_orientation.extent(0);
    ASSERT_EQ(group.h_theta.extent(0), n)
        << "h_theta extent(0) mismatch. Expected: " << n
        << ", Got: " << group.h_theta.extent(0);
    ASSERT_EQ(group.h_my_element.extent(0), n)
        << "h_my_element extent(0) mismatch. Expected: " << n
        << ", Got: " << group.h_my_element.extent(0);
    ASSERT_EQ(group.h_neighbor_element.extent(0), n)
        << "h_neighbor_element extent(0) mismatch. Expected: " << n
        << ", Got: " << group.h_neighbor_element.extent(0);

    for (unsigned int i = 0; i < n; ++i) {
      EXPECT_LE(group.h_theta(i), static_cast<unsigned char>(3))
          << "theta out of range [0,3] at face " << i
          << ". Got: " << static_cast<int>(group.h_theta(i));
    }

    for (size_t ei = 0; ei < faces.size(); ++ei) {
      const auto &expected_face = faces[ei];
      bool found = false;
      for (unsigned int ai = 0; ai < n; ++ai) {
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
 * @brief Expected data container for all face communication groups held by one
 * MPI process.
 *
 * Wraps a collection of ExpectedCommunicationGroup objects and validates them
 * against the full specfem::assembly::mpi communication interface. Groups are
 * matched by neighbor_rank, making the test independent of the order in which
 * face_groups are stored in the mpi object.
 */
struct ExpectedMPIFaceCommunicationGroups {
  std::vector<ExpectedFaceCommunicationGroup> groups;

  ExpectedMPIFaceCommunicationGroups(
      std::initializer_list<ExpectedFaceCommunicationGroup> groups)
      : groups(groups) {}

  void check(const specfem::assembly::mpi<specfem::element::dimension_tag::dim3>
                 &mpi_interfaces,
             unsigned int current_rank) const {

    std::vector<const ExpectedFaceCommunicationGroup *> relevant_groups;
    for (const auto &g : groups) {
      if (g.my_rank == current_rank) {
        relevant_groups.push_back(&g);
      }
    }

    ASSERT_GE(mpi_interfaces.face_groups.size(), relevant_groups.size())
        << "Fewer actual face groups than expected for rank " << current_rank
        << ". Expected at least: " << relevant_groups.size()
        << ", Got: " << mpi_interfaces.face_groups.size();

    for (const auto *expected_group : relevant_groups) {
      const auto it = std::find_if(
          mpi_interfaces.face_groups.begin(), mpi_interfaces.face_groups.end(),
          [&](const specfem::assembly::mpi_impl::face_communication_group &g) {
            return g.neighbor_rank == expected_group->neighbor_rank;
          });

      if (it == mpi_interfaces.face_groups.end()) {
        ADD_FAILURE() << "No face communication group found for neighbor_rank "
                      << expected_group->neighbor_rank;
        continue;
      }

      expected_group->check(*it);
    }
  }
};

// =============================================================================
// Edge communication group test data
// =============================================================================

/**
 * @brief Expected per-edge data for a single shared edge in an edge
 * communication group.
 *
 * Each field corresponds directly to the per-edge arrays stored in
 * specfem::assembly::mpi_impl::edge_communication_group. The `reflect` flag
 * encodes whether the 1D GLL traversal direction along this edge is reversed
 * relative to the neighbor partition.
 */
struct EdgeData {
  specfem::mesh_entity::dim3::type my_orientation;       ///< Edge type in local
                                                         ///< element
  specfem::mesh_entity::dim3::type neighbor_orientation; ///< Edge type in
                                                         ///< neighboring
                                                         ///< element
  bool reflect;   ///< true → reversed 1D GLL traversal relative to neighbor
  int my_element; ///< Spectral element index in the local partition
  int neighbor_element; ///< Spectral element index in the neighboring partition

  EdgeData(specfem::mesh_entity::dim3::type my_orientation,
           specfem::mesh_entity::dim3::type neighbor_orientation, bool reflect,
           int my_element, int neighbor_element)
      : my_orientation(my_orientation),
        neighbor_orientation(neighbor_orientation), reflect(reflect),
        my_element(my_element), neighbor_element(neighbor_element) {}
};

/**
 * @brief Expected data container for a single MPI edge communication group.
 *
 * Mirrors the scalar metadata and per-edge arrays of
 * specfem::assembly::mpi_impl::edge_communication_group. The check() method
 * performs a comprehensive validation:
 * - Scalar metadata equality (my_rank, neighbor_rank, n == nedges, ngll)
 * - Host-mirror view extents for all per-edge arrays including h_reflect
 * - Per-edge field equality for every edge in the group (unordered search)
 */
struct ExpectedEdgeCommunicationGroup {
  unsigned int my_rank;        ///< Expected MPI rank of the local process
  unsigned int neighbor_rank;  ///< Expected MPI rank of the neighboring process
  unsigned int nedges;         ///< Expected number of shared edges (= group.n)
  unsigned int ngll;           ///< Expected GLL points per edge dimension
  std::vector<EdgeData> edges; ///< Expected per-edge data (may be a subset of
                               ///< nedges)

  ExpectedEdgeCommunicationGroup(unsigned int my_rank,
                                 unsigned int neighbor_rank,
                                 unsigned int nedges, unsigned int ngll,
                                 std::initializer_list<EdgeData> edges)
      : my_rank(my_rank), neighbor_rank(neighbor_rank), nedges(nedges),
        ngll(ngll), edges(edges) {}

  void check(const specfem::assembly::mpi_impl::edge_communication_group &group)
      const {

    ASSERT_EQ(group.my_rank, my_rank)
        << "my_rank mismatch. Expected: " << my_rank
        << ", Got: " << group.my_rank;
    ASSERT_EQ(group.neighbor_rank, neighbor_rank)
        << "neighbor_rank mismatch. Expected: " << neighbor_rank
        << ", Got: " << group.neighbor_rank;
    ASSERT_EQ(group.n, nedges)
        << "nedges (n) mismatch. Expected: " << nedges << ", Got: " << group.n;
    ASSERT_EQ(group.ngll, ngll)
        << "ngll mismatch. Expected: " << ngll << ", Got: " << group.ngll;

    ASSERT_EQ(group.h_my_orientation.extent(0), nedges)
        << "h_my_orientation extent(0) mismatch. Expected: " << nedges
        << ", Got: " << group.h_my_orientation.extent(0);
    ASSERT_EQ(group.h_neighbor_orientation.extent(0), nedges)
        << "h_neighbor_orientation extent(0) mismatch. Expected: " << nedges
        << ", Got: " << group.h_neighbor_orientation.extent(0);
    ASSERT_EQ(group.h_reflect.extent(0), nedges)
        << "h_reflect extent(0) mismatch. Expected: " << nedges
        << ", Got: " << group.h_reflect.extent(0);
    ASSERT_EQ(group.h_my_element.extent(0), nedges)
        << "h_my_element extent(0) mismatch. Expected: " << nedges
        << ", Got: " << group.h_my_element.extent(0);
    ASSERT_EQ(group.h_neighbor_element.extent(0), nedges)
        << "h_neighbor_element extent(0) mismatch. Expected: " << nedges
        << ", Got: " << group.h_neighbor_element.extent(0);

    for (size_t ei = 0; ei < edges.size(); ++ei) {
      const auto &expected_edge = edges[ei];
      bool found = false;
      for (unsigned int ai = 0; ai < nedges; ++ai) {
        if (group.h_my_orientation(ai) == expected_edge.my_orientation &&
            group.h_neighbor_orientation(ai) ==
                expected_edge.neighbor_orientation &&
            group.h_reflect(ai) == expected_edge.reflect &&
            group.h_my_element(ai) == expected_edge.my_element &&
            group.h_neighbor_element(ai) == expected_edge.neighbor_element) {
          found = true;
          break;
        }
      }
      EXPECT_TRUE(found) << "Expected edge " << ei
                         << " not found in edge communication group "
                         << "(neighbor_rank=" << neighbor_rank << "): "
                         << "my_orientation="
                         << static_cast<int>(expected_edge.my_orientation)
                         << ", neighbor_orientation="
                         << static_cast<int>(expected_edge.neighbor_orientation)
                         << ", reflect=" << expected_edge.reflect
                         << ", my_element=" << expected_edge.my_element
                         << ", neighbor_element="
                         << expected_edge.neighbor_element;
    }

    SUCCEED() << "EdgeCommunicationGroup check passed for neighbor_rank "
              << group.neighbor_rank;
  }
};

/**
 * @brief Expected data container for all edge communication groups held by one
 * MPI process.
 *
 * Groups are matched by neighbor_rank, making the test independent of the order
 * in which edge_groups are stored in the mpi object.
 */
struct ExpectedMPIEdgeCommunicationGroups {
  std::vector<ExpectedEdgeCommunicationGroup> groups;

  ExpectedMPIEdgeCommunicationGroups(
      std::initializer_list<ExpectedEdgeCommunicationGroup> groups)
      : groups(groups) {}

  void check(const specfem::assembly::mpi<specfem::element::dimension_tag::dim3>
                 &mpi_interfaces,
             unsigned int current_rank) const {

    std::vector<const ExpectedEdgeCommunicationGroup *> relevant_groups;
    for (const auto &g : groups) {
      if (g.my_rank == current_rank) {
        relevant_groups.push_back(&g);
      }
    }

    ASSERT_GE(mpi_interfaces.edge_groups.size(), relevant_groups.size())
        << "Fewer actual edge groups than expected for rank " << current_rank
        << ". Expected at least: " << relevant_groups.size()
        << ", Got: " << mpi_interfaces.edge_groups.size();

    for (const auto *expected_group : relevant_groups) {
      const auto it = std::find_if(
          mpi_interfaces.edge_groups.begin(), mpi_interfaces.edge_groups.end(),
          [&](const specfem::assembly::mpi_impl::edge_communication_group &g) {
            return g.neighbor_rank == expected_group->neighbor_rank;
          });

      if (it == mpi_interfaces.edge_groups.end()) {
        ADD_FAILURE() << "No edge communication group found for neighbor_rank "
                      << expected_group->neighbor_rank;
        continue;
      }

      expected_group->check(*it);
    }
  }
};

// =============================================================================
// Corner communication group test data
// =============================================================================

/**
 * @brief Expected per-corner data for a single shared corner in a corner
 * communication group.
 *
 * Corners are single GLL points; no transformation metadata (theta or reflect)
 * is needed. The check validates only that the correct element and orientation
 * are recorded on both sides of the partition boundary.
 */
struct CornerData {
  specfem::mesh_entity::dim3::type my_orientation; ///< Corner type in local
                                                   ///< element
  specfem::mesh_entity::dim3::type neighbor_orientation; ///< Corner type in
                                                         ///< neighboring
                                                         ///< element
  int my_element;       ///< Spectral element index in the local partition
  int neighbor_element; ///< Spectral element index in the neighboring partition

  CornerData(specfem::mesh_entity::dim3::type my_orientation,
             specfem::mesh_entity::dim3::type neighbor_orientation,
             int my_element, int neighbor_element)
      : my_orientation(my_orientation),
        neighbor_orientation(neighbor_orientation), my_element(my_element),
        neighbor_element(neighbor_element) {}
};

/**
 * @brief Expected data container for a single MPI corner communication group.
 *
 * Mirrors the scalar metadata and per-corner arrays of
 * specfem::assembly::mpi_impl::corner_communication_group. The check() method
 * performs a comprehensive validation:
 * - Scalar metadata equality (my_rank, neighbor_rank, n == ncorners, ngll)
 * - Host-mirror view extents for all four per-corner arrays
 * - Per-corner field equality for every corner in the group (unordered search)
 */
struct ExpectedCornerCommunicationGroup {
  unsigned int my_rank;       ///< Expected MPI rank of the local process
  unsigned int neighbor_rank; ///< Expected MPI rank of the neighboring process
  unsigned int ncorners;      ///< Expected number of shared corners (= group.n)
  unsigned int ngll;          ///< Expected GLL points per edge dimension
  std::vector<CornerData> corners; ///< Expected per-corner data (may be a
                                   ///< subset of ncorners)

  ExpectedCornerCommunicationGroup(unsigned int my_rank,
                                   unsigned int neighbor_rank,
                                   unsigned int ncorners, unsigned int ngll,
                                   std::initializer_list<CornerData> corners)
      : my_rank(my_rank), neighbor_rank(neighbor_rank), ncorners(ncorners),
        ngll(ngll), corners(corners) {}

  void check(const specfem::assembly::mpi_impl::corner_communication_group
                 &group) const {

    ASSERT_EQ(group.my_rank, my_rank)
        << "my_rank mismatch. Expected: " << my_rank
        << ", Got: " << group.my_rank;
    ASSERT_EQ(group.neighbor_rank, neighbor_rank)
        << "neighbor_rank mismatch. Expected: " << neighbor_rank
        << ", Got: " << group.neighbor_rank;
    ASSERT_EQ(group.n, ncorners)
        << "ncorners (n) mismatch. Expected: " << ncorners
        << ", Got: " << group.n;
    ASSERT_EQ(group.ngll, ngll)
        << "ngll mismatch. Expected: " << ngll << ", Got: " << group.ngll;

    ASSERT_EQ(group.h_my_orientation.extent(0), ncorners)
        << "h_my_orientation extent(0) mismatch. Expected: " << ncorners
        << ", Got: " << group.h_my_orientation.extent(0);
    ASSERT_EQ(group.h_neighbor_orientation.extent(0), ncorners)
        << "h_neighbor_orientation extent(0) mismatch. Expected: " << ncorners
        << ", Got: " << group.h_neighbor_orientation.extent(0);
    ASSERT_EQ(group.h_my_element.extent(0), ncorners)
        << "h_my_element extent(0) mismatch. Expected: " << ncorners
        << ", Got: " << group.h_my_element.extent(0);
    ASSERT_EQ(group.h_neighbor_element.extent(0), ncorners)
        << "h_neighbor_element extent(0) mismatch. Expected: " << ncorners
        << ", Got: " << group.h_neighbor_element.extent(0);

    for (size_t ci = 0; ci < corners.size(); ++ci) {
      const auto &expected_corner = corners[ci];
      bool found = false;
      for (unsigned int ai = 0; ai < ncorners; ++ai) {
        if (group.h_my_orientation(ai) == expected_corner.my_orientation &&
            group.h_neighbor_orientation(ai) ==
                expected_corner.neighbor_orientation &&
            group.h_my_element(ai) == expected_corner.my_element &&
            group.h_neighbor_element(ai) == expected_corner.neighbor_element) {
          found = true;
          break;
        }
      }
      EXPECT_TRUE(found) << "Expected corner " << ci
                         << " not found in corner communication group "
                         << "(neighbor_rank=" << neighbor_rank << "): "
                         << "my_orientation="
                         << static_cast<int>(expected_corner.my_orientation)
                         << ", neighbor_orientation="
                         << static_cast<int>(
                                expected_corner.neighbor_orientation)
                         << ", my_element=" << expected_corner.my_element
                         << ", neighbor_element="
                         << expected_corner.neighbor_element;
    }

    SUCCEED() << "CornerCommunicationGroup check passed for neighbor_rank "
              << group.neighbor_rank;
  }
};

/**
 * @brief Expected data container for all corner communication groups held by
 * one MPI process.
 *
 * Groups are matched by neighbor_rank, making the test independent of the order
 * in which corner_groups are stored in the mpi object.
 */
struct ExpectedMPICornerCommunicationGroups {
  std::vector<ExpectedCornerCommunicationGroup> groups;

  ExpectedMPICornerCommunicationGroups(
      std::initializer_list<ExpectedCornerCommunicationGroup> groups)
      : groups(groups) {}

  void check(const specfem::assembly::mpi<specfem::element::dimension_tag::dim3>
                 &mpi_interfaces,
             unsigned int current_rank) const {

    std::vector<const ExpectedCornerCommunicationGroup *> relevant_groups;
    for (const auto &g : groups) {
      if (g.my_rank == current_rank) {
        relevant_groups.push_back(&g);
      }
    }

    ASSERT_GE(mpi_interfaces.corner_groups.size(), relevant_groups.size())
        << "Fewer actual corner groups than expected for rank " << current_rank
        << ". Expected at least: " << relevant_groups.size()
        << ", Got: " << mpi_interfaces.corner_groups.size();

    for (const auto *expected_group : relevant_groups) {
      const auto it = std::find_if(
          mpi_interfaces.corner_groups.begin(),
          mpi_interfaces.corner_groups.end(),
          [&](const specfem::assembly::mpi_impl::corner_communication_group
                  &g) {
            return g.neighbor_rank == expected_group->neighbor_rank;
          });

      if (it == mpi_interfaces.corner_groups.end()) {
        ADD_FAILURE()
            << "No corner communication group found for neighbor_rank "
            << expected_group->neighbor_rank;
        continue;
      }

      expected_group->check(*it);
    }
  }
};

} // namespace specfem::assembly_test
