#include "../mismatch_recorder.hpp"
#include "SPECFEM_Environment.hpp"
#include "specfem/io/impl/sub_cell_connectivity.hpp"
#include <gtest/gtest.h>

#include <array>
#include <set>
#include <tuple>

TEST(SubCellConnectivity, QuadrilateralsSpanNeighbouringGllPoints) {
  constexpr int number_of_elements = 2;
  // Local (ix, iz) offsets of the VTK quadrilateral nodes from the cell corner
  constexpr std::array<std::array<int, 2>, 4> node_offsets = {
    { { 0, 0 }, { 1, 0 }, { 1, 1 }, { 0, 1 } }
  };

  for (const int ngll : { 2, 5 }) {
    const specfem::mesh_entity::element_grid<
        specfem::element::dimension_tag::dim2>
        element_grid(ngll);
    const auto connectivity = specfem::io::impl::sub_cell_connectivity(
        number_of_elements, element_grid);

    const int number_of_points = number_of_elements * ngll * ngll;
    ASSERT_EQ(connectivity.extent(0),
              number_of_elements * (ngll - 1) * (ngll - 1));
    ASSERT_EQ(connectivity.extent(1), 4);

    std::set<std::tuple<int, int, int>> corners;
    specfem_test::MismatchRecorder element_mismatches;
    specfem_test::MismatchRecorder z_mismatches;
    specfem_test::MismatchRecorder x_mismatches;
    for (std::size_t icell = 0; icell < connectivity.extent(0); ++icell) {
      const int corner = connectivity(icell, 0);
      const int element = corner / (ngll * ngll);
      const int corner_z = (corner / ngll) % ngll;
      const int corner_x = corner % ngll;
      ASSERT_LT(corner_z, ngll - 1) << "cell " << icell;
      ASSERT_LT(corner_x, ngll - 1) << "cell " << icell;
      corners.insert({ element, corner_z, corner_x });

      for (int inode = 0; inode < 4; ++inode) {
        const int node = connectivity(icell, inode);
        ASSERT_GE(node, 0);
        ASSERT_LT(node, number_of_points);
        element_mismatches.equal(node / (ngll * ngll), element, icell, inode);
        z_mismatches.equal((node / ngll) % ngll,
                           corner_z + node_offsets[inode][1], icell, inode);
        x_mismatches.equal(node % ngll, corner_x + node_offsets[inode][0],
                           icell, inode);
      }
    }
    EXPECT_TRUE(element_mismatches.result()) << "ngll=" << ngll << " element";
    EXPECT_TRUE(z_mismatches.result()) << "ngll=" << ngll << " z";
    EXPECT_TRUE(x_mismatches.result()) << "ngll=" << ngll << " x";
    EXPECT_EQ(corners.size(), connectivity.extent(0));
  }
}

TEST(SubCellConnectivity, HexahedraSpanNeighbouringGllPoints) {
  constexpr int number_of_elements = 2;
  // Local (ix, iy, iz) offsets of the VTK hexahedron nodes from the corner
  constexpr std::array<std::array<int, 3>, 8> node_offsets = { { { 0, 0, 0 },
                                                                 { 1, 0, 0 },
                                                                 { 1, 1, 0 },
                                                                 { 0, 1, 0 },
                                                                 { 0, 0, 1 },
                                                                 { 1, 0, 1 },
                                                                 { 1, 1, 1 },
                                                                 { 0, 1,
                                                                   1 } } };

  for (const int ngll : { 2, 5 }) {
    const specfem::mesh_entity::element_grid<
        specfem::element::dimension_tag::dim3>
        element_grid(ngll, ngll, ngll);
    const auto connectivity = specfem::io::impl::sub_cell_connectivity(
        number_of_elements, element_grid);

    const int points_per_element = ngll * ngll * ngll;
    const int number_of_points = number_of_elements * points_per_element;
    ASSERT_EQ(connectivity.extent(0),
              number_of_elements * (ngll - 1) * (ngll - 1) * (ngll - 1));
    ASSERT_EQ(connectivity.extent(1), 8);

    std::set<std::tuple<int, int, int, int>> corners;
    specfem_test::MismatchRecorder element_mismatches;
    specfem_test::MismatchRecorder z_mismatches;
    specfem_test::MismatchRecorder y_mismatches;
    specfem_test::MismatchRecorder x_mismatches;
    for (std::size_t icell = 0; icell < connectivity.extent(0); ++icell) {
      const int corner = connectivity(icell, 0);
      const int element = corner / points_per_element;
      const int corner_z = (corner / (ngll * ngll)) % ngll;
      const int corner_y = (corner / ngll) % ngll;
      const int corner_x = corner % ngll;
      ASSERT_LT(corner_z, ngll - 1) << "cell " << icell;
      ASSERT_LT(corner_y, ngll - 1) << "cell " << icell;
      ASSERT_LT(corner_x, ngll - 1) << "cell " << icell;
      corners.insert({ element, corner_z, corner_y, corner_x });

      for (int inode = 0; inode < 8; ++inode) {
        const int node = connectivity(icell, inode);
        ASSERT_GE(node, 0);
        ASSERT_LT(node, number_of_points);
        element_mismatches.equal(node / points_per_element, element, icell,
                                 inode);
        z_mismatches.equal((node / (ngll * ngll)) % ngll,
                           corner_z + node_offsets[inode][2], icell, inode);
        y_mismatches.equal((node / ngll) % ngll,
                           corner_y + node_offsets[inode][1], icell, inode);
        x_mismatches.equal(node % ngll, corner_x + node_offsets[inode][0],
                           icell, inode);
      }
    }
    EXPECT_TRUE(element_mismatches.result()) << "ngll=" << ngll << " element";
    EXPECT_TRUE(z_mismatches.result()) << "ngll=" << ngll << " z";
    EXPECT_TRUE(y_mismatches.result()) << "ngll=" << ngll << " y";
    EXPECT_TRUE(x_mismatches.result()) << "ngll=" << ngll << " x";
    EXPECT_EQ(corners.size(), connectivity.extent(0));
  }
}

int main(int argc, char *argv[]) {
  ::testing::InitGoogleTest(&argc, argv);
  ::testing::AddGlobalTestEnvironment(new SPECFEMEnvironment);
  return RUN_ALL_TESTS();
}
