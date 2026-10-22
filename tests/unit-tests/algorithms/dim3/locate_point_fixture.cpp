#include "locate_point_fixture.hpp"

// Test locate_point core functionality with single unit cube
TEST_F(LocatePoint3D, CoreUnitCube) {
  auto geom = create_single_unit_cube_2x2x2();

  // Test point at center of unit cube (0.5, 0.5, 0.5)
  // Should map to local coordinates (0, 0, 0)
  specfem::point::global_coordinates<specfem::element::dimension_tag::dim3>
      target = { 0.5, 0.5, 0.5 };

  auto result = specfem::algorithms::locate_point_impl::locate_point_core(
      target, geom.global_coords, geom.index_mapping, geom.control_nodes,
      gll_points(geom.ngllx), geom.ngnod, geom.ngllx);

  // Should find element 0 with local coordinates near (0, 0, 0) for center
  // point
  EXPECT_EQ(result.ispec, 0);
  EXPECT_TRUE(is_close(result.xi, type_real{ 0.0 }))
      << expected_got(0.0, result.xi);
  EXPECT_TRUE(is_close(result.eta, type_real{ 0.0 }))
      << expected_got(0.0, result.eta);
  EXPECT_TRUE(is_close(result.gamma, type_real{ 0.0 }))
      << expected_got(0.0, result.gamma);

  // Test corner point (0, 0, 0) should map to (-1, -1, -1)
  target = { 0.0, 0.0, 0.0 };
  result = specfem::algorithms::locate_point_impl::locate_point_core(
      target, geom.global_coords, geom.index_mapping, geom.control_nodes,
      gll_points(geom.ngllx), geom.ngnod, geom.ngllx);

  EXPECT_EQ(result.ispec, 0);
  EXPECT_TRUE(is_close(result.xi, type_real{ -1.0 }))
      << expected_got(-1.0, result.xi);
  EXPECT_TRUE(is_close(result.eta, type_real{ -1.0 }))
      << expected_got(-1.0, result.eta);
  EXPECT_TRUE(is_close(result.gamma, type_real{ -1.0 }))
      << expected_got(-1.0, result.gamma);

  // Test corner point (1, 1, 1) should map to (1, 1, 1)
  target = { 1.0, 1.0, 1.0 };
  result = specfem::algorithms::locate_point_impl::locate_point_core(
      target, geom.global_coords, geom.index_mapping, geom.control_nodes,
      gll_points(geom.ngllx), geom.ngnod, geom.ngllx);

  EXPECT_EQ(result.ispec, 0);
  EXPECT_TRUE(is_close(result.xi, type_real{ 1.0 }))
      << expected_got(1.0, result.xi);
  EXPECT_TRUE(is_close(result.eta, type_real{ 1.0 }))
      << expected_got(1.0, result.eta);
  EXPECT_TRUE(is_close(result.gamma, type_real{ 1.0 }))
      << expected_got(1.0, result.gamma);
}

// Test locate_point with two adjacent 3D elements
TEST_F(LocatePoint3D, LocatePoint3DCoreTwoAdjacentElements) {
  auto geom = create_two_adjacent_elements_3d_2x2x2();

  // Test point in left element (0.25, 0.25, 0.25)
  specfem::point::global_coordinates<specfem::element::dimension_tag::dim3>
      target = { 0.25, 0.25, 0.25 };

  auto result = specfem::algorithms::locate_point_impl::locate_point_core(
      target, geom.global_coords, geom.index_mapping, geom.control_nodes,
      gll_points(geom.ngllx), geom.ngnod, geom.ngllx);

  EXPECT_EQ(result.ispec, 0);
  EXPECT_TRUE(is_close(result.xi, type_real{ 0.0 }))
      << expected_got(0.0, result.xi);
  EXPECT_TRUE(is_close(result.eta, type_real{ 0.0 }))
      << expected_got(0.0, result.eta);
  EXPECT_TRUE(is_close(result.gamma, type_real{ 0.0 }))
      << expected_got(0.0, result.gamma);

  // Test point in right element (0.75, 0.25, 0.25)
  target = { 0.75, 0.25, 0.25 };
  result = specfem::algorithms::locate_point_impl::locate_point_core(
      target, geom.global_coords, geom.index_mapping, geom.control_nodes,
      gll_points(geom.ngllx), geom.ngnod, geom.ngllx);

  EXPECT_EQ(result.ispec, 1);
  EXPECT_TRUE(is_close(result.xi, type_real{ 0.0 }))
      << expected_got(0.0, result.xi);
  EXPECT_TRUE(is_close(result.eta, type_real{ 0.0 }))
      << expected_got(0.0, result.eta);
  EXPECT_TRUE(is_close(result.gamma, type_real{ 0.0 }))
      << expected_got(0.0, result.gamma);

  // Test shared face point (0.5, 0.25, 0.25) - should find one of the elements
  target = { 0.5, 0.25, 0.25 };
  result = specfem::algorithms::locate_point_impl::locate_point_core(
      target, geom.global_coords, geom.index_mapping, geom.control_nodes,
      gll_points(geom.ngllx), geom.ngnod, geom.ngllx);

  EXPECT_TRUE(result.ispec == 0 ||
              result.ispec == 1); // Either element is valid
  EXPECT_TRUE(std::abs(std::abs(result.xi) - 1.0) < 1e-6); // Should be at face
                                                           // (±1)
}

// Test locate_point with 2x2x2 grid of elements
TEST_F(LocatePoint3D, Core2x2x2Grid) {
  auto geom = create_2x2x2_grid_elements_2x2x2();

  // Test points in different elements

  // Element 0: (0.5, 0.5, 0.5)
  specfem::point::global_coordinates<specfem::element::dimension_tag::dim3>
      target = { 0.5, 0.5, 0.5 };
  auto result = specfem::algorithms::locate_point_impl::locate_point_core(
      target, geom.global_coords, geom.index_mapping, geom.control_nodes,
      gll_points(geom.ngllx), geom.ngnod, geom.ngllx);
  EXPECT_EQ(result.ispec, 0);

  // Element 1: (1.5, 0.5, 0.5)
  target = { 1.5, 0.5, 0.5 };
  result = specfem::algorithms::locate_point_impl::locate_point_core(
      target, geom.global_coords, geom.index_mapping, geom.control_nodes,
      gll_points(geom.ngllx), geom.ngnod, geom.ngllx);
  EXPECT_EQ(result.ispec, 1);

  // Element 4: (0.5, 0.5, 1.5) - first element in upper layer
  target = { 0.5, 0.5, 1.5 };
  result = specfem::algorithms::locate_point_impl::locate_point_core(
      target, geom.global_coords, geom.index_mapping, geom.control_nodes,
      gll_points(geom.ngllx), geom.ngnod, geom.ngllx);
  EXPECT_EQ(result.ispec, 4);

  // Test shared corner point (1, 1, 1) - should find one of the elements
  target = { 1.0, 1.0, 1.0 };
  result = specfem::algorithms::locate_point_impl::locate_point_core(
      target, geom.global_coords, geom.index_mapping, geom.control_nodes,
      gll_points(geom.ngllx), geom.ngnod, geom.ngllx);

  EXPECT_TRUE(result.ispec >= 0 && result.ispec <= 7); // Should find one of the
                                                       // 8 elements
  // At corner, should have local coordinates (±1, ±1, ±1) depending on element
  EXPECT_TRUE(std::abs(std::abs(result.xi) - 1.0) < 1e-6);
  EXPECT_TRUE(std::abs(std::abs(result.eta) - 1.0) < 1e-6);
  EXPECT_TRUE(std::abs(std::abs(result.gamma) - 1.0) < 1e-6);
}

// Test error case: point outside mesh
TEST_F(LocatePoint3D, CoreOutsideMesh) {
  auto geom = create_single_unit_cube_2x2x2();

  // Point outside mesh should throw exception
  specfem::point::global_coordinates<specfem::element::dimension_tag::dim3>
      target = { 2.0, 2.0, 2.0 };

  {
    bool captured = false;
    try {
      specfem::algorithms::locate_point_impl::locate_point_core(
          target, geom.global_coords, geom.index_mapping, geom.control_nodes,
          gll_points(geom.ngllx), geom.ngnod, geom.ngllx);
    } catch (const std::runtime_error &) {
      captured = true;
    }
    if (!captured)
      ADD_FAILURE() << "Expected std::runtime_error";
  }
}

// Helper function tests - testing individual components of locate_point_core

// Test rough_location helper function for 3D
TEST_F(LocatePoint3D, RoughLocationSimple3D) {
  auto geom = create_two_adjacent_elements_3d_2x2x2();

  // Test point close to (0.1, 0.1, 0.1) - should find element 0
  specfem::point::global_coordinates<specfem::element::dimension_tag::dim3>
      test_point = { 0.1, 0.1, 0.1 };

  auto [ix, iy, iz, ispec] =
      specfem::algorithms::locate_point_impl::rough_location(
          test_point, geom.global_coords);

  EXPECT_EQ(ispec, 0);
  EXPECT_EQ(ix, 0);
  EXPECT_EQ(iy, 0);
  EXPECT_EQ(iz, 0);

  // Test point close to (0.75, 0.25, 0.25) - should find element 1 (right
  // element)
  test_point = { 0.75, 0.25, 0.25 };
  std::tie(ix, iy, iz, ispec) =
      specfem::algorithms::locate_point_impl::rough_location(
          test_point, geom.global_coords);

  // Should find element 1 (right element) - but rough_location may find either
  // element since they share the face at x=0.5
  EXPECT_TRUE(ispec == 0 || ispec == 1);
}

// Test get_best_candidates helper function for 3D
TEST_F(LocatePoint3D, GetBestCandidatesSimple3D) {
  auto geom = create_two_adjacent_elements_3d_2x2x2();

  // Test with element 0 - should return both elements 0 and 1 since they share
  // face
  auto candidates = specfem::algorithms::locate_point_impl::get_best_candidates(
      0, geom.index_mapping);

  // Should return both elements since they share face nodes
  EXPECT_EQ(candidates.size(), 2);
  EXPECT_EQ(candidates[0], 0); // Initial guess element
  EXPECT_EQ(candidates[1], 1); // Neighboring element
}

// Test get_local_coordinates helper function for 3D
TEST_F(LocatePoint3D, GetLocalCoordinatesUnitCube) {
  auto geom = create_single_unit_cube_2x2x2();

  // Create control node coordinates view for the single element
  const int ngnod = 8;
  const Kokkos::View<specfem::point::global_coordinates<
                         specfem::element::dimension_tag::dim3> *,
                     Kokkos::HostSpace>
      coorg("coorg", ngnod);

  // Unit cube control nodes [0,1] x [0,1] x [0,1]
  coorg(0) = { 0.0, 0.0, 0.0 }; // Bottom face
  coorg(1) = { 1.0, 0.0, 0.0 };
  coorg(2) = { 1.0, 1.0, 0.0 };
  coorg(3) = { 0.0, 1.0, 0.0 };
  coorg(4) = { 0.0, 0.0, 1.0 }; // Top face
  coorg(5) = { 1.0, 0.0, 1.0 };
  coorg(6) = { 1.0, 1.0, 1.0 };
  coorg(7) = { 0.0, 1.0, 1.0 };

  // Test point at center of unit cube (0.5, 0.5, 0.5)
  // Should map to local coordinates (0, 0, 0)
  specfem::point::global_coordinates<specfem::element::dimension_tag::dim3>
      target = { 0.5, 0.5, 0.5 };

  type_real xi_initial = 0.1;
  type_real eta_initial = 0.1;
  type_real gamma_initial = 0.1;

  auto [xi_final, eta_final, gamma_final] =
      specfem::algorithms::locate_point_impl::get_local_coordinates(
          shift_anchors(coorg, target), xi_initial, eta_initial, gamma_initial);

  // For a unit cube, center point (0.5, 0.5, 0.5) should map to (0, 0, 0) in
  // reference coords
  EXPECT_TRUE(is_close(xi_final, type_real{ 0.0 }))
      << expected_got(0.0, xi_final);
  EXPECT_TRUE(is_close(eta_final, type_real{ 0.0 }))
      << expected_got(0.0, eta_final);
  EXPECT_TRUE(is_close(gamma_final, type_real{ 0.0 }))
      << expected_got(0.0, gamma_final);

  // Test corner point (0, 0, 0) should map to (-1, -1, -1)
  target = { 0.0, 0.0, 0.0 };
  std::tie(xi_final, eta_final, gamma_final) =
      specfem::algorithms::locate_point_impl::get_local_coordinates(
          shift_anchors(coorg, target), 0.0, 0.0, 0.0);

  EXPECT_TRUE(is_close(xi_final, type_real{ -1.0 }))
      << expected_got(-1.0, xi_final);
  EXPECT_TRUE(is_close(eta_final, type_real{ -1.0 }))
      << expected_got(-1.0, eta_final);
  EXPECT_TRUE(is_close(gamma_final, type_real{ -1.0 }))
      << expected_got(-1.0, gamma_final);

  // Test corner point (1, 1, 1) should map to (1, 1, 1)
  target = { 1.0, 1.0, 1.0 };
  std::tie(xi_final, eta_final, gamma_final) =
      specfem::algorithms::locate_point_impl::get_local_coordinates(
          shift_anchors(coorg, target), 0.0, 0.0, 0.0);

  EXPECT_TRUE(is_close(xi_final, type_real{ 1.0 }))
      << expected_got(1.0, xi_final);
  EXPECT_TRUE(is_close(eta_final, type_real{ 1.0 }))
      << expected_got(1.0, eta_final);
  EXPECT_TRUE(is_close(gamma_final, type_real{ 1.0 }))
      << expected_got(1.0, gamma_final);
}

// Test locate_point with 3x3x3 GLL points (realistic spectral element
// resolution)
TEST_F(LocatePoint3D, Core3x3x3UnitCube) {
  auto geom = create_single_unit_cube_3x3x3();

  // Test point at center of unit cube (0.5, 0.5, 0.5)
  // Should map to local coordinates (0, 0, 0)
  specfem::point::global_coordinates<specfem::element::dimension_tag::dim3>
      target = { 0.5, 0.5, 0.5 };

  auto result = specfem::algorithms::locate_point_impl::locate_point_core(
      target, geom.global_coords, geom.index_mapping, geom.control_nodes,
      gll_points(geom.ngllx), geom.ngnod, geom.ngllx);

  // Should find element 0 with local coordinates near (0, 0, 0) for center
  // point
  EXPECT_EQ(result.ispec, 0);
  EXPECT_TRUE(is_close(result.xi, type_real{ 0.0 }))
      << expected_got(0.0, result.xi);
  EXPECT_TRUE(is_close(result.eta, type_real{ 0.0 }))
      << expected_got(0.0, result.eta);
  EXPECT_TRUE(is_close(result.gamma, type_real{ 0.0 }))
      << expected_got(0.0, result.gamma);

  // Test corner point (0, 0, 0) should map to (-1, -1, -1)
  target = { 0.0, 0.0, 0.0 };
  result = specfem::algorithms::locate_point_impl::locate_point_core(
      target, geom.global_coords, geom.index_mapping, geom.control_nodes,
      gll_points(geom.ngllx), geom.ngnod, geom.ngllx);

  EXPECT_EQ(result.ispec, 0);
  EXPECT_TRUE(is_close(result.xi, type_real{ -1.0 }))
      << expected_got(-1.0, result.xi);
  EXPECT_TRUE(is_close(result.eta, type_real{ -1.0 }))
      << expected_got(-1.0, result.eta);
  EXPECT_TRUE(is_close(result.gamma, type_real{ -1.0 }))
      << expected_got(-1.0, result.gamma);
}

// Test locate_point with 27-node control elements
TEST_F(LocatePoint3D, Core27NodeElement) {
  auto geom = create_single_unit_cube_2x2x2_27node();

  // Test point at center of unit cube (0.5, 0.5, 0.5)
  // Should map to local coordinates (0, 0, 0)
  specfem::point::global_coordinates<specfem::element::dimension_tag::dim3>
      target = { 0.5, 0.5, 0.5 };

  auto result = specfem::algorithms::locate_point_impl::locate_point_core(
      target, geom.global_coords, geom.index_mapping, geom.control_nodes,
      gll_points(geom.ngllx), geom.ngnod, geom.ngllx);

  // Should find element 0 with local coordinates near (0, 0, 0) for center
  // point
  EXPECT_EQ(result.ispec, 0);
  EXPECT_TRUE(is_close(result.xi, type_real{ 0.0 }))
      << expected_got(0.0, result.xi);
  EXPECT_TRUE(is_close(result.eta, type_real{ 0.0 }))
      << expected_got(0.0, result.eta);
  EXPECT_TRUE(is_close(result.gamma, type_real{ 0.0 }))
      << expected_got(0.0, result.gamma);
}

// Test locate_point with 3x3x3 GLL points AND 27-node control elements (most
// realistic case)
TEST_F(LocatePoint3D, Core3x3x3With27Node) {
  auto geom = create_single_unit_cube_3x3x3_27node();

  // Test point at center of unit cube (0.5, 0.5, 0.5)
  // Should map to local coordinates (0, 0, 0)
  specfem::point::global_coordinates<specfem::element::dimension_tag::dim3>
      target = { 0.5, 0.5, 0.5 };

  auto result = specfem::algorithms::locate_point_impl::locate_point_core(
      target, geom.global_coords, geom.index_mapping, geom.control_nodes,
      gll_points(geom.ngllx), geom.ngnod, geom.ngllx);

  // Should find element 0 with local coordinates near (0, 0, 0) for center
  // point
  EXPECT_EQ(result.ispec, 0);
  EXPECT_TRUE(is_close(result.xi, type_real{ 0.0 }))
      << expected_got(0.0, result.xi);
  EXPECT_TRUE(is_close(result.eta, type_real{ 0.0 }))
      << expected_got(0.0, result.eta);
  EXPECT_TRUE(is_close(result.gamma, type_real{ 0.0 }))
      << expected_got(0.0, result.gamma);

  // Test eighth point (0.25, 0.25, 0.25) should map to (-0.5, -0.5, -0.5)
  target = { 0.25, 0.25, 0.25 };
  result = specfem::algorithms::locate_point_impl::locate_point_core(
      target, geom.global_coords, geom.index_mapping, geom.control_nodes,
      gll_points(geom.ngllx), geom.ngnod, geom.ngllx);

  EXPECT_EQ(result.ispec, 0);
  EXPECT_TRUE(is_close(result.xi, type_real{ -0.5 }))
      << expected_got(-0.5, result.xi);
  EXPECT_TRUE(is_close(result.eta, type_real{ -0.5 }))
      << expected_got(-0.5, result.eta);
  EXPECT_TRUE(is_close(result.gamma, type_real{ -0.5 }))
      << expected_got(-0.5, result.gamma);
}

TEST_F(LocatePoint3D, EarthScaleOffsetRoundTrip) {
  constexpr double element_size = 1.0e4;
  constexpr double offset = 5.0e6;
  constexpr int grid = 3;

  std::vector<std::vector<std::tuple<double, double, double>>> element_coords;
  std::vector<std::vector<std::tuple<double, double, double>>> control_coords;
  for (int ez = 0; ez < grid; ez++) {
    for (int ey = 0; ey < grid; ey++) {
      for (int ex = 0; ex < grid; ex++) {
        const double x0 = offset + ex * element_size;
        const double y0 = offset + ey * element_size;
        const double z0 = offset + ez * element_size;
        element_coords.push_back(create_unit_cube(3, x0, x0 + element_size, y0,
                                                  y0 + element_size, z0,
                                                  z0 + element_size)[0]);
        control_coords.push_back(create_control_nodes(8, x0, x0 + element_size,
                                                      y0, y0 + element_size, z0,
                                                      z0 + element_size)[0]);
      }
    }
  }
  auto geom = create_element_geometry(element_coords, control_coords);

  struct Case {
    int ex, ey, ez;
    double xi, eta, gamma;
  };
  const std::vector<Case> cases = { { 0, 0, 0, 0.3, -0.2, 0.6 },
                                    { 1, 1, 1, -0.4, 0.5, -0.7 },
                                    { 2, 0, 1, 0.3, -0.2, 0.6 },
                                    { 1, 2, 2, 0.1, 0.8, -0.3 },
                                    { 0, 2, 0, -0.6, 0.3, 0.2 } };

  const auto points = gll_points(geom.ngllx);
  for (const auto &test_case : cases) {
    const int expected_ispec =
        test_case.ex + grid * test_case.ey + grid * grid * test_case.ez;
    SCOPED_TRACE("ispec " + std::to_string(expected_ispec));
    const double lower_x = offset + test_case.ex * element_size;
    const double lower_y = offset + test_case.ey * element_size;
    const double lower_z = offset + test_case.ez * element_size;
    const specfem::point::global_coordinates<
        specfem::element::dimension_tag::dim3>
        target = { static_cast<type_real>(lower_x + 0.5 * (test_case.xi + 1.0) *
                                                        element_size),
                   static_cast<type_real>(
                       lower_y + 0.5 * (test_case.eta + 1.0) * element_size),
                   static_cast<type_real>(lower_z +
                                          0.5 * (test_case.gamma + 1.0) *
                                              element_size) };

    const auto result =
        specfem::algorithms::locate_point_impl::locate_point_core(
            target, geom.global_coords, geom.index_mapping, geom.control_nodes,
            points, geom.ngnod, geom.ngllx);

    EXPECT_EQ(result.ispec, expected_ispec);

    Kokkos::View<specfem::point::global_coordinates<
                     specfem::element::dimension_tag::dim3> *,
                 Kokkos::HostSpace>
        anchors("anchors", geom.ngnod);
    for (int inode = 0; inode < geom.ngnod; inode++) {
      anchors(inode) = { geom.control_nodes(result.ispec, inode, 0),
                         geom.control_nodes(result.ispec, inode, 1),
                         geom.control_nodes(result.ispec, inode, 2) };
    }
    // Back-project in the target-shifted frame to avoid the float floor of
    // absolute coordinates at the offset.
    const auto back_projected = specfem::jacobian::compute_locations(
        shift_anchors(anchors, target), geom.ngnod, result.xi, result.eta,
        result.gamma);
    const type_real distance = std::sqrt(back_projected.x * back_projected.x +
                                         back_projected.y * back_projected.y +
                                         back_projected.z * back_projected.z);

    EXPECT_LT(distance, 5e-3);
    EXPECT_NEAR(result.xi, test_case.xi, 1e-4);
    EXPECT_NEAR(result.eta, test_case.eta, 1e-4);
    EXPECT_NEAR(result.gamma, test_case.gamma, 1e-4);
  }
}

int main(int argc, char *argv[]) {
  ::testing::InitGoogleTest(&argc, argv);
  ::testing::AddGlobalTestEnvironment(new SPECFEMEnvironment);
  return RUN_ALL_TESTS();
}
