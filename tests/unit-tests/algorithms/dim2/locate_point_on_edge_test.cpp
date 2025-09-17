#include "locate_point_fixture.hpp"

using specfem::utilities::is_close;

void populate_element_gcoord_array(
    const Kokkos::View<
        specfem::point::global_coordinates<specfem::dimension::type::dim2> *,
        Kokkos::HostSpace> &coorg,
    const Kokkos::View<type_real ***, Kokkos::LayoutLeft, Kokkos::HostSpace>
        &control_node_coord,
    const int &ispec) {
  const int ngnod = coorg.extent(0);
  for (int i = 0; i < ngnod; i++) {
    coorg(i).x = control_node_coord(0, ispec, i);
    coorg(i).z = control_node_coord(1, ispec, i);
  }
}

// Test locate_point core functionality with single unit square
TEST_F(LocatePoint2D, LocateEdgeOnCoreUnitSquare) {
  auto geom = create_single_unit_square_2x2();

  const Kokkos::View<
      specfem::point::global_coordinates<specfem::dimension::type::dim2> *,
      Kokkos::HostSpace>
      coorg("coorg", geom.ngnod);
  populate_element_gcoord_array(coorg, geom.control_nodes, 0);

  // Test point at center of each edge
  // Should map to local edge coordinate (0)
  for (auto [side, target] :
       std::vector<std::pair<specfem::mesh_entity::type,
                             specfem::point::global_coordinates<
                                 specfem::dimension::type::dim2> > >{
           { specfem::mesh_entity::type::bottom, { 0.5, 0.0 } }, // Bottom edge
                                                                 // center
           { specfem::mesh_entity::type::right, { 1.0, 0.5 } },  // Right edge
                                                                 // center
           { specfem::mesh_entity::type::top, { 0.5, 1.0 } }, // Top edge center
           { specfem::mesh_entity::type::left, { 0.0, 0.5 } } // Left edge
                                                              // center
       }) {

    auto result =
        specfem::algorithms::locate_point_impl::get_local_edge_coordinate(
            target, coorg, side, 0);

    // Should find element 0 with local coordinates near (0, ±1) or (±1, 0)
    EXPECT_TRUE(is_close(result.first, type_real{ 0.0 }))
        << expected_got(0.0, result.first);
    EXPECT_TRUE(result.second); // inside edge (not out-of-bounds)
  }

  // Test corner point (0, 0) should map to {left: -1, bottom: -1}
  specfem::point::global_coordinates<specfem::dimension::type::dim2> target = {
    0.0, 0.0
  };
  auto result =
      specfem::algorithms::locate_point_impl::get_local_edge_coordinate(
          target, coorg, specfem::mesh_entity::type::bottom, 0);

  EXPECT_TRUE(is_close(result.first, type_real{ -1 }))
      << expected_got(0.0, result.first);
  EXPECT_TRUE(result.second); // inside edge (not out-of-bounds)

  result = specfem::algorithms::locate_point_impl::get_local_edge_coordinate(
      target, coorg, specfem::mesh_entity::type::left, 0);
  EXPECT_TRUE(is_close(result.first, type_real{ -1 }))
      << expected_got(0.0, result.first);
  EXPECT_TRUE(result.second); // inside edge (not out-of-bounds)

  // Test corner point (1, 1) should map to {top: 1, right: 1}
  target = { 1.0, 1.0 };
  result = specfem::algorithms::locate_point_impl::get_local_edge_coordinate(
      target, coorg, specfem::mesh_entity::type::right, 0);
  EXPECT_TRUE(is_close(result.first, type_real{ 1 }))
      << expected_got(0.0, result.first);
  EXPECT_TRUE(result.second); // inside edge (not out-of-bounds)
  result = specfem::algorithms::locate_point_impl::get_local_edge_coordinate(
      target, coorg, specfem::mesh_entity::type::top, 0);
  EXPECT_TRUE(is_close(result.first, type_real{ 1 }))
      << expected_got(0.0, result.first);
  EXPECT_TRUE(result.second); // inside edge (not out-of-bounds)
}

// Test error case: point outside mesh
TEST_F(LocatePoint2D, LocateEdgeOnCoreOutsideMesh) {
  auto geom = create_single_unit_square_2x2();
  const Kokkos::View<
      specfem::point::global_coordinates<specfem::dimension::type::dim2> *,
      Kokkos::HostSpace>
      coorg("coorg", geom.ngnod);
  populate_element_gcoord_array(coorg, geom.control_nodes, 0);

  // Point away from edge (but distance minimized inside) should throw exception
  specfem::point::global_coordinates<specfem::dimension::type::dim2> target = {
    0.5, 0.5
  };

  for (auto side : std::vector<specfem::mesh_entity::type>{
           specfem::mesh_entity::type::bottom,
           specfem::mesh_entity::type::right, specfem::mesh_entity::type::top,
           specfem::mesh_entity::type::left }) {
    // target is center of element. Distance minimized on each edge at center,
    // so exception should be thrown
    EXPECT_THROW(
        specfem::algorithms::locate_point_impl::get_local_edge_coordinate(
            target, coorg, side, 0),
        std::runtime_error);
  }

  // Test corner point (1.5, -0.5) should map to (2, -2), out-of-bounds
  // for bottom, local coord will clamp to xi = 1, for right, gamma = -1
  target = { 1.5, -0.5 };
  auto result =
      specfem::algorithms::locate_point_impl::get_local_edge_coordinate(
          target, coorg, specfem::mesh_entity::type::right, 0);
  EXPECT_TRUE(is_close(result.first, type_real{ -1 }))
      << expected_got(0.0, result.first);
  EXPECT_FALSE(result.second); // inside edge (not out-of-bounds)
  result = specfem::algorithms::locate_point_impl::get_local_edge_coordinate(
      target, coorg, specfem::mesh_entity::type::bottom, 0);
  EXPECT_TRUE(is_close(result.first, type_real{ 1 }))
      << expected_got(0.0, result.first);
  EXPECT_FALSE(result.second); // inside edge (not out-of-bounds)
}
