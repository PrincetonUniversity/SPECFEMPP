#include "locate_point_fixture.hpp"
#include <gtest/gtest.h>
#include <sstream>

using specfem::utilities::is_close;

void populate_element_gcoord_array(
    const Kokkos::View<specfem::point::global_coordinates<
                           specfem::element::dimension_tag::dim3> *,
                       Kokkos::HostSpace> &coorg,
    const Kokkos::View<type_real ***, Kokkos::LayoutLeft, Kokkos::HostSpace>
        &control_node_coord,
    const int &ispec) {
  const int ngnod = coorg.extent(0);
  for (int i = 0; i < ngnod; i++) {
    coorg(i).x = control_node_coord(ispec, i, 0);
    coorg(i).y = control_node_coord(ispec, i, 1);
    coorg(i).z = control_node_coord(ispec, i, 2);
  }
}

// Test locate_point core functionality with single unit square
TEST_F(LocatePoint3D, LocateFaceOnCoreUnitCube) {
  auto geom = create_single_unit_cube_2x2x2();

  const Kokkos::View<specfem::point::global_coordinates<
                         specfem::element::dimension_tag::dim3> *,
                     Kokkos::HostSpace>
      coorg("coorg", geom.ngnod);
  populate_element_gcoord_array(coorg, geom.control_nodes, 0);

  // Test point at center of each edge
  // Should map to local edge coordinate (0)
  for (auto [side, target] :
       std::vector<std::pair<specfem::mesh_entity::dim3::type,
                             specfem::point::global_coordinates<
                                 specfem::element::dimension_tag::dim3>>>{
           { specfem::mesh_entity::dim3::type::bottom,
             { 0.5, 0.5, 0.0 } }, // Bottom
                                  // edge
                                  // center
           { specfem::mesh_entity::dim3::type::right,
             { 1.0, 0.5, 0.5 } }, // Right
                                  // edge
                                  // center
           { specfem::mesh_entity::dim3::type::top,
             { 0.5, 0.5, 1.0 } }, // Top edge
                                  // center
           { specfem::mesh_entity::dim3::type::left,
             { 0.0, 0.5, 0.5 } }, // Left edge
                                  // center

           { specfem::mesh_entity::dim3::type::front, { 0.5, 0.0, 0.5 } },

           { specfem::mesh_entity::dim3::type::back, { 0.5, 1.0, 0.5 } },
       }) {

    auto result =
        specfem::algorithms::locate_point_impl::get_local_face_coordinate(
            target, coorg, side, { 0.3, -0.3 });

    // Should find element 0 with local coordinates near (0, ±1) or (±1, 0)
    EXPECT_TRUE(is_close(result.first.first, type_real{ 0.0 }))
        << expected_got(0.0, result.first.first);
    EXPECT_TRUE(is_close(result.first.second, type_real{ 0.0 }))
        << expected_got(0.0, result.first.second);
    EXPECT_TRUE(result.second); // inside edge (not out-of-bounds)
  }

  // Test corner point (0, 0) should map to {left: -1, bottom: -1}
  specfem::point::global_coordinates<specfem::element::dimension_tag::dim3>
      target = { 0.0, 0.0, 0.0 };
  auto result =
      specfem::algorithms::locate_point_impl::get_local_face_coordinate(
          target, coorg, specfem::mesh_entity::dim3::type::bottom, { 0, 0 });

  EXPECT_TRUE(is_close(result.first.first, type_real{ -1 }))
      << expected_got(0.0, result.first.first);
  EXPECT_TRUE(is_close(result.first.second, type_real{ -1 }))
      << expected_got(0.0, result.first.second);
  EXPECT_TRUE(result.second); // inside edge (not out-of-bounds)

  result = specfem::algorithms::locate_point_impl::get_local_face_coordinate(
      target, coorg, specfem::mesh_entity::dim3::type::left, { 0, 0 });
  EXPECT_TRUE(is_close(result.first.first, type_real{ -1 }))
      << expected_got(0.0, result.first.first);
  EXPECT_TRUE(is_close(result.first.second, type_real{ -1 }))
      << expected_got(0.0, result.first.second);
  EXPECT_TRUE(result.second); // inside edge (not out-of-bounds)

  result = specfem::algorithms::locate_point_impl::get_local_face_coordinate(
      target, coorg, specfem::mesh_entity::dim3::type::front, { 0, 0 });
  EXPECT_TRUE(is_close(result.first.first, type_real{ -1 }))
      << expected_got(0.0, result.first.first);
  EXPECT_TRUE(is_close(result.first.second, type_real{ -1 }))
      << expected_got(0.0, result.first.second);
  EXPECT_TRUE(result.second); // inside edge (not out-of-bounds)

  // Test corner point (1, 1) should map to {top: 1, right: 1}
  target = { 1.0, 1.0, 1.0 };
  result = specfem::algorithms::locate_point_impl::get_local_face_coordinate(
      target, coorg, specfem::mesh_entity::dim3::type::right, { 0, 0 });
  EXPECT_TRUE(is_close(result.first.first, type_real{ 1 }))
      << expected_got(0.0, result.first.first);
  EXPECT_TRUE(is_close(result.first.second, type_real{ 1 }))
      << expected_got(0.0, result.first.second);
  EXPECT_TRUE(result.second); // inside edge (not out-of-bounds)
  result = specfem::algorithms::locate_point_impl::get_local_face_coordinate(
      target, coorg, specfem::mesh_entity::dim3::type::top, { 0, 0 });
  EXPECT_TRUE(is_close(result.first.first, type_real{ 1 }))
      << expected_got(0.0, result.first.first);
  EXPECT_TRUE(is_close(result.first.second, type_real{ 1 }))
      << expected_got(0.0, result.first.second);
  EXPECT_TRUE(result.second); // inside edge (not out-of-bounds)
  result = specfem::algorithms::locate_point_impl::get_local_face_coordinate(
      target, coorg, specfem::mesh_entity::dim3::type::back, { 0, 0 });
  EXPECT_TRUE(is_close(result.first.first, type_real{ 1 }))
      << expected_got(0.0, result.first.first);
  EXPECT_TRUE(is_close(result.first.second, type_real{ 1 }))
      << expected_got(0.0, result.first.second);
  EXPECT_TRUE(result.second); // inside edge (not out-of-bounds)
}

// Test error case: point outside mesh
TEST_F(LocatePoint3D, LocateFaceOnCoreOutsideMesh) {
  auto geom = create_single_unit_cube_2x2x2();
  const Kokkos::View<specfem::point::global_coordinates<
                         specfem::element::dimension_tag::dim3> *,
                     Kokkos::HostSpace>
      coorg("coorg", geom.ngnod);
  populate_element_gcoord_array(coorg, geom.control_nodes, 0);

  // Point away from edge (but distance minimized inside) should throw exception
  specfem::point::global_coordinates<specfem::element::dimension_tag::dim3>
      target = { 0.5, 0.5, 0.5 };

  for (auto side : std::vector<specfem::mesh_entity::dim3::type>{
           specfem::mesh_entity::dim3::type::bottom,
           specfem::mesh_entity::dim3::type::right,
           specfem::mesh_entity::dim3::type::top,
           specfem::mesh_entity::dim3::type::left,
           specfem::mesh_entity::dim3::type::front,
           specfem::mesh_entity::dim3::type::back,
       }) {
    // target is center of element. Distance minimized on each edge at center,
    // so exception should be thrown
    {
      bool captured = false;
      try {
        specfem::algorithms::locate_point_impl::get_local_face_coordinate(
            target, coorg, side, { 0.3, -0.5 });
      } catch (const std::runtime_error &) {
        captured = true;
      }
      if (!captured)
        ADD_FAILURE() << "Expected std::runtime_error";
    }
  }

  // Test corner point (1.5, -0.5, 0.25) should map to (2, -2, -0.5),
  // out-of-bounds for bottom, local coord will clamp to (1, -1, -0.5)
  // remember: for local face coordinates, second coordinate first.
  target = { 1.5, -0.5, 0.25 };
  type_real target_clamped[3] = { 1, -1, -0.5 };
  std::vector<std::string> axis_names = { "xi", "eta", "gamma" };
  std::vector<specfem::mesh_entity::dim3::type> sides_to_test = {
    specfem::mesh_entity::dim3::type::right,
    specfem::mesh_entity::dim3::type::front,
    specfem::mesh_entity::dim3::type::bottom
  };
  // face-local to element-local for each side in sides_to_test.
  std::vector<std::array<int, 2>> local_axes = { { 2, 1 }, { 2, 0 }, { 1, 0 } };

  std::ostringstream fail_case_expand;
  bool fails = false;
  for (int iside = 0; iside < sides_to_test.size(); iside++) {
    int axis[2] = { local_axes[iside][0], local_axes[iside][1] };
    type_real expect[2] = { target_clamped[axis[0]], target_clamped[axis[1]] };

    auto result =
        specfem::algorithms::locate_point_impl::get_local_face_coordinate(
            target, coorg, sides_to_test[iside], { 0, 0 });

    bool correct_coords = is_close(result.first.first, expect[0]) &&
                          is_close(result.first.second, expect[1]);
    bool correct_isinside_flag = !result.second;

    if (!(correct_coords && correct_isinside_flag)) {
      fails = true;
      fail_case_expand
          << "  - Failure on get_local_face_coordinate() for right side:\n";
      if (!correct_coords) {
        fail_case_expand << "   [✗] Expects: (" << axis_names[axis[0]] << " = "
                         << expect[0] << ", " << axis_names[axis[1]] << " = "
                         << expect[1] << ")\n"
                         << "           Got: (" << result.first.first << ","
                         << result.first.second << ")\n";
      } else {
        fail_case_expand << "   [✓] Got correct coords (" << axis_names[axis[0]]
                         << " = " << expect[0] << ", " << axis_names[axis[1]]
                         << " = " << expect[1] << ")\n";
      }
      if (result.second) {
        fail_case_expand
            << "   [✗] Expected out-of-bounds. Second return value was instead "
               "true (is inside face)\n";
      } else {
        fail_case_expand
            << "   [✓] Expected out-of-bounds. Second return value correctly "
               "false (is not in face)\n";
      }
    }
  }

  if (fails) {
    ADD_FAILURE() << "global(1.5, -0.5, 0.25) -> local(2, -2, -0.5)\n"
                  << fail_case_expand.str();
  }
}
