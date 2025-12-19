#include "../../SPECFEM_Environment.hpp"
#include "../../mesh_utilities/mapping.hpp"
#include "../../test_macros.hpp"
#include "algorithms/locate_point.hpp"
#include "algorithms/locate_point_impl.hpp"
#include "kokkos_abstractions.h"
#include "specfem/point.hpp"
#include "utilities/utilities.hpp"
#include <Kokkos_Core.hpp>
#include <cmath>
#include <gtest/gtest.h>

using specfem::utilities::is_close;

// Use test-specific mesh utilities for coordinate generation
using namespace specfem::test::mesh_utilities;

// Test fixture for 3D locate_point algorithms using tested mesh utilities
class LocatePoint3D : public ::testing::Test {
protected:
  struct ElementGeometry {
    Kokkos::View<type_real *****, Kokkos::LayoutLeft, Kokkos::HostSpace>
        global_coords;
    Kokkos::View<int ****, Kokkos::LayoutLeft, Kokkos::HostSpace> index_mapping;
    Kokkos::View<type_real ***, Kokkos::LayoutLeft, Kokkos::HostSpace>
        control_nodes; // (ispec, inode, icoord)
    int nspec;
    int ngllx;
    int nglly;
    int ngllz;
    int ngnod;
    int nglob;
    type_real xmin, xmax, ymin, ymax, zmin, zmax;
  };

  void SetUp() override {}

  void TearDown() override {}

  // Helper to create unit cube coordinates programmatically
  std::vector<std::vector<std::tuple<double, double, double> > >
  create_unit_cube(int ngll, double xmin = -1.0, double xmax = 1.0,
                   double ymin = -1.0, double ymax = 1.0, double zmin = -1.0,
                   double zmax = 1.0) {
    std::vector<std::tuple<double, double, double> > coords;
    for (int iz = 0; iz < ngll; iz++) {
      for (int iy = 0; iy < ngll; iy++) {
        for (int ix = 0; ix < ngll; ix++) {
          double x = xmin + (xmax - xmin) * ix / (ngll - 1);
          double y = ymin + (ymax - ymin) * iy / (ngll - 1);
          double z = zmin + (zmax - zmin) * iz / (ngll - 1);
          coords.push_back({ x, y, z });
        }
      }
    }
    return { coords };
  }

  // Helper to create control nodes for 8-node or 27-node elements
  std::vector<std::vector<std::tuple<double, double, double> > >
  create_control_nodes(int ngnod, double xmin = 0.0, double xmax = 1.0,
                       double ymin = 0.0, double ymax = 1.0, double zmin = 0.0,
                       double zmax = 1.0) {
    std::vector<std::tuple<double, double, double> > nodes;

    if (ngnod == 8) {
      // 8-node hexahedral: corners only - standard hexahedral ordering
      nodes = {
        { xmin, ymin, zmin }, // 0: (0,0,0)
        { xmax, ymin, zmin }, // 1: (1,0,0)
        { xmax, ymax, zmin }, // 2: (1,1,0)
        { xmin, ymax, zmin }, // 3: (0,1,0)
        { xmin, ymin, zmax }, // 4: (0,0,1)
        { xmax, ymin, zmax }, // 5: (1,0,1)
        { xmax, ymax, zmax }, // 6: (1,1,1)
        { xmin, ymax, zmax }  // 7: (0,1,1)
      };
    } else if (ngnod == 27) {
      // 27-node hexahedral: matching SPECFEM shape function ordering
      double xmid = (xmin + xmax) / 2.0;
      double ymid = (ymin + ymax) / 2.0;
      double zmid = (zmin + zmax) / 2.0;
      nodes = {
        // 8 corner nodes (0-7) - same as 8-node ordering
        { xmin, ymin, zmin }, // 0: (-1,-1,-1) -> (0,0,0)
        { xmax, ymin, zmin }, // 1: (+1,-1,-1) -> (1,0,0)
        { xmax, ymax, zmin }, // 2: (+1,+1,-1) -> (1,1,0)
        { xmin, ymax, zmin }, // 3: (-1,+1,-1) -> (0,1,0)
        { xmin, ymin, zmax }, // 4: (-1,-1,+1) -> (0,0,1)
        { xmax, ymin, zmax }, // 5: (+1,-1,+1) -> (1,0,1)
        { xmax, ymax, zmax }, // 6: (+1,+1,+1) -> (1,1,1)
        { xmin, ymax, zmax }, // 7: (-1,+1,+1) -> (0,1,1)
        // 12 mid-edge nodes (8-19)
        { xmid, ymin, zmin }, // 8: ( 0,-1,-1) -> (0.5,0,0) - bottom front edge
        { xmax, ymid, zmin }, // 9: (+1, 0,-1) -> (1,0.5,0) - bottom right edge
        { xmid, ymax, zmin }, // 10: ( 0,+1,-1) -> (0.5,1,0) - bottom back edge
        { xmin, ymid, zmin }, // 11: (-1, 0,-1) -> (0,0.5,0) - bottom left edge
        { xmin, ymin, zmid }, // 12: (-1,-1, 0) -> (0,0,0.5) - front left edge
        { xmax, ymin, zmid }, // 13: (+1,-1, 0) -> (1,0,0.5) - front right edge
        { xmax, ymax, zmid }, // 14: (+1,+1, 0) -> (1,1,0.5) - back right edge
        { xmin, ymax, zmid }, // 15: (-1,+1, 0) -> (0,1,0.5) - back left edge
        { xmid, ymin, zmax }, // 16: ( 0,-1,+1) -> (0.5,0,1) - top front edge
        { xmax, ymid, zmax }, // 17: (+1, 0,+1) -> (1,0.5,1) - top right edge
        { xmid, ymax, zmax }, // 18: ( 0,+1,+1) -> (0.5,1,1) - top back edge
        { xmin, ymid, zmax }, // 19: (-1, 0,+1) -> (0,0.5,1) - top left edge
        // 6 face centers (20-25)
        { xmid, ymid, zmin }, // 20: ( 0, 0,-1) -> (0.5,0.5,0) - bottom face
        { xmid, ymin, zmid }, // 21: ( 0,-1, 0) -> (0.5,0,0.5) - front face
        { xmax, ymid, zmid }, // 22: (+1, 0, 0) -> (1,0.5,0.5) - right face
        { xmid, ymax, zmid }, // 23: ( 0,+1, 0) -> (0.5,1,0.5) - back face
        { xmin, ymid, zmid }, // 24: (-1, 0, 0) -> (0,0.5,0.5) - left face
        { xmid, ymid, zmax }, // 25: ( 0, 0,+1) -> (0.5,0.5,1) - top face
        // 1 volume center (26)
        { xmid, ymid, zmid } // 26: ( 0, 0, 0) -> (0.5,0.5,0.5) - center
      };
    } else {
      throw std::runtime_error("Unsupported ngnod: " + std::to_string(ngnod));
    }

    return { nodes };
  }

  // Helper to create coordinate array from element corners and ngll points
  Kokkos::View<type_real *****, Kokkos::LayoutLeft, Kokkos::HostSpace>
  create_coordinate_array(
      const std::vector<std::vector<std::tuple<double, double, double> > >
          &element_coords) {
    int nspec = element_coords.size();
    int ngll = std::cbrt(element_coords[0].size()); // cube root for 3D

    Kokkos::View<type_real *****, Kokkos::LayoutLeft, Kokkos::HostSpace> coords(
        "coords", nspec, ngll, ngll, ngll, 3);

    for (int ispec = 0; ispec < nspec; ispec++) {
      int idx = 0;
      for (int iz = 0; iz < ngll; iz++) {
        for (int iy = 0; iy < ngll; iy++) {
          for (int ix = 0; ix < ngll; ix++) {
            coords(ispec, iz, iy, ix, 0) =
                std::get<0>(element_coords[ispec][idx]); // x
            coords(ispec, iz, iy, ix, 1) =
                std::get<1>(element_coords[ispec][idx]); // y
            coords(ispec, iz, iy, ix, 2) =
                std::get<2>(element_coords[ispec][idx]); // z
            idx++;
          }
        }
      }
    }
    return coords;
  }

  // Create element geometry using tested utility functions
  ElementGeometry create_element_geometry(
      const std::vector<std::vector<std::tuple<double, double, double> > >
          &element_coords,
      const std::vector<std::vector<std::tuple<double, double, double> > >
          &control_coords) {

    ElementGeometry geom;
    geom.nspec = element_coords.size();
    geom.ngllx = std::cbrt(element_coords[0].size());
    geom.nglly = geom.ngllx;
    geom.ngllz = geom.ngllx;
    geom.ngnod = control_coords[0].size();
    int ngllxyz = geom.ngllx * geom.nglly * geom.ngllz;

    // Create coordinate array
    auto coords_double = create_coordinate_array(element_coords);

    // Use tested utility functions to replicate assign_numbering logic
    auto points = flatten_coordinates(coords_double);
    auto sorted_points = points;
    sort_points_spatially(sorted_points);
    type_real tolerance =
        compute_spatial_tolerance(sorted_points, geom.nspec, ngllxyz);
    int nglob = assign_global_numbering(sorted_points, tolerance);
    auto reordered_points = reorder_to_original_layout(sorted_points);
    auto bbox = compute_bounding_box(reordered_points);

    // Use the create_coordinate_arrays function to get proper index_mapping and
    // coordinates
    auto [index_mapping, global_coords, nglob_actual] =
        create_coordinate_arrays(reordered_points, geom.nspec, geom.ngllx,
                                 nglob);

    // Assign to geometry structure
    geom.index_mapping = index_mapping;
    geom.global_coords = global_coords;
    geom.nglob = nglob_actual;

    // Set up control nodes - Note: access pattern is (ispec, inode, icoord)
    geom.control_nodes =
        Kokkos::View<type_real ***, Kokkos::LayoutLeft, Kokkos::HostSpace>(
            "control_nodes", geom.nspec, geom.ngnod, 3);
    for (int ispec = 0; ispec < geom.nspec; ispec++) {
      for (int inode = 0; inode < geom.ngnod; inode++) {
        geom.control_nodes(ispec, inode, 0) =
            std::get<0>(control_coords[ispec][inode]);
        geom.control_nodes(ispec, inode, 1) =
            std::get<1>(control_coords[ispec][inode]);
        geom.control_nodes(ispec, inode, 2) =
            std::get<2>(control_coords[ispec][inode]);
      }
    }

    // Set bounding box
    geom.xmin = bbox.xmin;
    geom.xmax = bbox.xmax;
    geom.ymin = bbox.ymin;
    geom.ymax = bbox.ymax;
    geom.zmin = bbox.zmin;
    geom.zmax = bbox.zmax;

    return geom;
  }

  // Create a single unit cube element [0,1] x [0,1] x [0,1] with 2x2x2 GLL
  // points
  ElementGeometry create_single_unit_cube_2x2x2() {
    auto element_coords = create_unit_cube(2, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0);
    auto control_coords = create_control_nodes(8, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0);
    return create_element_geometry(element_coords, control_coords);
  }

  // Create two adjacent elements in x-direction with 2x2x2 GLL points
  ElementGeometry create_two_adjacent_elements_3d_2x2x2() {
    std::vector<std::vector<std::tuple<double, double, double> > >
        element_coords = {
          create_unit_cube(2, 0.0, 0.5, 0.0, 0.5, 0.0, 0.5)[0], // Left element
          create_unit_cube(2, 0.5, 1.0, 0.0, 0.5, 0.0,
                           0.5)[0] // Right element (shares face at x=0.5)
        };
    auto control_coords =
        std::vector<std::vector<std::tuple<double, double, double> > >{
          create_control_nodes(8, 0.0, 0.5, 0.0, 0.5, 0.0, 0.5)[0], // Left
                                                                    // element
          create_control_nodes(8, 0.5, 1.0, 0.0, 0.5, 0.0, 0.5)[0]  // Right
                                                                    // element
        };
    return create_element_geometry(element_coords, control_coords);
  }

  // Create 2x2x2 grid of elements with 2x2x2 GLL points each
  ElementGeometry create_2x2x2_grid_elements_2x2x2() {
    std::vector<std::vector<std::tuple<double, double, double> > >
        element_coords;
    std::vector<std::vector<std::tuple<double, double, double> > >
        control_coords;

    // Create 8 elements in 2x2x2 grid
    for (int ez = 0; ez < 2; ez++) {
      for (int ey = 0; ey < 2; ey++) {
        for (int ex = 0; ex < 2; ex++) {
          element_coords.push_back(
              create_unit_cube(2, ex, ex + 1, ey, ey + 1, ez, ez + 1)[0]);
          control_coords.push_back(
              create_control_nodes(8, ex, ex + 1, ey, ey + 1, ez, ez + 1)[0]);
        }
      }
    }
    return create_element_geometry(element_coords, control_coords);
  }

  // Create a single unit cube element [0,1] x [0,1] x [0,1] with 3x3x3 GLL
  // points
  ElementGeometry create_single_unit_cube_3x3x3() {
    auto element_coords = create_unit_cube(3, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0);
    auto control_coords = create_control_nodes(8, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0);
    return create_element_geometry(element_coords, control_coords);
  }

  // Create a single unit cube element with 2x2x2 GLL points and 27 control
  // nodes
  ElementGeometry create_single_unit_cube_2x2x2_27node() {
    auto element_coords = create_unit_cube(2, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0);
    auto control_coords =
        create_control_nodes(27, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0);
    return create_element_geometry(element_coords, control_coords);
  }

  // Create a single unit cube element with 3x3x3 GLL points and 27 control
  // nodes
  ElementGeometry create_single_unit_cube_3x3x3_27node() {
    auto element_coords = create_unit_cube(3, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0);
    auto control_coords =
        create_control_nodes(27, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0);
    return create_element_geometry(element_coords, control_coords);
  }
};

// Test locate_point core functionality with single unit cube
TEST_F(LocatePoint3D, CoreUnitCube) {
  auto geom = create_single_unit_cube_2x2x2();

  // Test point at center of unit cube (0.5, 0.5, 0.5)
  // Should map to local coordinates (0, 0, 0)
  specfem::point::global_coordinates<specfem::dimension::type::dim3> target = {
    0.5, 0.5, 0.5
  };

  auto result = specfem::algorithms::locate_point_impl::locate_point_core(
      target, geom.global_coords, geom.index_mapping, geom.control_nodes,
      geom.ngnod, geom.ngllx);

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
      geom.ngnod, geom.ngllx);

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
      geom.ngnod, geom.ngllx);

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
  specfem::point::global_coordinates<specfem::dimension::type::dim3> target = {
    0.25, 0.25, 0.25
  };

  auto result = specfem::algorithms::locate_point_impl::locate_point_core(
      target, geom.global_coords, geom.index_mapping, geom.control_nodes,
      geom.ngnod, geom.ngllx);

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
      geom.ngnod, geom.ngllx);

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
      geom.ngnod, geom.ngllx);

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
  specfem::point::global_coordinates<specfem::dimension::type::dim3> target = {
    0.5, 0.5, 0.5
  };
  auto result = specfem::algorithms::locate_point_impl::locate_point_core(
      target, geom.global_coords, geom.index_mapping, geom.control_nodes,
      geom.ngnod, geom.ngllx);
  EXPECT_EQ(result.ispec, 0);

  // Element 1: (1.5, 0.5, 0.5)
  target = { 1.5, 0.5, 0.5 };
  result = specfem::algorithms::locate_point_impl::locate_point_core(
      target, geom.global_coords, geom.index_mapping, geom.control_nodes,
      geom.ngnod, geom.ngllx);
  EXPECT_EQ(result.ispec, 1);

  // Element 4: (0.5, 0.5, 1.5) - first element in upper layer
  target = { 0.5, 0.5, 1.5 };
  result = specfem::algorithms::locate_point_impl::locate_point_core(
      target, geom.global_coords, geom.index_mapping, geom.control_nodes,
      geom.ngnod, geom.ngllx);
  EXPECT_EQ(result.ispec, 4);

  // Test shared corner point (1, 1, 1) - should find one of the elements
  target = { 1.0, 1.0, 1.0 };
  result = specfem::algorithms::locate_point_impl::locate_point_core(
      target, geom.global_coords, geom.index_mapping, geom.control_nodes,
      geom.ngnod, geom.ngllx);

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
  specfem::point::global_coordinates<specfem::dimension::type::dim3> target = {
    2.0, 2.0, 2.0
  };

  EXPECT_THROW(specfem::algorithms::locate_point_impl::locate_point_core(
                   target, geom.global_coords, geom.index_mapping,
                   geom.control_nodes, geom.ngnod, geom.ngllx),
               std::runtime_error);
}

// Helper function tests - testing individual components of locate_point_core

// Test rough_location helper function for 3D
TEST_F(LocatePoint3D, RoughLocationSimple3D) {
  auto geom = create_two_adjacent_elements_3d_2x2x2();

  // Test point close to (0.1, 0.1, 0.1) - should find element 0
  specfem::point::global_coordinates<specfem::dimension::type::dim3>
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
  const Kokkos::View<
      specfem::point::global_coordinates<specfem::dimension::type::dim3> *,
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
  specfem::point::global_coordinates<specfem::dimension::type::dim3> target = {
    0.5, 0.5, 0.5
  };

  type_real xi_initial = 0.1;
  type_real eta_initial = 0.1;
  type_real gamma_initial = 0.1;

  auto [xi_final, eta_final, gamma_final] =
      specfem::algorithms::locate_point_impl::get_local_coordinates(
          target, coorg, xi_initial, eta_initial, gamma_initial);

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
          target, coorg, 0.0, 0.0, 0.0);

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
          target, coorg, 0.0, 0.0, 0.0);

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
  specfem::point::global_coordinates<specfem::dimension::type::dim3> target = {
    0.5, 0.5, 0.5
  };

  auto result = specfem::algorithms::locate_point_impl::locate_point_core(
      target, geom.global_coords, geom.index_mapping, geom.control_nodes,
      geom.ngnod, geom.ngllx);

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
      geom.ngnod, geom.ngllx);

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
  specfem::point::global_coordinates<specfem::dimension::type::dim3> target = {
    0.5, 0.5, 0.5
  };

  auto result = specfem::algorithms::locate_point_impl::locate_point_core(
      target, geom.global_coords, geom.index_mapping, geom.control_nodes,
      geom.ngnod, geom.ngllx);

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
  specfem::point::global_coordinates<specfem::dimension::type::dim3> target = {
    0.5, 0.5, 0.5
  };

  auto result = specfem::algorithms::locate_point_impl::locate_point_core(
      target, geom.global_coords, geom.index_mapping, geom.control_nodes,
      geom.ngnod, geom.ngllx);

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
      geom.ngnod, geom.ngllx);

  EXPECT_EQ(result.ispec, 0);
  EXPECT_TRUE(is_close(result.xi, type_real{ -0.5 }))
      << expected_got(-0.5, result.xi);
  EXPECT_TRUE(is_close(result.eta, type_real{ -0.5 }))
      << expected_got(-0.5, result.eta);
  EXPECT_TRUE(is_close(result.gamma, type_real{ -0.5 }))
      << expected_got(-0.5, result.gamma);
}

int main(int argc, char *argv[]) {
  ::testing::InitGoogleTest(&argc, argv);
  ::testing::AddGlobalTestEnvironment(new SPECFEMEnvironment);
  return RUN_ALL_TESTS();
}
