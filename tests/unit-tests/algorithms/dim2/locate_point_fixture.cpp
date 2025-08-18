#include "../../test_macros.hpp"
#include "Kokkos_Environment.hpp"
#include "MPI_environment.hpp"
#include "algorithms/locate_point.hpp"
#include "algorithms/locate_point_impl.hpp"
#include "kokkos_abstractions.h"
#include "parallel_configuration/chunk_config.hpp"
#include "specfem/assembly/mesh/dim2/impl/utilities.hpp"
#include "specfem/point.hpp"
#include "utilities/utilities.hpp"
#include <Kokkos_Core.hpp>
#include <gtest/gtest.h>

using specfem::utilities::is_close;

// Use tested mesh utilities for coordinate generation
using namespace specfem::assembly::mesh_impl::dim2;

// Test fixture for 2D locate_point algorithms using tested mesh utilities
class LocatePoint2D : public ::testing::Test {
protected:
  struct ElementGeometry {
    specfem::kokkos::HostView4d<type_real> global_coords;
    Kokkos::View<int ***, Kokkos::LayoutLeft, Kokkos::HostSpace> index_mapping;
    Kokkos::View<type_real ***, Kokkos::LayoutLeft, Kokkos::HostSpace>
        control_nodes;
    int nspec;
    int ngllx;
    int ngllz;
    int ngnod;
    int nglob;
    type_real xmin, xmax, zmin, zmax;
  };

  void SetUp() override {
    // Initialize Kokkos if not already done
    if (!Kokkos::is_initialized()) {
      Kokkos::initialize();
    }
  }

  void TearDown() override {
    // Kokkos cleanup handled by test environment
  }

  // Helper to create unit square coordinates programmatically
  std::vector<std::vector<std::pair<double, double> > >
  create_unit_square(int ngll, double xmin = -1.0, double xmax = 1.0,
                     double zmin = -1.0, double zmax = 1.0) {
    std::vector<std::pair<double, double> > coords;
    for (int iz = 0; iz < ngll; iz++) {
      for (int ix = 0; ix < ngll; ix++) {
        double x = xmin + (xmax - xmin) * ix / (ngll - 1);
        double z = zmin + (zmax - zmin) * iz / (ngll - 1);
        coords.push_back({ x, z });
      }
    }
    return { coords };
  }

  // Helper to create control nodes for 4-node or 9-node elements
  std::vector<std::vector<std::pair<double, double> > >
  create_control_nodes(int ngnod, double xmin = 0.0, double xmax = 1.0,
                       double zmin = 0.0, double zmax = 1.0) {
    std::vector<std::pair<double, double> > nodes;

    if (ngnod == 4) {
      // 4-node quadrilateral: corners only
      nodes = {
        { xmin, zmin }, // Bottom-left
        { xmax, zmin }, // Bottom-right
        { xmax, zmax }, // Top-right
        { xmin, zmax }  // Top-left
      };
    } else if (ngnod == 9) {
      // 9-node quadrilateral: corners + edge midpoints + center
      double xmid = (xmin + xmax) / 2.0;
      double zmid = (zmin + zmax) / 2.0;
      nodes = {
        { xmin, zmin }, // 0: Bottom-left corner
        { xmax, zmin }, // 1: Bottom-right corner
        { xmax, zmax }, // 2: Top-right corner
        { xmin, zmax }, // 3: Top-left corner
        { xmid, zmin }, // 4: Bottom edge midpoint
        { xmax, zmid }, // 5: Right edge midpoint
        { xmid, zmax }, // 6: Top edge midpoint
        { xmin, zmid }, // 7: Left edge midpoint
        { xmid, zmid }  // 8: Center
      };
    } else {
      throw std::runtime_error("Unsupported ngnod: " + std::to_string(ngnod));
    }

    return { nodes };
  }

  // Helper to create coordinate array from element corners and ngll points
  specfem::kokkos::HostView4d<double> create_coordinate_array(
      const std::vector<std::vector<std::pair<double, double> > >
          &element_coords) {
    int nspec = element_coords.size();
    int ngll = std::sqrt(element_coords[0].size());

    specfem::kokkos::HostView4d<double> coords("coords", nspec, ngll, ngll, 2);

    for (int ispec = 0; ispec < nspec; ispec++) {
      int idx = 0;
      for (int iz = 0; iz < ngll; iz++) {
        for (int ix = 0; ix < ngll; ix++) {
          coords(ispec, iz, ix, 0) = element_coords[ispec][idx].first;  // x
          coords(ispec, iz, ix, 1) = element_coords[ispec][idx].second; // z
          idx++;
        }
      }
    }
    return coords;
  }

  // Create element geometry using tested utility functions
  ElementGeometry create_element_geometry(
      const std::vector<std::vector<std::pair<double, double> > >
          &element_coords,
      const std::vector<std::vector<std::pair<double, double> > >
          &control_coords) {

    ElementGeometry geom;
    geom.nspec = element_coords.size();
    geom.ngllx = std::sqrt(element_coords[0].size());
    geom.ngllz = geom.ngllx;
    geom.ngnod = control_coords[0].size();
    int ngllxz = geom.ngllx * geom.ngllz;

    // Create coordinate array
    auto coords_double = create_coordinate_array(element_coords);

    // Use tested utility functions to replicate assign_numbering logic
    auto points = flatten_coordinates(coords_double);
    auto sorted_points = points;
    sort_points_spatially(sorted_points);
    type_real tolerance =
        compute_spatial_tolerance(sorted_points, geom.nspec, ngllxz);
    int nglob = assign_global_numbering(sorted_points, tolerance);
    auto reordered_points = reorder_to_original_layout(sorted_points);
    auto bbox = compute_bounding_box(reordered_points);

    // Use the create_coordinate_arrays function to get proper index_mapping and
    // coordinates
    auto [index_mapping_host, coord_host, nglob_actual] =
        create_coordinate_arrays(reordered_points, geom.nspec, geom.ngllx,
                                 nglob);

    geom.nglob = nglob_actual;

    // Convert coordinates to the format expected by locate_point (different
    // layout)
    geom.global_coords = specfem::kokkos::HostView4d<type_real>(
        "global_coords", 2, geom.nspec, geom.ngllz, geom.ngllx);
    geom.index_mapping =
        Kokkos::View<int ***, Kokkos::LayoutLeft, Kokkos::HostSpace>(
            "index_mapping", geom.nspec, geom.ngllz, geom.ngllx);

    // Copy data with correct layout transformation
    for (int ispec = 0; ispec < geom.nspec; ispec++) {
      for (int iz = 0; iz < geom.ngllz; iz++) {
        for (int ix = 0; ix < geom.ngllx; ix++) {
          geom.global_coords(0, ispec, iz, ix) = coord_host(0, ispec, iz, ix);
          geom.global_coords(1, ispec, iz, ix) = coord_host(1, ispec, iz, ix);
          geom.index_mapping(ispec, iz, ix) = index_mapping_host(ispec, iz, ix);
        }
      }
    }

    // Set up control nodes
    geom.control_nodes =
        Kokkos::View<type_real ***, Kokkos::LayoutLeft, Kokkos::HostSpace>(
            "control_nodes", 2, geom.nspec, geom.ngnod);
    for (int ispec = 0; ispec < geom.nspec; ispec++) {
      for (int inode = 0; inode < geom.ngnod; inode++) {
        geom.control_nodes(0, ispec, inode) =
            control_coords[ispec][inode].first;
        geom.control_nodes(1, ispec, inode) =
            control_coords[ispec][inode].second;
      }
    }

    // Set bounding box
    geom.xmin = bbox.xmin;
    geom.xmax = bbox.xmax;
    geom.zmin = bbox.zmin;
    geom.zmax = bbox.zmax;

    return geom;
  }

  // Create a single unit square element [0,1] x [0,1] with 2x2 GLL points
  ElementGeometry create_single_unit_square_2x2() {
    auto element_coords = create_unit_square(2, 0.0, 1.0, 0.0, 1.0);
    auto control_coords = create_control_nodes(4, 0.0, 1.0, 0.0, 1.0);
    return create_element_geometry(element_coords, control_coords);
  }

  // Create two adjacent elements [0,0.5]x[0,0.5] and [0.5,1]x[0,0.5] with 2x2
  // GLL points
  ElementGeometry create_two_adjacent_elements_2x2() {
    std::vector<std::vector<std::pair<double, double> > > element_coords = {
      create_unit_square(2, 0.0, 0.5, 0.0, 0.5)[0], // Left element
      create_unit_square(2, 0.5, 1.0, 0.0, 0.5)[0] // Right element (shares edge
                                                   // at x=0.5)
    };
    auto control_coords = std::vector<std::vector<std::pair<double, double> > >{
      create_control_nodes(4, 0.0, 0.5, 0.0, 0.5)[0], // Left element
      create_control_nodes(4, 0.5, 1.0, 0.0, 0.5)[0]  // Right element
    };
    return create_element_geometry(element_coords, control_coords);
  }

  // Create 2x2 grid of elements with 2x2 GLL points each
  ElementGeometry create_2x2_grid_elements_2x2() {
    std::vector<std::vector<std::pair<double, double> > > element_coords = {
      create_unit_square(2, 0.0, 1.0, 0.0, 1.0)[0], // Element 0: bottom-left
      create_unit_square(2, 1.0, 2.0, 0.0, 1.0)[0], // Element 1: bottom-right
      create_unit_square(2, 0.0, 1.0, 1.0, 2.0)[0], // Element 2: top-left
      create_unit_square(2, 1.0, 2.0, 1.0, 2.0)[0]  // Element 3: top-right
    };
    auto control_coords = std::vector<std::vector<std::pair<double, double> > >{
      create_control_nodes(4, 0.0, 1.0, 0.0, 1.0)[0], // Element 0
      create_control_nodes(4, 1.0, 2.0, 0.0, 1.0)[0], // Element 1
      create_control_nodes(4, 0.0, 1.0, 1.0, 2.0)[0], // Element 2
      create_control_nodes(4, 1.0, 2.0, 1.0, 2.0)[0]  // Element 3
    };
    return create_element_geometry(element_coords, control_coords);
  }

  // Create a single unit square element [0,1] x [0,1] with 5x5 GLL points
  ElementGeometry create_single_unit_square_5x5() {
    auto element_coords = create_unit_square(5, 0.0, 1.0, 0.0, 1.0);
    auto control_coords = create_control_nodes(4, 0.0, 1.0, 0.0, 1.0);
    return create_element_geometry(element_coords, control_coords);
  }

  // Create a single unit square element [0,1] x [0,1] with 2x2 GLL points and 9
  // control nodes
  ElementGeometry create_single_unit_square_2x2_9node() {
    auto element_coords = create_unit_square(2, 0.0, 1.0, 0.0, 1.0);
    auto control_coords = create_control_nodes(9, 0.0, 1.0, 0.0, 1.0);
    return create_element_geometry(element_coords, control_coords);
  }

  // Create a single unit square element [0,1] x [0,1] with 5x5 GLL points and 9
  // control nodes
  ElementGeometry create_single_unit_square_5x5_9node() {
    auto element_coords = create_unit_square(5, 0.0, 1.0, 0.0, 1.0);
    auto control_coords = create_control_nodes(9, 0.0, 1.0, 0.0, 1.0);
    return create_element_geometry(element_coords, control_coords);
  }

  // Create two adjacent elements with 5x5 GLL points
  ElementGeometry create_two_adjacent_elements_5x5() {
    std::vector<std::vector<std::pair<double, double> > > element_coords = {
      create_unit_square(5, 0.0, 0.5, 0.0, 0.5)[0], // Left element
      create_unit_square(5, 0.5, 1.0, 0.0, 0.5)[0] // Right element (shares edge
                                                   // at x=0.5)
    };
    auto control_coords = std::vector<std::vector<std::pair<double, double> > >{
      create_control_nodes(4, 0.0, 0.5, 0.0, 0.5)[0], // Left element
      create_control_nodes(4, 0.5, 1.0, 0.0, 0.5)[0]  // Right element
    };
    return create_element_geometry(element_coords, control_coords);
  }
};

// Test locate_point core functionality with single unit square
TEST_F(LocatePoint2D, CoreUnitSquare) {
  auto geom = create_single_unit_square_2x2();

  // Test point at center of unit square (0.5, 0.5)
  // Should map to local coordinates (0, 0)
  specfem::point::global_coordinates<specfem::dimension::type::dim2> target = {
    0.5, 0.5
  };

  auto result = specfem::algorithms::locate_point_impl::locate_point_core(
      target, geom.global_coords, geom.index_mapping, geom.control_nodes,
      geom.ngnod, geom.ngllx);

  // Should find element 0 with local coordinates near (0, 0) for center point
  EXPECT_EQ(result.ispec, 0);
  EXPECT_TRUE(is_close(result.xi, type_real{ 0.0 }))
      << expected_got(0.0, result.xi);
  EXPECT_TRUE(is_close(result.gamma, type_real{ 0.0 }))
      << expected_got(0.0, result.gamma);

  // Test corner point (0, 0) should map to (-1, -1)
  target = { 0.0, 0.0 };
  result = specfem::algorithms::locate_point_impl::locate_point_core(
      target, geom.global_coords, geom.index_mapping, geom.control_nodes,
      geom.ngnod, geom.ngllx);

  EXPECT_EQ(result.ispec, 0);
  EXPECT_TRUE(is_close(result.xi, type_real{ -1.0 }))
      << expected_got(-1.0, result.xi);
  EXPECT_TRUE(is_close(result.gamma, type_real{ -1.0 }))
      << expected_got(-1.0, result.gamma);

  // Test corner point (1, 1) should map to (1, 1)
  target = { 1.0, 1.0 };
  result = specfem::algorithms::locate_point_impl::locate_point_core(
      target, geom.global_coords, geom.index_mapping, geom.control_nodes,
      geom.ngnod, geom.ngllx);

  EXPECT_EQ(result.ispec, 0);
  EXPECT_TRUE(is_close(result.xi, type_real{ 1.0 }))
      << expected_got(1.0, result.xi);
  EXPECT_TRUE(is_close(result.gamma, type_real{ 1.0 }))
      << expected_got(1.0, result.gamma);
}

// Test locate_point with two adjacent elements
TEST_F(LocatePoint2D, LocatePoint2DCoreTwoAdjacentElements) {
  auto geom = create_two_adjacent_elements_2x2();

  // Test point in left element (0.25, 0.25)
  specfem::point::global_coordinates<specfem::dimension::type::dim2> target = {
    0.25, 0.25
  };

  auto result = specfem::algorithms::locate_point_impl::locate_point_core(
      target, geom.global_coords, geom.index_mapping, geom.control_nodes,
      geom.ngnod, geom.ngllx);

  EXPECT_EQ(result.ispec, 0);
  EXPECT_TRUE(is_close(result.xi, type_real{ 0.0 }))
      << expected_got(0.0, result.xi);
  EXPECT_TRUE(is_close(result.gamma, type_real{ 0.0 }))
      << expected_got(0.0, result.gamma);

  // Test point in right element (0.75, 0.25)
  target = { 0.75, 0.25 };
  result = specfem::algorithms::locate_point_impl::locate_point_core(
      target, geom.global_coords, geom.index_mapping, geom.control_nodes,
      geom.ngnod, geom.ngllx);

  EXPECT_EQ(result.ispec, 1);
  EXPECT_TRUE(is_close(result.xi, type_real{ 0.0 }))
      << expected_got(0.0, result.xi);
  EXPECT_TRUE(is_close(result.gamma, type_real{ 0.0 }))
      << expected_got(0.0, result.gamma);

  // Test shared edge point (0.5, 0.25) - should find one of the elements
  target = { 0.5, 0.25 };
  result = specfem::algorithms::locate_point_impl::locate_point_core(
      target, geom.global_coords, geom.index_mapping, geom.control_nodes,
      geom.ngnod, geom.ngllx);

  EXPECT_TRUE(result.ispec == 0 ||
              result.ispec == 1); // Either element is valid
  EXPECT_TRUE(std::abs(std::abs(result.xi) - 1.0) < 1e-6); // Should be at edge
                                                           // (±1)
}

// Test locate_point with 2x2 grid of elements
TEST_F(LocatePoint2D, Core2x2Grid) {
  auto geom = create_2x2_grid_elements_2x2();

  // Test points in each element

  // Element 0: bottom-left (0.5, 0.5)
  specfem::point::global_coordinates<specfem::dimension::type::dim2> target = {
    0.5, 0.5
  };
  auto result = specfem::algorithms::locate_point_impl::locate_point_core(
      target, geom.global_coords, geom.index_mapping, geom.control_nodes,
      geom.ngnod, geom.ngllx);
  EXPECT_EQ(result.ispec, 0);

  // Element 1: bottom-right (1.5, 0.5)
  target = { 1.5, 0.5 };
  result = specfem::algorithms::locate_point_impl::locate_point_core(
      target, geom.global_coords, geom.index_mapping, geom.control_nodes,
      geom.ngnod, geom.ngllx);
  EXPECT_EQ(result.ispec, 1);

  // Element 2: top-left (0.5, 1.5)
  target = { 0.5, 1.5 };
  result = specfem::algorithms::locate_point_impl::locate_point_core(
      target, geom.global_coords, geom.index_mapping, geom.control_nodes,
      geom.ngnod, geom.ngllx);
  EXPECT_EQ(result.ispec, 2);

  // Element 3: top-right (1.5, 1.5)
  target = { 1.5, 1.5 };
  result = specfem::algorithms::locate_point_impl::locate_point_core(
      target, geom.global_coords, geom.index_mapping, geom.control_nodes,
      geom.ngnod, geom.ngllx);
  EXPECT_EQ(result.ispec, 3);

  // Test shared corner point (1, 1) - should find one of the elements that
  // share this corner
  target = { 1.0, 1.0 };
  result = specfem::algorithms::locate_point_impl::locate_point_core(
      target, geom.global_coords, geom.index_mapping, geom.control_nodes,
      geom.ngnod, geom.ngllx);

  EXPECT_TRUE(result.ispec >= 0 && result.ispec <= 3); // Should find one of the
                                                       // four elements
  // At corner, should have local coordinates (±1, ±1) depending on element
  EXPECT_TRUE(std::abs(std::abs(result.xi) - 1.0) < 1e-6);
  EXPECT_TRUE(std::abs(std::abs(result.gamma) - 1.0) < 1e-6);
}

// Test error case: point outside mesh
TEST_F(LocatePoint2D, CoreOutsideMesh) {
  auto geom = create_single_unit_square_2x2();

  // Point outside mesh should throw exception
  specfem::point::global_coordinates<specfem::dimension::type::dim2> target = {
    2.0, 2.0
  };

  EXPECT_THROW(specfem::algorithms::locate_point_impl::locate_point_core(
                   target, geom.global_coords, geom.index_mapping,
                   geom.control_nodes, geom.ngnod, geom.ngllx),
               std::runtime_error);
}

// Helper function tests - testing individual components of locate_point_core

// Test rough_location helper function
TEST_F(LocatePoint2D, RoughLocationSimple) {
  auto geom = create_two_adjacent_elements_2x2();

  // Test point close to (0.1, 0.1) - should find element 0, point (0,0)
  specfem::point::global_coordinates<specfem::dimension::type::dim2>
      test_point = { 0.1, 0.1 };

  auto [ix, iz, ispec] = specfem::algorithms::locate_point_impl::rough_location(
      test_point, geom.global_coords);

  EXPECT_EQ(ispec, 0);
  EXPECT_EQ(ix, 0);
  EXPECT_EQ(iz, 0);

  // Test point close to (0.75, 0.25) - should find element 1 (right element)
  // Two adjacent elements: left [0,0.5]x[0,0.5] and right [0.5,1]x[0,0.5]
  test_point = { 0.75, 0.25 };
  std::tie(ix, iz, ispec) =
      specfem::algorithms::locate_point_impl::rough_location(
          test_point, geom.global_coords);

  // Should find element 1 (right element) - but rough_location may find either
  // element since they share the edge at x=0.5. The original test also had this
  // ambiguity.
  EXPECT_TRUE(ispec == 0 || ispec == 1);
}

// Test get_best_candidates helper function
TEST_F(LocatePoint2D, GetBestCandidatesSimple) {
  auto geom = create_two_adjacent_elements_2x2();

  // Test with element 0 - should return both elements 0 and 1 since they share
  // nodes
  auto candidates = specfem::algorithms::locate_point_impl::get_best_candidates(
      0, geom.index_mapping);

  // Should return both elements since they share edge nodes
  EXPECT_EQ(candidates.size(), 2);
  EXPECT_EQ(candidates[0], 0); // Initial guess element
  EXPECT_EQ(candidates[1], 1); // Neighboring element
}

// Test get_local_coordinates helper function
TEST_F(LocatePoint2D, GetLocalCoordinatesUnitSquare) {
  auto geom = create_single_unit_square_2x2();

  // Create control node coordinates view for the single element
  const int ngnod = 4;
  const Kokkos::View<
      specfem::point::global_coordinates<specfem::dimension::type::dim2> *,
      Kokkos::HostSpace>
      coorg("coorg", ngnod);

  // Unit square control nodes [0,1] x [0,1]
  coorg(0) = { 0.0, 0.0 }; // Bottom-left
  coorg(1) = { 1.0, 0.0 }; // Bottom-right
  coorg(2) = { 1.0, 1.0 }; // Top-right
  coorg(3) = { 0.0, 1.0 }; // Top-left

  // Test point at center of unit square (0.5, 0.5)
  // Should map to local coordinates (0, 0)
  specfem::point::global_coordinates<specfem::dimension::type::dim2> target = {
    0.5, 0.5
  };

  type_real xi_initial = 0.1; // Start with slight offset
  type_real gamma_initial = 0.1;

  auto [xi_final, gamma_final] =
      specfem::algorithms::locate_point_impl::get_local_coordinates(
          target, coorg, xi_initial, gamma_initial);

  // For a unit square, center point (0.5, 0.5) should map to (0, 0) in
  // reference coords
  EXPECT_TRUE(is_close(xi_final, type_real{ 0.0 }))
      << expected_got(0.0, xi_final);
  EXPECT_TRUE(is_close(gamma_final, type_real{ 0.0 }))
      << expected_got(0.0, gamma_final);

  // Test corner point (0, 0) should map to (-1, -1)
  target = { 0.0, 0.0 };
  std::tie(xi_final, gamma_final) =
      specfem::algorithms::locate_point_impl::get_local_coordinates(
          target, coorg, 0.0, 0.0);

  EXPECT_TRUE(is_close(xi_final, type_real{ -1.0 }))
      << expected_got(-1.0, xi_final);
  EXPECT_TRUE(is_close(gamma_final, type_real{ -1.0 }))
      << expected_got(-1.0, gamma_final);

  // Test corner point (1, 1) should map to (1, 1)
  target = { 1.0, 1.0 };
  std::tie(xi_final, gamma_final) =
      specfem::algorithms::locate_point_impl::get_local_coordinates(
          target, coorg, 0.0, 0.0);

  EXPECT_TRUE(is_close(xi_final, type_real{ 1.0 }))
      << expected_got(1.0, xi_final);
  EXPECT_TRUE(is_close(gamma_final, type_real{ 1.0 }))
      << expected_got(1.0, gamma_final);
}

// Test locate_point with 5x5 GLL points (realistic spectral element resolution)
TEST_F(LocatePoint2D, Core5x5UnitSquare) {
  auto geom = create_single_unit_square_5x5();

  // Test point at center of unit square (0.5, 0.5)
  // Should map to local coordinates (0, 0)
  specfem::point::global_coordinates<specfem::dimension::type::dim2> target = {
    0.5, 0.5
  };

  auto result = specfem::algorithms::locate_point_impl::locate_point_core(
      target, geom.global_coords, geom.index_mapping, geom.control_nodes,
      geom.ngnod, geom.ngllx);

  // Should find element 0 with local coordinates near (0, 0) for center point
  EXPECT_EQ(result.ispec, 0);
  EXPECT_TRUE(is_close(result.xi, type_real{ 0.0 }))
      << expected_got(0.0, result.xi);
  EXPECT_TRUE(is_close(result.gamma, type_real{ 0.0 }))
      << expected_got(0.0, result.gamma);

  // Test corner point (0, 0) should map to (-1, -1)
  target = { 0.0, 0.0 };
  result = specfem::algorithms::locate_point_impl::locate_point_core(
      target, geom.global_coords, geom.index_mapping, geom.control_nodes,
      geom.ngnod, geom.ngllx);

  EXPECT_EQ(result.ispec, 0);
  EXPECT_TRUE(is_close(result.xi, type_real{ -1.0 }))
      << expected_got(-1.0, result.xi);
  EXPECT_TRUE(is_close(result.gamma, type_real{ -1.0 }))
      << expected_got(-1.0, result.gamma);
}

// Test locate_point with 9-node control elements
TEST_F(LocatePoint2D, Core9NodeElement) {
  auto geom = create_single_unit_square_2x2_9node();

  // Test point at center of unit square (0.5, 0.5)
  // Should map to local coordinates (0, 0)
  specfem::point::global_coordinates<specfem::dimension::type::dim2> target = {
    0.5, 0.5
  };

  auto result = specfem::algorithms::locate_point_impl::locate_point_core(
      target, geom.global_coords, geom.index_mapping, geom.control_nodes,
      geom.ngnod, geom.ngllx);

  // Should find element 0 with local coordinates near (0, 0) for center point
  EXPECT_EQ(result.ispec, 0);
  EXPECT_TRUE(is_close(result.xi, type_real{ 0.0 }))
      << expected_got(0.0, result.xi);
  EXPECT_TRUE(is_close(result.gamma, type_real{ 0.0 }))
      << expected_got(0.0, result.gamma);
}

// Test locate_point with 5x5 GLL points AND 9-node control elements (most
// realistic case)
TEST_F(LocatePoint2D, Core5x5With9Node) {
  auto geom = create_single_unit_square_5x5_9node();

  // Test point at center of unit square (0.5, 0.5)
  // Should map to local coordinates (0, 0)
  specfem::point::global_coordinates<specfem::dimension::type::dim2> target = {
    0.5, 0.5
  };

  auto result = specfem::algorithms::locate_point_impl::locate_point_core(
      target, geom.global_coords, geom.index_mapping, geom.control_nodes,
      geom.ngnod, geom.ngllx);

  // Should find element 0 with local coordinates near (0, 0) for center point
  EXPECT_EQ(result.ispec, 0);
  EXPECT_TRUE(is_close(result.xi, type_real{ 0.0 }))
      << expected_got(0.0, result.xi);
  EXPECT_TRUE(is_close(result.gamma, type_real{ 0.0 }))
      << expected_got(0.0, result.gamma);

  // Test quarter point (0.25, 0.25) should map to (-0.5, -0.5)
  target = { 0.25, 0.25 };
  result = specfem::algorithms::locate_point_impl::locate_point_core(
      target, geom.global_coords, geom.index_mapping, geom.control_nodes,
      geom.ngnod, geom.ngllx);

  EXPECT_EQ(result.ispec, 0);
  EXPECT_TRUE(is_close(result.xi, type_real{ -0.5 }))
      << expected_got(-0.5, result.xi);
  EXPECT_TRUE(is_close(result.gamma, type_real{ -0.5 }))
      << expected_got(-0.5, result.gamma);
}

// Test locate_point with two adjacent 5x5 elements
TEST_F(LocatePoint2D, LocatePoint2DCoreTwoAdjacent5x5Elements) {
  auto geom = create_two_adjacent_elements_5x5();

  // Test point in left element (0.25, 0.25)
  specfem::point::global_coordinates<specfem::dimension::type::dim2> target = {
    0.25, 0.25
  };

  auto result = specfem::algorithms::locate_point_impl::locate_point_core(
      target, geom.global_coords, geom.index_mapping, geom.control_nodes,
      geom.ngnod, geom.ngllx);

  EXPECT_EQ(result.ispec, 0);
  EXPECT_TRUE(is_close(result.xi, type_real{ 0.0 }))
      << expected_got(0.0, result.xi);
  EXPECT_TRUE(is_close(result.gamma, type_real{ 0.0 }))
      << expected_got(0.0, result.gamma);

  // Test point in right element (0.75, 0.25)
  target = { 0.75, 0.25 };
  result = specfem::algorithms::locate_point_impl::locate_point_core(
      target, geom.global_coords, geom.index_mapping, geom.control_nodes,
      geom.ngnod, geom.ngllx);

  EXPECT_EQ(result.ispec, 1);
  EXPECT_TRUE(is_close(result.xi, type_real{ 0.0 }))
      << expected_got(0.0, result.xi);
  EXPECT_TRUE(is_close(result.gamma, type_real{ 0.0 }))
      << expected_got(0.0, result.gamma);
}

int main(int argc, char *argv[]) {
  ::testing::InitGoogleTest(&argc, argv);
  ::testing::AddGlobalTestEnvironment(new MPIEnvironment);
  ::testing::AddGlobalTestEnvironment(new KokkosEnvironment);
  return RUN_ALL_TESTS();
}
