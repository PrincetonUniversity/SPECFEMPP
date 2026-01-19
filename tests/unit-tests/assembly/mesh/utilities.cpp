#include "specfem/assembly/mesh/dim2/impl/utilities.hpp"
#include "kokkos_abstractions.h"
#include "specfem/parallel_configuration.hpp"
#include "specfem/utilities.hpp"
#include "test_macros.hpp"
#include <gtest/gtest.h>
#include <set>
#include <vector>

using namespace specfem::assembly::mesh_impl::dim2;

class MeshUtilitiesTest : public ::testing::Test {
protected:
  void SetUp() override {
    // Initialize Kokkos if not already done
    if (!Kokkos::is_initialized()) {
      Kokkos::initialize();
    }
  }

  void TearDown() override {
    // Kokkos cleanup handled by test environment
  }

  // Helper to create 4D coordinate array
  specfem::kokkos::HostView4d<double>
  create_coordinates(const std::vector<std::vector<std::pair<double, double> > >
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
};

class MeshNumberingTests : public MeshUtilitiesTest {
protected:
  // Helper to create unit square coordinates
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

  // Helper to create two adjacent squares sharing an edge
  std::vector<std::vector<std::pair<double, double> > >
  create_two_adjacent_squares_2x2() {
    auto left_square =
        create_unit_square(2, -2.0, 0.0, -1.0, 1.0)[0]; // Left element
    auto right_square = create_unit_square(
        2, 0.0, 2.0, -1.0, 1.0)[0]; // Right element (shares x=0 edge)
    return { left_square, right_square };
  }

  std::vector<std::vector<std::pair<double, double> > >
  create_two_adjacent_squares_5x5() {
    auto left_square =
        create_unit_square(5, -2.0, 2.0, -1.0, 1.0)[0]; // Left element
    auto right_square = create_unit_square(
        5, 2.0, 6.0, -1.0, 1.0)[0]; // Right element (shares x=2 edge)
    return { left_square, right_square };
  }

  // Pre-built geometries for convenience
  std::vector<std::vector<std::pair<double, double> > > unit_square_2x2 =
      create_unit_square(2);
  std::vector<std::vector<std::pair<double, double> > > unit_square_5x5 =
      create_unit_square(5);
  std::vector<std::vector<std::pair<double, double> > >
      two_adjacent_squares_2x2 = create_two_adjacent_squares_2x2();
  std::vector<std::vector<std::pair<double, double> > >
      two_adjacent_squares_5x5 = create_two_adjacent_squares_5x5();

  // Helper to create a 2x2 grid of elements
  std::vector<std::vector<std::pair<double, double> > >
  create_2x2_grid(int ngll) {
    std::vector<std::vector<std::pair<double, double> > > grid;
    // Element layout: [2][3]
    //                 [0][1]
    grid.push_back(create_unit_square(ngll, 0.0, 1.0, 0.0,
                                      1.0)[0]); // Element 0: bottom-left
    grid.push_back(create_unit_square(ngll, 1.0, 2.0, 0.0,
                                      1.0)[0]); // Element 1: bottom-right
    grid.push_back(
        create_unit_square(ngll, 0.0, 1.0, 1.0, 2.0)[0]); // Element 2: top-left
    grid.push_back(create_unit_square(ngll, 1.0, 2.0, 1.0,
                                      2.0)[0]); // Element 3: top-right
    return grid;
  }

  // Sheared element coordinates (manually specified since it's non-regular)
  std::vector<std::vector<std::pair<double, double> > > sheared_element_2x2 = {
    {
        { -1.0, -1.0 },
        { 2.0, -1.0 }, // iz=0: ix=0,1
        { 0.0, 2.0 },
        { 3.0, 2.0 } // iz=1: ix=0,1
    }
  };

  // Pre-built grid geometry
  std::vector<std::vector<std::pair<double, double> > > grid_2x2_elements_2x2 =
      create_2x2_grid(2);
};

// Test flatten_coordinates function
TEST_F(MeshNumberingTests, FlattenCoordinatesUnitSquare) {
  auto coords = create_coordinates(unit_square_2x2);
  auto flattened = flatten_coordinates(coords);

  ASSERT_EQ(flattened.size(), 4); // 1 element * 2*2 points

  // Check coordinates are preserved
  EXPECT_DOUBLE_EQ(flattened[0].x, -1.0);
  EXPECT_DOUBLE_EQ(flattened[0].z, -1.0);
  EXPECT_EQ(flattened[0].iloc, 0);

  EXPECT_DOUBLE_EQ(flattened[3].x, 1.0);
  EXPECT_DOUBLE_EQ(flattened[3].z, 1.0);
  EXPECT_EQ(flattened[3].iloc, 3);
}

TEST_F(MeshNumberingTests, FlattenCoordinatesMultipleElements) {
  auto coords = create_coordinates(two_adjacent_squares_2x2);
  auto flattened = flatten_coordinates(coords);

  ASSERT_EQ(flattened.size(), 8); // 2 elements * 2*2 points

  // Check that we have coordinates from both elements
  // The exact ordering depends on chunked iteration but coordinates should be
  // preserved
  bool found_element1_corner = false;
  for (const auto &p : flattened) {
    if (p.x == 0.0 && p.z == -1.0) {
      found_element1_corner = true;
      break;
    }
  }
  EXPECT_TRUE(found_element1_corner);
}

// Test flatten_coordinates with 5x5 GLL points
TEST_F(MeshNumberingTests, FlattenCoordinatesUnitSquare5x5) {
  auto coords = create_coordinates(unit_square_5x5);
  auto flattened = flatten_coordinates(coords);

  ASSERT_EQ(flattened.size(), 25); // 1 element * 5*5 points

  // Check corners are preserved
  EXPECT_DOUBLE_EQ(flattened[0].x, -1.0); // First point
  EXPECT_DOUBLE_EQ(flattened[0].z, -1.0);
  EXPECT_EQ(flattened[0].iloc, 0);

  EXPECT_DOUBLE_EQ(flattened[24].x, 1.0); // Last point
  EXPECT_DOUBLE_EQ(flattened[24].z, 1.0);
  EXPECT_EQ(flattened[24].iloc, 24);

  // Check center point
  EXPECT_DOUBLE_EQ(flattened[12].x, 0.0); // Center point (iz=2, ix=2)
  EXPECT_DOUBLE_EQ(flattened[12].z, 0.0);
  EXPECT_EQ(flattened[12].iloc, 12);
}

TEST_F(MeshNumberingTests, FlattenCoordinatesMultipleElements5x5) {
  auto coords = create_coordinates(two_adjacent_squares_5x5);
  auto flattened = flatten_coordinates(coords);

  ASSERT_EQ(flattened.size(), 50); // 2 elements * 5*5 points

  // Check that we have coordinates from both elements
  bool found_shared_edge = false;
  int shared_count = 0;
  for (const auto &p : flattened) {
    if (p.x == 2.0) { // Shared edge at x=2
      shared_count++;
      found_shared_edge = true;
    }
  }
  EXPECT_TRUE(found_shared_edge);
  EXPECT_EQ(shared_count, 10); // 5 points on each element share x=2
}

// Test spatial sorting
TEST_F(MeshNumberingTests, SortPointsSpatiallyUnitSquare) {
  auto coords = create_coordinates(unit_square_2x2);
  auto points = flatten_coordinates(coords);

  sort_points_spatially(points);

  // Should be sorted by x, then z
  // Expected order: (-1,-1), (-1,1), (1,-1), (1,1)
  EXPECT_DOUBLE_EQ(points[0].x, -1.0);
  EXPECT_DOUBLE_EQ(points[0].z, -1.0);

  EXPECT_DOUBLE_EQ(points[1].x, -1.0);
  EXPECT_DOUBLE_EQ(points[1].z, 1.0);

  EXPECT_DOUBLE_EQ(points[2].x, 1.0);
  EXPECT_DOUBLE_EQ(points[2].z, -1.0);

  EXPECT_DOUBLE_EQ(points[3].x, 1.0);
  EXPECT_DOUBLE_EQ(points[3].z, 1.0);
}

// Test tolerance calculation
TEST_F(MeshNumberingTests, ComputeSpatialToleranceUnitSquare) {
  auto coords = create_coordinates(unit_square_2x2);
  auto points = flatten_coordinates(coords);

  type_real tolerance = compute_spatial_tolerance(points, 1, 4);

  // For unit square: min dimension = 2.0, tolerance = 1e-6 * 2.0
  EXPECT_TRUE(specfem::utilities::is_close(tolerance, type_real(2e-6)))
      << expected_got(2e-6, tolerance);
}

TEST_F(MeshNumberingTests, ComputeSpatialToleranceSheared) {
  auto coords = create_coordinates(sheared_element_2x2);
  auto points = flatten_coordinates(coords);

  type_real tolerance = compute_spatial_tolerance(points, 1, 4);

  // For sheared element: x range = 4.0, z range = 3.0, min = 3.0
  EXPECT_TRUE(specfem::utilities::is_close(tolerance, type_real(3e-6)))
      << expected_got(3e-6, tolerance);
}

// Test global numbering assignment
TEST_F(MeshNumberingTests, AssignGlobalNumberingUnitSquare) {
  auto coords = create_coordinates(unit_square_2x2);
  auto points = flatten_coordinates(coords);
  sort_points_spatially(points);

  type_real tolerance = compute_spatial_tolerance(points, 1, 4);
  int nglob = assign_global_numbering(points, tolerance);

  // All points are distinct, should get 4 unique global numbers
  EXPECT_EQ(nglob, 4);

  // Check numbering is sequential
  EXPECT_EQ(points[0].iglob, 0);
  EXPECT_EQ(points[1].iglob, 1);
  EXPECT_EQ(points[2].iglob, 2);
  EXPECT_EQ(points[3].iglob, 3);
}

TEST_F(MeshNumberingTests, AssignGlobalNumberingSharedPoints) {
  auto coords = create_coordinates(two_adjacent_squares_2x2);
  auto points = flatten_coordinates(coords);
  sort_points_spatially(points);

  type_real tolerance = compute_spatial_tolerance(points, 2, 4);
  int nglob = assign_global_numbering(points, tolerance);

  // Elements share edge (2 points), so 8 - 2 = 6 unique points
  EXPECT_EQ(nglob, 6);

  // Find shared points (x=0.0, z=-1.0) and (x=0.0, z=1.0)
  int shared_count = 0;
  for (int i = 0; i < points.size(); i++) {
    for (int j = i + 1; j < points.size(); j++) {
      if (points[i].iglob == points[j].iglob) {
        shared_count++;
        // Verify they are actually the same coordinate
        EXPECT_DOUBLE_EQ(points[i].x, points[j].x);
        EXPECT_DOUBLE_EQ(points[i].z, points[j].z);
      }
    }
  }
  EXPECT_EQ(shared_count, 2); // Two shared points
}

// Critical test: Shared points with 5x5 GLL points
TEST_F(MeshNumberingTests, AssignGlobalNumberingSharedPoints5x5) {
  auto coords = create_coordinates(two_adjacent_squares_5x5);
  auto points = flatten_coordinates(coords);
  sort_points_spatially(points);

  type_real tolerance = compute_spatial_tolerance(points, 2, 25);
  int nglob = assign_global_numbering(points, tolerance);

  // Elements share edge (5 points along x=2), so 50 - 5 = 45 unique points
  EXPECT_EQ(nglob, 45);

  // Find shared points along x=2.0 edge
  int shared_pairs = 0;
  for (int i = 0; i < points.size(); i++) {
    for (int j = i + 1; j < points.size(); j++) {
      if (points[i].iglob == points[j].iglob) {
        shared_pairs++;
        // Verify they are actually the same coordinate
        EXPECT_TRUE(specfem::utilities::is_close(points[i].x, points[j].x))
            << expected_got(points[j].x, points[i].x);
        EXPECT_TRUE(specfem::utilities::is_close(points[i].z, points[j].z))
            << expected_got(points[j].z, points[i].z);
        // All shared points should be at x=2.0
        EXPECT_TRUE(specfem::utilities::is_close(points[i].x, type_real(2.0)))
            << expected_got(2.0, points[i].x);
      }
    }
  }
  EXPECT_EQ(shared_pairs, 5); // Five shared points along edge
}

TEST_F(MeshNumberingTests, AssignGlobalNumberingGrid2x2) {
  auto coords = create_coordinates(grid_2x2_elements_2x2);
  auto points = flatten_coordinates(coords);
  sort_points_spatially(points);

  type_real tolerance = compute_spatial_tolerance(points, 4, 4);
  int nglob = assign_global_numbering(points, tolerance);

  // 2x2 grid should have 3x3 = 9 unique global points
  EXPECT_EQ(nglob, 9);
}

// Test point reordering
TEST_F(MeshNumberingTests, ReorderToOriginalLayout) {
  auto coords = create_coordinates(unit_square_2x2);
  auto points = flatten_coordinates(coords);
  auto original_order = points;

  sort_points_spatially(points);

  type_real tolerance = compute_spatial_tolerance(points, 1, 4);
  assign_global_numbering(points, tolerance);

  auto reordered = reorder_to_original_layout(points);

  ASSERT_EQ(reordered.size(), original_order.size());

  // Check that iloc indices match original
  for (int i = 0; i < reordered.size(); i++) {
    EXPECT_EQ(reordered[i].iloc, i);
    // Coordinates should match original positions
    EXPECT_DOUBLE_EQ(reordered[i].x, original_order[i].x);
    EXPECT_DOUBLE_EQ(reordered[i].z, original_order[i].z);
  }
}

// Test bounding box calculation
TEST_F(MeshNumberingTests, ComputeBoundingBoxUnitSquare) {
  auto coords = create_coordinates(unit_square_2x2);
  auto points = flatten_coordinates(coords);

  auto bbox = compute_bounding_box(points);

  EXPECT_DOUBLE_EQ(bbox.xmin, -1.0);
  EXPECT_DOUBLE_EQ(bbox.xmax, 1.0);
  EXPECT_DOUBLE_EQ(bbox.zmin, -1.0);
  EXPECT_DOUBLE_EQ(bbox.zmax, 1.0);
}

TEST_F(MeshNumberingTests, ComputeBoundingBoxSheared) {
  auto coords = create_coordinates(sheared_element_2x2);
  auto points = flatten_coordinates(coords);

  auto bbox = compute_bounding_box(points);

  EXPECT_DOUBLE_EQ(bbox.xmin, -1.0);
  EXPECT_DOUBLE_EQ(bbox.xmax, 3.0);
  EXPECT_DOUBLE_EQ(bbox.zmin, -1.0);
  EXPECT_DOUBLE_EQ(bbox.zmax, 2.0);
}

TEST_F(MeshNumberingTests, ComputeBoundingBoxGrid) {
  auto coords = create_coordinates(grid_2x2_elements_2x2);
  auto points = flatten_coordinates(coords);

  auto bbox = compute_bounding_box(points);

  EXPECT_DOUBLE_EQ(bbox.xmin, 0.0);
  EXPECT_DOUBLE_EQ(bbox.xmax, 2.0);
  EXPECT_DOUBLE_EQ(bbox.zmin, 0.0);
  EXPECT_DOUBLE_EQ(bbox.zmax, 2.0);
}

// Integration test for full workflow - tests complete utility pipeline
TEST_F(MeshNumberingTests, FullWorkflowIntegration) {
  auto coords = create_coordinates(two_adjacent_squares_2x2);
  int nspec = 2;
  int ngll = 2;

  // Extract coordinates into testable utility functions
  auto points = flatten_coordinates(coords);

  // Sort points spatially
  auto sorted_points = points;
  sort_points_spatially(sorted_points);

  // Compute spatial tolerance
  type_real tolerance =
      compute_spatial_tolerance(sorted_points, nspec, ngll * ngll);

  // Assign global numbering
  int nglob = assign_global_numbering(sorted_points, tolerance);

  // Reorder points to original layout
  auto reordered = reorder_to_original_layout(sorted_points);

  // Calculate bounding box
  auto bbox = compute_bounding_box(reordered);

  // Verify end-to-end results
  EXPECT_EQ(nglob, 6); // 8 points - 2 shared = 6 unique
  EXPECT_EQ(reordered.size(), 8);

  EXPECT_DOUBLE_EQ(bbox.xmin, -2.0);
  EXPECT_DOUBLE_EQ(bbox.xmax, 2.0);
  EXPECT_DOUBLE_EQ(bbox.zmin, -1.0);
  EXPECT_DOUBLE_EQ(bbox.zmax, 1.0);

  // Verify shared points have same iglob
  int shared_pairs = 0;
  for (int i = 0; i < reordered.size(); i++) {
    for (int j = i + 1; j < reordered.size(); j++) {
      if (std::abs(reordered[i].x - reordered[j].x) < tolerance &&
          std::abs(reordered[i].z - reordered[j].z) < tolerance) {
        EXPECT_EQ(reordered[i].iglob, reordered[j].iglob);
        shared_pairs++;
      }
    }
  }
  EXPECT_EQ(shared_pairs, 2); // Exactly 2 shared points (edge points)
}

// Integration test for 5x5 GLL points - critical for spectral elements
TEST_F(MeshNumberingTests, FullWorkflowIntegration5x5) {
  auto coords = create_coordinates(two_adjacent_squares_5x5);
  int nspec = 2;
  int ngll = 5;

  // Extract coordinates into testable utility functions
  auto points = flatten_coordinates(coords);

  // Sort points spatially
  auto sorted_points = points;
  sort_points_spatially(sorted_points);

  // Compute spatial tolerance
  type_real tolerance =
      compute_spatial_tolerance(sorted_points, nspec, ngll * ngll);

  // Assign global numbering
  int nglob = assign_global_numbering(sorted_points, tolerance);

  // Reorder points to original layout
  auto reordered = reorder_to_original_layout(sorted_points);

  // Calculate bounding box
  auto bbox = compute_bounding_box(reordered);

  // Verify end-to-end results for 5x5 GLL points
  EXPECT_EQ(nglob, 45); // 50 total - 5 shared edge points
  EXPECT_EQ(reordered.size(), 50);

  EXPECT_DOUBLE_EQ(bbox.xmin, -2.0);
  EXPECT_DOUBLE_EQ(bbox.xmax, 6.0);
  EXPECT_DOUBLE_EQ(bbox.zmin, -1.0);
  EXPECT_DOUBLE_EQ(bbox.zmax, 1.0);

  // Verify edge sharing is correct - count points at shared edge x=2
  int points_at_shared_edge = 0;
  std::set<int> unique_iglobs_at_edge;

  for (const auto &p : reordered) {
    if (specfem::utilities::is_close(p.x, type_real(2.0))) {
      points_at_shared_edge++;
      unique_iglobs_at_edge.insert(p.iglob);
    }
  }

  EXPECT_EQ(points_at_shared_edge, 10); // 5 from each element
  EXPECT_EQ(unique_iglobs_at_edge.size(),
            5); // But only 5 unique global numbers

  // Verify all points have been assigned valid global numbers
  for (const auto &p : reordered) {
    EXPECT_GE(p.iglob, 0);
    EXPECT_LT(p.iglob, nglob);
  }

  // Verify shared edge points have exactly 5 pairs
  int shared_pairs = 0;
  for (int i = 0; i < reordered.size(); i++) {
    for (int j = i + 1; j < reordered.size(); j++) {
      if (reordered[i].iglob == reordered[j].iglob) {
        shared_pairs++;
        // All shared points should be at x=2.0
        EXPECT_TRUE(
            specfem::utilities::is_close(reordered[i].x, type_real(2.0)))
            << expected_got(2.0, reordered[i].x);
        EXPECT_TRUE(
            specfem::utilities::is_close(reordered[j].x, type_real(2.0)))
            << expected_got(2.0, reordered[j].x);
      }
    }
  }
  EXPECT_EQ(shared_pairs, 5); // Exactly 5 shared points along edge
}

// Test edge cases and error conditions
TEST_F(MeshNumberingTests, EdgeCaseEmptyPoints) {
  std::vector<point> empty_points;

  // Empty points should return 0 global points
  type_real tolerance = 1e-6;
  int nglob = assign_global_numbering(empty_points, tolerance);
  EXPECT_EQ(nglob, 0);

  // Bounding box of empty points should have max/min limits
  auto bbox = compute_bounding_box(empty_points);
  EXPECT_EQ(bbox.xmin, std::numeric_limits<type_real>::max());
  EXPECT_EQ(bbox.xmax, std::numeric_limits<type_real>::min());
}

TEST_F(MeshNumberingTests, EdgeCaseSinglePoint) {
  std::vector<point> single_point = { { 1.0, 2.0, 0, 0 } };

  type_real tolerance = 1e-6;
  int nglob = assign_global_numbering(single_point, tolerance);
  EXPECT_EQ(nglob, 1);
  EXPECT_EQ(single_point[0].iglob, 0);

  auto bbox = compute_bounding_box(single_point);
  EXPECT_DOUBLE_EQ(bbox.xmin, 1.0);
  EXPECT_DOUBLE_EQ(bbox.xmax, 1.0);
  EXPECT_DOUBLE_EQ(bbox.zmin, 2.0);
  EXPECT_DOUBLE_EQ(bbox.zmax, 2.0);
}

TEST_F(MeshNumberingTests, EdgeCaseIdenticalPoints) {
  std::vector<point> identical_points = { { 1.0, 2.0, 0, 0 },
                                          { 1.0, 2.0, 1, 0 },
                                          { 1.0, 2.0, 2, 0 } };

  sort_points_spatially(identical_points);

  type_real tolerance = 1e-6;
  int nglob = assign_global_numbering(identical_points, tolerance);

  // All identical points should get same global number
  EXPECT_EQ(nglob, 1);
  for (const auto &p : identical_points) {
    EXPECT_EQ(p.iglob, 0);
  }
}
