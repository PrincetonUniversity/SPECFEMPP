#include "../test_macros.hpp"
#include "mapping.hpp"
#include "specfem/utilities.hpp"
#include <gtest/gtest.h>
#include <set>
#include <vector>

using namespace specfem::test::mesh_utilities;

class TestMeshUtilities3DTest : public ::testing::Test {
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

  // Helper to create 5D coordinate array for 3D elements
  HostView5d create_coordinates(
      const std::vector<std::vector<std::tuple<double, double, double> > >
          &element_coords) {
    int nspec = element_coords.size();
    int ngll = std::cbrt(element_coords[0].size()); // cube root for 3D

    HostView5d coords("coords", nspec, ngll, ngll, ngll, 3);

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
};

class TestMeshNumbering3DTests : public TestMeshUtilities3DTest {
protected:
  // Helper to create unit cube coordinates
  std::vector<std::vector<std::tuple<double, double, double> > >
  create_unit_cube(int ngll, double xmin = -1.0, double xmax = 1.0,
                   double ymin = -1.0, double ymax = 1.0, double zmin = -1.0,
                   double zmax = 1.0) {
    std::vector<std::tuple<double, double, double> > coords;
    for (int iy = 0; iy < ngll; iy++) {
      for (int iz = 0; iz < ngll; iz++) {
        for (int ix = 0; ix < ngll; ix++) {
          double x = xmin + (xmax - xmin) * ix / (ngll - 1);
          double y = ymin + (ymax - ymin) * iy / (ngll - 1);
          double z = zmin + (zmax - zmin) * iz / (ngll - 1);
          coords.push_back(std::make_tuple(x, y, z));
        }
      }
    }
    return { coords };
  }

  // Helper to create 2x2x2 grid of elements
  std::vector<std::vector<std::tuple<double, double, double> > >
  create_2x2x2_grid(int ngll) {
    std::vector<std::vector<std::tuple<double, double, double> > > grid;

    // Create 8 elements in a 2x2x2 configuration
    // Elements numbered as follows:
    // Bottom layer (z=0 to 1): [0,1,2,3]
    // Top layer (z=1 to 2):    [4,5,6,7]

    // Element layout for each layer:
    // [2][3]
    // [0][1]

    // Bottom layer (z = 0 to 1)
    grid.push_back(create_unit_cube(ngll, 0.0, 1.0, 0.0, 1.0, 0.0,
                                    1.0)[0]); // Element 0: x[0,1], y[0,1],
                                              // z[0,1]
    grid.push_back(create_unit_cube(ngll, 1.0, 2.0, 0.0, 1.0, 0.0,
                                    1.0)[0]); // Element 1: x[1,2], y[0,1],
                                              // z[0,1]
    grid.push_back(create_unit_cube(ngll, 0.0, 1.0, 1.0, 2.0, 0.0,
                                    1.0)[0]); // Element 2: x[0,1], y[1,2],
                                              // z[0,1]
    grid.push_back(create_unit_cube(ngll, 1.0, 2.0, 1.0, 2.0, 0.0,
                                    1.0)[0]); // Element 3: x[1,2], y[1,2],
                                              // z[0,1]

    // Top layer (z = 1 to 2)
    grid.push_back(create_unit_cube(ngll, 0.0, 1.0, 0.0, 1.0, 1.0,
                                    2.0)[0]); // Element 4: x[0,1], y[0,1],
                                              // z[1,2]
    grid.push_back(create_unit_cube(ngll, 1.0, 2.0, 0.0, 1.0, 1.0,
                                    2.0)[0]); // Element 5: x[1,2], y[0,1],
                                              // z[1,2]
    grid.push_back(create_unit_cube(ngll, 0.0, 1.0, 1.0, 2.0, 1.0,
                                    2.0)[0]); // Element 6: x[0,1], y[1,2],
                                              // z[1,2]
    grid.push_back(create_unit_cube(ngll, 1.0, 2.0, 1.0, 2.0, 1.0,
                                    2.0)[0]); // Element 7: x[1,2], y[1,2],
                                              // z[1,2]

    return grid;
  }

  // Pre-built geometries for convenience
  std::vector<std::vector<std::tuple<double, double, double> > > unit_cube_5x5 =
      create_unit_cube(5);
  std::vector<std::vector<std::tuple<double, double, double> > >
      grid_2x2x2_elements_5x5 = create_2x2x2_grid(5);
};

// Test flatten_coordinates function for 3D
TEST_F(TestMeshNumbering3DTests, FlattenCoordinatesUnitCube5x5) {
  auto coords = create_coordinates(unit_cube_5x5);
  auto flattened = flatten_coordinates(coords);

  ASSERT_EQ(flattened.size(), 125); // 1 element * 5*5*5 points

  // Check corners are preserved
  EXPECT_DOUBLE_EQ(flattened[0].x, -1.0); // First point
  EXPECT_DOUBLE_EQ(flattened[0].y, -1.0);
  EXPECT_DOUBLE_EQ(flattened[0].z, -1.0);
  EXPECT_EQ(flattened[0].iloc, 0);

  EXPECT_DOUBLE_EQ(flattened[124].x, 1.0); // Last point
  EXPECT_DOUBLE_EQ(flattened[124].y, 1.0);
  EXPECT_DOUBLE_EQ(flattened[124].z, 1.0);
  EXPECT_EQ(flattened[124].iloc, 124);

  // Check center point (ix=2, iz=2, iy=2 -> idx = 2*25 + 2*5 + 2 = 62)
  EXPECT_DOUBLE_EQ(flattened[62].x, 0.0); // Center point
  EXPECT_DOUBLE_EQ(flattened[62].y, 0.0);
  EXPECT_DOUBLE_EQ(flattened[62].z, 0.0);
  EXPECT_EQ(flattened[62].iloc, 62);
}

// Test spatial sorting for 3D
TEST_F(TestMeshNumbering3DTests, SortPointsSpatiallyUnitCube) {
  auto coords = create_coordinates(unit_cube_5x5);
  auto points = flatten_coordinates(coords);

  sort_points_spatially(points);

  // Should be sorted by x, then y, then z
  // For a 5x5x5 cube, first few points should be at x=-1
  EXPECT_DOUBLE_EQ(points[0].x, -1.0);
  EXPECT_DOUBLE_EQ(points[0].y, -1.0);
  EXPECT_DOUBLE_EQ(points[0].z, -1.0);

  // Check that sorting is correct (x first, then y, then z)
  for (int i = 1; i < points.size(); i++) {
    if (points[i].x != points[i - 1].x) {
      EXPECT_GT(points[i].x, points[i - 1].x);
    } else if (points[i].y != points[i - 1].y) {
      EXPECT_GE(points[i].y, points[i - 1].y);
    } else {
      EXPECT_GE(points[i].z, points[i - 1].z);
    }
  }
}

// Test tolerance calculation for 3D
TEST_F(TestMeshNumbering3DTests, ComputeSpatialToleranceUnitCube) {
  auto coords = create_coordinates(unit_cube_5x5);
  auto points = flatten_coordinates(coords);

  type_real tolerance = compute_spatial_tolerance(points, 1, 125); // 5^3 = 125

  // For unit cube: min dimension = 2.0, tolerance = 1e-6 * 2.0
  EXPECT_TRUE(specfem::utilities::is_close(tolerance, type_real(2e-6)))
      << expected_got(2e-6, tolerance);
}

// Test global numbering assignment for 3D
TEST_F(TestMeshNumbering3DTests, AssignGlobalNumberingUnitCube) {
  auto coords = create_coordinates(unit_cube_5x5);
  auto points = flatten_coordinates(coords);
  sort_points_spatially(points);

  type_real tolerance = compute_spatial_tolerance(points, 1, 125);
  int nglob = assign_global_numbering(points, tolerance);

  // All points are distinct in a single cube, should get 125 unique global
  // numbers
  EXPECT_EQ(nglob, 125);

  // Check numbering is sequential for distinct points
  for (int i = 0; i < points.size(); i++) {
    EXPECT_EQ(points[i].iglob, i);
  }
}

// Critical test: 2x2x2 grid with shared points and 5x5x5 GLL points
TEST_F(TestMeshNumbering3DTests, AssignGlobalNumbering2x2x2Grid5x5) {
  auto coords = create_coordinates(grid_2x2x2_elements_5x5);
  auto points = flatten_coordinates(coords);
  sort_points_spatially(points);

  type_real tolerance =
      compute_spatial_tolerance(points, 8, 125); // 8 elements, 125 points each
  int nglob = assign_global_numbering(points, tolerance);

  // For a 2x2x2 grid with 5x5x5 GLL points per element:
  // - Total points: 8 * 125 = 1000
  // - Each element shares faces with neighbors
  // - Expected unique points: should be much less than 1000
  // - For a 2x2x2 grid, we expect approximately: (2*4+1)^3 = 9^3 = 729 unique
  // points

  EXPECT_LT(nglob, 1000); // Should have fewer than total due to sharing
  EXPECT_GT(nglob, 500);  // But still a substantial number

  // Verify all points have valid global numbers
  for (const auto &p : points) {
    EXPECT_GE(p.iglob, 0);
    EXPECT_LT(p.iglob, nglob);
  }
}

// Test bounding box calculation for 3D
TEST_F(TestMeshNumbering3DTests, ComputeBoundingBoxUnitCube) {
  auto coords = create_coordinates(unit_cube_5x5);
  auto points = flatten_coordinates(coords);

  auto bbox = compute_bounding_box(points);

  EXPECT_DOUBLE_EQ(bbox.xmin, -1.0);
  EXPECT_DOUBLE_EQ(bbox.xmax, 1.0);
  EXPECT_DOUBLE_EQ(bbox.ymin, -1.0);
  EXPECT_DOUBLE_EQ(bbox.ymax, 1.0);
  EXPECT_DOUBLE_EQ(bbox.zmin, -1.0);
  EXPECT_DOUBLE_EQ(bbox.zmax, 1.0);
}

TEST_F(TestMeshNumbering3DTests, ComputeBoundingBox2x2x2Grid) {
  auto coords = create_coordinates(grid_2x2x2_elements_5x5);
  auto points = flatten_coordinates(coords);

  auto bbox = compute_bounding_box(points);

  EXPECT_DOUBLE_EQ(bbox.xmin, 0.0);
  EXPECT_DOUBLE_EQ(bbox.xmax, 2.0);
  EXPECT_DOUBLE_EQ(bbox.ymin, 0.0);
  EXPECT_DOUBLE_EQ(bbox.ymax, 2.0);
  EXPECT_DOUBLE_EQ(bbox.zmin, 0.0);
  EXPECT_DOUBLE_EQ(bbox.zmax, 2.0);
}

// Integration test for full 3D workflow
TEST_F(TestMeshNumbering3DTests, FullWorkflow3DIntegration) {
  auto coords = create_coordinates(grid_2x2x2_elements_5x5);
  int nspec = 8;
  int ngll = 5;

  // Extract coordinates into testable utility functions
  auto points = flatten_coordinates(coords);

  // Sort points spatially
  auto sorted_points = points;
  sort_points_spatially(sorted_points);

  // Compute spatial tolerance
  type_real tolerance =
      compute_spatial_tolerance(sorted_points, nspec, ngll * ngll * ngll);

  // Assign global numbering
  int nglob = assign_global_numbering(sorted_points, tolerance);

  // Reorder points to original layout
  auto reordered = reorder_to_original_layout(sorted_points);

  // Calculate bounding box
  auto bbox = compute_bounding_box(reordered);

  // Verify end-to-end results
  EXPECT_LT(nglob, 1000);            // Should have shared points
  EXPECT_GT(nglob, 500);             // But still substantial
  EXPECT_EQ(reordered.size(), 1000); // 8 elements * 125 points each

  EXPECT_DOUBLE_EQ(bbox.xmin, 0.0);
  EXPECT_DOUBLE_EQ(bbox.xmax, 2.0);
  EXPECT_DOUBLE_EQ(bbox.ymin, 0.0);
  EXPECT_DOUBLE_EQ(bbox.ymax, 2.0);
  EXPECT_DOUBLE_EQ(bbox.zmin, 0.0);
  EXPECT_DOUBLE_EQ(bbox.zmax, 2.0);

  // Verify all points have been assigned valid global numbers
  for (const auto &p : reordered) {
    EXPECT_GE(p.iglob, 0);
    EXPECT_LT(p.iglob, nglob);
  }
}

// Test edge cases for 3D
TEST_F(TestMeshNumbering3DTests, EdgeCaseEmptyPoints3D) {
  std::vector<point_3d> empty_points;

  // Empty points should return 0 global points
  type_real tolerance = 1e-6;
  int nglob = assign_global_numbering(empty_points, tolerance);
  EXPECT_EQ(nglob, 0);

  // Bounding box of empty points should have max/min limits
  auto bbox = compute_bounding_box(empty_points);
  EXPECT_EQ(bbox.xmin, std::numeric_limits<type_real>::max());
  EXPECT_EQ(bbox.xmax, std::numeric_limits<type_real>::min());
  EXPECT_EQ(bbox.ymin, std::numeric_limits<type_real>::max());
  EXPECT_EQ(bbox.ymax, std::numeric_limits<type_real>::min());
  EXPECT_EQ(bbox.zmin, std::numeric_limits<type_real>::max());
  EXPECT_EQ(bbox.zmax, std::numeric_limits<type_real>::min());
}

TEST_F(TestMeshNumbering3DTests, EdgeCaseSinglePoint3D) {
  std::vector<point_3d> single_point = { { 1.0, 2.0, 3.0, 0, 0 } };

  type_real tolerance = 1e-6;
  int nglob = assign_global_numbering(single_point, tolerance);
  EXPECT_EQ(nglob, 1);
  EXPECT_EQ(single_point[0].iglob, 0);

  auto bbox = compute_bounding_box(single_point);
  EXPECT_DOUBLE_EQ(bbox.xmin, 1.0);
  EXPECT_DOUBLE_EQ(bbox.xmax, 1.0);
  EXPECT_DOUBLE_EQ(bbox.ymin, 2.0);
  EXPECT_DOUBLE_EQ(bbox.ymax, 2.0);
  EXPECT_DOUBLE_EQ(bbox.zmin, 3.0);
  EXPECT_DOUBLE_EQ(bbox.zmax, 3.0);
}

//-------------------------- create_coordinate_arrays Tests
//----------------------//

// Test create_coordinate_arrays with simple 3D single element
TEST_F(TestMeshUtilities3DTest, CreateCoordinateArraysSingleElement3D) {
  // Create a single 3D element with 2x2x2 GLL points (unit cube)
  std::vector<std::vector<std::tuple<double, double, double> > >
      element_coords = { {
          { 0.0, 0.0, 0.0 },
          { 1.0, 0.0, 0.0 },
          { 0.0, 1.0, 0.0 },
          { 1.0, 1.0, 0.0 }, // z=0 plane
          { 0.0, 0.0, 1.0 },
          { 1.0, 0.0, 1.0 },
          { 0.0, 1.0, 1.0 },
          { 1.0, 1.0, 1.0 } // z=1 plane
      } };

  auto coords = create_coordinates(element_coords);
  auto points = flatten_coordinates(coords);

  // Process points through the mapping pipeline
  sort_points_spatially(points);
  int nglob = assign_global_numbering(points, 1e-6);
  auto reordered_points = reorder_to_original_layout(points);

  // Test create_coordinate_arrays
  int nspec = 1;
  int ngll = 2;

  auto [index_mapping, coord, actual_nglob] =
      create_coordinate_arrays(reordered_points, nspec, ngll, nglob);

  // Verify dimensions
  EXPECT_EQ(index_mapping.extent(0), nspec);
  EXPECT_EQ(index_mapping.extent(1), ngll);
  EXPECT_EQ(index_mapping.extent(2), ngll);
  EXPECT_EQ(index_mapping.extent(3), ngll);

  EXPECT_EQ(coord.extent(0), nspec);
  EXPECT_EQ(coord.extent(1), ngll);
  EXPECT_EQ(coord.extent(2), ngll);
  EXPECT_EQ(coord.extent(3), ngll);
  EXPECT_EQ(coord.extent(4), 3); // 3 coordinates (x, y, z)

  EXPECT_EQ(actual_nglob, nglob);
  EXPECT_EQ(actual_nglob, 8); // 8 unique points for single 3D element

  // Verify corner coordinates
  EXPECT_EQ(coord(0, 0, 0, 0, 0), 0.0); // x-coord at (0,0,0)
  EXPECT_EQ(coord(0, 0, 0, 0, 1), 0.0); // y-coord at (0,0,0)
  EXPECT_EQ(coord(0, 0, 0, 0, 2), 0.0); // z-coord at (0,0,0)

  EXPECT_EQ(coord(0, 0, 0, 1, 0), 1.0); // x-coord at (0,0,1)
  EXPECT_EQ(coord(0, 0, 0, 1, 1), 0.0); // y-coord at (0,0,1)
  EXPECT_EQ(coord(0, 0, 0, 1, 2), 0.0); // z-coord at (0,0,1)

  EXPECT_EQ(coord(0, 1, 1, 1, 0), 1.0); // x-coord at (1,1,1)
  EXPECT_EQ(coord(0, 1, 1, 1, 1), 1.0); // y-coord at (1,1,1)
  EXPECT_EQ(coord(0, 1, 1, 1, 2), 1.0); // z-coord at (1,1,1)

  // Verify all global indices are unique for single element
  std::set<int> unique_indices;
  for (int iz = 0; iz < ngll; iz++) {
    for (int iy = 0; iy < ngll; iy++) {
      for (int ix = 0; ix < ngll; ix++) {
        unique_indices.insert(index_mapping(0, iz, iy, ix));
      }
    }
  }
  EXPECT_EQ(unique_indices.size(), 8);
}

// Test create_coordinate_arrays with two adjacent 3D elements
TEST_F(TestMeshUtilities3DTest, CreateCoordinateArraysTwoAdjacentElements3D) {
  // Create two adjacent 3D elements sharing a face
  std::vector<std::vector<std::tuple<double, double, double> > >
      element_coords = { { // Element 0: [0,0.5] x [0,0.5] x [0,0.5]
                           { 0.0, 0.0, 0.0 },
                           { 0.5, 0.0, 0.0 },
                           { 0.0, 0.5, 0.0 },
                           { 0.5, 0.5, 0.0 },
                           { 0.0, 0.0, 0.5 },
                           { 0.5, 0.0, 0.5 },
                           { 0.0, 0.5, 0.5 },
                           { 0.5, 0.5, 0.5 } },
                         { // Element 1: [0.5,1] x [0,0.5] x [0,0.5] (adjacent
                           // in x-direction)
                           { 0.5, 0.0, 0.0 },
                           { 1.0, 0.0, 0.0 },
                           { 0.5, 0.5, 0.0 },
                           { 1.0, 0.5, 0.0 },
                           { 0.5, 0.0, 0.5 },
                           { 1.0, 0.0, 0.5 },
                           { 0.5, 0.5, 0.5 },
                           { 1.0, 0.5, 0.5 } } };

  auto coords = create_coordinates(element_coords);
  auto points = flatten_coordinates(coords);

  sort_points_spatially(points);
  int nglob = assign_global_numbering(points, 1e-6);
  auto reordered_points = reorder_to_original_layout(points);

  int nspec = 2;
  int ngll = 2;

  auto [index_mapping, coord, actual_nglob] =
      create_coordinate_arrays(reordered_points, nspec, ngll, nglob);

  // Verify dimensions
  EXPECT_EQ(index_mapping.extent(0), nspec);
  EXPECT_EQ(index_mapping.extent(1), ngll);
  EXPECT_EQ(index_mapping.extent(2), ngll);
  EXPECT_EQ(index_mapping.extent(3), ngll);

  EXPECT_EQ(coord.extent(0), nspec);
  EXPECT_EQ(coord.extent(1), ngll);
  EXPECT_EQ(coord.extent(2), ngll);
  EXPECT_EQ(coord.extent(3), ngll);
  EXPECT_EQ(coord.extent(4), 3);

  EXPECT_EQ(actual_nglob, nglob);

  // Should have 12 unique points (16 total - 4 shared on face)
  EXPECT_EQ(nglob, 12);

  // Check that adjacent elements share face points
  EXPECT_EQ(index_mapping(0, 0, 0, 1),
            index_mapping(1, 0, 0, 0)); // Point (0.5, 0.0, 0.0)
  EXPECT_EQ(index_mapping(0, 0, 1, 1),
            index_mapping(1, 0, 1, 0)); // Point (0.5, 0.5, 0.0)
  EXPECT_EQ(index_mapping(0, 1, 0, 1),
            index_mapping(1, 1, 0, 0)); // Point (0.5, 0.0, 0.5)
  EXPECT_EQ(index_mapping(0, 1, 1, 1),
            index_mapping(1, 1, 1, 0)); // Point (0.5, 0.5, 0.5)

  // Verify coordinate values for shared points
  type_real x_shared = coord(0, 0, 0, 1, 0);
  EXPECT_NEAR(x_shared, 0.5, 1e-12); // Should be at x=0.5
}

// Test create_coordinate_arrays with higher order 3D elements (3x3x3 GLL
// points)
TEST_F(TestMeshUtilities3DTest, CreateCoordinateArraysHigherOrder3D) {
  // Create single element with 3x3x3 GLL points
  std::vector<std::vector<std::tuple<double, double, double> > > element_coords;
  std::vector<std::tuple<double, double, double> > elem_points;

  int ngll = 3;
  for (int iz = 0; iz < ngll; iz++) {
    for (int iy = 0; iy < ngll; iy++) {
      for (int ix = 0; ix < ngll; ix++) {
        double x = static_cast<double>(ix) / (ngll - 1);
        double y = static_cast<double>(iy) / (ngll - 1);
        double z = static_cast<double>(iz) / (ngll - 1);
        elem_points.push_back({ x, y, z });
      }
    }
  }
  element_coords.push_back(elem_points);

  auto coords = create_coordinates(element_coords);
  auto points = flatten_coordinates(coords);

  sort_points_spatially(points);
  int nglob = assign_global_numbering(points, 1e-6);
  auto reordered_points = reorder_to_original_layout(points);

  int nspec = 1;

  auto [index_mapping, coord, actual_nglob] =
      create_coordinate_arrays(reordered_points, nspec, ngll, nglob);

  EXPECT_EQ(actual_nglob, nglob);
  EXPECT_EQ(actual_nglob, 27); // All 27 points should be unique

  // Check corner coordinates
  EXPECT_EQ(coord(0, 0, 0, 0, 0), 0.0); // Corner (0,0,0) x
  EXPECT_EQ(coord(0, 0, 0, 0, 1), 0.0); // Corner (0,0,0) y
  EXPECT_EQ(coord(0, 0, 0, 0, 2), 0.0); // Corner (0,0,0) z

  EXPECT_EQ(coord(0, 2, 2, 2, 0), 1.0); // Corner (1,1,1) x
  EXPECT_EQ(coord(0, 2, 2, 2, 1), 1.0); // Corner (1,1,1) y
  EXPECT_EQ(coord(0, 2, 2, 2, 2), 1.0); // Corner (1,1,1) z

  // Check center point
  EXPECT_EQ(coord(0, 1, 1, 1, 0), 0.5); // Center x
  EXPECT_EQ(coord(0, 1, 1, 1, 1), 0.5); // Center y
  EXPECT_EQ(coord(0, 1, 1, 1, 2), 0.5); // Center z
}

// Test create_coordinate_arrays layout consistency for 3D
TEST_F(TestMeshUtilities3DTest, CreateCoordinateArraysLayoutConsistency3D) {
  std::vector<std::vector<std::tuple<double, double, double> > >
      element_coords = { { { 0.0, 0.0, 0.0 },
                           { 1.0, 0.0, 0.0 },
                           { 0.0, 1.0, 0.0 },
                           { 1.0, 1.0, 0.0 },
                           { 0.0, 0.0, 1.0 },
                           { 1.0, 0.0, 1.0 },
                           { 0.0, 1.0, 1.0 },
                           { 1.0, 1.0, 1.0 } } };

  auto coords = create_coordinates(element_coords);
  auto points = flatten_coordinates(coords);

  sort_points_spatially(points);
  int nglob = assign_global_numbering(points, 1e-6);
  auto reordered_points = reorder_to_original_layout(points);

  int nspec = 1;
  int ngll = 2;

  auto [index_mapping, coord, actual_nglob] =
      create_coordinate_arrays(reordered_points, nspec, ngll, nglob);

  // Test coordinate access pattern (should not throw)
  for (int ispec = 0; ispec < nspec; ispec++) {
    for (int iz = 0; iz < ngll; iz++) {
      for (int iy = 0; iy < ngll; iy++) {
        for (int ix = 0; ix < ngll; ix++) {
          type_real x = coord(ispec, iz, iy, ix, 0);
          type_real y = coord(ispec, iz, iy, ix, 1);
          type_real z = coord(ispec, iz, iy, ix, 2);

          // Verify coordinates are reasonable
          EXPECT_GE(x, 0.0);
          EXPECT_LE(x, 1.0);
          EXPECT_GE(y, 0.0);
          EXPECT_LE(y, 1.0);
          EXPECT_GE(z, 0.0);
          EXPECT_LE(z, 1.0);
        }
      }
    }
  }
}

// Test create_coordinate_arrays with 2x2x2 grid of 3D elements
TEST_F(TestMeshUtilities3DTest, CreateCoordinateArrays2x2x2Grid3D) {
  // Create 8 elements in a 2x2x2 grid, each with 2x2x2 GLL points
  std::vector<std::vector<std::tuple<double, double, double> > > element_coords;

  int ngll = 2;
  for (int elem_z = 0; elem_z < 2; elem_z++) {
    for (int elem_y = 0; elem_y < 2; elem_y++) {
      for (int elem_x = 0; elem_x < 2; elem_x++) {
        std::vector<std::tuple<double, double, double> > elem_points;

        for (int iz = 0; iz < ngll; iz++) {
          for (int iy = 0; iy < ngll; iy++) {
            for (int ix = 0; ix < ngll; ix++) {
              double x = elem_x + static_cast<double>(ix) / (ngll - 1);
              double y = elem_y + static_cast<double>(iy) / (ngll - 1);
              double z = elem_z + static_cast<double>(iz) / (ngll - 1);
              elem_points.push_back({ x, y, z });
            }
          }
        }
        element_coords.push_back(elem_points);
      }
    }
  }

  auto coords = create_coordinates(element_coords);
  auto points = flatten_coordinates(coords);

  sort_points_spatially(points);
  int nglob = assign_global_numbering(points, 1e-6);
  auto reordered_points = reorder_to_original_layout(points);

  int nspec = 8;

  auto [index_mapping, coord, actual_nglob] =
      create_coordinate_arrays(reordered_points, nspec, ngll, nglob);

  EXPECT_EQ(actual_nglob, nglob);

  // For a 2x2x2 grid of 2x2x2 elements, total unique points should be 3x3x3 =
  // 27
  EXPECT_EQ(nglob, 27);

  // Check that adjacent elements share faces/edges/corners
  // Elements 0 and 1 should share their x-face
  for (int iz = 0; iz < ngll; iz++) {
    for (int iy = 0; iy < ngll; iy++) {
      EXPECT_EQ(index_mapping(0, iz, iy, 1), index_mapping(1, iz, iy, 0));
    }
  }

  // Elements 0 and 2 should share their y-face
  for (int iz = 0; iz < ngll; iz++) {
    for (int ix = 0; ix < ngll; ix++) {
      EXPECT_EQ(index_mapping(0, iz, 1, ix), index_mapping(2, iz, 0, ix));
    }
  }

  // Elements 0 and 4 should share their z-face
  for (int iy = 0; iy < ngll; iy++) {
    for (int ix = 0; ix < ngll; ix++) {
      EXPECT_EQ(index_mapping(0, 1, iy, ix), index_mapping(4, 0, iy, ix));
    }
  }
}
