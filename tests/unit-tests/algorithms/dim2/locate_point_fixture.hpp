#pragma once
#include "../../mesh_utilities/mapping.hpp"
#include "../../test_macros.hpp"
#include "SPECFEM_Environment.hpp"
#include "algorithms/locate_point.hpp"
#include "algorithms/locate_point_impl.hpp"
#include "kokkos_abstractions.h"
#include "specfem/point.hpp"
#include "utilities/utilities.hpp"
#include <Kokkos_Core.hpp>
#include <gtest/gtest.h>

// Use test-specific mesh utilities for coordinate generation
using namespace specfem::test::mesh_utilities;

// Test fixture for 2D locate_point algorithms using tested mesh utilities
class LocatePoint2D : public ::testing::Test {
protected:
  struct ElementGeometry {
    Kokkos::View<type_real ****, Kokkos::LayoutRight, Kokkos::HostSpace>
        global_coords;
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

  void SetUp() override {}

  void TearDown() override {}

  // Helper to create unit square coordinates programmatically
  std::vector<std::vector<std::pair<double, double> > >
  create_unit_square(int ngll, double xmin = -1.0, double xmax = 1.0,
                     double zmin = -1.0, double zmax = 1.0);

  // Helper to create control nodes for 4-node or 9-node elements
  std::vector<std::vector<std::pair<double, double> > >
  create_control_nodes(int ngnod, double xmin = 0.0, double xmax = 1.0,
                       double zmin = 0.0, double zmax = 1.0);

  // Helper to create coordinate array from element corners and ngll points
  HostView4d create_coordinate_array(
      const std::vector<std::vector<std::pair<double, double> > >
          &element_coords);

  // Create element geometry using tested utility functions
  ElementGeometry create_element_geometry(
      const std::vector<std::vector<std::pair<double, double> > >
          &element_coords,
      const std::vector<std::vector<std::pair<double, double> > >
          &control_coords);

  // Create a single unit square element [0,1] x [0,1] with 2x2 GLL points
  ElementGeometry create_single_unit_square_2x2();

  // Create two adjacent elements [0,0.5]x[0,0.5] and [0.5,1]x[0,0.5] with 2x2
  // GLL points
  ElementGeometry create_two_adjacent_elements_2x2();

  // Create 2x2 grid of elements with 2x2 GLL points each
  ElementGeometry create_2x2_grid_elements_2x2();

  // Create a single unit square element [0,1] x [0,1] with 5x5 GLL points
  ElementGeometry create_single_unit_square_5x5();

  // Create a single unit square element [0,1] x [0,1] with 2x2 GLL points and 9
  // control nodes
  ElementGeometry create_single_unit_square_2x2_9node();

  // Create a single unit square element [0,1] x [0,1] with 5x5 GLL points and 9
  // control nodes
  ElementGeometry create_single_unit_square_5x5_9node();

  // Create two adjacent elements with 5x5 GLL points
  ElementGeometry create_two_adjacent_elements_5x5();
};
