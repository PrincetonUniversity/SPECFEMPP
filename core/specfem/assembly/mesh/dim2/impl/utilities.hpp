#pragma once

#include "kokkos_abstractions.h"
#include <limits>
#include <vector>

/*
 * Utility functions for 2D mesh operations
 * This header is technically part of the implementation details, but the
 * functions declared here are intended for testing as well and therefore
 * exposed.
 */

namespace specfem {
namespace assembly {
namespace mesh_impl {
namespace dim2 {
namespace utilities {

struct point {
  type_real x = 0, z = 0;
  int iloc = 0, iglob = 0;
};

struct bounding_box {
  type_real xmin = std::numeric_limits<type_real>::max();
  type_real xmax = std::numeric_limits<type_real>::min();
  type_real zmin = std::numeric_limits<type_real>::max();
  type_real zmax = std::numeric_limits<type_real>::min();
};

type_real compute_spatial_tolerance(const std::vector<point> &points, int nspec,
                                    int ngllxz);

std::vector<point> flatten_coordinates(
    const specfem::kokkos::HostView4d<double> &global_coordinates);

void sort_points_spatially(std::vector<point> &points);

int assign_global_numbering(std::vector<point> &points, type_real tolerance);

std::vector<point>
reorder_to_original_layout(const std::vector<point> &sorted_points);

bounding_box compute_bounding_box(const std::vector<point> &points);

} // namespace utilities
} // namespace dim2
} // namespace mesh_impl
} // namespace assembly
} // namespace specfem
