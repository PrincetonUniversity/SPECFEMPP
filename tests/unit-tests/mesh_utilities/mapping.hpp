#pragma once

#include "enumerations/dimension.hpp"
#include "specfem_setup.hpp"
#include <Kokkos_Core.hpp>
#include <limits>
#include <tuple>
#include <vector>

// Type aliases for Kokkos host views
using HostView4d =
    Kokkos::View<type_real ****, Kokkos::LayoutRight, Kokkos::HostSpace>;
using HostView5d =
    Kokkos::View<type_real *****, Kokkos::LayoutLeft, Kokkos::HostSpace>;

/*
 * Test-specific mesh utilities for locate_point testing
 * These are copies of the mesh utilities functionality to avoid dependency
 * on core implementation that will be deprecated.
 * Templated for 2D and 3D support using specfem::dimension::type.
 */

namespace specfem::test {
namespace mesh_utilities {

/**
 * @brief Templated point structure for 2D and 3D coordinates
 * @tparam DimensionTag dimension type (dim2 or dim3)
 */
template <specfem::dimension::type DimensionTag> struct point;

/**
 * @brief 2D point specialization
 */
template <> struct point<specfem::dimension::type::dim2> {
  type_real x = 0, z = 0;
  int iloc = 0, iglob = 0;
};

/**
 * @brief 3D point specialization
 */
template <> struct point<specfem::dimension::type::dim3> {
  type_real x = 0, y = 0, z = 0;
  int iloc = 0, iglob = 0;
};

/**
 * @brief Templated bounding box structure for 2D and 3D
 * @tparam DimensionTag dimension type (dim2 or dim3)
 */
template <specfem::dimension::type DimensionTag> struct bounding_box;

/**
 * @brief 2D bounding box specialization
 */
template <> struct bounding_box<specfem::dimension::type::dim2> {
  type_real xmin = std::numeric_limits<type_real>::max();
  type_real xmax = std::numeric_limits<type_real>::min();
  type_real zmin = std::numeric_limits<type_real>::max();
  type_real zmax = std::numeric_limits<type_real>::min();
};

/**
 * @brief 3D bounding box specialization
 */
template <> struct bounding_box<specfem::dimension::type::dim3> {
  type_real xmin = std::numeric_limits<type_real>::max();
  type_real xmax = std::numeric_limits<type_real>::min();
  type_real ymin = std::numeric_limits<type_real>::max();
  type_real ymax = std::numeric_limits<type_real>::min();
  type_real zmin = std::numeric_limits<type_real>::max();
  type_real zmax = std::numeric_limits<type_real>::min();
};

//-------------------------- Type Aliases for Convenience
//----------------------//
using point_2d = point<specfem::dimension::type::dim2>;
using bounding_box_2d = bounding_box<specfem::dimension::type::dim2>;

using point_3d = point<specfem::dimension::type::dim3>;
using bounding_box_3d = bounding_box<specfem::dimension::type::dim3>;

//-------------------------- Function Overloads for 2D and 3D
//----------------------//

// 2D function overloads
type_real compute_spatial_tolerance(const std::vector<point_2d> &points,
                                    int nspec, int ngllxz);

void sort_points_spatially(std::vector<point_2d> &points);

int assign_global_numbering(std::vector<point_2d> &points, type_real tolerance);

std::vector<point_2d>
reorder_to_original_layout(const std::vector<point_2d> &sorted_points);

bounding_box_2d compute_bounding_box(const std::vector<point_2d> &points);

std::tuple<Kokkos::View<int ***, Kokkos::LayoutLeft, Kokkos::HostSpace>,
           Kokkos::View<type_real ****, Kokkos::LayoutRight, Kokkos::HostSpace>,
           int>
create_coordinate_arrays(const std::vector<point_2d> &reordered_points,
                         int nspec, int ngll, int nglob);

// 3D function overloads
type_real compute_spatial_tolerance(const std::vector<point_3d> &points,
                                    int nspec, int ngllxyz);

std::vector<point_2d> flatten_coordinates(const HostView4d &global_coordinates);
std::vector<point_3d>
flatten_coordinates(const HostView5d &global_coordinates); // DEPRECATED: Use
                                                           // LayoutLeft version

void sort_points_spatially(std::vector<point_3d> &points);

int assign_global_numbering(std::vector<point_3d> &points, type_real tolerance);

std::vector<point_3d>
reorder_to_original_layout(const std::vector<point_3d> &sorted_points);

bounding_box_3d compute_bounding_box(const std::vector<point_3d> &points);

std::tuple<Kokkos::View<int ****, Kokkos::LayoutLeft, Kokkos::HostSpace>,
           Kokkos::View<type_real *****, Kokkos::LayoutLeft, Kokkos::HostSpace>,
           int>
create_coordinate_arrays(const std::vector<point_3d> &reordered_points,
                         int nspec, int ngll, int nglob);

} // namespace mesh_utilities
} // namespace specfem::test
