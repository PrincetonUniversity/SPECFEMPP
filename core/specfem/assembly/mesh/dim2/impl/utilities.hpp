#pragma once

#include "kokkos_abstractions.h"

#include <limits>
#include <vector>

namespace specfem {
namespace assembly {
namespace mesh_impl {

/**
 * @namespace specfem::assembly::mesh_impl::dim2
 *
 * @brief Utility functions for 2D mesh operations
 *
 * Utility functions for 2D mesh operations
 * This header is technically part of the implementation details, but the
 * functions declared here are intended for testing as well and therefore
 * exposed.
 */
namespace dim2 {

/**
 * @brief Represents a point in 2D space with coordinates and
 * global/local indices
 */
struct point {
  type_real x = 0; ///< x coordinate
  type_real z = 0; ///< z coordinate
  int iloc = 0;    ///< local index
  int iglob = 0;   ///< global index
};

/**
 * @brief Represents a bounding box in 2D space
 *
 * This structure defines the minimum and maximum extents of a 2D region.
 * required by the index mapping procedure.
 */
struct bounding_box {
  type_real xmin =
      std::numeric_limits<type_real>::max(); ///< Minimum x coordinate
  type_real xmax =
      std::numeric_limits<type_real>::min(); ///< Maximum x coordinate
  type_real zmin =
      std::numeric_limits<type_real>::max(); ///< Minimum z coordinate
  type_real zmax =
      std::numeric_limits<type_real>::min(); ///< Maximum z coordinate
};

/**
 * @brief Computes the spatial tolerance for a set of 2D points
 *
 * This function calculates the spatial tolerance based on the distribution
 * of points in the mesh. It is used to determine the acceptable level of
 * precision for spatial operations.
 *
 * @param points The 2D points to analyze
 * @param nspec The number of spectral elements
 * @param ngllxz The number of Gauss-Lobatto-Legendre points in the x and z
 * directions
 * @return type_real The computed spatial tolerance
 */
type_real compute_spatial_tolerance(const std::vector<point> &points, int nspec,
                                    int ngllxz);

/**
 * @brief Flattens a 4D array of global coordinates into a 2D vector of points
 *
 * This function takes a 4D array of global coordinates and converts it into
 * a 2D vector of point structures, which contain the relevant coordinate
 * information for each point in the mesh.
 *
 * @param global_coordinates The 4D array of global coordinates
 * @return std::vector<point> The flattened vector of points
 */
std::vector<point> flatten_coordinates(
    const specfem::kokkos::HostView4d<double> &global_coordinates);

/**
 * @brief Sorts a vector of 2D points spatially
 *
 * This function sorts the points in the vector based on their spatial
 * coordinates (x and z). The sorting is performed in-place.
 *
 * @param points The vector of 2D points to sort
 */
void sort_points_spatially(std::vector<point> &points);

/**
 * @brief Assigns global numbering to a vector of 2D points
 *
 * This function assigns a global index to each point in the vector based
 * on its spatial location. The global numbering is used to establish a
 * consistent ordering of points across different processes.
 *
 * @param points The vector of 2D points to assign global numbering
 * @param tolerance The spatial tolerance for determining point proximity
 * @return int The number of unique global indices assigned
 */
int assign_global_numbering(std::vector<point> &points, type_real tolerance);

/**
 * @brief Reorders a vector of 2D points to match the original layout
 *
 * This function takes a vector of sorted 2D points and reorders them
 * to match the original layout before sorting. This is useful for
 * maintaining consistency between different representations of the mesh.
 *
 * @param sorted_points The vector of sorted 2D points
 * @return std::vector<point> The reordered vector of points
 */
std::vector<point>
reorder_to_original_layout(const std::vector<point> &sorted_points);

/**
 * @brief Computes the bounding box for a set of 2D points
 *
 * This function calculates the minimum and maximum extents of the
 * points in the x and z dimensions, effectively creating a bounding
 * box that encloses all the points.
 *
 * @param points The vector of 2D points to analyze
 * @return bounding_box The computed bounding box
 */
bounding_box compute_bounding_box(const std::vector<point> &points);

/**
 * @brief Create a coordinate arrays object
 *
 * Given the reordered 2D points and the mesh specifications, this function
 * creates the necessary unique coordinate arrays for the simulation, as well as
 * the associated mapping from local coordinates local to global coordinates,
 * number of total points.
 *
 * @param reordered_points The vector of reordered 2D points
 * @param nspec The number of spectral elements
 * @param ngll The number of Gauss-Lobatto-Legendre points
 * @param nglob The number of global points
 * @return std::tuple<Kokkos::View<int ***, Kokkos::LayoutLeft,
 * Kokkos::HostSpace>, Kokkos::View<type_real ****, Kokkos::LayoutRight,
 * Kokkos::HostSpace>, int>
 */
std::tuple<Kokkos::View<int ***, Kokkos::LayoutLeft, Kokkos::HostSpace>,
           Kokkos::View<type_real ****, Kokkos::LayoutRight, Kokkos::HostSpace>,
           int>
create_coordinate_arrays(const std::vector<point> &reordered_points, int nspec,
                         int ngll, int nglob);

} // namespace dim2
} // namespace mesh_impl
} // namespace assembly
} // namespace specfem
