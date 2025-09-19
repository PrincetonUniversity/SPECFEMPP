#pragma once

#include "mesh_entities.hpp"
#include <functional>
#include <stdexcept>
#include <tuple>
#include <unordered_map>

namespace specfem::connections {

/**
 * @brief Enumeration of connection types for mesh connectivity
 *
 * Defines the types of connections that can exist between mesh elements
 * in the spectral element method.
 */
enum class type : int {
  /// @brief Strongly conforming connection where nodes match exactly
  strongly_conforming = 1,
  /// @brief Weakly conforming connection where nodes match, but the shape
  /// function can be discontinuous. (example: coupling across different media,
  /// kinematic faults).
  weakly_conforming = 2,
  /// @brief Nonconforming connections have no matching nodes, but are
  /// geometrically (spatially) adjacent
  nonconforming = 3
};

/**
 * @brief Recovers a human-readable string for a given connection
 *
 */
const std::string to_string(const specfem::connections::type &conn);

/**
 * @class connection_mapping
 * @brief Provides mapping utilities between mesh entities (edges, corners) and
 * their coordinates in a structured grid, supporting operations for spectral
 * element methods.
 *
 * This class encapsulates the logic required to map between different mesh
 * entities (such as edges and corners) and their corresponding coordinates in a
 * 2D structured grid, typically used in spectral element methods. It provides
 * functions to:
 *   - Determine the number of points along a given orientation (edge).
 *   - Map coordinates between different mesh entities.
 *   - Find the location of a corner on a given edge.
 *   - Retrieve coordinates for a point on an edge or at a corner.
 *
 * The class stores the number of grid points in the x and z directions, and
 * maintains lookup tables for corner and edge coordinate mappings.
 *
 * @note This class is intended for internal use within the mesh connectivity
 * logic of SPECFEM.
 */
class connection_mapping {
public:
  /**
   * @brief Constructs a connection mapping for a structured grid
   *
   * @param ngllx Number of grid points in the x-direction
   * @param ngllz Number of grid points in the z-direction
   *
   * @note The grid points include boundary points, so for a spectral element
   *       with polynomial degree N, ngllx = ngllz = N+1
   */
  connection_mapping(const int ngllx, const int ngllz);

  /**
   * @brief Returns the number of points along a given mesh entity
   *
   * @param edge The mesh entity (edge or corner) to query
   * @return int Number of points along the entity
   *
   * For edges:
   * - Top/bottom edges: returns ngllx
   * - Left/right edges: returns ngllz
   * - Corners: returns 1
   *
   * @throws std::runtime_error if the edge type is invalid
   */
  int number_of_points_on_orientation(
      const specfem::mesh_entity::type &edge) const;

  /**
   * @brief Maps coordinates between two mesh entities
   *
   * @param from Source mesh entity (edge or corner)
   * @param to Target mesh entity (edge or corner)
   * @param point Point index along the source entity
   * @return std::tuple<std::tuple<int, int>, std::tuple<int, int>>
   *         Pair of coordinate tuples: (from_coords, to_coords)
   *         Each coordinate tuple is (z, x) in the 2D grid
   *
   * This function maps a point on one mesh entity to the corresponding
   * point on another mesh entity, accounting for orientation flips
   * when necessary.
   *
   * @throws std::runtime_error if corner-to-edge mapping is attempted
   * @throws std::runtime_error if point index is invalid for corner-to-corner
   * mapping
   */
  std::tuple<std::tuple<int, int>, std::tuple<int, int> >
  map_coordinates(const specfem::mesh_entity::type &from,
                  const specfem::mesh_entity::type &to, const int point) const;

  /**
   * @brief Finds the index of a corner point on a given edge
   *
   * @param corner The corner mesh entity
   * @param edge The edge mesh entity
   * @return int The point index of the corner along the edge
   *
   * This function determines at which point index along an edge
   * a specific corner is located.
   *
   * @throws std::runtime_error if the first argument is not a corner
   * @throws std::runtime_error if the second argument is not an edge
   * @throws std::runtime_error if the corner does not belong to the edge
   */
  int find_corner_on_edge(const specfem::mesh_entity::type &corner,
                          const specfem::mesh_entity::type &edge) const;

  /**
   * @brief Returns the coordinates of a point along an edge
   *
   * @param edge The edge mesh entity
   * @param point The point index along the edge
   * @return std::tuple<int, int> The (z, x) coordinates of the point
   *
   * @throws std::runtime_error if the edge type is invalid
   */
  std::tuple<int, int>
  coordinates_at_edge(const specfem::mesh_entity::type &edge,
                      const int point) const;

  /**
   * @brief Returns the coordinates of a corner
   *
   * @param corner The corner mesh entity
   * @return std::tuple<int, int> The (z, x) coordinates of the corner
   *
   * @throws std::runtime_error if the corner type is invalid
   */
  std::tuple<int, int>
  coordinates_at_corner(const specfem::mesh_entity::type &corner) const;

  /**
   * @brief Helper function to determine if orientation mapping requires
   * coordinate flipping
   *
   * @param from Source mesh entity orientation
   * @param to Target mesh entity orientation
   * @return bool True if coordinates should be flipped during mapping
   *
   * This function implements the logic for determining when coordinate
   * mappings between edges require flipping to maintain proper orientation.
   * The flipping rules ensure consistent connectivity across mesh elements.
   */
  bool flip_orientation(const specfem::mesh_entity::type &from,
                        const specfem::mesh_entity::type &to) const;

private:
  /// @brief Number of grid points in the x-direction
  int ngllx;

  /// @brief Number of grid points in the z-direction
  int ngllz;

  /**
   * @brief Lookup table for corner coordinates
   *
   * Maps each corner mesh entity to its (z, x) coordinates in the grid.
   * The coordinate system follows the convention:
   * - z=0 is bottom, z=ngllz-1 is top
   * - x=0 is left, x=ngllx-1 is right
   */
  std::unordered_map<specfem::mesh_entity::type, std::tuple<int, int> >
      corner_coordinates;

  /**
   * @brief Function to get the coordinates along an edge
   *
   * Key: edge type
   * Value: function that takes point index and returns the coordinates (z, x)
   *
   * Each edge is parameterized by a point index that varies from 0 to
   * the number of points on that edge minus 1. The lambda functions
   * convert this parameterization to actual (z, x) grid coordinates.
   *
   * Usage example:
   * @code
   * const auto [iz, ix] = edge_coordinates.at(edge)(point);
   * @endcode
   *
   */
  std::unordered_map<specfem::mesh_entity::type,
                     std::function<std::tuple<int, int>(int)> >
      edge_coordinates;
};

} // namespace specfem::connections
