#pragma once

#include "dimension.hpp"
#include <Kokkos_Core.hpp>
#include <algorithm>
#include <list>
#include <stdexcept>
#include <string>

/**
 * @namespace specfem::mesh_entity
 * @brief Defines mesh entity types and utilities for spectral element method
 *
 * This namespace provides enumerations and utility functions for working with
 * mesh entities in 2D spectral element grids, including edges and corners
 * of quadrilateral elements.
 */
namespace specfem::mesh_entity {

/**
 * @brief Enumeration of mesh entity types for 2D quadrilateral elements
 *
 * Defines the different types of mesh entities that can exist on the boundary
 * of a quadrilateral element in a 2D spectral element mesh. The numbering
 * follows the convention:
 *
 * @code
 * 8 --- 3 --- 7
 * |           |
 * 4           2
 * |           |
 * 5 --- 1 --- 6
 * @endcode
 *
 * Where:
 * - 1 = bottom edge, 2 = top edge, 3 = left edge, 4 = right edge
 * - 5 = bottom_left, 6 = bottom_right, 7 = top_right, 8 = top_left corners
 */
enum class type : int {
  bottom = 1,       ///< Bottom edge of the element
  right = 2,        ///< Top edge of the element
  top = 3,          ///< Left edge of the element
  left = 4,         ///< Right edge of the element
  bottom_left = 5,  ///< Bottom-left corner of the element
  bottom_right = 6, ///< Bottom-right corner of the element
  top_right = 7,    ///< Top-right corner of the element
  top_left = 8      ///< Top-left corner of the element
};

/**
 * @brief Recovers a human-readable string for a given mesh entity.
 *
 */
const std::string to_string(const specfem::mesh_entity::type &entity);

/**
 * @brief List of all edge types in a quadrilateral element
 *
 * Contains all edge mesh entities in counter-clockwise order starting from the
 * top edge. This list is useful for iterating over all edges of an element.
 */
const std::list<type> edges = { type::top, type::right, type::bottom,
                                type::left };

/**
 * @brief List of all corner types in a quadrilateral element
 *
 * Contains all corner mesh entities in counter-clockwise order starting from
 * the top-left corner. This list is useful for iterating over all corners of an
 * element.
 */
const std::list<type> corners = { type::top_left, type::top_right,
                                  type::bottom_right, type::bottom_left };

/**
 * @brief Generic utility function to check if a container contains a specific
 * mesh entity type
 *
 * @tparam T Container type that supports begin() and end() iterators
 * @param list The container to search in
 * @param value The mesh entity type to search for
 * @return bool True if the value is found in the container, false otherwise
 *
 * This template function provides a generic way to check membership in any
 * container of mesh entity types. It's commonly used with the predefined
 * edges and corners lists.
 *
 * @code
 * if (contains(edges, type::top)) {
 *     // Handle edge case
 * }
 * @endcode
 */
template <typename T> bool contains(const T &list, const type &value) {
  return std::find(list.begin(), list.end(), value) != list.end();
}

/**
 * @brief Returns the edges that form a given corner
 *
 * @param corner The corner mesh entity type
 * @return std::list<type> List of edge types that meet at the specified corner
 *
 * For each corner of a quadrilateral element, this function returns the two
 * edges that meet at that corner. The edges are returned in a consistent order.
 *
 * Corner-to-edge mappings:
 * - top_left: [top, left]
 * - top_right: [top, right]
 * - bottom_right: [bottom, right]
 * - bottom_left: [bottom, left]
 *
 * @throws std::runtime_error if the input is not a valid corner type
 */
std::list<type> edges_of_corner(const type &corner);

/**
 * @brief Returns the corners that are adjacent to a given edge.
 *
 * For a specified edge of a quadrilateral element, this function returns the
 * two corners that are connected to that edge. The corners are returned in a
 * consistent order.
 *
 * Edge-to-corner mappings:
 * - top: [top_left, top_right]
 * - right: [top_right, bottom_right]
 * - bottom: [bottom_right, bottom_left]
 * - left: [bottom_left, top_left]
 *
 * @param edge The edge mesh entity type.
 * @return std::list<type> List of corner types adjacent to the specified edge.
 *
 * @throws std::runtime_error if the input is not a valid edge type.
 */
std::list<type> corners_of_edge(const type &edge);

struct edge {
  specfem::mesh_entity::type edge_type;
  int ispec;
  bool reverse_orientation;

  KOKKOS_INLINE_FUNCTION
  edge(const int ispec, const specfem::mesh_entity::type edge_type,
       const bool reverse_orientation = false)
      : edge_type(edge_type), ispec(ispec),
        reverse_orientation(reverse_orientation) {}

  KOKKOS_INLINE_FUNCTION
  edge() = default;
};
/**
 * @brief Mesh element structure for a specific dimension
 *
 * @tparam Dimension The dimension type (e.g., dim2, dim3)
 */
template <specfem::dimension::type Dimension> struct element;

/**
 * @brief Mesh element structure for 2D elements (Specialization)
 */
template <> struct element<specfem::dimension::type::dim2> {

public:
  int ngllz;  ///< Number of Gauss-Lobatto-Legendre points in the z-direction
  int ngllx;  ///< Number of Gauss-Lobatto-Legendre points in the x-direction
  int orderz; ///< Polynomial order of the element
  int orderx; ///< Polynomial order of the element
  int size;   ///< Total number of GLL points in the element

  /**
   * @brief Default constructor for the element struct
   */
  element() = default;

  /**
   * @brief Constructs an element entity given the number of
   * Gauss-Lobatto-Legendre points
   *
   * @param ngll The number of Gauss-Lobatto-Legendre points
   */
  element(const int ngll)
      : ngllz(ngll), ngllx(ngll), orderz(ngll - 1), orderx(ngll - 1),
        size(ngll * ngll) {}

  /**
   * @brief Constructs an element entity given the number of
   * Gauss-Lobatto-Legendre points in each dimension
   *
   * @param ngll The number of Gauss-Lobatto-Legendre points
   */
  element(const int ngllz, const int ngllx)
      : ngllz(ngllz), ngllx(ngllx), orderz(ngllz - 1), orderx(ngllx - 1),
        size(ngllz * ngllx) {
    if (ngllz != ngllx) {
      throw std::invalid_argument(
          "Different number of GLL points for Z and X are not supported.");
    }
  };

  /**
   * @brief Checks if the element is consistent across dimensions against a
   *        specific number of GLL points.
   *
   * @param ngll_in The number of Gauss-Lobatto-Legendre points
   * @return true If all dimensions match the specified number of GLL points
   * @return false If any dimension does not match
   */
  bool operator==(const int ngll_in) const {
    return ngll_in == this->ngllz && ngll_in == this->ngllx;
  }

  /**
   * @brief Checks if the element is consistent across dimensions against a
   *        specific number of GLL points.
   *
   * @param ngll_in The number of Gauss-Lobatto-Legendre points
   * @return false If all dimensions match the specified number of GLL points
   * @return true If any dimension does not match
   *
   */
  bool operator!=(const int ngll_in) const { return !(*this == ngll_in); }
};

template <> struct element<specfem::dimension::type::dim3> {

public:
  int ngllz;  ///< Number of Gauss-Lobatto-Legendre points in the z-direction
  int nglly;  ///< Number of Gauss-Lobatto-Legendre points in the y-direction
  int ngllx;  ///< Number of Gauss-Lobatto-Legendre points in the x-direction
  int orderz; ///< Polynomial order of the element
  int ordery; ///< Polynomial order of the element
  int orderx; ///< Polynomial order of the element
  int size;   ///< Total number of GLL points in the element

  /**
   * @brief Default constructor for the element struct
   */
  element() = default;

  /**
   * @brief Constructs an element entity given the number of
   * Gauss-Lobatto-Legendre points
   *
   * @param ngll The number of Gauss-Lobatto-Legendre points
   */
  element(const int ngll)
      : ngllx(ngll), nglly(ngll), ngllz(ngll), orderz(ngll - 1),
        ordery(nglly - 1), orderx(ngllx - 1), size(ngll * ngll * ngll) {};

  /**
   * @brief Constructs an element entity given individual GLL points for each
   * dimension
   *
   * @param ngll The base number of Gauss-Lobatto-Legendre points
   * @param ngllz The number of Gauss-Lobatto-Legendre points in the z-direction
   * @param nglly The number of Gauss-Lobatto-Legendre points in the y-direction
   * @param ngllx The number of Gauss-Lobatto-Legendre points in the x-direction
   */
  element(const int ngllz, const int nglly, const int ngllx)
      : ngllz(ngllz), nglly(nglly), ngllx(ngllx), orderz(ngllz - 1),
        ordery(nglly - 1), orderx(ngllx - 1), size(ngllz * nglly * ngllx) {
    if (ngllz != nglly || ngllz != ngllx) {
      throw std::invalid_argument("Inconsistent number of GLL points");
    }
  };

  /**
   * @brief Check if the GLL number of point is consistent against input ngll
   *
   * @param ngll The number of Gauss-Lobatto-Legendre points
   * @return true If all dimensions match the specified number of GLL points
   * @return false If any dimension does not match
   */
  bool operator==(const int ngll) const {
    return ngll == ngllz && ngll == nglly && ngll == ngllx;
  }

  /**
   * @brief Check if the GLL number of points is _not_ consistent against input
   *        number of GLL points
   *
   * @param ngll The number of Gauss-Lobatto-Legendre points
   * @return false If all dimensions match the specified number of GLL points
   * @return true If any dimension does not match
   */
  bool operator!=(const int ngll) const { return !(*this == ngll); }
};

} // namespace specfem::mesh_entity
