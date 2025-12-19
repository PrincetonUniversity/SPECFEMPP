#pragma once

#include "enumerations/dimension.hpp"
#include <Kokkos_Core.hpp>
#include <algorithm>
#include <list>
#include <stdexcept>
#include <string>

namespace specfem::mesh_entity {

/**
 * @brief 2D quadrilateral mesh entity types and utilities.
 */
namespace dim2 {

/**
 * @brief Mesh entity types for 2D quadrilateral elements.
 *
 * Numbering: edges (1-4), corners (5-8)
 * @code
 * 8 --- 3 --- 7
 * |           |
 * 4           2
 * |           |
 * 5 --- 1 --- 6
 * @endcode
 */
enum class type : int {
  bottom = 1,       ///< Bottom edge
  right = 2,        ///< Right edge
  top = 3,          ///< Top edge
  left = 4,         ///< Left edge
  bottom_left = 5,  ///< Bottom-left corner
  bottom_right = 6, ///< Bottom-right corner
  top_right = 7,    ///< Top-right corner
  top_left = 8      ///< Top-left corner
};

/**
 * @brief Convert entity type to string.
 * @param entity Mesh entity type
 * @return String representation
 */
const std::string to_string(const specfem::mesh_entity::dim2::type &entity);

/// All edge types in counter-clockwise order
const std::list<type> edges = { type::top, type::right, type::bottom,
                                type::left };

/// All corner types in counter-clockwise order
const std::list<type> corners = { type::top_left, type::top_right,
                                  type::bottom_right, type::bottom_left };

} // namespace dim2
/**
 * @brief Get edges that meet at a corner.
 * @param corner Corner entity type
 * @return List of two edges meeting at the corner
 * @throws std::runtime_error if invalid corner type
 */
std::list<dim2::type> edges_of_corner(const dim2::type &corner);

/**
 * @brief Get corners connected to an edge.
 * @param edge Edge entity type
 * @return List of two corners connected to the edge
 * @throws std::runtime_error if invalid edge type
 */
std::list<dim2::type> corners_of_edge(const dim2::type &edge);

/**
 * @brief Check if container contains a mesh entity type.
 * @tparam T Container type with iterators
 * @param list Container to search
 * @param value Entity type to find
 * @return True if found, false otherwise
 */
template <typename T> bool contains(const T &list, const dim2::type &value) {
  return std::find(list.begin(), list.end(), value) != list.end();
}

std::vector<int>
nodes_on_orientation(const specfem::mesh_entity::dim2::type &entity);

/**
 * @brief 2D edge entity for quadrilateral elements.
 */
template <> struct edge<specfem::dimension::type::dim2> {
  specfem::mesh_entity::dim2::type edge_type; ///< Edge type
  int ispec;                                  ///< Element index
  int iedge;                                  ///< Local edge index
  bool reverse_orientation;                   ///< Orientation flag

  KOKKOS_INLINE_FUNCTION
  edge(const int ispec, const int iedge,
       const specfem::mesh_entity::dim2::type edge_type,
       const bool reverse_orientation = false)
      : edge_type(edge_type), ispec(ispec), iedge(iedge),
        reverse_orientation(reverse_orientation) {}

  /**
   * @brief Default constructor.
   */
  KOKKOS_INLINE_FUNCTION
  edge() = default;
};

/**
 * @brief 2D element grid with GLL point configuration.
 */
template <> struct element_grid<specfem::dimension::type::dim2> {

public:
  int ngllz;  ///< Number of GLL points in z-direction
  int ngllx;  ///< Number of GLL points in x-direction
  int orderz; ///< Polynomial order in z
  int orderx; ///< Polynomial order in x
  int size;   ///< Total number of GLL points

  /**
   * @brief Default constructor.
   */
  element_grid() = default;

  /**
   * @brief Construct with uniform GLL points.
   * @param ngll Number of GLL points in both directions
   */
  element_grid(const int ngll)
      : ngllz(ngll), ngllx(ngll), orderz(ngll - 1), orderx(ngll - 1),
        size(ngll * ngll) {}

  /**
   * @brief Construct with different GLL points per direction.
   * @param ngllz Number of GLL points in z-direction
   * @param ngllx Number of GLL points in x-direction
   * @throws std::invalid_argument if ngllz != ngllx
   */
  element_grid(const int ngllz, const int ngllx)
      : ngllz(ngllz), ngllx(ngllx), orderz(ngllz - 1), orderx(ngllx - 1),
        size(ngllz * ngllx) {
    if (ngllz != ngllx) {
      throw std::invalid_argument(
          "Different number of GLL points for Z and X are not supported.");
    }
  };

  /**
   * @brief Check if element matches specified GLL point count.
   * @param ngll_in Number of GLL points to compare against
   * @return True if all dimensions match the specified count
   */
  bool operator==(const int ngll_in) const {
    return ngll_in == this->ngllz && ngll_in == this->ngllx;
  }

  /**
   * @brief Check if element does not match specified GLL point count.
   * @param ngll_in Number of GLL points to compare against
   * @return True if any dimension does not match the specified count
   */
  bool operator!=(const int ngll_in) const { return !(*this == ngll_in); }
};

/**
 * @brief 2D element with coordinate mapping capabilities.
 */
template <>
struct element<specfem::dimension::type::dim2>
    : public element_grid<specfem::dimension::type::dim2> {
private:
  using base = element_grid<specfem::dimension::type::dim2>;

public:
  /**
   * @brief Default constructor.
   */
  element() = default;
  /**
   * @brief Construct with uniform GLL points.
   * @param ngll Number of GLL points in both directions
   */
  element(const int ngll);
  /**
   * @brief Construct with different GLL points per direction.
   * @param ngllz Number of GLL points in z-direction
   * @param ngllx Number of GLL points in x-direction
   * @throws std::invalid_argument if ngllz != ngllx
   */
  element(const int ngllz, const int ngllx);

  /**
   * @brief Get number of GLL points on a mesh entity.
   * @param entity Mesh entity type (edge or corner)
   * @return Number of GLL points on the specified entity
   */
  int number_of_points_on_orientation(
      const specfem::mesh_entity::dim2::type &entity) const;

  /**
   * @brief Map point index to element coordinates.
   * @param entity Mesh entity type
   * @param point Point index on the entity
   * @return Tuple of (iz, ix) coordinates
   */
  std::tuple<int, int>
  map_coordinates(const specfem::mesh_entity::dim2::type &entity,
                  const int point) const;

  /**
   * @brief Get corner coordinates.
   * @param corner Corner entity type
   * @return Tuple of (iz, ix) coordinates for the corner
   */
  std::tuple<int, int>
  map_coordinates(const specfem::mesh_entity::dim2::type &corner) const;

private:
  std::unordered_map<specfem::mesh_entity::dim2::type,
                     std::function<std::tuple<int, int>(int)> >
      edge_coordinates; ///< Maps edge types to coordinate functions
  std::unordered_map<specfem::mesh_entity::dim2::type, std::tuple<int, int> >
      corner_coordinates; ///< Maps corner types to coordinates
};

} // namespace specfem::mesh_entity
