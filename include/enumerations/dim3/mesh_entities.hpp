#pragma once

#include "enumerations/dimension.hpp"
#include <Kokkos_Core.hpp>
#include <algorithm>
#include <list>
#include <stdexcept>
#include <string>

namespace specfem::mesh_entity {
/**
 * @brief 3D hexahedral mesh entity types and utilities.
 */
namespace dim3 {

/**
 * @brief Mesh entity types for 3D hexahedral elements.
 *
 * 6 faces (1-6), 12 edges (7-18), 8 corners (19-26)
 * @code
 *     8 -------- 7
 *    /|         /|
 *   / |        / |
 *  5 -------- 6  |
 *  |  |       |  |
 *  |  4 ------|--3
 *  | /        | /
 *  |/         |/
 *  1 -------- 2
 * @endcode
 */
enum class type : int {
  bottom = 1,              ///< Bottom face of the element
  right = 2,               ///< Right face of the element
  top = 3,                 ///< Top face of the element
  left = 4,                ///< Left face of the element
  front = 5,               ///< Front face of the element
  back = 6,                ///< Back face of the element
  bottom_left = 7,         ///< Bottom-left edge of the element
  bottom_right = 8,        ///< Bottom-right edge of the element
  top_right = 9,           ///< Top-right edge of the element
  top_left = 10,           ///< Top-left edge of the element
  front_bottom = 11,       ///< Front-bottom edge of the element
  front_top = 12,          ///< Front-top edge of the element
  front_left = 13,         ///< Front-left edge of the element
  front_right = 14,        ///< Front-right edge of the element
  back_bottom = 15,        ///< Back-bottom edge of the element
  back_top = 16,           ///< Back-top edge of the element
  back_left = 17,          ///< Back-left edge of the element
  back_right = 18,         ///< Back-right edge of the element
  bottom_front_left = 19,  ///< Bottom-front-left corner of the element
  bottom_front_right = 20, ///< Bottom-front-right corner of the element
  bottom_back_left = 21,   ///< Bottom-back-left corner of the element
  bottom_back_right = 22,  ///< Bottom-back-right corner of the element
  top_front_left = 23,     ///< Top-front-left corner of the element
  top_front_right = 24,    ///< Top-front-right corner of the element
  top_back_left = 25,      ///< Top-back-left corner of the element
  top_back_right = 26      ///< Top-back-right corner of the element
};

/**
 * @brief Convert entity type to string.
 * @param entity Mesh entity type
 * @return String representation
 */
const std::string to_string(const specfem::mesh_entity::dim3::type &entity);

/// All face types (6 boundary surfaces)
const std::list<specfem::mesh_entity::dim3::type> faces = {
  type::bottom, type::right, type::top, type::left, type::front, type::back
};

/// All edge types (12 connecting edges)
const std::list<specfem::mesh_entity::dim3::type> edges = {
  type::bottom_left,  type::bottom_right, type::top_right,  type::top_left,
  type::front_bottom, type::front_top,    type::front_left, type::front_right,
  type::back_bottom,  type::back_top,     type::back_left,  type::back_right
};

/// All corner types (8 vertices)
const std::list<specfem::mesh_entity::dim3::type> corners = {
  type::bottom_front_left, type::bottom_front_right, type::bottom_back_left,
  type::bottom_back_right, type::top_front_left,     type::top_front_right,
  type::top_back_left,     type::top_back_right
};

/**
 * @brief Get faces connected by an edge.
 * @param edge Edge entity type
 * @return List of two faces sharing the edge
 * @throws std::runtime_error if invalid edge type
 */
const std::list<specfem::mesh_entity::dim3::type>
faces_of_edge(const specfem::mesh_entity::dim3::type &edge);

/**
 * @brief Get edges meeting at a corner.
 * @param corner Corner entity type
 * @return List of three edges meeting at the corner
 * @throws std::runtime_error if invalid corner type
 */
const std::list<specfem::mesh_entity::dim3::type>
edges_of_corner(const specfem::mesh_entity::dim3::type &corner);

/**
 * @brief Get corners defining a face.
 * @param face Face entity type
 * @return List of four corners outlining the face
 * @throws std::runtime_error if invalid face type
 */
const std::list<specfem::mesh_entity::dim3::type>
corners_of_face(const specfem::mesh_entity::dim3::type &face);

} // namespace dim3

/**
 * @brief Check if container contains a mesh entity type.
 * @tparam T Container type with iterators
 * @param list Container to search
 * @param value Entity type to find
 * @return True if found, false otherwise
 */
template <typename T> bool contains(const T &list, const dim3::type &value) {
  return std::find(list.begin(), list.end(), value) != list.end();
}

/**
 * @brief Get node indices for a mesh entity orientation.
 * @param entity Mesh entity type (face, edge, corner)
 * @return Vector of node indices on the entity
 */
std::vector<int>
nodes_on_orientation(const specfem::mesh_entity::dim3::type &entity);

template <specfem::dimension::type Dimension> struct edge;

/**
 * @brief 3D edge entity for hexahedral elements.
 */
template <> struct edge<specfem::dimension::type::dim3> {
  specfem::mesh_entity::dim3::type edge_type; ///< Edge type
  int ispec;                                  ///< Element index
  bool reverse_orientation;                   ///< Orientation flag

  /**
   * @brief Construct edge with parameters.
   * @param ispec Element index
   * @param edge_type Edge type
   * @param reverse_orientation Orientation flag (default: false)
   */
  KOKKOS_INLINE_FUNCTION
  edge(const int ispec, const specfem::mesh_entity::dim3::type edge_type,
       const bool reverse_orientation = false)
      : edge_type(edge_type), ispec(ispec),
        reverse_orientation(reverse_orientation) {}

  /**
   * @brief Default constructor.
   */
  KOKKOS_INLINE_FUNCTION
  edge() = default;
};

template <specfem::dimension::type Dimension> struct element;
template <specfem::dimension::type Dimension> struct element_grid;

/**
 * @brief 3D element grid with GLL point configuration.
 */
template <> struct element_grid<specfem::dimension::type::dim3> {

public:
  int ngllz;  ///< Number of GLL points in z-direction
  int nglly;  ///< Number of GLL points in y-direction
  int ngllx;  ///< Number of GLL points in x-direction
  int orderz; ///< Polynomial order in z
  int ordery; ///< Polynomial order in y
  int orderx; ///< Polynomial order in x
  int size;   ///< Total number of GLL points

  /**
   * @brief Default constructor.
   */
  element_grid() = default;

  /**
   * @brief Construct with uniform GLL points.
   * @param ngll Number of GLL points in all directions
   */
  element_grid(const int ngll)
      : ngllx(ngll), nglly(ngll), ngllz(ngll), orderz(ngll - 1),
        ordery(nglly - 1), orderx(ngllx - 1), size(ngll * ngll * ngll) {};

  /**
   * @brief Construct with different GLL points per direction.
   * @param ngllz Number of GLL points in z-direction
   * @param nglly Number of GLL points in y-direction
   * @param ngllx Number of GLL points in x-direction
   * @throws std::runtime_error if dimensions differ or < 2
   */
  element_grid(const int ngllz, const int nglly, const int ngllx)
      : ngllz(ngllz), nglly(nglly), ngllx(ngllx), orderz(ngllz - 1),
        ordery(nglly - 1), orderx(ngllx - 1), size(ngllz * nglly * ngllx) {
    if (ngllz < 2 || nglly < 2 || ngllx < 2) {
      throw std::runtime_error(
          "ngllz, nglly, and ngllx must be at least 2 to define a 3D element");
    }

    if (ngllz != nglly || ngllz != ngllx) {
      throw std::runtime_error(
          "ngllz, nglly, and ngllx must be equal for a cubic 3D element");
    }
  }

  /**
   * @brief Check if element matches specified GLL point count.
   * @param ngll Number of GLL points to compare against
   * @return True if all dimensions match the specified count
   */
  bool operator==(const int ngll) const {
    return ngll == ngllz && ngll == nglly && ngll == ngllx;
  }

  /**
   * @brief Check if element does not match specified GLL point count.
   * @param ngll Number of GLL points to compare against
   * @return True if any dimension does not match the specified count
   */
  bool operator!=(const int ngll) const { return !(*this == ngll); }
};

/**
 * @brief 3D element with coordinate mapping capabilities.
 */
template <>
struct element<specfem::dimension::type::dim3>
    : element_grid<specfem::dimension::type::dim3> {

public:
  /**
   * @brief Default constructor.
   */
  element() = default;
  /**
   * @brief Construct with uniform GLL points.
   * @param ngll Number of GLL points in all directions
   */
  element(const int ngll);
  /**
   * @brief Construct with different GLL points per direction.
   * @param ngllz Number of GLL points in z-direction
   * @param nglly Number of GLL points in y-direction
   * @param ngllx Number of GLL points in x-direction
   * @throws std::runtime_error if dimensions differ or < 2
   */
  element(const int ngllz, const int nglly, const int ngllx);

  /**
   * @brief Get number of GLL points on a mesh entity.
   * @param entity Mesh entity type (face, edge, corner)
   * @return Number of GLL points on the specified entity
   */
  int number_of_points_on_orientation(
      const specfem::mesh_entity::dim3::type &entity) const;

  /**
   * @brief Map point index to element coordinates.
   * @param entity Mesh entity type
   * @param point Point index on the entity
   * @return Tuple of (iz, iy, ix) coordinates
   */
  std::tuple<int, int, int>
  map_coordinates(const specfem::mesh_entity::dim3::type &entity,
                  const int point) const;

  /**
   * @brief Get corner coordinates.
   * @param corner Corner entity type
   * @return Tuple of (iz, iy, ix) coordinates for the corner
   */
  std::tuple<int, int, int>
  map_coordinates(const specfem::mesh_entity::dim3::type &corner) const;

private:
  int ngll2d; ///< Points per 2D face
  int ngll;   ///< Points per direction

  /**
   * @brief Get face coordinates by 2D indices.
   * @param face Face entity type
   * @param ipoint First face coordinate index
   * @param jpoint Second face coordinate index
   * @return Tuple of (iz, iy, ix) coordinates
   */
  std::tuple<int, int, int>
  get_face_coordinates(const specfem::mesh_entity::dim3::type &face,
                       const int ipoint, const int jpoint) const;

  /**
   * @brief Get edge coordinates by 1D index.
   * @param edge Edge entity type
   * @param point Point index along the edge
   * @return Tuple of (iz, iy, ix) coordinates
   */
  std::tuple<int, int, int>
  get_edge_coordinates(const specfem::mesh_entity::dim3::type &edge,
                       const int point) const;

  /**
   * @brief Get corner coordinates.
   * @param corner Corner entity type
   * @return Tuple of (iz, iy, ix) coordinates for the corner
   */
  std::tuple<int, int, int>
  get_corner_coordinates(const specfem::mesh_entity::dim3::type &corner) const;
};

/**
 * @brief Get edges defining a face.
 * @param face Face entity type
 * @return Array of four edges outlining the face
 * @throws std::runtime_error if invalid face type
 */
std::array<specfem::mesh_entity::dim3::type, 4>
edges_of_face(const specfem::mesh_entity::dim3::type &face);

/**
 * @brief Get corners defining a face.
 * @param face Face entity type
 * @return Array of four corners outlining the face
 * @throws std::runtime_error if invalid face type
 */
std::array<specfem::mesh_entity::dim3::type, 4>
corners_of_face(const specfem::mesh_entity::dim3::type &face);

/**
 * @brief Get faces meeting at a corner.
 * @param corner Corner entity type
 * @return List of three faces at the corner
 * @throws std::runtime_error if invalid corner type
 */
std::list<specfem::mesh_entity::dim3::type>
faces_of_corner(const specfem::mesh_entity::dim3::type &corner);

/**
 * @brief Get edges meeting at a corner.
 * @param corner Corner entity type
 * @return List of three edges at the corner
 * @throws std::runtime_error if invalid corner type
 */
std::list<specfem::mesh_entity::dim3::type>
edges_of_corner(const specfem::mesh_entity::dim3::type &corner);

/**
 * @brief Get faces connected by an edge.
 * @param edge Edge entity type
 * @return List of two faces sharing the edge
 * @throws std::runtime_error if invalid edge type
 */
std::list<specfem::mesh_entity::dim3::type>
faces_of_edge(const specfem::mesh_entity::dim3::type &edge);
} // namespace specfem::mesh_entity
