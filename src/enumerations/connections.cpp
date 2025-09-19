
#include "enumerations/connections.hpp"
#include "enumerations/mesh_entities.hpp"
#include <functional>
#include <stdexcept>
#include <string>
#include <tuple>
#include <unordered_map>

const std::string
specfem::connections::to_string(const specfem::connections::type &conn) {
  switch (conn) {
  case specfem::connections::type::strongly_conforming:
    return "strongly_conforming";
  case specfem::connections::type::weakly_conforming:
    return "weakly_conforming";
  case specfem::connections::type::nonconforming:
    return "nonconforming";
  default:
    throw std::runtime_error(
        std::string("specfem::connections::to_string does not handle ") +
        std::to_string(static_cast<int>(conn)));
    return "!ERR";
  }
}

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
bool specfem::connections::connection_mapping::flip_orientation(
    const specfem::mesh_entity::type &from,
    const specfem::mesh_entity::type &to) const {
  if ((from == specfem::mesh_entity::type::top &&
       to == specfem::mesh_entity::type::bottom) ||
      (from == specfem::mesh_entity::type::bottom &&
       to == specfem::mesh_entity::type::top) ||
      (from == specfem::mesh_entity::type::left &&
       to == specfem::mesh_entity::type::right) ||
      (from == specfem::mesh_entity::type::right &&
       to == specfem::mesh_entity::type::left)) {
    return false;
  }

  if ((from == specfem::mesh_entity::type::top &&
       to == specfem::mesh_entity::type::right) ||
      (from == specfem::mesh_entity::type::right &&
       to == specfem::mesh_entity::type::top) ||
      (from == specfem::mesh_entity::type::left &&
       to == specfem::mesh_entity::type::bottom) ||
      (from == specfem::mesh_entity::type::bottom &&
       to == specfem::mesh_entity::type::left)) {
    return false;
  }

  return true;
}

specfem::connections::connection_mapping::connection_mapping(const int ngllx,
                                                             const int ngllz)
    : ngllx(ngllx), ngllz(ngllz) {
  // corner coordinates
  corner_coordinates[specfem::mesh_entity::type::top_left] =
      std::make_tuple(ngllz - 1, 0);
  corner_coordinates[specfem::mesh_entity::type::top_right] =
      std::make_tuple(ngllz - 1, ngllx - 1);
  corner_coordinates[specfem::mesh_entity::type::bottom_right] =
      std::make_tuple(0, ngllx - 1);
  corner_coordinates[specfem::mesh_entity::type::bottom_left] =
      std::make_tuple(0, 0);

  // coordinates along edges
  // The element is defined as (z, x) in 2D
  // Where the diagram is as follows:
  // 3 --- 2
  // |     |
  // |     |
  // 0 --- 1

  edge_coordinates[specfem::mesh_entity::type::top] =
      [ngllx, ngllz](int point) { return std::make_tuple(ngllz - 1, point); };

  edge_coordinates[specfem::mesh_entity::type::bottom] =
      [ngllx, ngllz](int point) { return std::make_tuple(0, point); };

  edge_coordinates[specfem::mesh_entity::type::left] =
      [ngllx, ngllz](int point) { return std::make_tuple(point, 0); };

  edge_coordinates[specfem::mesh_entity::type::right] =
      [ngllx, ngllz](int point) { return std::make_tuple(point, ngllx - 1); };
}

std::tuple<std::tuple<int, int>, std::tuple<int, int> >
specfem::connections::connection_mapping::map_coordinates(
    const specfem::mesh_entity::type &from,
    const specfem::mesh_entity::type &to, const int point) const {

  if (corner_coordinates.find(from) != corner_coordinates.end() &&
      corner_coordinates.find(to) != corner_coordinates.end()) {
    // both are corner points
    if (point != 0) {
      throw std::runtime_error(
          "Point index should be 0 when both from and to are corner points");
    }
    return std::make_tuple(corner_coordinates.at(from),
                           corner_coordinates.at(to));
  }

  if (corner_coordinates.find(from) != corner_coordinates.end() ||
      corner_coordinates.find(to) != corner_coordinates.end()) {
    throw std::runtime_error(
        "Corner is connecting to an edge which is not allowed");
  }

  const auto total_points_on_to = number_of_points_on_orientation(to);

  const auto coord_from = edge_coordinates.at(from)(point);

  const auto flip = this->flip_orientation(from, to);

  const auto coord_to =
      flip ? edge_coordinates.at(to)(total_points_on_to - 1 - point)
           : edge_coordinates.at(to)(point);

  return std::make_tuple(coord_from, coord_to);
}

int specfem::connections::connection_mapping::number_of_points_on_orientation(
    const specfem::mesh_entity::type &edge) const {
  if ((edge == specfem::mesh_entity::type::top) ||
      (edge == specfem::mesh_entity::type::bottom)) {
    return ngllx;
  }

  if ((edge == specfem::mesh_entity::type::left) ||
      (edge == specfem::mesh_entity::type::right)) {
    return ngllz;
  }

  // Corner points
  if (specfem::mesh_entity::contains(specfem::mesh_entity::corners, edge)) {
    return 1;
  }

  throw std::runtime_error("Invalid edge orientation");
}

int specfem::connections::connection_mapping::find_corner_on_edge(
    const specfem::mesh_entity::type &corner,
    const specfem::mesh_entity::type &edge) const {
  // Check if the corner is a corner
  if (!specfem::mesh_entity::contains(specfem::mesh_entity::corners, corner)) {
    throw std::runtime_error("The first argument is not a corner");
  }

  // Check if the edge is an edge
  if (!specfem::mesh_entity::contains(specfem::mesh_entity::edges, edge)) {
    throw std::runtime_error("The second argument is not an edge");
  }

  // Check if the corner belongs to the edge
  const auto edges_of_the_corner =
      specfem::mesh_entity::edges_of_corner(corner);
  if (std::find(edges_of_the_corner.begin(), edges_of_the_corner.end(), edge) ==
      edges_of_the_corner.end()) {
    throw std::runtime_error("The corner does not belong to the edge");
  }

  // Find the index of the corner on the edge
  if (corner == specfem::mesh_entity::type::top_left) {
    if (edge == specfem::mesh_entity::type::top) {
      return 0;
    } else if (edge == specfem::mesh_entity::type::left) {
      return ngllz - 1;
    }
  }

  if (corner == specfem::mesh_entity::type::top_right) {
    if (edge == specfem::mesh_entity::type::top) {
      return ngllx - 1;
    } else if (edge == specfem::mesh_entity::type::right) {
      return ngllz - 1;
    }
  }

  if (corner == specfem::mesh_entity::type::bottom_right) {
    if (edge == specfem::mesh_entity::type::bottom) {
      return ngllx - 1;
    } else if (edge == specfem::mesh_entity::type::right) {
      return 0;
    }
  }

  if (corner == specfem::mesh_entity::type::bottom_left) {
    if (edge == specfem::mesh_entity::type::bottom) {
      return 0;
    } else if (edge == specfem::mesh_entity::type::left) {
      return 0;
    }
  }

  throw std::runtime_error("The corner does not belong to the edge");
}

std::tuple<int, int>
specfem::connections::connection_mapping::coordinates_at_edge(
    const specfem::mesh_entity::type &edge, const int point) const {
  if (!specfem::mesh_entity::contains(specfem::mesh_entity::edges, edge)) {
    throw std::runtime_error("The argument is not an edge");
  }

  return edge_coordinates.at(edge)(point);
}

std::tuple<int, int>
specfem::connections::connection_mapping::coordinates_at_corner(
    const specfem::mesh_entity::type &corner) const {
  if (!specfem::mesh_entity::contains(specfem::mesh_entity::corners, corner)) {
    throw std::runtime_error("The argument is not a corner");
  }

  return corner_coordinates.at(corner);
}

std::list<specfem::mesh_entity::type>
specfem::mesh_entity::corners_of_edge(const specfem::mesh_entity::type &edge) {
  if (!specfem::mesh_entity::contains(specfem::mesh_entity::edges, edge)) {
    throw std::runtime_error("The argument is not an edge");
  }

  if (edge == specfem::mesh_entity::type::top) {
    return { specfem::mesh_entity::type::top_left,
             specfem::mesh_entity::type::top_right };
  } else if (edge == specfem::mesh_entity::type::right) {
    return { specfem::mesh_entity::type::top_right,
             specfem::mesh_entity::type::bottom_right };
  } else if (edge == specfem::mesh_entity::type::bottom) {
    return { specfem::mesh_entity::type::bottom_right,
             specfem::mesh_entity::type::bottom_left };
  } else if (edge == specfem::mesh_entity::type::left) {
    return { specfem::mesh_entity::type::bottom_left,
             specfem::mesh_entity::type::top_left };
  }

  throw std::runtime_error("The edge does not have corners");
}
