
#include "connections.hpp"
static bool flip_orientations(const specfem::connections::orientation &from,
                              const specfem::connections::orientation &to) {
  if ((from == specfem::connections::orientation::top &&
       to == specfem::connections::orientation::bottom) ||
      (from == specfem::connections::orientation::bottom &&
       to == specfem::connections::orientation::top) ||
      (from == specfem::connections::orientation::left &&
       to == specfem::connections::orientation::right) ||
      (from == specfem::connections::orientation::right &&
       to == specfem::connections::orientation::left)) {
    return false;
  }

  if ((from == specfem::connections::orientation::top &&
       to == specfem::connections::orientation::right) ||
      (from == specfem::connections::orientation::right &&
       to == specfem::connections::orientation::top) ||
      (from == specfem::connections::orientation::left &&
       to == specfem::connections::orientation::bottom) ||
      (from == specfem::connections::orientation::bottom &&
       to == specfem::connections::orientation::left)) {
    return false;
  }

  return true;
}

specfem::connections::connection_mapping::coordinate_mappings(const int ngllx,
                                                              const int ngllz)
    : ngllx(ngllx), ngllz(ngllz) {
  // corner coordinates
  coordinates_at_corner[orientation::top_left] = std::make_tuple(0, ngllz - 1);
  coordinates_at_corner[orientation::top_right] =
      std::make_tuple(ngllx - 1, ngllz - 1);
  coordinates_at_corner[orientation::bottom_right] =
      std::make_tuple(ngllx - 1, 0);
  coordinates_at_corner[orientation::bottom_left] = std::make_tuple(0, 0);

  // coordinates along edges
  coordinates_along_edge[orientation::top] = [ngllx](int point) {
    return std::make_tuple(point, ngllx - 1);
  };

  coordinates_along_edge[orientation::bottom] = [ngllx](int point) {
    return std::make_tuple(point, 0);
  };

  coordinates_along_edge[orientation::left] =
      [this](int point) { return std::make_tuple(0, point); }

  coordinates_along_edge[orientation::right] =
      [this](int point) { return std::make_tuple(ngllx - 1, point); };
}

std::tuple<std::tuple<int, int>, std::tuple<int, int> >
specfem::connections::connection_mapping::map_coordinates(
    const orientation &from, const orientation &to, const int point) const {

  if (coordinates_at_corner.find(from) != coordinates_at_corner.end() &&
      coordinates_at_corner.find(to) != coordinates_at_corner.end()) {
    // both are corner points
    if (point != 0) {
      throw std::runtime_error(
          "Point index should be 0 when both from and to are corner points");
    }
    return std::make_tuple(coordinates_at_corner.at(from),
                           coordinates_at_corner.at(to));
  }

  if (coordinates_at_corner.find(from) != coordinates_at_corner.end() ||
      coordinates_at_corner.find(to) != coordinates_at_corner.end()) {
    throw std::runtime_error(
        "Corner is connecting to an edge which is not allowed");
  }

  const auto total_points_on_to = number_of_points_on_orientation(to);

  const auto coord_from = coordinates_along_edge.at(from)[point];

  const auto flip = flip_orientations(from, to);

  if (flip) {
    const auto coord_to =
        coordinates_along_edge.at(to)(total_points_on_to - point - 1);
  } else {
    const auto coord_to = coordinates_along_edge.at(to)[point];
  }

  return std::make_tuple(coord_from, coord_to);
}

int specfem::connections::connection_mapping::number_of_points_on_orientation(
    const orientation &edge) const {
  if ((edge == orientation::top) || (edge == orientation::bottom)) {
    return ngllx;
  }

  if ((edge == orientation::left) || (edge == orientation::right)) {
    return ngllz;
  }

  // Corner points
  if (corners.find(edge) != corners.end()) {
    return 1;
  }

  throw std::runtime_error("Invalid edge orientation");
}

std::array<specfem::connections::orientation, 2>
specfem::connections::connection_mapping::edges_of_corner(
    const orientation &corner) const {

  if (corners.find(corner) == corners.end()) {
    throw std::runtime_error("The argument is not a corner");
  }

  if (corner == orientation::top_left) {
    return { orientation::top, orientation::left };
  }

  if (corner == orientation::top_right) {
    return { orientation::top, orientation::right };
  }

  if (corner == orientation::bottom_right) {
    return { orientation::bottom, orientation::right };
  }

  if (corner == orientation::bottom_left) {
    return { orientation::bottom, orientation::left };
  }
}

int specfem::connections::connection_mapping::corner_index_on_edge(
    const orientation &corner, const orientation &edge) const {
  // Check if the corner is a corner
  if (corners.find(corner) == corners.end()) {
    throw std::runtime_error("The first argument is not a corner");
  }

  // Check if the edge is an edge
  if (edges.find(edge) == edges.end()) {
    throw std::runtime_error("The second argument is not an edge");
  }

  // Check if the corner belongs to the edge
  const auto edges_of_the_corner = this->edges_of_corner(corner);
  if (std::find(edges_of_the_corner.begin(), edges_of_the_corner.end(), edge) ==
      edges_of_the_corner.end()) {
    throw std::runtime_error("The corner does not belong to the edge");
  }

  // Find the index of the corner on the edge
  if (corner == orientation::top_left) {
    if (edge == orientation::top) {
      return 0;
    } else if (edge == orientation::left) {
      return ngllz - 1;
    }
  }

  if (corner == orientation::top_right) {
    if (edge == orientation::top) {
      return ngllx - 1;
    } else if (edge == orientation::right) {
      return ngllz - 1;
    }
  }

  if (corner == orientation::bottom_right) {
    if (edge == orientation::bottom) {
      return ngllx - 1;
    } else if (edge == orientation::right) {
      return 0;
    }
  }

  if (corner == orientation::bottom_left) {
    if (edge == orientation::bottom) {
      return 0;
    } else if (edge == orientation::left) {
      return 0;
    }
  }

  throw std::runtime_error("The corner does not belong to the edge");
}
