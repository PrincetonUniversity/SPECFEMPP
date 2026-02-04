#include "specfem/mesh_entity.hpp"
#include <list>
#include <stdexcept>

std::list<specfem::mesh_entity::dim2::type>
specfem::mesh_entity::edges_of_corner(
    const specfem::mesh_entity::dim2::type &corner) {
  switch (corner) {
  case specfem::mesh_entity::dim2::type::top_left:
    return { specfem::mesh_entity::dim2::type::top,
             specfem::mesh_entity::dim2::type::left };
  case specfem::mesh_entity::dim2::type::top_right:
    return { specfem::mesh_entity::dim2::type::top,
             specfem::mesh_entity::dim2::type::right };
  case specfem::mesh_entity::dim2::type::bottom_right:
    return { specfem::mesh_entity::dim2::type::bottom,
             specfem::mesh_entity::dim2::type::right };
  case specfem::mesh_entity::dim2::type::bottom_left:
    return { specfem::mesh_entity::dim2::type::bottom,
             specfem::mesh_entity::dim2::type::left };
  default:
    throw std::runtime_error("Invalid corner type");
  }
}

std::list<specfem::mesh_entity::dim2::type>
specfem::mesh_entity::corners_of_edge(
    const specfem::mesh_entity::dim2::type &edge) {
  switch (edge) {
  case specfem::mesh_entity::dim2::type::top:
    return { specfem::mesh_entity::dim2::type::top_left,
             specfem::mesh_entity::dim2::type::top_right };
  case specfem::mesh_entity::dim2::type::right:
    return { specfem::mesh_entity::dim2::type::top_right,
             specfem::mesh_entity::dim2::type::bottom_right };
  case specfem::mesh_entity::dim2::type::bottom:
    return { specfem::mesh_entity::dim2::type::bottom_right,
             specfem::mesh_entity::dim2::type::bottom_left };
  case specfem::mesh_entity::dim2::type::left:
    return { specfem::mesh_entity::dim2::type::bottom_left,
             specfem::mesh_entity::dim2::type::top_left };
  default:
    throw std::runtime_error("Invalid edge type");
  }
}

const std::string specfem::mesh_entity::dim2::to_string(
    const specfem::mesh_entity::dim2::type &entity) {
  switch (entity) {
  case specfem::mesh_entity::dim2::type::bottom:
    return "bottom";
  case specfem::mesh_entity::dim2::type::right:
    return "right";
  case specfem::mesh_entity::dim2::type::top:
    return "top";
  case specfem::mesh_entity::dim2::type::left:
    return "left";
  case specfem::mesh_entity::dim2::type::bottom_left:
    return "bottom_left";
  case specfem::mesh_entity::dim2::type::bottom_right:
    return "bottom_right";
  case specfem::mesh_entity::dim2::type::top_right:
    return "top_right";
  case specfem::mesh_entity::dim2::type::top_left:
    return "top_left";
  default:
    throw std::runtime_error(
        std::string("specfem::mesh_entity::dim2::to_string does not handle ") +
        std::to_string(static_cast<int>(entity)));
    return "!ERR";
  }
}

std::vector<int> specfem::mesh_entity::nodes_on_orientation(
    const specfem::mesh_entity::dim2::type &entity) {
  switch (entity) {
  case specfem::mesh_entity::dim2::type::bottom:
    return { 0, 1 };
  case specfem::mesh_entity::dim2::type::right:
    return { 1, 2 };
  case specfem::mesh_entity::dim2::type::top:
    return { 3, 2 };
  case specfem::mesh_entity::dim2::type::left:
    return { 0, 3 };
  case specfem::mesh_entity::dim2::type::bottom_left:
    return { 0 };
  case specfem::mesh_entity::dim2::type::bottom_right:
    return { 1 };
  case specfem::mesh_entity::dim2::type::top_right:
    return { 2 };
  case specfem::mesh_entity::dim2::type::top_left:
    return { 3 };
  default:
    throw std::runtime_error("Invalid mesh entity type");
  }
}

specfem::mesh_entity::element<specfem::element::dimension_tag::dim2>::element(
    const int ngllz, const int ngllx)
    : base(ngllz, ngllx) {

  // corner coordinates
  corner_coordinates[specfem::mesh_entity::dim2::type::top_left] =
      std::make_tuple(ngllz - 1, 0);
  corner_coordinates[specfem::mesh_entity::dim2::type::top_right] =
      std::make_tuple(ngllz - 1, ngllx - 1);
  corner_coordinates[specfem::mesh_entity::dim2::type::bottom_right] =
      std::make_tuple(0, ngllx - 1);
  corner_coordinates[specfem::mesh_entity::dim2::type::bottom_left] =
      std::make_tuple(0, 0);

  // coordinates along edges
  // The element is defined as (z, x) in 2D
  // Where the diagram is as follows:
  // 3 --- 2
  // |     |
  // |     |
  // 0 --- 1

  edge_coordinates[specfem::mesh_entity::dim2::type::top] =
      [ngllx, ngllz](int point) { return std::make_tuple(ngllz - 1, point); };

  edge_coordinates[specfem::mesh_entity::dim2::type::bottom] =
      [ngllx, ngllz](int point) { return std::make_tuple(0, point); };

  edge_coordinates[specfem::mesh_entity::dim2::type::left] =
      [ngllx, ngllz](int point) { return std::make_tuple(point, 0); };

  edge_coordinates[specfem::mesh_entity::dim2::type::right] =
      [ngllx, ngllz](int point) { return std::make_tuple(point, ngllx - 1); };
}

specfem::mesh_entity::element<specfem::element::dimension_tag::dim2>::element(
    const int ngll)
    : element(ngll, ngll) {}

int specfem::mesh_entity::element<specfem::element::dimension_tag::dim2>::
    number_of_points_on_orientation(
        const specfem::mesh_entity::dim2::type &entity) const {
  if (specfem::mesh_entity::contains(specfem::mesh_entity::dim2::edges,
                                     entity)) {
    // edges
    return this->ngllx;
  } else if (specfem::mesh_entity::contains(specfem::mesh_entity::dim2::corners,
                                            entity)) {
    // corners
    return 1;
  } else {
    throw std::runtime_error("The argument is not a valid mesh entity");
  }
}

std::tuple<int, int>
specfem::mesh_entity::element<specfem::element::dimension_tag::dim2>::
    map_coordinates(const specfem::mesh_entity::dim2::type &entity,
                    const int point) const {
  if (specfem::mesh_entity::contains(specfem::mesh_entity::dim2::edges, entity))
    return edge_coordinates.at(entity)(point);

  if (specfem::mesh_entity::contains(specfem::mesh_entity::dim2::corners,
                                     entity)) {
    throw std::runtime_error(
        "For corner entities, use the map_coordinates overload that only takes "
        "the corner type.");
  }

  throw std::runtime_error("The argument is not a valid mesh entity");
}

std::tuple<int, int>
specfem::mesh_entity::element<specfem::element::dimension_tag::dim2>::
    map_coordinates(const specfem::mesh_entity::dim2::type &corner) const {
  if (specfem::mesh_entity::contains(specfem::mesh_entity::dim2::corners,
                                     corner)) {
    return corner_coordinates.at(corner);
  }

  throw std::runtime_error("The argument is not a corner entity");
}
