#include "specfem/element_connections.hpp"
#include "specfem/mesh_entities.hpp"
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

template <typename ViewType>
std::array<int, 2> get_edge_nodes(const specfem::mesh_entity::dim2::type &edge,
                                  const ViewType element) {
  if (specfem::mesh_entity::contains(specfem::mesh_entity::dim2::edges, edge)) {
    auto nodes = specfem::mesh_entity::nodes_on_orientation(edge);
    return { element(nodes[0]), element(nodes[1]) };
  }

  throw std::runtime_error("The provided entity is not an edge.");
}

int edge_transform(const std::array<int, 2> &from_nodes,
                   const std::array<int, 2> &to_nodes, const int index,
                   const int ngll) {
  if (from_nodes[0] == to_nodes[0] && from_nodes[1] == to_nodes[1]) {
    return index;
  } else if (from_nodes[0] == to_nodes[1] && from_nodes[1] == to_nodes[0]) {
    return ngll - 1 - index;
  } else {
    throw std::runtime_error("Edges do not match for transformation.");
  }
}

std::tuple<int, int> specfem::connections::
    connection_mapping<specfem::element::dimension_tag::dim2>::map_coordinates(
        const specfem::mesh_entity::dim2::type &from,
        const specfem::mesh_entity::dim2::type &to, const int iz,
        const int ix) const {

  // get nodes associated with edges
  const auto edge1_nodes = get_edge_nodes(from, element1);
  const auto edge2_nodes = get_edge_nodes(to, element2);

  const auto [i, n] = [=]() {
    switch (from) {
    case specfem::mesh_entity::dim2::type::bottom:
    case specfem::mesh_entity::dim2::type::top:
      return std::make_pair(ix, ngllx);
    case specfem::mesh_entity::dim2::type::left:
    case specfem::mesh_entity::dim2::type::right:
      return std::make_pair(iz, ngllz);
    default:
      throw std::runtime_error("Invalid edge orientation.");
    }
  }();

  const int i_prime = edge_transform(edge1_nodes, edge2_nodes, i, n);

  return [=](const int i_prime) {
    switch (to) {
    case specfem::mesh_entity::dim2::type::bottom:
      return std::make_tuple(0, i_prime);
    case specfem::mesh_entity::dim2::type::top:
      return std::make_tuple(ngllz - 1, i_prime);
    case specfem::mesh_entity::dim2::type::left:
      return std::make_tuple(i_prime, 0);
    case specfem::mesh_entity::dim2::type::right:
      return std::make_tuple(i_prime, ngllx - 1);
    default:
      throw std::runtime_error("Invalid edge orientation.");
    }
  }(i_prime);
}

std::tuple<int, int> specfem::connections::
    connection_mapping<specfem::element::dimension_tag::dim2>::map_coordinates(
        const specfem::mesh_entity::dim2::type &from,
        const specfem::mesh_entity::dim2::type &to) const {
  // Implementation of coordinate mapping logic for 2D entities without point
  // specification goes here
  if (!(specfem::mesh_entity::contains(specfem::mesh_entity::dim2::corners,
                                       from) &&
        specfem::mesh_entity::contains(specfem::mesh_entity::dim2::corners,
                                       to)))
    throw std::runtime_error("Both entities must be corners for this mapping.");

  return specfem::mesh_entity::element(ngllz, ngllx).map_coordinates(to);
}
