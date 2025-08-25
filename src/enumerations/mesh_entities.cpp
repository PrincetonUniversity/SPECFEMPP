#include "enumerations/mesh_entities.hpp"
#include <list>
#include <stdexcept>

std::list<specfem::mesh_entity::type> specfem::mesh_entity::edges_of_corner(
    const specfem::mesh_entity::type &corner) {
  switch (corner) {
  case specfem::mesh_entity::type::top_left:
    return { specfem::mesh_entity::type::top,
             specfem::mesh_entity::type::left };
  case specfem::mesh_entity::type::top_right:
    return { specfem::mesh_entity::type::top,
             specfem::mesh_entity::type::right };
  case specfem::mesh_entity::type::bottom_right:
    return { specfem::mesh_entity::type::bottom,
             specfem::mesh_entity::type::right };
  case specfem::mesh_entity::type::bottom_left:
    return { specfem::mesh_entity::type::bottom,
             specfem::mesh_entity::type::left };
  default:
    throw std::runtime_error("Invalid corner type");
  }
}

const std::string
specfem::mesh_entity::to_string(const specfem::mesh_entity::type &entity) {
  switch (entity) {
  case type::bottom:
    return "bottom";
  case type::right:
    return "right";
  case type::top:
    return "top";
  case type::left:
    return "left";
  case type::bottom_left:
    return "bottom_left";
  case type::bottom_right:
    return "bottom_right";
  case type::top_right:
    return "top_right";
  case type::top_left:
    return "top_left";
  default:
    throw std::runtime_error(
        std::string("specfem::mesh_entity::to_string does not handle ") +
        std::to_string(static_cast<int>(entity)));
    return "!ERR";
  }
}
