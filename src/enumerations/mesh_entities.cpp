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
