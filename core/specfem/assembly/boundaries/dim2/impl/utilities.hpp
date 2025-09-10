#pragma once

#include "enumerations/interface.hpp"

namespace {
bool is_on_boundary(specfem::mesh_entity::type type, int iz, int ix, int ngllz,
                    int ngllx) {
  return (type == specfem::mesh_entity::type::top && iz == ngllz - 1) ||
         (type == specfem::mesh_entity::type::bottom && iz == 0) ||
         (type == specfem::mesh_entity::type::left && ix == 0) ||
         (type == specfem::mesh_entity::type::right && ix == ngllx - 1) ||
         (type == specfem::mesh_entity::type::bottom_right && iz == 0 &&
          ix == ngllx - 1) ||
         (type == specfem::mesh_entity::type::bottom_left && iz == 0 &&
          ix == 0) ||
         (type == specfem::mesh_entity::type::top_right && iz == ngllz - 1 &&
          ix == ngllx - 1) ||
         (type == specfem::mesh_entity::type::top_left && iz == ngllz - 1 &&
          ix == 0);
}

std::tuple<std::array<type_real, 2>, type_real> get_boundary_edge_and_weight(
    specfem::mesh_entity::type type, const std::array<type_real, 2> &weights,
    const specfem::point::jacobian_matrix<specfem::dimension::type::dim2, true,
                                          false> &point_jacobian_matrix) {

  if (type == specfem::mesh_entity::type::bottom_left ||
      type == specfem::mesh_entity::type::top_left ||
      type == specfem::mesh_entity::type::left) {
    const auto normal =
        point_jacobian_matrix.compute_normal(specfem::mesh_entity::type::left);
    const std::array<type_real, 2> edge_normal = { normal(0), normal(1) };
    return std::make_tuple(edge_normal, weights[1]);
  }

  if (type == specfem::mesh_entity::type::bottom_right ||
      type == specfem::mesh_entity::type::top_right ||
      type == specfem::mesh_entity::type::right) {
    const auto normal =
        point_jacobian_matrix.compute_normal(specfem::mesh_entity::type::right);
    const std::array<type_real, 2> edge_normal = { normal(0), normal(1) };
    return std::make_tuple(edge_normal, weights[1]);
  }

  if (type == specfem::mesh_entity::type::top) {
    const auto normal =
        point_jacobian_matrix.compute_normal(specfem::mesh_entity::type::top);
    const std::array<type_real, 2> edge_normal = { normal(0), normal(1) };
    return std::make_tuple(edge_normal, weights[0]);
  }

  if (type == specfem::mesh_entity::type::bottom) {
    const auto normal = point_jacobian_matrix.compute_normal(
        specfem::mesh_entity::type::bottom);
    const std::array<type_real, 2> edge_normal = { normal(0), normal(1) };
    return std::make_tuple(edge_normal, weights[0]);
  }

  throw std::invalid_argument("Error: Unknown boundary type");
}
} // namespace
