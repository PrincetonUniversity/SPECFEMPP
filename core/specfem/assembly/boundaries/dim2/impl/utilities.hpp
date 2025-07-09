#pragma once

#include "enumerations/interface.hpp"

namespace {
bool is_on_boundary(specfem::enums::boundaries::type type, int iz, int ix,
                    int ngllz, int ngllx) {
  return (type == specfem::enums::boundaries::type::TOP && iz == ngllz - 1) ||
         (type == specfem::enums::boundaries::type::BOTTOM && iz == 0) ||
         (type == specfem::enums::boundaries::type::LEFT && ix == 0) ||
         (type == specfem::enums::boundaries::type::RIGHT && ix == ngllx - 1) ||
         (type == specfem::enums::boundaries::type::BOTTOM_RIGHT && iz == 0 &&
          ix == ngllx - 1) ||
         (type == specfem::enums::boundaries::type::BOTTOM_LEFT && iz == 0 &&
          ix == 0) ||
         (type == specfem::enums::boundaries::type::TOP_RIGHT &&
          iz == ngllz - 1 && ix == ngllx - 1) ||
         (type == specfem::enums::boundaries::type::TOP_LEFT &&
          iz == ngllz - 1 && ix == 0);
}

std::tuple<std::array<type_real, 2>, type_real> get_boundary_edge_and_weight(
    specfem::enums::boundaries::type type,
    const std::array<type_real, 2> &weights,
    const specfem::point::jacobian_matrix<specfem::dimension::type::dim2, true,
                                          false> &point_jacobian_matrix) {

  if (type == specfem::enums::boundaries::type::BOTTOM_LEFT ||
      type == specfem::enums::boundaries::type::TOP_LEFT ||
      type == specfem::enums::boundaries::type::LEFT) {
    const auto normal =
        point_jacobian_matrix.compute_normal(specfem::enums::edge::type::LEFT);
    const std::array<type_real, 2> edge_normal = { normal(0), normal(1) };
    return std::make_tuple(edge_normal, weights[1]);
  }

  if (type == specfem::enums::boundaries::type::BOTTOM_RIGHT ||
      type == specfem::enums::boundaries::type::TOP_RIGHT ||
      type == specfem::enums::boundaries::type::RIGHT) {
    const auto normal =
        point_jacobian_matrix.compute_normal(specfem::enums::edge::type::RIGHT);
    const std::array<type_real, 2> edge_normal = { normal(0), normal(1) };
    return std::make_tuple(edge_normal, weights[1]);
  }

  if (type == specfem::enums::boundaries::type::TOP) {
    const auto normal =
        point_jacobian_matrix.compute_normal(specfem::enums::edge::type::TOP);
    const std::array<type_real, 2> edge_normal = { normal(0), normal(1) };
    return std::make_tuple(edge_normal, weights[0]);
  }

  if (type == specfem::enums::boundaries::type::BOTTOM) {
    const auto normal = point_jacobian_matrix.compute_normal(
        specfem::enums::edge::type::BOTTOM);
    const std::array<type_real, 2> edge_normal = { normal(0), normal(1) };
    return std::make_tuple(edge_normal, weights[0]);
  }

  throw std::invalid_argument("Error: Unknown boundary type");
}
} // namespace
