#pragma once

#include "enumerations/interface.hpp"

namespace {
bool is_on_boundary(specfem::mesh_entity::dim3::type type, int iz, int iy,
                    int ix, int ngllz, int nglly, int ngllx) {
  return (type == specfem::mesh_entity::dim3::type::top && iz == ngllz - 1) ||
         (type == specfem::mesh_entity::dim3::type::bottom && iz == 0) ||
         (type == specfem::mesh_entity::dim3::type::left && ix == 0) ||
         (type == specfem::mesh_entity::dim3::type::right && ix == ngllx - 1) ||
         (type == specfem::mesh_entity::dim3::type::front && iy == nglly - 1) ||
         (type == specfem::mesh_entity::dim3::type::back && iy == 0) ||
         // Edges
         (type == specfem::mesh_entity::dim3::type::bottom_left && iz == 0 &&
          ix == 0) ||
         (type == specfem::mesh_entity::dim3::type::bottom_right && iz == 0 &&
          ix == ngllx - 1) ||
         (type == specfem::mesh_entity::dim3::type::top_right &&
          iz == ngllz - 1 && ix == ngllx - 1) ||
         (type == specfem::mesh_entity::dim3::type::top_left &&
          iz == ngllz - 1 && ix == 0) ||
         (type == specfem::mesh_entity::dim3::type::front_bottom &&
          iy == nglly - 1 && iz == 0) ||
         (type == specfem::mesh_entity::dim3::type::front_top &&
          iy == nglly - 1 && iz == ngllz - 1) ||
         (type == specfem::mesh_entity::dim3::type::front_left &&
          iy == nglly - 1 && ix == 0) ||
         (type == specfem::mesh_entity::dim3::type::front_right &&
          iy == nglly - 1 && ix == ngllx - 1) ||
         (type == specfem::mesh_entity::dim3::type::back_bottom && iy == 0 &&
          iz == 0) ||
         (type == specfem::mesh_entity::dim3::type::back_top && iy == 0 &&
          iz == ngllz - 1) ||
         (type == specfem::mesh_entity::dim3::type::back_left && iy == 0 &&
          ix == 0) ||
         (type == specfem::mesh_entity::dim3::type::back_right && iy == 0 &&
          ix == ngllx - 1) ||
         // Corners
         (type == specfem::mesh_entity::dim3::type::bottom_front_left &&
          iz == 0 && iy == nglly - 1 && ix == 0) ||
         (type == specfem::mesh_entity::dim3::type::bottom_front_right &&
          iz == 0 && iy == nglly - 1 && ix == ngllx - 1) ||
         (type == specfem::mesh_entity::dim3::type::bottom_back_left &&
          iz == 0 && iy == 0 && ix == 0) ||
         (type == specfem::mesh_entity::dim3::type::bottom_back_right &&
          iz == 0 && iy == 0 && ix == ngllx - 1) ||
         (type == specfem::mesh_entity::dim3::type::top_front_left &&
          iz == ngllz - 1 && iy == nglly - 1 && ix == 0) ||
         (type == specfem::mesh_entity::dim3::type::top_front_right &&
          iz == ngllz - 1 && iy == nglly - 1 && ix == ngllx - 1) ||
         (type == specfem::mesh_entity::dim3::type::top_back_left &&
          iz == ngllz - 1 && iy == 0 && ix == 0) ||
         (type == specfem::mesh_entity::dim3::type::top_back_right &&
          iz == ngllz - 1 && iy == 0 && ix == ngllx - 1);
}

std::tuple<std::array<type_real, 3>, type_real> get_boundary_face_and_weight(
    specfem::mesh_entity::dim3::type type,
    const std::array<type_real, 3> &weights,
    const specfem::point::jacobian_matrix<specfem::dimension::type::dim3, true,
                                          false> &point_jacobian_matrix) {

  if (type == specfem::mesh_entity::dim3::type::bottom_front_left ||
      type == specfem::mesh_entity::dim3::type::bottom_back_left ||
      type == specfem::mesh_entity::dim3::type::top_front_left ||
      type == specfem::mesh_entity::dim3::type::top_back_left ||
      type == specfem::mesh_entity::dim3::type::front_left ||
      type == specfem::mesh_entity::dim3::type::back_left ||
      type == specfem::mesh_entity::dim3::type::bottom_left ||
      type == specfem::mesh_entity::dim3::type::top_left ||
      type == specfem::mesh_entity::dim3::type::left) {
    const auto normal = point_jacobian_matrix.compute_normal(
        specfem::mesh_entity::dim3::type::left);
    const std::array<type_real, 3> face_normal = { normal(0), normal(1),
                                                   normal(2) };
    return std::make_tuple(face_normal, weights[1] * weights[2]);
  }

  if (type == specfem::mesh_entity::dim3::type::bottom_front_right ||
      type == specfem::mesh_entity::dim3::type::bottom_back_right ||
      type == specfem::mesh_entity::dim3::type::top_front_right ||
      type == specfem::mesh_entity::dim3::type::top_back_right ||
      type == specfem::mesh_entity::dim3::type::front_right ||
      type == specfem::mesh_entity::dim3::type::back_right ||
      type == specfem::mesh_entity::dim3::type::bottom_right ||
      type == specfem::mesh_entity::dim3::type::top_right ||
      type == specfem::mesh_entity::dim3::type::right) {
    const auto normal = point_jacobian_matrix.compute_normal(
        specfem::mesh_entity::dim3::type::right);
    const std::array<type_real, 3> face_normal = { normal(0), normal(1),
                                                   normal(2) };
    return std::make_tuple(face_normal, weights[1] * weights[2]);
  }

  if (type == specfem::mesh_entity::dim3::type::top_front_left ||
      type == specfem::mesh_entity::dim3::type::top_front_right ||
      type == specfem::mesh_entity::dim3::type::top_back_left ||
      type == specfem::mesh_entity::dim3::type::top_back_right ||
      type == specfem::mesh_entity::dim3::type::front_top ||
      type == specfem::mesh_entity::dim3::type::back_top ||
      type == specfem::mesh_entity::dim3::type::top_left ||
      type == specfem::mesh_entity::dim3::type::top_right ||
      type == specfem::mesh_entity::dim3::type::top) {
    const auto normal = point_jacobian_matrix.compute_normal(
        specfem::mesh_entity::dim3::type::top);
    const std::array<type_real, 3> face_normal = { normal(0), normal(1),
                                                   normal(2) };
    return std::make_tuple(face_normal, weights[0] * weights[2]);
  }

  if (type == specfem::mesh_entity::dim3::type::bottom_front_left ||
      type == specfem::mesh_entity::dim3::type::bottom_front_right ||
      type == specfem::mesh_entity::dim3::type::bottom_back_left ||
      type == specfem::mesh_entity::dim3::type::bottom_back_right ||
      type == specfem::mesh_entity::dim3::type::front_bottom ||
      type == specfem::mesh_entity::dim3::type::back_bottom ||
      type == specfem::mesh_entity::dim3::type::bottom_left ||
      type == specfem::mesh_entity::dim3::type::bottom_right ||
      type == specfem::mesh_entity::dim3::type::bottom) {
    const auto normal = point_jacobian_matrix.compute_normal(
        specfem::mesh_entity::dim3::type::bottom);
    const std::array<type_real, 3> face_normal = { normal(0), normal(1),
                                                   normal(2) };
    return std::make_tuple(face_normal, weights[0] * weights[2]);
  }

  if (type == specfem::mesh_entity::dim3::type::bottom_front_left ||
      type == specfem::mesh_entity::dim3::type::bottom_front_right ||
      type == specfem::mesh_entity::dim3::type::top_front_left ||
      type == specfem::mesh_entity::dim3::type::top_front_right ||
      type == specfem::mesh_entity::dim3::type::front_left ||
      type == specfem::mesh_entity::dim3::type::front_right ||
      type == specfem::mesh_entity::dim3::type::front_bottom ||
      type == specfem::mesh_entity::dim3::type::front_top ||
      type == specfem::mesh_entity::dim3::type::front) {
    const auto normal = point_jacobian_matrix.compute_normal(
        specfem::mesh_entity::dim3::type::front);
    const std::array<type_real, 3> face_normal = { normal(0), normal(1),
                                                   normal(2) };
    return std::make_tuple(face_normal, weights[0] * weights[1]);
  }

  if (type == specfem::mesh_entity::dim3::type::bottom_back_left ||
      type == specfem::mesh_entity::dim3::type::bottom_back_right ||
      type == specfem::mesh_entity::dim3::type::top_back_left ||
      type == specfem::mesh_entity::dim3::type::top_back_right ||
      type == specfem::mesh_entity::dim3::type::back_left ||
      type == specfem::mesh_entity::dim3::type::back_right ||
      type == specfem::mesh_entity::dim3::type::back_bottom ||
      type == specfem::mesh_entity::dim3::type::back_top ||
      type == specfem::mesh_entity::dim3::type::back) {
    const auto normal = point_jacobian_matrix.compute_normal(
        specfem::mesh_entity::dim3::type::back);
    const std::array<type_real, 3> face_normal = { normal(0), normal(1),
                                                   normal(2) };
    return std::make_tuple(face_normal, weights[0] * weights[1]);
  }

  throw std::invalid_argument("Error: Unknown boundary type");
}
} // namespace
