#pragma once

#include "jacobian_matrix.hpp"

template <bool UseSIMD>
specfem::datatype::VectorPointViewType<type_real, 2, UseSIMD>
specfem::point::jacobian_matrix<
    specfem::dimension::type::dim2, true,
    UseSIMD>::compute_normal(const specfem::mesh_entity::dim2::type &type) const {
  switch (type) {
  case specfem::mesh_entity::dim2::type::bottom:
    return this->impl_compute_normal_bottom();
  case specfem::mesh_entity::dim2::type::top:
    return this->impl_compute_normal_top();
  case specfem::mesh_entity::dim2::type::left:
    return this->impl_compute_normal_left();
  case specfem::mesh_entity::dim2::type::right:
    return this->impl_compute_normal_right();
  default:
    return this->impl_compute_normal_bottom();
  }
}

template <bool UseSIMD>
specfem::datatype::VectorPointViewType<type_real, 3, UseSIMD>
specfem::point::jacobian_matrix<
    specfem::dimension::type::dim3, true,
    UseSIMD>::compute_normal(const specfem::mesh_entity::dim3::type &type) const {
  // For 3D, we handle all the face types (including edges and corners)
  // by returning the normal for the primary face direction

  // Bottom faces and related edges/corners
  if (type == specfem::mesh_entity::dim3::type::bottom ||
      type == specfem::mesh_entity::dim3::type::bottom_left ||
      type == specfem::mesh_entity::dim3::type::bottom_right ||
      type == specfem::mesh_entity::dim3::type::bottom_front_left ||
      type == specfem::mesh_entity::dim3::type::bottom_front_right ||
      type == specfem::mesh_entity::dim3::type::bottom_back_left ||
      type == specfem::mesh_entity::dim3::type::bottom_back_right ||
      type == specfem::mesh_entity::dim3::type::front_bottom ||
      type == specfem::mesh_entity::dim3::type::back_bottom) {
    return this->impl_compute_normal_bottom();
  }

  // Top faces and related edges/corners
  if (type == specfem::mesh_entity::dim3::type::top ||
      type == specfem::mesh_entity::dim3::type::top_left ||
      type == specfem::mesh_entity::dim3::type::top_right ||
      type == specfem::mesh_entity::dim3::type::top_front_left ||
      type == specfem::mesh_entity::dim3::type::top_front_right ||
      type == specfem::mesh_entity::dim3::type::top_back_left ||
      type == specfem::mesh_entity::dim3::type::top_back_right ||
      type == specfem::mesh_entity::dim3::type::front_top ||
      type == specfem::mesh_entity::dim3::type::back_top) {
    return this->impl_compute_normal_top();
  }

  // Left faces and related edges/corners
  if (type == specfem::mesh_entity::dim3::type::left ||
      type == specfem::mesh_entity::dim3::type::front_left ||
      type == specfem::mesh_entity::dim3::type::back_left) {
    return this->impl_compute_normal_left();
  }

  // Right faces and related edges/corners
  if (type == specfem::mesh_entity::dim3::type::right ||
      type == specfem::mesh_entity::dim3::type::front_right ||
      type == specfem::mesh_entity::dim3::type::back_right) {
    return this->impl_compute_normal_right();
  }

  // Front faces
  if (type == specfem::mesh_entity::dim3::type::front) {
    return this->impl_compute_normal_front();
  }

  // Back faces
  if (type == specfem::mesh_entity::dim3::type::back) {
    return this->impl_compute_normal_back();
  }

  // Default case (should not be reached)
  return this->impl_compute_normal_bottom();
}
