#pragma once

#include "jacobian_matrix.hpp"

template <bool UseSIMD>
specfem::datatype::VectorPointViewType<type_real, 2, UseSIMD>
specfem::point::jacobian_matrix<
    specfem::dimension::type::dim2, true,
    UseSIMD>::compute_normal(const specfem::mesh_entity::type &type) const {
  switch (type) {
  case specfem::mesh_entity::type::bottom:
    return this->impl_compute_normal_bottom();
  case specfem::mesh_entity::type::top:
    return this->impl_compute_normal_top();
  case specfem::mesh_entity::type::left:
    return this->impl_compute_normal_left();
  case specfem::mesh_entity::type::right:
    return this->impl_compute_normal_right();
  default:
    return this->impl_compute_normal_bottom();
  }
}
