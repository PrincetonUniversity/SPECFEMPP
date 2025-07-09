#pragma once

#include "jacobian_matrix.hpp"

template <bool UseSIMD>
specfem::datatype::VectorPointViewType<type_real, 2, UseSIMD>
specfem::point::jacobian_matrix<
    specfem::dimension::type::dim2, true,
    UseSIMD>::compute_normal(const specfem::enums::edge::type &type) const {
  switch (type) {
  case specfem::enums::edge::type::BOTTOM:
    return this->impl_compute_normal_bottom();
  case specfem::enums::edge::type::TOP:
    return this->impl_compute_normal_top();
  case specfem::enums::edge::type::LEFT:
    return this->impl_compute_normal_left();
  case specfem::enums::edge::type::RIGHT:
    return this->impl_compute_normal_right();
  default:
    return this->impl_compute_normal_bottom();
  }
}
