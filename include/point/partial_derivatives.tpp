#ifndef _POINT_PARTIAL_DERIVATIVES_TPP
#define _POINT_PARTIAL_DERIVATIVES_TPP

#include "enumerations/specfem_enums.hpp"
#include "kokkos_abstractions.h"
#include "partial_derivatives.hpp"
#include "specfem_setup.hpp"

// template <bool UseSIMD>
// KOKKOS_FUNCTION specfem::point::partial_derivatives2<UseSIMD, false>
// specfem::point::operator+(
//     const specfem::point::partial_derivatives2<UseSIMD, false> &lhs,
//     const specfem::point::partial_derivatives2<UseSIMD, false> &rhs) {
//   return specfem::point::partial_derivatives2<UseSIMD, false>(
//       lhs.xix + rhs.xix, lhs.gammax + rhs.gammax, lhs.xiz + rhs.xiz,
//       lhs.gammaz + rhs.gammaz);
// }

// template <bool UseSIMD>
// KOKKOS_FUNCTION specfem::point::partial_derivatives2<UseSIMD, false> &
// specfem::point::operator+=(
//     specfem::point::partial_derivatives2<UseSIMD, false> &lhs,
//     const specfem::point::partial_derivatives2<UseSIMD, false> &rhs) {
//   lhs.xix += rhs.xix;
//   lhs.gammax += rhs.gammax;
//   lhs.xiz += rhs.xiz;
//   lhs.gammaz += rhs.gammaz;
//   return lhs;
// }

// template <bool UseSIMD>
// KOKKOS_FUNCTION specfem::point::partial_derivatives2<UseSIMD, false>
// specfem::point::operator*(
//     const type_real &lhs,
//     const specfem::point::partial_derivatives2<UseSIMD, false> &rhs) {
//   return specfem::point::partial_derivatives2<UseSIMD, false>(
//       lhs * rhs.xix, lhs * rhs.gammax, lhs * rhs.xiz, lhs * rhs.gammaz);
// }

// template <bool UseSIMD>
// KOKKOS_FUNCTION specfem::point::partial_derivatives2<UseSIMD, false>
// specfem::point::operator*(
//     const specfem::point::partial_derivatives2<UseSIMD, false> &lhs,
//     const type_real &rhs) {
//   return specfem::point::partial_derivatives2<UseSIMD, false>(
//       lhs.xix * rhs, lhs.gammax * rhs, lhs.xiz * rhs, lhs.gammaz * rhs);
// }

template <bool UseSIMD>
KOKKOS_FUNCTION specfem::datatype::ScalarPointViewType<type_real, 2, UseSIMD>
specfem::point::partial_derivatives2<UseSIMD, true>::compute_normal(
    const specfem::enums::edge::type &type) const {
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

template <bool UseSIMD>
KOKKOS_FUNCTION specfem::datatype::ScalarPointViewType<type_real, 2, UseSIMD>
specfem::point::partial_derivatives2<UseSIMD, true>::compute_normal(
    const specfem::edge::interface &interface) const {
  return this->compute_normal(interface.type);
}

#endif
