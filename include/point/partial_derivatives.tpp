#ifndef _POINT_PARTIAL_DERIVATIVES_TPP
#define _POINT_PARTIAL_DERIVATIVES_TPP

#include "enumerations/specfem_enums.hpp"
#include "kokkos_abstractions.h"
#include "partial_derivatives.hpp"
#include "specfem_setup.hpp"

template <>
KOKKOS_INLINE_FUNCTION specfem::kokkos::array_type<type_real, 2>
specfem::point::partial_derivatives2<true>::impl_compute_normal<
    specfem::enums::edge::type::BOTTOM>() const {
  specfem::kokkos::array_type<type_real, 2> dn;
  dn[0] = -1.0 * this->gammax * this->jacobian;
  dn[1] = -1.0 * this->gammaz * this->jacobian;
  return dn;
};

template <>
KOKKOS_INLINE_FUNCTION specfem::kokkos::array_type<type_real, 2>
specfem::point::partial_derivatives2<true>::impl_compute_normal<
    specfem::enums::edge::type::TOP>() const {
  specfem::kokkos::array_type<type_real, 2> dn;
  dn[0] = this->gammax * this->jacobian;
  dn[1] = this->gammaz * this->jacobian;
  return dn;
};

template <>
KOKKOS_INLINE_FUNCTION specfem::kokkos::array_type<type_real, 2>
specfem::point::partial_derivatives2<true>::impl_compute_normal<
    specfem::enums::edge::type::LEFT>() const {
  specfem::kokkos::array_type<type_real, 2> dn;
  dn[0] = -1.0 * this->xix * this->jacobian;
  dn[1] = -1.0 * this->xiz * this->jacobian;
  return dn;
};

template <>
KOKKOS_INLINE_FUNCTION specfem::kokkos::array_type<type_real, 2>
specfem::point::partial_derivatives2<true>::impl_compute_normal<
    specfem::enums::edge::type::RIGHT>() const {
  specfem::kokkos::array_type<type_real, 2> dn;
  dn[0] = this->xix * this->jacobian;
  dn[1] = this->xiz * this->jacobian;
  return dn;
};

#endif
