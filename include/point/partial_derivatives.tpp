#ifndef _POINT_PARTIAL_DERIVATIVES_TPP
#define _POINT_PARTIAL_DERIVATIVES_TPP

#include "enumerations/specfem_enums.hpp"
#include "kokkos_abstractions.h"
#include "partial_derivatives.hpp"
#include "specfem_setup.hpp"

template <>
KOKKOS_INLINE_FUNCTION specfem::datatype::ScalarPointViewType<type_real, 2>
specfem::point::partial_derivatives2<true>::impl_compute_normal<
    specfem::enums::edge::type::BOTTOM>() const {
  return { static_cast<type_real>(-1.0 * this->gammax * this->jacobian),
           static_cast<type_real>(-1.0 * this->gammaz * this->jacobian) };
};

template <>
KOKKOS_INLINE_FUNCTION specfem::datatype::ScalarPointViewType<type_real, 2>
specfem::point::partial_derivatives2<true>::impl_compute_normal<
    specfem::enums::edge::type::TOP>() const {
  return { static_cast<type_real>(this->gammax * this->jacobian),
           static_cast<type_real>(this->gammaz * this->jacobian) };
};

template <>
KOKKOS_INLINE_FUNCTION specfem::datatype::ScalarPointViewType<type_real, 2>
specfem::point::partial_derivatives2<true>::impl_compute_normal<
    specfem::enums::edge::type::LEFT>() const {
  return { static_cast<type_real>(-1.0 * this->xix * this->jacobian),
           static_cast<type_real>(-1.0 * this->xiz * this->jacobian) };
};

template <>
KOKKOS_INLINE_FUNCTION specfem::datatype::ScalarPointViewType<type_real, 2>
specfem::point::partial_derivatives2<true>::impl_compute_normal<
    specfem::enums::edge::type::RIGHT>() const {
  return { static_cast<type_real>(this->xix * this->jacobian),
           static_cast<type_real>(this->xiz * this->jacobian) };
};

#endif
