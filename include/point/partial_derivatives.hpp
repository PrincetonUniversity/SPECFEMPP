#ifndef _POINT_PARTIAL_DERIVATIVES_HPP
#define _POINT_PARTIAL_DERIVATIVES_HPP

#include "enumerations/specfem_enums.hpp"
#include "kokkos_abstractions.h"
#include "macros.hpp"
#include "specfem_setup.hpp"

namespace specfem {
namespace point {

struct partial_derivatives2 {
  type_real xix;
  type_real gammax;
  type_real xiz;
  type_real gammaz;
  type_real jacobian;

  KOKKOS_FUNCTION
  partial_derivatives2() = default;

  KOKKOS_FUNCTION
  partial_derivatives2(const type_real &xix, const type_real &gammax,
                       const type_real &xiz, const type_real &gammaz)
      : xix(xix), gammax(gammax), xiz(xiz), gammaz(gammaz) {}

  KOKKOS_FUNCTION
  partial_derivatives2(const type_real &xix, const type_real &gammax,
                       const type_real &xiz, const type_real &gammaz,
                       const type_real &jacobian)
      : xix(xix), gammax(gammax), xiz(xiz), gammaz(gammaz), jacobian(jacobian) {
  }

  template <specfem::enums::boundaries::type type>
  KOKKOS_INLINE_FUNCTION specfem::kokkos::array_type<type_real, 2>
  compute_normal() const {
    ASSERT(false, "Invalid boundary type");
    return specfem::kokkos::array_type<type_real, 2>();
  };
};
} // namespace point
} // namespace specfem

#endif
