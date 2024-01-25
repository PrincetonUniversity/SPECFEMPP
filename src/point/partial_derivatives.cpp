
#include "point/partial_derivatives.hpp"
#include <Kokkos_Core.hpp>

// operator+
KOKKOS_FUNCTION
specfem::point::partial_derivatives2
operator+(const specfem::point::partial_derivatives2 &lhs,
          const specfem::point::partial_derivatives2 &rhs) {
  return specfem::point::partial_derivatives2(
      lhs.xix + rhs.xix, lhs.gammax + rhs.gammax, lhs.xiz + rhs.xiz,
      lhs.gammaz + rhs.gammaz);
}

// operator+=
KOKKOS_FUNCTION
specfem::point::partial_derivatives2 &
operator+=(specfem::point::partial_derivatives2 &lhs,
           const specfem::point::partial_derivatives2 &rhs) {
  lhs.xix += rhs.xix;
  lhs.gammax += rhs.gammax;
  lhs.xiz += rhs.xiz;
  lhs.gammaz += rhs.gammaz;
  return lhs;
}

// operator*
KOKKOS_FUNCTION
specfem::point::partial_derivatives2
operator*(const type_real &lhs,
          const specfem::point::partial_derivatives2 &rhs) {
  return specfem::point::partial_derivatives2(lhs * rhs.xix, lhs * rhs.gammax,
                                              lhs * rhs.xiz, lhs * rhs.gammaz);
}

// operator*
KOKKOS_FUNCTION
specfem::point::partial_derivatives2
operator*(const specfem::point::partial_derivatives2 &lhs,
          const type_real &rhs) {
  return specfem::point::partial_derivatives2(lhs.xix * rhs, lhs.gammax * rhs,
                                              lhs.xiz * rhs, lhs.gammaz * rhs);
}
