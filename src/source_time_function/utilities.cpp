#include "specfem_setup.hpp"
#include <Kokkos_Core.hpp>
#include <cmath>

KOKKOS_INLINE_FUNCTION
type_real gaussian(const type_real t, const type_real f0) {
  // Gaussian wavelet i.e. second integral of a Ricker wavelet
  constexpr auto pi = Kokkos::numbers::pi_v<type_real>;
  type_real a = pi * pi * f0 * f0;
  type_real gaussian = -1.0 * Kokkos::exp(-a * t * t) / (2.0 * a);

  return gaussian;
}

KOKKOS_INLINE_FUNCTION
type_real d2gaussian(const type_real t, const type_real f0) {
  constexpr auto pi = Kokkos::numbers::pi_v<type_real>;

  type_real a = pi * pi * f0 * f0;
  type_real d2gaussian = (1.0 - 2.0 * a * t * t) * Kokkos::exp(-a * t * t);

  return d2gaussian;
}

KOKKOS_INLINE_FUNCTION
type_real d4gaussian(const type_real t, const type_real f0) {

  constexpr auto pi = Kokkos::numbers::pi_v<type_real>;

  type_real a = pi * pi * f0 * f0;
  // d4gaussian = - 2.d0 * a * (3.d0 - 12.d0 * a * t*t + 4.d0 * a**2 * t*t*t*t)
  // * exp( -a * t*t )

  type_real d4gaussian =
      -2.0 * a * (3.0 - 12.0 * a * t * t + 4.0 * a * a * t * t * t * t) *
      Kokkos::exp(-a * t * t);

  return d4gaussian;
}
