#include "specfem_setup.hpp"
#include <Kokkos_Core.hpp>
#include <cmath>

KOKKOS_INLINE_FUNCTION
type_real gaussian(type_real t, type_real f0) {
  // Gaussian wavelet i.e. second integral of a Ricker wavelet
  constexpr auto pi = Kokkos::numbers::pi_v<type_real>;
  type_real a = pi * pi * f0 * f0;
  type_real gaussian = -1.0 * Kokkos::exp(-a * t * t) / (2.0 * a);

  return gaussian;
}

KOKKOS_INLINE_FUNCTION
type_real d2gaussian(type_real t, type_real f0) {
  constexpr auto pi = Kokkos::numbers::pi_v<type_real>;

  type_real a = pi * pi * f0 * f0;
  type_real d2gaussian = (1.0 - 2.0 * a * t * t) * Kokkos::exp(-a * t * t);

  return d2gaussian;
}
