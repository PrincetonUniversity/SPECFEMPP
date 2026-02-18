#pragma once

#include "specfem/setup.hpp"
#include <Kokkos_Core.hpp>
#include <cmath>

namespace specfem::utilities {
/**
 * @brief Compute logarithmically spaced values between min and
 * max
 *
 * @tparam N NUmber of points to generate
 * @param min Minimum value (must be > 0)
 * @param max Maximum value (must be > 0)
 * @return Kokkos::View<type_real[N], Kokkos::LayoutRight, Kokkos::HostSpace>
 */
template <int N>
Kokkos::View<type_real[N], Kokkos::LayoutRight, Kokkos::HostSpace>
logspace(type_real min, type_real max) {

  if (min <= 0 || max <= 0) {
    throw std::invalid_argument(
        "Min and max must be greater than 0 for logspace.");
  }

  Kokkos::View<type_real[N], Kokkos::LayoutRight, Kokkos::HostSpace> f(
      "logspace_values");
  const type_real log_f1 = std::log10(min);
  const type_real log_f2 = std::log10(max);
  const type_real d_log_f = (log_f2 - log_f1) / (static_cast<type_real>(N) - 1);

  for (int i = 0; i < N; ++i) {
    f(i) = std::pow(10.0, log_f1 + i * d_log_f);
  }

  return f;
}

} // namespace specfem::utilities
