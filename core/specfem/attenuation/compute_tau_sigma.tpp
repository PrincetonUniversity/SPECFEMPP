#pragma once
#include "specfem/constants.hpp"
#include "specfem/units.hpp"
#include "specfem/setup.hpp"
#include "specfem/utilities/band.hpp"
#include <Kokkos_Core.hpp>
#include <cmath>

namespace specfem {
namespace attenuation {

template <int N_SLS, typename T>
Kokkos::View<type_real[N_SLS], Kokkos::LayoutRight, Kokkos::HostSpace>
compute_tau_sigma(const specfem::utilities::Band<T> band) {
  static_assert(N_SLS > 1, "N_SLS must be greater than 1 to avoid division by zero");

  // If the band is not convertible to a frequency band, this will throw an
  // exception. This is intentional.
  const specfem::utilities::Band<specfem::units::Hertz> frequency_band(band);

  // Extract the minimum and maximum frequencies from the band
  type_real min_frequency = frequency_band.min.raw();
  type_real max_frequency = frequency_band.max.raw();

  // logarithms of the input frequencies
  const type_real exp1 = std::log10(min_frequency);
  const type_real exp2 = std::log10(max_frequency);

  // equally spaced in log10 frequency
  const type_real dexpval =
      (exp2 - exp1) / (static_cast<type_real>(N_SLS) - 1);

  Kokkos::View<type_real[N_SLS], Kokkos::LayoutRight, Kokkos::HostSpace> tau_s(
      "tau_sigma");

  for (int i = 0; i < N_SLS; ++i) {
    tau_s(i) = 1.0 / (specfem::constants::pi * 2.0 * std::pow(10.0, exp1 + i * dexpval));
  }

  return tau_s;
}

} // namespace attenuation
} // namespace specfem
