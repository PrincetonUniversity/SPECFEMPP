#pragma once
#include "compute_band.hpp"
#include "specfem/setup.hpp"
#include "specfem/utilities/frequency_band.hpp"
#include <cmath>

namespace specfem {
namespace attenuation {

template <int N_SLS>
specfem::utilities::FrequencyBand<specfem::datatype::Omega>
compute_band(const specfem::datatype::Seconds min_resolved_period) {
  static_assert(N_SLS >= 2 && N_SLS <= 5,
                "N_SLS must be between 2 and 5 (inclusive)");

  // Decade-width values minimising spectral flatness of the absorption band.
  // Index corresponds to N_SLS (THETA(N_SLS) in the original Fortran).
  constexpr double theta[6] = { 0.00, 0.00, 0.75, 1.75, 2.25, 2.85 };

  const specfem::datatype::Seconds max_period =
      static_cast<specfem::datatype::Seconds>(min_resolved_period *
                                              std::pow(10.0, theta[N_SLS]));
  return specfem::utilities::FrequencyBand<specfem::datatype::Omega>{
    min_resolved_period, max_period
  };
}

} // namespace attenuation
} // namespace specfem
