#pragma once
#include "compute_band.hpp"
#include "specfem/setup.hpp"
#include "specfem/utilities/band.hpp"
#include <cmath>

namespace specfem {
namespace attenuation {

template <int N_SLS>
specfem::utilities::Band<specfem::units::Hertz>
compute_band(const specfem::units::Seconds min_resolved_period) {
  static_assert(N_SLS >= 2 && N_SLS <= 5,
                "N_SLS must be between 2 and 5 (inclusive)");

  // Decade-width values minimising spectral flatness of the absorption band.
  // Index corresponds to N_SLS (THETA(N_SLS) in the original Fortran).
  constexpr double theta[6] = { 0.00, 0.00, 0.75, 1.75, 2.25, 2.85 };

  const specfem::units::Seconds max_period =
      min_resolved_period * static_cast<type_real>(std::pow(10.0, theta[N_SLS]));

  return specfem::utilities::Band<specfem::units::Hertz>(1.0/max_period, 1.0/min_resolved_period);
}

} // namespace attenuation
} // namespace specfem
