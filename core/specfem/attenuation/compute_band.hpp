#pragma once

#include "specfem/constants.hpp"
#include "specfem/setup.hpp"
#include "specfem/units.hpp"
#include "specfem/utilities/frequency_band.hpp"

namespace specfem {
namespace attenuation {

/**
 * @brief Compute the attenuation period band from the minimum resolved period.
 *
 * The minimum attenuation period is set equal to @p min_resolved_period.
 * The maximum attenuation period is
 * @f[
 *   T_{\text{max}} = T_{\text{min}} \times 10^{\Theta(N\_SLS)}
 * @f]
 * where @f$\Theta@f$ are empirically determined decade-width values that
 * minimise the spectral flatness of the absorption band for each SLS count:
 *
 * | N_SLS | @f$\Theta@f$ |
 * |-------|-------------|
 * | 2     | 0.75        |
 * | 3     | 1.75        |
 * | 4     | 2.25        |
 * | 5     | 2.85        |
 *
 * @tparam N_SLS Number of standard linear solids (must be 2–5)
 * @param min_resolved_period Minimum period resolved by the mesh (s)
 * @return FrequencyBand with @c min_period and @c max_period
 */
template <int N_SLS>
specfem::utilities::FrequencyBand<specfem::units::Omega>
compute_band(specfem::units::Seconds min_resolved_period);

} // namespace attenuation
} // namespace specfem

#include "compute_band.tpp"
