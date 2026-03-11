#pragma once

#include "specfem/constants.hpp"
#include "specfem/setup.hpp"
#include "specfem/utilities/frequency_band.hpp"

namespace specfem {
namespace attenuation {

/**
 * @brief Compute the attenuation period band from the minimum resolved period.
 *
 * The attenuation band defines the frequency range over which viscoelastic
 * attenuation is computed using standard linear solids (SLS). The minimum
 * attenuation period is set equal to @p min_resolved_period, while the maximum
 * period is determined by empirically optimized decade-width values.
 *
 * The maximum attenuation period is computed as:
 * @f[
 *   T_{\text{max}} = T_{\text{min}} \times 10^{\Theta(N\_SLS)}
 * @f]
 * where @f$\Theta@f$ are empirically determined decade-width values that
 * minimise the spectral flatness of the absorption band for each SLS count:
 *
 * | N_SLS | @f$\Theta@f$ | Max/Min Ratio |
 * |-------|--------------|---------------|
 * | 2     | 0.75        | ~5.6          |
 * | 3     | 1.75        | ~56           |
 * | 4     | 2.25        | ~178          |
 * | 5     | 2.85        | ~708          |
 *
 * @tparam N_SLS Number of standard linear solids (must be 2–5)
 * @param min_resolved_period Minimum period resolved by the mesh (seconds)
 * @return FrequencyBand containing the computed attenuation band limits
 *
 * @code
 * // Compute attenuation band for 3 SLS mechanisms
 * constexpr type_real min_period = 0.5; // seconds
 * auto band = specfem::attenuation::compute_band<3>(min_period);
 *
 * // Access band limits
 * type_real f_min = band.min_frequency();
 * type_real f_max = band.max_frequency();
 * type_real T_min = band.min_period();  // equals min_period
 * type_real T_max = band.max_period();  // min_period * 10^1.75
 * @endcode
 *
 * @see tests/unit-tests/attenuation/compute_band_tests.cpp for usage examples
 */
template <int N_SLS>
specfem::utilities::FrequencyBand compute_band(type_real min_resolved_period);

} // namespace attenuation
} // namespace specfem

#include "compute_band.tpp"
