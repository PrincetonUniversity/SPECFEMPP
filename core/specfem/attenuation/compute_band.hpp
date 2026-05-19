#pragma once

#include "specfem/constants.hpp"
#include "specfem/setup.hpp"
#include "specfem/units.hpp"
#include "specfem/utilities/band.hpp"

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
 * @param min_resolved_period Minimum period resolved by the mesh (s)
 * @return Band in Hertz with @c min and @c max
 */
template <int N_SLS>
specfem::utilities::Band<specfem::units::Hertz>
compute_band(specfem::units::Seconds min_resolved_period);

} // namespace attenuation
} // namespace specfem

#include "compute_band.tpp"
