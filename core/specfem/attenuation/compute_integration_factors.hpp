#pragma once

#include "specfem/constants.hpp"
#include "specfem/setup.hpp"
#include <Kokkos_Core.hpp>

namespace specfem {
namespace attenuation {

/**
 * @brief Runge-Kutta memory-variable update coefficients per SLS mechanism.
 *
 * Holds the three coefficient arrays @f$ \alpha_j @f$, @f$ \beta_j @f$,
 * @f$ \gamma_j @f$ that advance each standard linear solid one time step
 * via the low-storage fourth-order Runge-Kutta scheme of
 * Savage et al. (BSSA 2010, eq. 11).
 *
 * The coefficients are used to update memory variables in the attenuation
 * implementation, providing stable time integration of the viscoelastic
 * stress-strain relationships.
 *
 * @tparam N_SLS Number of standard linear solids
 *
 * @code
 * // Example usage with computed integration factors
 * constexpr int N_SLS = 3;
 * auto factors = compute_integration_factors<N_SLS>(tau_sigma, dt);
 *
 * // Access coefficient arrays
 * for (int j = 0; j < N_SLS; ++j) {
 *   type_real alpha_j = factors.alpha(j);
 *   type_real beta_j  = factors.beta(j);
 *   type_real gamma_j = factors.gamma(j);
 * }
 * @endcode
 */
template <int N_SLS> struct IntegrationFactors {
  Kokkos::View<type_real[N_SLS], Kokkos::LayoutRight, Kokkos::HostSpace>
      alpha; ///< Fourth-order alpha coefficients
  Kokkos::View<type_real[N_SLS], Kokkos::LayoutRight, Kokkos::HostSpace>
      beta; ///< Fourth-order beta coefficients
  Kokkos::View<type_real[N_SLS], Kokkos::LayoutRight, Kokkos::HostSpace>
      gamma; ///< Fourth-order gamma coefficients
};

/**
 * @brief Compute Runge-Kutta coefficients for memory-variable time stepping.
 *
 * Computes the fourth-order Runge-Kutta integration coefficients used to
 * advance memory variables for viscoelastic attenuation. This function
 * implements the computation of the integration coefficients (Savage et al.
 * BSSA 2010, eq. 11).
 *
 * For each SLS mechanism @f$ j @f$ with stress relaxation time
 * @f$ \tau_{\sigma,j} @f$, the coefficients are computed as:
 * @f[ \alpha_j = 1 - \frac{\Delta t}{\tau_{\sigma,j}}
 *   + \frac{(\Delta t)^2}{2\,\tau_{\sigma,j}^2}
 *   - \frac{(\Delta t)^3}{6\,\tau_{\sigma,j}^3}
 *   + \frac{(\Delta t)^4}{24\,\tau_{\sigma,j}^4}
 * @f]
 * @f[ \beta_j  = \frac{\Delta t}{2}
 *   - \frac{(\Delta t)^2}{3\,\tau_{\sigma,j}}
 *   + \frac{(\Delta t)^3}{8\,\tau_{\sigma,j}^2}
 *   - \frac{(\Delta t)^4}{24\,\tau_{\sigma,j}^3}
 * @f]
 * @f[ \gamma_j = \frac{\Delta t}{2}
 *   - \frac{(\Delta t)^2}{6\,\tau_{\sigma,j}}
 *   + \frac{(\Delta t)^3}{24\,\tau_{\sigma,j}^2}
 * @f]
 *
 * This notation is a departure from the Fortran implementation (and Savage et
 * al. 2010), which sets the `tauinv` variable to @f$ - 1 / \tau_{\sigma,j} @f$
 * so that the sign switch is absorbed. Here, we follow a more standard
 * mathematical convention where @f$ 1/\tau_{\sigma,j} @f$ remains positive, and
 * the negative signs are explicitly included in the formula for the
 * coefficients.
 *
 * @tparam N_SLS Number of standard linear solids
 * @param tau_sigma Stress relaxation times @f$ \tau_\sigma @f$ (positive
 * values, seconds)
 * @param deltat    Time step @f$ \Delta t @f$ (seconds)
 * @return IntegrationFactors containing the three coefficient arrays
 *
 * @code
 * // Setup relaxation times and time step constexpr int N_SLS = 3;
 * Kokkos::View<type_real[N_SLS], Kokkos::LayoutRight, Kokkos::HostSpace>
 * tau_sigma("tau_sigma"); tau_sigma(0) = 1.0;  // 1 second tau_sigma(1) = 2.0;
 * // 2 seconds
 * tau_sigma(2) = 5.0;  // 5 seconds
 *
 * type_real dt = 0.01; // 10 ms time step
 *
 * // Compute integration factors auto factors =
 * specfem::attenuation::compute_integration_factors<N_SLS>( tau_sigma, dt);
 *
 * // Use coefficients in time stepping for (int j = 0; j < N_SLS; ++j) { //
 * Apply Runge-Kutta update using factors.alpha(j), beta(j), gamma(j)
 * }
 * @endcode
 *
 * @see tests/unit-tests/attenuation/compute_integration_factors_tests.cpp for
 * detailed examples
 */
template <int N_SLS>
IntegrationFactors<N_SLS> compute_integration_factors(
    Kokkos::View<type_real[N_SLS], Kokkos::LayoutRight, Kokkos::HostSpace>
        tau_sigma,
    type_real deltat);

} // namespace attenuation
} // namespace specfem

#include "compute_integration_factors.tpp"
