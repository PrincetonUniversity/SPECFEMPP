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
 * @tparam N_SLS Number of standard linear solids
 */
template <int N_SLS> struct IntegrationFactors {
  Kokkos::View<type_real[N_SLS], Kokkos::LayoutRight, Kokkos::HostSpace> alpha;
  Kokkos::View<type_real[N_SLS], Kokkos::LayoutRight, Kokkos::HostSpace> beta;
  Kokkos::View<type_real[N_SLS], Kokkos::LayoutRight, Kokkos::HostSpace> gamma;
};

/**
 * @brief Compute Runge-Kutta coefficients for memory-variable time stepping.
 *
 * Translates the Fortran routine @c get_attenuation_memory_values from
 * @c get_attenuation_model.f90 (Savage et al. BSSA 2010, eq. 11).
 *
 * For each SLS mechanism @f$ j @f$ with relaxation time
 * @f$ \tau_{\sigma,j} @f$ the coefficients are
 * @f[
 *   \tau_{\text{inv},j} = -1/\tau_{\sigma,j}   \quad \text{(negative sign,
 *                         Fortran convention)}
 * @f]
 * @f[
 *   \alpha_j = 1 + \Delta t\,\tau_{\text{inv}}
 *                + \tfrac{1}{2}(\Delta t)^2\,\tau_{\text{inv}}^2
 *                + \tfrac{1}{6}(\Delta t)^3\,\tau_{\text{inv}}^3
 *                + \tfrac{1}{24}(\Delta t)^4\,\tau_{\text{inv}}^4
 * @f]
 * @f[
 *   \beta_j  = \tfrac{\Delta t}{2}
 *              + \tfrac{1}{3}(\Delta t)^2\,\tau_{\text{inv}}
 *              + \tfrac{1}{8}(\Delta t)^3\,\tau_{\text{inv}}^2
 *              + \tfrac{1}{24}(\Delta t)^4\,\tau_{\text{inv}}^3
 * @f]
 * @f[
 *   \gamma_j = \tfrac{\Delta t}{2}
 *              + \tfrac{1}{6}(\Delta t)^2\,\tau_{\text{inv}}
 *              + \tfrac{1}{24}(\Delta t)^3\,\tau_{\text{inv}}^2
 * @f]
 *
 * @tparam N_SLS Number of standard linear solids
 * @param tau_sigma Stress relaxation times @f$ \tau_\sigma @f$ (positive,
 *                  in seconds)
 * @param deltat    Time step @f$ \Delta t @f$ (seconds)
 * @return IntegrationFactors containing the three coefficient arrays
 */
template <int N_SLS>
IntegrationFactors<N_SLS> compute_integration_factors(
    Kokkos::View<type_real[N_SLS], Kokkos::LayoutRight, Kokkos::HostSpace>
        tau_sigma,
    type_real deltat);

} // namespace attenuation
} // namespace specfem

#include "compute_integration_factors.tpp"
