#pragma once

#include "specfem/assembly/attenuation.hpp"
#include "specfem/constants.hpp"
#include "specfem/enums.hpp"
#include "specfem/setup.hpp"
#include <Kokkos_Core.hpp>

namespace specfem::assembly {

/**
 * @defgroup AttenuationDataAccess Attenuation Data Access Functions
 */

/**
 * @brief Load attenuation state for a GLL point from the assembly attenuation
 *        container into a point-local attenuation struct (device kernel).
 *
 * Populates all fields of @p point:
 *  - kappa and mu relaxation rates (per-element constants read from device
 * views inside the matching @c attenuation_medium),
 *  - the current SLS memory variables (e.g. Rxx, Rxz, Rkappa for dim2), and
 *  - the global Runge-Kutta coefficients (alpha/beta/gamma, broadcast from the
 *    @c Attenuation device views to all active SIMD lanes).
 *
 * @tparam PointAttenuationType  Point attenuation struct
 *                               (specfem::point::attenuation<...>).
 *                               Must have @c medium_tag, @c property_tag, and
 *                               @c attenuation_tag == constant_isotropic.
 * @tparam IndexType             Point index struct
 *                               (specfem::point::index<...>).
 *                               @c IndexType::using_simd must match
 *                               @c PointAttenuationType::using_simd.
 *
 * @param lcoord      GLL point index (global ispec, iz, [iy,] ix).
 * @param attenuation Assembly attenuation container (device-accessible).
 * @param point       Output: populated point attenuation struct.
 *
 * @ingroup AttenuationDataAccess
 */
template <typename PointAttenuationType, typename IndexType,
          typename std::enable_if_t<
              IndexType::using_simd == PointAttenuationType::using_simd &&
                  PointAttenuationType::attenuation_tag ==
                      specfem::element::attenuation_tag::constant_isotropic,
              int> = 0>
KOKKOS_FORCEINLINE_FUNCTION void load_on_device(
    const IndexType &lcoord,
    const specfem::assembly::Attenuation<PointAttenuationType::dimension_tag>
        &attenuation,
    PointAttenuationType &point) {

  constexpr auto MediumTag = PointAttenuationType::medium_tag;
  constexpr auto PropertyTag = PointAttenuationType::property_tag;
  constexpr int N_SLS = PointAttenuationType::N_SLS;

  // Load per-element data: relaxation rates and memory variables
  attenuation.template get_medium<MediumTag, PropertyTag>().load_device_values(
      lcoord, point);

  // Broadcast global RK coefficients (identical for every element in a run)
  for (int j = 0; j < N_SLS; ++j) {
    point.alpha_rk(j) = attenuation.d_alpha_rk(j);
    point.beta_rk(j) = attenuation.d_beta_rk(j);
    point.gamma_rk(j) = attenuation.d_gamma_rk(j);
  }
}

} // namespace specfem::assembly
