#pragma once

#include "specfem/assembly/attenuation.hpp"
#include "specfem/enums.hpp"
#include <Kokkos_Core.hpp>

namespace specfem::assembly {

/**
 * @brief Store evolved SLS memory variables from a point-local attenuation
 *        struct back to the assembly attenuation container (device kernel).
 *
 * Only the memory variables (e.g. Rxx, Rxz, Rkappa for dim2; additionally
 * Ryy, Rxy, Rxz, Ryz for dim3) are written back. The relaxation rates and
 * Runge-Kutta coefficients are simulation-lifetime constants and are never
 * written back by this function.
 *
 * @note For dim3, @c memory_variable_Rzz = -(Rxx + Ryy) is not stored in the
 *       point type and must be updated separately by the caller if required.
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
 * @param index       GLL point index (global ispec, iz, [iy,] ix).
 * @param attenuation Assembly attenuation container (device-accessible).
 * @param point       Input: point attenuation struct with updated memory vars.
 *
 * @ingroup AttenuationDataAccess
 */
template <typename PointAttenuationType, typename IndexType,
          typename std::enable_if_t<
              IndexType::using_simd == PointAttenuationType::using_simd &&
                  PointAttenuationType::attenuation_tag ==
                      specfem::element::attenuation_tag::constant_isotropic,
              int> = 0>
KOKKOS_FORCEINLINE_FUNCTION void store_on_device(
    const IndexType &index,
    const specfem::assembly::Attenuation<PointAttenuationType::dimension_tag>
        &attenuation,
    const PointAttenuationType &point) {

  constexpr auto MediumTag = PointAttenuationType::medium_tag;
  constexpr auto PropertyTag = PointAttenuationType::property_tag;

  // Store memory variables back to device views
  attenuation.template get_container<MediumTag, PropertyTag>()
      .store_device_values(index, point);
}

} // namespace specfem::assembly
