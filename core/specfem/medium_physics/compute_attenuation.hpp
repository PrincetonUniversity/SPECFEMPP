#pragma once

#include "specfem/medium/dim2/elastic/isotropic/attenuation.hpp"
#include <Kokkos_Core.hpp>

namespace specfem {
namespace medium_physics {

/**
 * @brief No-op attenuation update for elements with attenuation_tag::none.
 *
 * Compiles to nothing when attenuation is disabled, so the gradient callback
 * can call compute_attenuation unconditionally.
 */
template <typename Tags, typename IndexType, typename GradientPackType,
          typename AttenuationContainer,
          std::enable_if_t<Tags::attenuation_tag ==
                               specfem::element::attenuation_tag::none,
                           int> = 0>
KOKKOS_INLINE_FUNCTION void
compute_attenuation(const IndexType &, specfem::point::stress<Tags> &,
                    const GradientPackType &, const AttenuationContainer &) {}

/**
 * @brief Dispatch attenuation update for constant_isotropic attenuation.
 *
 * Calls the medium-specific impl_compute_attenuation which loads state,
 * modifies the stress tensor, advances memory variables, and writes back.
 *
 * @tparam Tags                       Element tags (must include
 * attenuation_tag)
 * @tparam IndexType                  GLL point index type
 * @tparam GradientPackType stiffness_gradient_pack<constant_isotropic,...>
 * @tparam AttenuationContainer       Assembly attenuation container
 *
 * @param index                   GLL point index
 * @param point_stress            Elastic stress tensor (modified in-place)
 * @param grad_pack               Gradient pack (contains du and dv)
 * @param attenuation             Assembly attenuation container
 */
template <
    typename Tags, typename IndexType, typename GradientPackType,
    typename AttenuationContainer,
    std::enable_if_t<Tags::attenuation_tag ==
                         specfem::element::attenuation_tag::constant_isotropic,
                     int> = 0>
KOKKOS_INLINE_FUNCTION void
compute_attenuation(const IndexType &index,
                    specfem::point::stress<Tags> &point_stress,
                    const GradientPackType &grad_pack,
                    const AttenuationContainer &attenuation) {
  specfem::medium_physics::impl_compute_attenuation<Tags>(
      index, point_stress, grad_pack, attenuation);
}

} // namespace medium_physics
} // namespace specfem
