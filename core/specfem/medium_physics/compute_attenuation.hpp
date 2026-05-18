#pragma once

#include "specfem/element/attributes.hpp"
#include "specfem/medium/dim2/elastic/isotropic/attenuation.hpp"
#include "specfem/medium/dim3/elastic/isotropic/attenuation.hpp"
#include <Kokkos_Core.hpp>

namespace specfem {
namespace medium_physics {

/**
 * @brief No-op: attenuation_tag::none, non-elastic, or non-isotropic.
 */
template <typename Tags, typename IndexType, typename GradientPackType,
          typename AttenuationContainer,
          std::enable_if_t<Tags::attenuation_tag ==
                               specfem::element::attenuation_tag::none,
                           int> = 0>
KOKKOS_INLINE_FUNCTION void
compute_attenuation(const IndexType &index,
                    specfem::point::stress<Tags> &point_stress,
                    const GradientPackType &grad_pack,
                    const AttenuationContainer &attenuation) {}

/**
 * @brief Active attenuation update: elastic + isotropic + any attenuation.
 *
 * Delegates to the medium-specific impl_compute_attenuation.
 */
template <typename Tags, typename IndexType, typename GradientPackType,
          typename AttenuationContainer,
          std::enable_if_t<Tags::attenuation_tag !=
                               specfem::element::attenuation_tag::none,
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
