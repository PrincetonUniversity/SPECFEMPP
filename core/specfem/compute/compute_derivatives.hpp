#pragma once

#include "impl/compute_material_derivatives.hpp"
#include "specfem/assembly/assembly.hpp"
#include "specfem/element.hpp"
#include "specfem/setup.hpp"
#include "specfem/tag_dispatch.hpp"
#include "specfem/tags.hpp"

namespace specfem {
namespace compute {

/**
 * @brief Compute Frechet derivatives.
 *
 * @tparam DimensionTag Dimension of the problem.
 * @tparam NGLL Number of GLL points.
 * @param assembly Assembly object.
 * @param dt Time interval.
 */
template <int NGLL, typename Tags>
void compute_derivatives(
    const specfem::assembly::assembly<Tags::dimension_tag> &assembly,
    const type_real &dt) {
  constexpr auto property_set = []() {
    if constexpr (Tags::dimension_tag ==
                  specfem::element::dimension_tag::dim3) {
      return PROPERTY_SET(isotropic);
    } else {
      return PROPERTY_SET(isotropic, anisotropic);
    }
  }();

  specfem::tag_dispatch::for_each(
      specfem::tag_dispatch::dimension_set<Tags::dimension_tag>{} *
          MEDIUM_SET(elastic, elastic_psv, elastic_sh) * property_set *
          ATTENUATION_SET(none, constant_isotropic),
      [&]<typename ElementTags>() {
        impl::compute_material_derivatives<NGLL, ElementTags>(assembly, dt);
      });
}

} // namespace compute
} // namespace specfem
