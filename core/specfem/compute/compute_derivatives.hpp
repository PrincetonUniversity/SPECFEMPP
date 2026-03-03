#pragma once

#include "impl/compute_material_derivatives.hpp"
#include "specfem/assembly.hpp"
#include "specfem/element.hpp"
#include "specfem/macros.hpp"
#include "specfem/setup.hpp"
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
  constexpr auto DimensionTag = Tags::dimension_tag;
  FOR_EACH_IN_PRODUCT(
      (DIMENSION_TAG(DIM2),
       MEDIUM_TAG(ELASTIC_PSV, ELASTIC_SH, ACOUSTIC, POROELASTIC),
       PROPERTY_TAG(ISOTROPIC, ANISOTROPIC), ATTENUATION_TAG(NONE)),
      {
        if constexpr (DimensionTag == _dimension_tag_) {
          impl::compute_material_derivatives<
              NGLL,
              specfem::tags::Tags<DimensionTag, _medium_tag_, _property_tag_> >(
              assembly, dt);
        }
      })
}

} // namespace compute
} // namespace specfem
