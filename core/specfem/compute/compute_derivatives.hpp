#pragma once

#include "enumerations/dimension.hpp"
#include "impl/compute_material_derivatives.hpp"
#include "specfem/assembly.hpp"
#include "specfem/element.hpp"
#include "specfem/macros.hpp"

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
template <specfem::dimension::type DimensionTag, int NGLL>
void compute_derivatives(
    const specfem::assembly::assembly<DimensionTag> &assembly,
    const type_real &dt) {
  FOR_EACH_IN_PRODUCT(
      (DIMENSION_TAG(DIM2),
       MEDIUM_TAG(ELASTIC_PSV, ELASTIC_SH, ACOUSTIC, POROELASTIC),
       PROPERTY_TAG(ISOTROPIC, ANISOTROPIC), ATTENUATION_TAG(NONE)),
      {
        if constexpr (DimensionTag == _dimension_tag_) {
          impl::compute_material_derivatives<DimensionTag, NGLL, _medium_tag_,
                                             _property_tag_>(assembly, dt);
        }
      })
}

} // namespace compute
} // namespace specfem
