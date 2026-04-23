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
  specfem::tag_dispatch::for_each(
      specfem::tag_dispatch::dimension_set<Tags::dimension_tag>{} *
          MEDIUM_SET(elastic_psv, elastic_sh, acoustic, poroelastic) *
          PROPERTY_SET(isotropic, anisotropic) * ATTENUATION_SET(none),
      [&]<typename ElementTags>() {
        impl::compute_material_derivatives<
            NGLL,
            specfem::tags::Tags<Tags::dimension_tag, ElementTags::medium_tag,
                                ElementTags::property_tag> >(assembly, dt);
      });
}

} // namespace compute
} // namespace specfem
