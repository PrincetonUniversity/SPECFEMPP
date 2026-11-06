#pragma once

#include "specfem/element.hpp"
#include "specfem/point.hpp"
#include <Kokkos_Core.hpp>

namespace specfem {
namespace medium_physics {

/**
 * @defgroup specfem_constitutive_tensor_dim3_elastic_isotropic
 *
 */

/**
 * @ingroup specfem_constitutive_tensor_dim3_elastic_isotropic
 * @brief Compute a single entry of the fourth-order constitutive tensor for
 * 3D elastic isotropic media.
 *
 * The constitutive tensor \f$ C_{akbl} \f$ relates the stress tensor to the
 * displacement gradient through
 * \f$ \sigma_{ak} = \sum_{b,l} C_{akbl} \, \partial u_b / \partial x_l \f$,
 * where \f$ a \f$ is the stress component index, \f$ k \f$ is the
 * corresponding derivative direction, \f$ b \f$ is the displacement
 * component index, and \f$ l \f$ is the derivative direction of the
 * displacement gradient \c du(b, l). This convention matches
 * `specfem::medium_physics::compute_stress`, which must agree with the
 * contraction of this tensor against the displacement gradient.
 *
 * For an isotropic material with Lamé parameters \f$ \lambda \f$ and
 * \f$ \mu \f$, the entries are given by
 * \f[
 * C_{akbl} = \lambda \, \delta_{ak} \delta_{bl} + \mu \, (\delta_{ab}
 * \delta_{kl} + \delta_{al} \delta_{kb})
 * \f]
 * where \f$ \delta \f$ is the Kronecker delta.
 *
 * The tensor satisfies the minor and major symmetries
 * \f$ C_{akbl} = C_{kabl} = C_{blak} \f$.
 *
 * @tparam Tags Element tags identifying the dim3 elastic isotropic medium
 * @param properties Material properties (\f$ \lambda \f$, \f$ \mu \f$)
 * @param a Stress component index
 * @param k Stress derivative direction index
 * @param b Displacement component index
 * @param l Displacement derivative direction index
 * @return Value of the constitutive tensor entry \f$ C_{akbl} \f$
 */
template <
    typename Tags,
    std::enable_if_t<
        Tags::dimension_tag == specfem::element::dimension_tag::dim3 &&
            Tags::medium_tag == specfem::element::medium_tag::elastic &&
            Tags::property_tag == specfem::element::property_tag::isotropic,
        int> = 0>
KOKKOS_INLINE_FUNCTION typename specfem::point::properties<Tags>::value_type
constitutive_tensor(const specfem::point::properties<Tags> &properties,
                    const int a, const int k, const int b, const int l) {

  const type_real d_ak = (a == k) ? 1.0 : 0.0;
  const type_real d_bl = (b == l) ? 1.0 : 0.0;
  const type_real d_ab = (a == b) ? 1.0 : 0.0;
  const type_real d_kl = (k == l) ? 1.0 : 0.0;
  const type_real d_al = (a == l) ? 1.0 : 0.0;
  const type_real d_kb = (k == b) ? 1.0 : 0.0;

  return properties.lambda() * (d_ak * d_bl) +
         properties.mu() * (d_ab * d_kl + d_al * d_kb);
}

} // namespace medium_physics
} // namespace specfem
