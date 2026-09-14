#pragma once

#include "specfem/element.hpp"
#include <Kokkos_Core.hpp>

namespace specfem {
namespace medium_physics {

/**
 * @defgroup specfem_cosserat_stress_computation_dim3_elastic_isotropic_cosserat
 *
 */

/**
 * @ingroup specfem_cosserat_stress_computation_dim3_elastic_isotropic_cosserat
 * @brief Compute Cosserat stress contribution for 3D elastic isotropic
 * micropolar media.
 *
 * Implements asymmetric stress correction for a Cosserat continuum with
 * rotational degrees of freedom. Adds coupling between the rotation field
 * and the asymmetric Cosserat stress tensor to capture microstructural
 * effects.
 *
 * **Stress corrections:**
 * - \f$ \sigma_{xy} = \sigma_{xy}^{classical} - 2\nu\phi_z \f$
 * - \f$ \sigma_{yx} = \sigma_{yx}^{classical} + 2\nu\phi_z \f$
 * - \f$ \sigma_{xz} = \sigma_{xz}^{classical} + 2\nu\phi_y \f$
 * - \f$ \sigma_{zx} = \sigma_{zx}^{classical} - 2\nu\phi_y \f$
 * - \f$ \sigma_{yz} = \sigma_{yz}^{classical} - 2\nu\phi_x \f$
 * - \f$ \sigma_{zy} = \sigma_{zy}^{classical} + 2\nu\phi_x \f$
 *
 * where:
 * - \f$ \nu \f$: Cosserat coupling parameter
 * - \f$ \phi \f$: rotation vector (microrotation field)
 * - Asymmetric tensor: \f$ \sigma_{yx} \neq \sigma_{xy} \f$
 * @param properties Cosserat material properties including the Cosserat
 * coupling parameter \f$ \nu \f$ (nu)
 * @param u Displacement field [u_x, u_y, u_z, φ_x, φ_y, φ_z]
 * @param point_stress[in,out] Stress tensor (modified by Cosserat effects)
 */
template <typename PointPropertiesType, typename PointDisplacementType,
          typename PointStressType>
KOKKOS_INLINE_FUNCTION void impl_compute_cosserat_stress(
    std::true_type,
    const std::integral_constant<specfem::element::dimension_tag,
                                 specfem::element::dimension_tag::dim3>,
    const std::integral_constant<specfem::element::medium_tag,
                                 specfem::element::medium_tag::elastic_spin>,
    const std::integral_constant<
        specfem::element::property_tag,
        specfem::element::property_tag::isotropic_cosserat>,
    const PointPropertiesType &properties, const PointDisplacementType &u,
    PointStressType &point_stress) {

  using value_type = typename PointStressType::simd::datatype;

  // Stress and diplacement alias
  auto &T = point_stress.T;

  // Here we also have to remember that we are getting the stress transposed
  // T(0, 1) = sigma_xy, but the spin notes have the divergence act on the first
  // component. So, sigma_xy is actually sigma_yx. And we have to add the
  // spin contribution from the notes
  // sigma_yx = ... + 2 \nu \phi_{y}
  T(0, 1) += static_cast<value_type>(2.0) * properties.nu() * u(5);
  T(1, 0) -= static_cast<value_type>(2.0) * properties.nu() * u(5);
  T(0, 2) -= static_cast<value_type>(2.0) * properties.nu() * u(4);
  T(2, 0) += static_cast<value_type>(2.0) * properties.nu() * u(4);
  T(1, 2) += static_cast<value_type>(2.0) * properties.nu() * u(3);
  T(2, 1) -= static_cast<value_type>(2.0) * properties.nu() * u(3);

  return;
};

} // namespace medium_physics
} // namespace specfem
