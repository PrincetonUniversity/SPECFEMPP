#pragma once

#include "enumerations/dimension.hpp"
#include "enumerations/medium.hpp"
#include <Kokkos_Core.hpp>

namespace specfem {
namespace medium {

/**
 * @defgroup specfem_cosserat_stress_computation_dim2_elastic_isotropic_cosserat
 *
 */

/**
 * @ingroup specfem_cosserat_stress_computation_dim2_elastic_isotropic_cosserat
 * @brief Compute Cosserat stress contribution for 2D elastic isotropic
 * micropolar media.
 *
 * Implements asymmetric stress correction for a Cosserat continuum with
 * rotational degrees of freedom. Adds coupling between the rotation field
 * and the asymmetric Cosserat stress tensor to capture microstructural
 * effects.
 *
 * **Stress corrections:**
 * - \f$ \sigma_{zx} = \sigma_{zx}^{classical} - 2\nu\phi_y \f$
 * - \f$ \sigma_{xz} = \sigma_{xz}^{classical} + 2\nu\phi_y \f$
 *
 * where:
 * - \f$ \nu \f$: Cosserat coupling parameter
 * - \f$ \phi_y \f$: rotation about y-axis (out-of-plane)
 * - Asymmetric tensor: \f$ \sigma_{zx} \neq \sigma_{xz} \f$
 * @param properties Cosserat material properties including the Cosserat
 * coupling parameter \f$ \nu \f$ (nu)
 * @param u Displacement field [u_x, u_z, φ_y]
 * @param point_stress[in,out] Stress tensor (modified by Cosserat effects)
 */
template <typename PointPropertiesType, typename PointDisplacementType,
          typename PointStressType>
KOKKOS_INLINE_FUNCTION void impl_compute_cosserat_stress(
    std::true_type,
    const std::integral_constant<specfem::dimension::type,
                                 specfem::dimension::type::dim2>,
    const std::integral_constant<specfem::element::medium_tag,
                                 specfem::element::medium_tag::elastic_psv_t>,
    const std::integral_constant<
        specfem::element::property_tag,
        specfem::element::property_tag::isotropic_cosserat>,
    const PointPropertiesType &properties, const PointDisplacementType &u,
    PointStressType &point_stress) {

  using value_type = typename PointStressType::simd::datatype;

  // Stress and diplacement alias
  auto &T = point_stress.T;

  const value_type factor =
      static_cast<type_real>(2.0) * properties.nu() * u(2);

  // Here we also have to remember that we are getting the stress transposed
  // T(0, 1) = sigma_xz, but the spin notes have the divergence act on the first
  // component. So, sigma_xz is actually sigma_zx. And we have to add the
  // spin contribution from the notes
  // sigma_zx = ... - 2 \nu \phi_{y}
  T(0, 1) -= factor;

  // The converse is true for the second component
  // sigma_xz = ... + 2 \nu \phi_{y}
  T(1, 0) += factor;

  return;
};

} // namespace medium
} // namespace specfem
