#pragma once

#include "enumerations/medium.hpp"
#include "specfem/point.hpp"
#include <Kokkos_Core.hpp>

namespace specfem {
namespace medium {

/**
 * @defgroup specfem_stress_computation_dim2_elastic_isotropic
 *
 */

/**
 * @ingroup specfem_stress_computation_dim2_elastic_isotropic
 * @brief Compute stress tensor for 2D elastic isotropic P-SV waves.
 *
 * Implements constitutive relation for 2D elastic P-SV (pressure-shear
 * vertical) wave propagation in isotropic media. Computes full stress tensor
 * from displacement gradients and Lamé parameters.
 *
 * **Stress components:**
 * - \f$ \sigma_{xx} = (\lambda + 2\mu) \frac{\partial u_x}{\partial x} +
 * \lambda \frac{\partial u_z}{\partial z} \f$
 * - \f$ \sigma_{zz} = (\lambda + 2\mu) \frac{\partial u_z}{\partial z} +
 * \lambda \frac{\partial u_x}{\partial x} \f$
 * - \f$ \sigma_{xz} = \mu \left( \frac{\partial u_x}{\partial z} +
 * \frac{\partial u_z}{\partial x} \right) \f$
 *
 * @tparam UseSIMD Enable SIMD vectorization for performance
 * @param properties Material properties (\f$\lambda\f$, \f$\mu\f$, \f$\rho\f$)
 * @param field_derivatives Displacement gradients (\f$\frac{\partial
 * u_i}{\partial x_j}\f$)
 * @return 2x2 symmetric stress tensor [\f$\sigma_{xx}\f$, \f$\sigma_{xz}\f$;
 * \f$\sigma_{xz}\f$, \f$\sigma_{zz}\f$]
 */
template <bool UseSIMD>
KOKKOS_INLINE_FUNCTION
    specfem::point::stress<specfem::dimension::type::dim2,
                           specfem::element::medium_tag::elastic_psv, UseSIMD>
    impl_compute_stress(
        const specfem::point::properties<
            specfem::dimension::type::dim2,
            specfem::element::medium_tag::elastic_psv,
            specfem::element::property_tag::isotropic, UseSIMD> &properties,
        const specfem::point::field_derivatives<
            specfem::dimension::type::dim2,
            specfem::element::medium_tag::elastic_psv, UseSIMD>
            &field_derivatives) {

  using datatype =
      typename specfem::datatype::simd<type_real, UseSIMD>::datatype;
  const auto &du = field_derivatives.du;

  datatype sigma_xx, sigma_zz, sigma_xz;

  // P_SV case
  // sigma_xx
  sigma_xx =
      properties.lambdaplus2mu() * du(0, 0) + properties.lambda() * du(1, 1);

  // sigma_zz
  sigma_zz =
      properties.lambdaplus2mu() * du(1, 1) + properties.lambda() * du(0, 0);

  // sigma_xz
  sigma_xz = properties.mu() * (du(0, 1) + du(1, 0));

  specfem::datatype::TensorPointViewType<type_real, 2, 2, UseSIMD> T;

  T(0, 0) = sigma_xx;
  T(0, 1) = sigma_xz;
  T(1, 0) = sigma_xz;
  T(1, 1) = sigma_zz;

  return { T };
}

/**
 * @ingroup specfem_stress_computation_dim2_elastic_isotropic
 * @brief Compute stress tensor for 2D elastic SH waves.
 *
 * Implements constitutive relation for 2D elastic SH (shear horizontal)
 * wave propagation in isotropic media. Computes shear stress components from
 * displacement gradients and shear modulus.
 *
 * **Stress components:**
 * - \f$ \sigma_{xy} = \mu \frac{\partial u_y}{\partial x} \f$
 * - \f$ \sigma_{zy} = \mu \frac{\partial u_y}{\partial z} \f$
 *
 * @tparam UseSIMD Enable SIMD vectorization for performance
 * @param properties Material properties (\f$\mu\f$, \f$\rho\f$)
 * @param field_derivatives Displacement gradients (\f$\frac{\partial
 * u_y}{\partial x_j}\f$)
 * @return 1x2 shear stress tensor [\f$\sigma_{xy}\f$, \f$\sigma_{zy}\f$]
 */
template <bool UseSIMD>
KOKKOS_INLINE_FUNCTION specfem::point::stress<
    specfem::dimension::type::dim2, specfem::element::medium_tag::elastic_sh,
    UseSIMD>
impl_compute_stress(
    const specfem::point::properties<specfem::dimension::type::dim2,
                                     specfem::element::medium_tag::elastic_sh,
                                     specfem::element::property_tag::isotropic,
                                     UseSIMD> &properties,
    const specfem::point::field_derivatives<
        specfem::dimension::type::dim2,
        specfem::element::medium_tag::elastic_sh, UseSIMD> &field_derivatives) {

  using datatype =
      typename specfem::datatype::simd<type_real, UseSIMD>::datatype;
  const auto &du = field_derivatives.du;

  datatype sigma_xy, sigma_zy;

  // SH-case
  // sigma_xy
  sigma_xy = properties.mu() * du(0, 0); // would be sigma_xy in
                                         // CPU-version

  // sigma_zy
  sigma_zy = properties.mu() * du(0, 1); // sigma_zy
  specfem::datatype::TensorPointViewType<type_real, 1, 2, UseSIMD> T;

  T(0, 0) = sigma_xy;
  T(0, 1) = sigma_zy;
  return { T };
}

} // namespace medium
} // namespace specfem
