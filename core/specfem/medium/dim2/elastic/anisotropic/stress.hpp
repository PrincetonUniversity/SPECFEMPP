#pragma once

#include "enumerations/medium.hpp"
#include "specfem/point.hpp"

namespace specfem {
namespace medium {

/**
 * @defgroup specfem_stress_computation_dim2_elastic_anisotropic
 *
 */

/**
 * @ingroup specfem_stress_computation_dim2_elastic_anisotropic
 * @brief Compute stress tensor for 2D elastic anisotropic P-SV waves.
 *
 * Implements constitutive relation for 2D elastic P-SV wave propagation
 * in anisotropic media using full elastic stiffness tensor. Accounts for
 * directionally-dependent wave velocities and coupling between normal
 * and shear components.
 *
 * **Stress components:**
 * - \f$\sigma_{xx} = c_{11}\frac{\partial u_x}{\partial x} +
 * c_{13}\frac{\partial u_z}{\partial z} + c_{15}\left(\frac{\partial
 * u_x}{\partial z} + \frac{\partial u_z}{\partial x}\right)\f$
 * - \f$\sigma_{zz} = c_{13}\frac{\partial u_x}{\partial x} +
 * c_{33}\frac{\partial u_z}{\partial z} + c_{35}\left(\frac{\partial
 * u_x}{\partial z} + \frac{\partial u_z}{\partial x}\right)\f$
 * - \f$\sigma_{xz} = c_{15}\frac{\partial u_x}{\partial x} +
 * c_{35}\frac{\partial u_z}{\partial z} + c_{55}\left(\frac{\partial
 * u_x}{\partial z} + \frac{\partial u_z}{\partial x}\right)\f$
 *
 * @tparam UseSIMD Enable SIMD vectorization for performance
 * @param properties Anisotropic material properties (\f$c_{11}, c_{13}, c_{15},
 * c_{33}, c_{35}, c_{55}\f$)
 * @param field_derivatives Displacement gradients (\f$\frac{\partial
 * u_i}{\partial x_j}\f$)
 * @return 2x2 symmetric stress tensor [\f$\sigma_{xx}, \sigma_{xz};
 * \sigma_{xz}, \sigma_{zz}\f$]
 */
template <bool UseSIMD>
KOKKOS_INLINE_FUNCTION
    specfem::point::stress<specfem::dimension::type::dim2,
                           specfem::element::medium_tag::elastic_psv, UseSIMD>
    impl_compute_stress(
        const specfem::point::properties<
            specfem::dimension::type::dim2,
            specfem::element::medium_tag::elastic_psv,
            specfem::element::property_tag::anisotropic, UseSIMD> &properties,
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
  sigma_xx = properties.c11() * du(0, 0) + properties.c13() * du(1, 1) +
             properties.c15() * (du(1, 0) + du(0, 1));

  // sigma_zz
  sigma_zz = properties.c13() * du(0, 0) + properties.c33() * du(1, 1) +
             properties.c35() * (du(1, 0) + du(0, 1));

  // sigma_xz
  sigma_xz = properties.c15() * du(0, 0) + properties.c35() * du(1, 1) +
             properties.c55() * (du(1, 0) + du(0, 1));

  specfem::datatype::TensorPointViewType<type_real, 2, 2, UseSIMD> T;

  T(0, 0) = sigma_xx;
  T(0, 1) = sigma_xz;
  T(1, 0) = sigma_xz;
  T(1, 1) = sigma_zz;

  return { T };
}

/**
 * @ingroup specfem_stress_computation_dim2_elastic_anisotropic
 * @brief Compute stress tensor for 2D elastic anisotropic SH waves.
 *
 * Implements constitutive relation for 2D elastic SH wave propagation
 * in anisotropic media. For SH waves, only the c55 stiffness component
 * is relevant, simplifying the anisotropic case to isotropic behavior.
 *
 * **Stress components:**
 * - \f$\sigma_{xy} = c_{55} \frac{\partial u_y}{\partial x}\f$ (stored as
 * T(0,0))
 * - \f$\sigma_{zy} = c_{55} \frac{\partial u_y}{\partial z}\f$ (stored as
 * T(0,1))
 *
 * @tparam UseSIMD Enable SIMD vectorization for performance
 * @param properties Anisotropic material properties (c55 used only)
 * @param field_derivatives Displacement gradients (\f$\frac{\partial
 * u_y}{\partial x_j}\f$)
 * @return 1x2 shear stress tensor [\f$\sigma_{xy}\f$, \f$\sigma_{zy}\f$]
 */
template <bool UseSIMD>
KOKKOS_INLINE_FUNCTION
    specfem::point::stress<specfem::dimension::type::dim2,
                           specfem::element::medium_tag::elastic_sh, UseSIMD>
    impl_compute_stress(
        const specfem::point::properties<
            specfem::dimension::type::dim2,
            specfem::element::medium_tag::elastic_sh,
            specfem::element::property_tag::anisotropic, UseSIMD> &properties,
        const specfem::point::field_derivatives<
            specfem::dimension::type::dim2,
            specfem::element::medium_tag::elastic_sh, UseSIMD>
            &field_derivatives) {

  using datatype =
      typename specfem::datatype::simd<type_real, UseSIMD>::datatype;
  const auto &du = field_derivatives.du;

  datatype sigma_xy, sigma_zy;

  // SH-case
  // sigma_xy
  sigma_xy = properties.c55() * du(0, 0);
  // sigma_zy
  sigma_zy = properties.c55() * du(0, 1);

  specfem::datatype::TensorPointViewType<type_real, 1, 2, UseSIMD> T;

  T(0, 0) = sigma_xy;
  T(0, 1) = sigma_zy;

  return { T };
}

} // namespace medium
} // namespace specfem
