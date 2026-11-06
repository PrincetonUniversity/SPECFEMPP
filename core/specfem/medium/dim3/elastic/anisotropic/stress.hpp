#pragma once

#include "specfem/element.hpp"
#include "specfem/point.hpp"
#include <Kokkos_Core.hpp>

namespace specfem {
namespace medium_physics {

/**
 * @defgroup specfem_stress_computation_dim3_elastic_anisotropic
 *
 */

/**
 * @ingroup specfem_stress_computation_dim3_elastic_anisotropic
 * @brief Compute stress tensor for 3D elastic anisotropic media.
 *
 * Implements the general constitutive relation \f$\sigma_I = c_{IJ}
 * \varepsilon_J\f$ over the 21 independent entries of the symmetric
 * \f$6\times6\f$ Voigt stiffness matrix.
 *
 * Voigt indices follow the usual convention \f$1\!=\!xx\f$, \f$2\!=\!yy\f$,
 * \f$3\!=\!zz\f$, \f$4\!=\!yz\f$, \f$5\!=\!xz\f$, \f$6\!=\!xy\f$, with
 * *engineering* shear strains
 *
 * \f[
 * \varepsilon_4 = \frac{\partial u_y}{\partial z} + \frac{\partial
 * u_z}{\partial y}, \quad \varepsilon_5 = \frac{\partial u_x}{\partial z} +
 * \frac{\partial u_z}{\partial x}, \quad \varepsilon_6 = \frac{\partial
 * u_x}{\partial y} + \frac{\partial u_y}{\partial x},
 * \f]
 *
 * matching the convention used by the anisotropic Fréchet derivative in
 * `frechet_derivative.hpp`, whose \f$c_{IJ}\f$ kernels carry the corresponding
 * factors of two and four on the shear terms.
 *
 * For a medium whose stiffnesses are the isotropic-equivalent Voigt matrix
 * (\f$c_{11}=c_{22}=c_{33}=\lambda+2\mu\f$, \f$c_{12}=c_{13}=c_{23}=\lambda\f$,
 * \f$c_{44}=c_{55}=c_{66}=\mu\f$, all others zero) this reduces exactly to the
 * isotropic Hooke's law in `../isotropic/stress.hpp`.
 *
 * @tparam Tags Point tags selecting 3D elastic anisotropic properties
 * @param properties Anisotropic material properties (\f$c_{11} \ldots
 * c_{66}\f$)
 * @param field_derivatives Displacement gradients (\f$\frac{\partial
 * u_i}{\partial x_j}\f$)
 * @return 3x3 symmetric stress tensor
 */
template <
    typename Tags,
    std::enable_if_t<
        Tags::dimension_tag == specfem::element::dimension_tag::dim3 &&
            Tags::medium_tag == specfem::element::medium_tag::elastic &&
            Tags::property_tag == specfem::element::property_tag::anisotropic,
        int> = 0>
KOKKOS_INLINE_FUNCTION specfem::point::stress<Tags> compute_stress(
    const specfem::point::properties<Tags> &properties,
    const specfem::point::field_derivatives<Tags> &field_derivatives) {

  using datatype =
      typename specfem::datatype::simd<type_real, Tags::using_simd>::datatype;
  const auto &du = field_derivatives.du;

  // Voigt strain vector; shear entries are engineering strains.
  const datatype epsilon_xx = du(0, 0);
  const datatype epsilon_yy = du(1, 1);
  const datatype epsilon_zz = du(2, 2);
  const datatype gamma_yz = du(1, 2) + du(2, 1);
  const datatype gamma_xz = du(0, 2) + du(2, 0);
  const datatype gamma_xy = du(0, 1) + du(1, 0);

  const datatype sigma_xx =
      properties.c11() * epsilon_xx + properties.c12() * epsilon_yy +
      properties.c13() * epsilon_zz + properties.c14() * gamma_yz +
      properties.c15() * gamma_xz + properties.c16() * gamma_xy;

  const datatype sigma_yy =
      properties.c12() * epsilon_xx + properties.c22() * epsilon_yy +
      properties.c23() * epsilon_zz + properties.c24() * gamma_yz +
      properties.c25() * gamma_xz + properties.c26() * gamma_xy;

  const datatype sigma_zz =
      properties.c13() * epsilon_xx + properties.c23() * epsilon_yy +
      properties.c33() * epsilon_zz + properties.c34() * gamma_yz +
      properties.c35() * gamma_xz + properties.c36() * gamma_xy;

  const datatype sigma_yz =
      properties.c14() * epsilon_xx + properties.c24() * epsilon_yy +
      properties.c34() * epsilon_zz + properties.c44() * gamma_yz +
      properties.c45() * gamma_xz + properties.c46() * gamma_xy;

  const datatype sigma_xz =
      properties.c15() * epsilon_xx + properties.c25() * epsilon_yy +
      properties.c35() * epsilon_zz + properties.c45() * gamma_yz +
      properties.c55() * gamma_xz + properties.c56() * gamma_xy;

  const datatype sigma_xy =
      properties.c16() * epsilon_xx + properties.c26() * epsilon_yy +
      properties.c36() * epsilon_zz + properties.c46() * gamma_yz +
      properties.c56() * gamma_xz + properties.c66() * gamma_xy;

  specfem::datatype::TensorPointViewType<type_real, 3, 3, Tags::using_simd> T;

  T(0, 0) = sigma_xx;
  T(1, 1) = sigma_yy;
  T(2, 2) = sigma_zz;
  T(0, 1) = sigma_xy;
  T(1, 0) = sigma_xy;
  T(0, 2) = sigma_xz;
  T(2, 0) = sigma_xz;
  T(1, 2) = sigma_yz;
  T(2, 1) = sigma_yz;

  return { T };
}

} // namespace medium_physics
} // namespace specfem
