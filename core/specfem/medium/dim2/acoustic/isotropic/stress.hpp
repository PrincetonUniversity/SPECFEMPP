#pragma once

#include "specfem/element.hpp"
#include "specfem/point.hpp"
#include <Kokkos_Core.hpp>

namespace specfem {
namespace medium_physics {

/**
 * @defgroup specfem_stress_computation_dim2_acoustic
 *
 */

/**
 * @ingroup specfem_stress_computation_dim2_acoustic
 * @brief Compute stress tensor for 2D acoustic isotropic media.
 *
 * Implements constitutive relation for 2D acoustic wave propagation in
 * isotropic fluids. Computes stress (pressure) from displacement gradients
 * and bulk modulus, assuming no shear wave propagation.
 *
 * **Stress components:**
 * - \f$\sigma_{xx} = \rho^{-1} \frac{\partial u_x}{\partial x}\f$
 * - \f$\sigma_{zz} = \rho^{-1} \frac{\partial u_z}{\partial z}\f$
 *
 * @tparam UseSIMD Enable SIMD vectorization for performance
 * @param properties Material properties (\f$\rho^{-1}\f$, \f$\kappa\f$)
 * @param field_derivatives Displacement gradients (\f$\frac{\partial
 * u_i}{\partial x_j}\f$)
 * @return 1x2 stress tensor [\f$\sigma_{xx}\f$, \f$\sigma_{zz}\f$]
 */
template <
    typename Tags,
    std::enable_if_t<
        Tags::dimension_tag == specfem::element::dimension_tag::dim2 &&
            Tags::medium_tag == specfem::element::medium_tag::acoustic &&
            Tags::property_tag == specfem::element::property_tag::isotropic,
        int> = 0>
KOKKOS_INLINE_FUNCTION specfem::point::stress<Tags> compute_stress(
    const specfem::point::properties<Tags> &properties,
    const specfem::point::field_derivatives<Tags> &field_derivatives) {

  const auto &du = field_derivatives.du;

  specfem::datatype::TensorPointViewType<type_real, 1, 2, Tags::using_simd> T;

  T(0, 0) = properties.rho_inverse() * du(0, 0);
  T(0, 1) = properties.rho_inverse() * du(0, 1);

  return { T };
}

} // namespace medium_physics
} // namespace specfem
