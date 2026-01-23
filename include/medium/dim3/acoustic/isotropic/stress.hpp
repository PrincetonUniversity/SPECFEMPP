#pragma once

#include "enumerations/dimension.hpp"
#include "enumerations/medium.hpp"
#include "specfem/datatype.hpp"
#include "specfem/point.hpp"
#include <Kokkos_Core.hpp>

namespace specfem::medium {

/**
 * @defgroup specfem_stress_computation_dim3_acoustic
 *
 */

/**
 * @ingroup specfem_stress_computation_dim3_acoustic
 * @brief Compute stress components for 3D acoustic isotropic media.
 *
 * Computes diagonal stress components for acoustic wave propagation.
 * In acoustic media, stress relates to field gradients through density:
 *
 * **Stress components:**
 * - \f$\sigma_{xx} = \frac{1}{\rho} \frac{\partial u}{\partial x}\f$
 * - \f$\sigma_{yy} = \frac{1}{\rho} \frac{\partial u}{\partial y}\f$
 * - \f$\sigma_{zz} = \frac{1}{\rho} \frac{\partial u}{\partial z}\f$
 *
 *
 * @tparam UseSIMD Enable SIMD vectorization for performance
 * @param properties Material properties (density \f$\rho\f$)
 * @param field_derivatives Field gradients (\f$\frac{\partial u}{\partial
 * x_i}\f$)
 * @return 1x3 stress tensor with diagonal components
 */
template <bool UseSIMD>
KOKKOS_INLINE_FUNCTION specfem::point::stress<
    specfem::dimension::type::dim3, specfem::element::medium_tag::acoustic,
    UseSIMD>
impl_compute_stress(
    const specfem::point::properties<
        specfem::dimension::type::dim3, specfem::element::medium_tag::acoustic,
        specfem::element::property_tag::isotropic, UseSIMD> &properties,
    const specfem::point::field_derivatives<
        specfem::dimension::type::dim3, specfem::element::medium_tag::acoustic,
        UseSIMD> &field_derivatives) {

  const auto &du = field_derivatives.du;

  specfem::datatype::TensorPointViewType<type_real, 1, 3, UseSIMD> T;
  T(0, 0) = properties.rho_inverse() * du(0, 0);
  T(0, 1) = properties.rho_inverse() * du(0, 1);
  T(0, 2) = properties.rho_inverse() * du(0, 2);
  return { T };
}

} // namespace specfem::medium
