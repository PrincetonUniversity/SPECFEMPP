#pragma once

#include "specfem/datatype.hpp"
#include "specfem/element.hpp"
#include "specfem/point/field_derivatives.hpp"
#include <Kokkos_Core.hpp>

namespace specfem {
namespace medium_physics {

/**
 * @defgroup specfem_strain_computation_dim2_elastic_isotropic 2D Elastic Strain
 * Computation
 * @brief Strain tensor computation for 2D elastic wave propagation.
 */

/**
 * @ingroup specfem_strain_computation_dim2_elastic_isotropic
 * @brief Compute symmetric strain tensor for 2D elastic P-SV waves.
 *
 * Computes the symmetric strain tensor \f$\boldsymbol{\varepsilon}\f$ from
 * displacement gradients:
 * \f[
 * \boldsymbol{\varepsilon} = \begin{pmatrix}
 * \varepsilon_{xx} & \varepsilon_{xz} \\
 * \varepsilon_{xz} & \varepsilon_{zz}
 * \end{pmatrix} = \begin{pmatrix}
 * \frac{\partial u_x}{\partial x} & \frac{1}{2}\left(\frac{\partial
 * u_x}{\partial z} + \frac{\partial u_z}{\partial x}\right) \\
 * \frac{1}{2}\left(\frac{\partial u_x}{\partial z} + \frac{\partial
 * u_z}{\partial x}\right) & \frac{\partial u_z}{\partial z}
 * \end{pmatrix}
 * \f]
 *
 * @tparam UseSIMD Enable SIMD vectorization
 * @param field_derivatives Displacement gradients \f$\nabla \mathbf{u}\f$
 * @return 2×2 symmetric strain tensor \f$\boldsymbol{\varepsilon}\f$
 */
template <bool UseSIMD>
KOKKOS_INLINE_FUNCTION
    specfem::datatype::TensorPointViewType<type_real, 2, 2, UseSIMD>
    impl_compute_strain(const specfem::point::field_derivatives<
                        specfem::element::dimension_tag::dim2,
                        specfem::element::medium_tag::elastic_psv, UseSIMD>
                            &field_derivatives) {

  using datatype =
      typename specfem::datatype::simd<type_real, UseSIMD>::datatype;
  const auto &du = field_derivatives.du;

  specfem::datatype::TensorPointViewType<type_real, 2, 2, UseSIMD> T;

  T(0, 0) = du(0, 0); // ε_xx = ∂u_x/∂x
  T(1, 1) = du(1, 1); // ε_zz = ∂u_z/∂z

  const datatype eps_xz = static_cast<type_real>(0.5) * (du(0, 1) + du(1, 0));
  T(0, 1) = eps_xz;
  T(1, 0) = eps_xz;

  return T;
}

/**
 * @ingroup specfem_strain_computation_dim2_elastic_isotropic
 * @brief Compute deviatoric strain tensor for 2D elastic P-SV waves.
 *
 * Computes the deviatoric strain \f$\boldsymbol{\varepsilon}'\f$ by removing
 * the volumetric component:
 * \f[
 * \boldsymbol{\varepsilon}' = \boldsymbol{\varepsilon} -
 * \frac{\text{tr}(\boldsymbol{\varepsilon})}{3}\mathbf{I}
 * \f]
 * where \f$\text{tr}(\boldsymbol{\varepsilon}) = \varepsilon_{xx} +
 * \varepsilon_{zz}\f$.
 *
 * Under plane-strain conditions, the missing \f$\varepsilon'_{yy} =
 * -\frac{\text{tr}(\boldsymbol{\varepsilon})}{3}\f$ ensures
 * \f$\text{tr}(\boldsymbol{\varepsilon}') = 0\f$ in 3D.
 *
 * @tparam UseSIMD Enable SIMD vectorization
 * @param field_derivatives Displacement gradients \f$\nabla \mathbf{u}\f$
 * @return 2×2 deviatoric strain tensor \f$\boldsymbol{\varepsilon}'\f$
 */
template <bool UseSIMD>
KOKKOS_INLINE_FUNCTION
    specfem::datatype::TensorPointViewType<type_real, 2, 2, UseSIMD>
    impl_compute_deviatoric_strain(const specfem::point::field_derivatives<
                                   specfem::element::dimension_tag::dim2,
                                   specfem::element::medium_tag::elastic_psv,
                                   UseSIMD> &field_derivatives) {

  auto epsilon = impl_compute_strain(field_derivatives);

  const auto trace = epsilon(0, 0) + epsilon(1, 1);
  const auto third_trace = static_cast<type_real>(1.0 / 3.0) * trace;

  epsilon(0, 0) = epsilon(0, 0) - third_trace;
  epsilon(1, 1) = epsilon(1, 1) - third_trace;

  return epsilon;
}

/**
 * @ingroup specfem_strain_computation_dim2_elastic_isotropic
 * @brief Compute strain tensor for 2D elastic SH waves.
 *
 * For SH waves, the single displacement component \f$u_y\f$ produces a 1×2
 * strain vector:
 * \f[
 * \boldsymbol{\varepsilon} = \begin{pmatrix} \frac{\partial u_y}{\partial x} &
 * \frac{\partial u_y}{\partial z} \end{pmatrix}
 * \f]
 *
 * @tparam UseSIMD Enable SIMD vectorization
 * @param field_derivatives Displacement gradients \f$\nabla u_y\f$
 * @return 1×2 strain tensor \f$\boldsymbol{\varepsilon}\f$
 */
template <bool UseSIMD>
KOKKOS_INLINE_FUNCTION
    specfem::datatype::TensorPointViewType<type_real, 1, 2, UseSIMD>
    impl_compute_strain(const specfem::point::field_derivatives<
                        specfem::element::dimension_tag::dim2,
                        specfem::element::medium_tag::elastic_sh, UseSIMD>
                            &field_derivatives) {

  const auto &du = field_derivatives.du;

  specfem::datatype::TensorPointViewType<type_real, 1, 2, UseSIMD> T;
  T(0, 0) = du(0, 0); // ∂u_y/∂x
  T(0, 1) = du(0, 1); // ∂u_y/∂z

  return T;
}

/**
 * @ingroup specfem_strain_computation_dim2_elastic_isotropic
 * @brief Compute deviatoric strain tensor for 2D elastic SH waves.
 *
 * For SH waves, \f$\text{tr}(\boldsymbol{\varepsilon}) = 0\f$ since there are
 * no diagonal strain components. Therefore, \f$\boldsymbol{\varepsilon}' =
 * \boldsymbol{\varepsilon}\f$.
 *
 * @tparam UseSIMD Enable SIMD vectorization
 * @param field_derivatives Displacement gradients \f$\nabla u_y\f$
 * @return 1×2 deviatoric strain tensor \f$\boldsymbol{\varepsilon}' =
 * \boldsymbol{\varepsilon}\f$
 */
template <bool UseSIMD>
KOKKOS_INLINE_FUNCTION
    specfem::datatype::TensorPointViewType<type_real, 1, 2, UseSIMD>
    impl_compute_deviatoric_strain(const specfem::point::field_derivatives<
                                   specfem::element::dimension_tag::dim2,
                                   specfem::element::medium_tag::elastic_sh,
                                   UseSIMD> &field_derivatives) {

  return impl_compute_strain(field_derivatives);
}

} // namespace medium_physics
} // namespace specfem
