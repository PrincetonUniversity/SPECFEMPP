#pragma once

#include "specfem/element.hpp"
#include "specfem/point.hpp"
#include <Kokkos_Core.hpp>

namespace specfem::medium_physics {

namespace frechet_derivative_impl {

/**
 * @brief Compute absolute-parameter Fréchet kernels for a 3D fully
 * anisotropic elastic medium.
 *
 * Implements the 21 symmetric strain products used by SPECFEM3D for the
 * stiffness kernels in Voigt order. Normal strains use unit weight, mixed
 * normal/shear terms use a factor of two, and shear/shear terms use a factor
 * of four. The result is the gradient with respect to an absolute perturbation
 * of each stiffness, not a logarithmic perturbation.
 *
 * @tparam Tags Compile-time dimension, medium, property, and SIMD tags.
 * @param adjoint_acceleration Adjoint acceleration.
 * @param backward_displacement Backward/reconstructed displacement.
 * @param adjoint_derivatives Spatial derivatives of the adjoint displacement.
 * @param backward_derivatives Spatial derivatives of the backward displacement.
 * @param dt Time step.
 * @return Density and 21 independent stiffness-kernel contributions.
 */
template <typename Tags>
  requires(Tags::dimension_tag == specfem::element::dimension_tag::dim3 &&
           Tags::medium_tag == specfem::element::medium_tag::elastic &&
           Tags::property_tag == specfem::element::property_tag::anisotropic)
KOKKOS_FUNCTION specfem::point::kernels<Tags>
compute_anisotropic_frechet_derivatives(
    const specfem::point::acceleration<Tags> &adjoint_acceleration,
    const specfem::point::displacement<Tags> &backward_displacement,
    const specfem::point::field_derivatives<Tags> &adjoint_derivatives,
    const specfem::point::field_derivatives<Tags> &backward_derivatives,
    const type_real &dt) {

  const auto ad_xx = adjoint_derivatives.du(0, 0);
  const auto ad_yy = adjoint_derivatives.du(1, 1);
  const auto ad_zz = adjoint_derivatives.du(2, 2);
  const auto ad_yz =
      static_cast<type_real>(0.5) *
      (adjoint_derivatives.du(2, 1) + adjoint_derivatives.du(1, 2));
  const auto ad_xz =
      static_cast<type_real>(0.5) *
      (adjoint_derivatives.du(2, 0) + adjoint_derivatives.du(0, 2));
  const auto ad_xy =
      static_cast<type_real>(0.5) *
      (adjoint_derivatives.du(1, 0) + adjoint_derivatives.du(0, 1));

  const auto b_xx = backward_derivatives.du(0, 0);
  const auto b_yy = backward_derivatives.du(1, 1);
  const auto b_zz = backward_derivatives.du(2, 2);
  const auto b_yz =
      static_cast<type_real>(0.5) *
      (backward_derivatives.du(2, 1) + backward_derivatives.du(1, 2));
  const auto b_xz =
      static_cast<type_real>(0.5) *
      (backward_derivatives.du(2, 0) + backward_derivatives.du(0, 2));
  const auto b_xy =
      static_cast<type_real>(0.5) *
      (backward_derivatives.du(1, 0) + backward_derivatives.du(0, 1));

  const auto scale = -dt;
  const auto normal_shear_scale = static_cast<type_real>(2.0) * scale;
  const auto shear_scale = static_cast<type_real>(4.0) * scale;

  const auto rho = adjoint_acceleration.get_data() *
                   backward_displacement.get_data() * scale;
  const auto c11 = scale * ad_xx * b_xx;
  const auto c12 = scale * (ad_xx * b_yy + ad_yy * b_xx);
  const auto c13 = scale * (ad_xx * b_zz + ad_zz * b_xx);
  const auto c14 = normal_shear_scale * (ad_xx * b_yz + ad_yz * b_xx);
  const auto c15 = normal_shear_scale * (ad_xx * b_xz + ad_xz * b_xx);
  const auto c16 = normal_shear_scale * (ad_xx * b_xy + ad_xy * b_xx);
  const auto c22 = scale * ad_yy * b_yy;
  const auto c23 = scale * (ad_yy * b_zz + ad_zz * b_yy);
  const auto c24 = normal_shear_scale * (ad_yy * b_yz + ad_yz * b_yy);
  const auto c25 = normal_shear_scale * (ad_yy * b_xz + ad_xz * b_yy);
  const auto c26 = normal_shear_scale * (ad_yy * b_xy + ad_xy * b_yy);
  const auto c33 = scale * ad_zz * b_zz;
  const auto c34 = normal_shear_scale * (ad_zz * b_yz + ad_yz * b_zz);
  const auto c35 = normal_shear_scale * (ad_zz * b_xz + ad_xz * b_zz);
  const auto c36 = normal_shear_scale * (ad_zz * b_xy + ad_xy * b_zz);
  const auto c44 = shear_scale * ad_yz * b_yz;
  const auto c45 = shear_scale * (ad_yz * b_xz + ad_xz * b_yz);
  const auto c46 = shear_scale * (ad_yz * b_xy + ad_xy * b_yz);
  const auto c55 = shear_scale * ad_xz * b_xz;
  const auto c56 = shear_scale * (ad_xz * b_xy + ad_xy * b_xz);
  const auto c66 = shear_scale * ad_xy * b_xy;

  return { rho, c11, c12, c13, c14, c15, c16, c22, c23, c24, c25,
           c26, c33, c34, c35, c36, c44, c45, c46, c55, c56, c66 };
}

} // namespace frechet_derivative_impl

/**
 * @brief Compute absolute-parameter Fréchet kernels for a 3D fully
 * anisotropic elastic medium.
 *
 * @tparam Tags Compile-time dimension, medium, property, and SIMD tags.
 * @param properties Material properties; unused for absolute-parameter
 * kernels.
 * @param adjoint_velocity Adjoint velocity; unused by displacement kernels.
 * @param adjoint_acceleration Adjoint acceleration.
 * @param backward_displacement Backward/reconstructed displacement.
 * @param adjoint_derivatives Spatial derivatives of the adjoint displacement.
 * @param backward_derivatives Spatial derivatives of the backward displacement.
 * @param dt Time step.
 * @return Density and 21 independent stiffness-kernel contributions.
 */
template <typename Tags>
  requires(Tags::dimension_tag == specfem::element::dimension_tag::dim3 &&
           Tags::medium_tag == specfem::element::medium_tag::elastic &&
           Tags::property_tag == specfem::element::property_tag::anisotropic)
KOKKOS_FUNCTION specfem::point::kernels<Tags> compute_frechet_derivatives(
    const specfem::point::properties<Tags> &properties,
    const specfem::point::velocity<Tags> &adjoint_velocity,
    const specfem::point::acceleration<Tags> &adjoint_acceleration,
    const specfem::point::displacement<Tags> &backward_displacement,
    const specfem::point::field_derivatives<Tags> &adjoint_derivatives,
    const specfem::point::field_derivatives<Tags> &backward_derivatives,
    const type_real &dt) {
  static_cast<void>(properties);
  static_cast<void>(adjoint_velocity);
  return frechet_derivative_impl::compute_anisotropic_frechet_derivatives<Tags>(
      adjoint_acceleration, backward_displacement, adjoint_derivatives,
      backward_derivatives, dt);
}

} // namespace specfem::medium_physics
