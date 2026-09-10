#pragma once

#include "specfem/element.hpp"
#include "specfem/point.hpp"
#include <Kokkos_Core.hpp>

namespace specfem {
namespace medium_physics {

/**
 * @defgroup specfem_medium_frechet_derivative_dim3_elastic_anisotropic
 *
 */

/**
 * @ingroup specfem_medium_frechet_derivative_dim3_elastic_anisotropic
 * @brief Compute Fréchet derivatives for 3D elastic anisotropic media.
 *
 * Calculates the absolute-parameter density kernel and the 21 independent
 * stiffness kernels in Voigt order. This follows SPECFEM3D's fully
 * anisotropic kernel expressions. Normal strain products use unit weight,
 * mixed normal/shear products use a factor of two, and shear/shear products
 * use a factor of four.
 *
 * @tparam Tags Compile-time tag bundle (dimension, medium, property, SIMD)
 *
 * @param properties Elastic material properties (unused for absolute kernels)
 * @param adjoint_velocity Adjoint velocity field (unused)
 * @param adjoint_acceleration Adjoint acceleration field
 * @param backward_displacement Backward displacement field
 * @param adjoint_derivatives Spatial derivatives of adjoint field
 * @param backward_derivatives Spatial derivatives of backward field
 * @param dt Time step size
 * @return Point kernels containing density and stiffness sensitivities
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

  (void)properties;
  (void)adjoint_velocity;

  const auto adjoint_xx = adjoint_derivatives.du(0, 0);
  const auto adjoint_yy = adjoint_derivatives.du(1, 1);
  const auto adjoint_zz = adjoint_derivatives.du(2, 2);
  const auto adjoint_yz =
      static_cast<type_real>(0.5) *
      (adjoint_derivatives.du(2, 1) + adjoint_derivatives.du(1, 2));
  const auto adjoint_xz =
      static_cast<type_real>(0.5) *
      (adjoint_derivatives.du(2, 0) + adjoint_derivatives.du(0, 2));
  const auto adjoint_xy =
      static_cast<type_real>(0.5) *
      (adjoint_derivatives.du(1, 0) + adjoint_derivatives.du(0, 1));

  const auto backward_xx = backward_derivatives.du(0, 0);
  const auto backward_yy = backward_derivatives.du(1, 1);
  const auto backward_zz = backward_derivatives.du(2, 2);
  const auto backward_yz =
      static_cast<type_real>(0.5) *
      (backward_derivatives.du(2, 1) + backward_derivatives.du(1, 2));
  const auto backward_xz =
      static_cast<type_real>(0.5) *
      (backward_derivatives.du(2, 0) + backward_derivatives.du(0, 2));
  const auto backward_xy =
      static_cast<type_real>(0.5) *
      (backward_derivatives.du(1, 0) + backward_derivatives.du(0, 1));

  const auto scale = static_cast<type_real>(-1.0) * dt;
  const auto normal_shear_scale = static_cast<type_real>(2.0) * scale;
  const auto shear_scale = static_cast<type_real>(4.0) * scale;

  const auto rho_kl = adjoint_acceleration.get_data() *
                      backward_displacement.get_data() * scale;
  const auto c11_kl = scale * adjoint_xx * backward_xx;
  const auto c12_kl =
      scale * (adjoint_xx * backward_yy + adjoint_yy * backward_xx);
  const auto c13_kl =
      scale * (adjoint_xx * backward_zz + adjoint_zz * backward_xx);
  const auto c14_kl = normal_shear_scale *
                      (adjoint_xx * backward_yz + adjoint_yz * backward_xx);
  const auto c15_kl = normal_shear_scale *
                      (adjoint_xx * backward_xz + adjoint_xz * backward_xx);
  const auto c16_kl = normal_shear_scale *
                      (adjoint_xx * backward_xy + adjoint_xy * backward_xx);
  const auto c22_kl = scale * adjoint_yy * backward_yy;
  const auto c23_kl =
      scale * (adjoint_yy * backward_zz + adjoint_zz * backward_yy);
  const auto c24_kl = normal_shear_scale *
                      (adjoint_yy * backward_yz + adjoint_yz * backward_yy);
  const auto c25_kl = normal_shear_scale *
                      (adjoint_yy * backward_xz + adjoint_xz * backward_yy);
  const auto c26_kl = normal_shear_scale *
                      (adjoint_yy * backward_xy + adjoint_xy * backward_yy);
  const auto c33_kl = scale * adjoint_zz * backward_zz;
  const auto c34_kl = normal_shear_scale *
                      (adjoint_zz * backward_yz + adjoint_yz * backward_zz);
  const auto c35_kl = normal_shear_scale *
                      (adjoint_zz * backward_xz + adjoint_xz * backward_zz);
  const auto c36_kl = normal_shear_scale *
                      (adjoint_zz * backward_xy + adjoint_xy * backward_zz);
  const auto c44_kl = shear_scale * adjoint_yz * backward_yz;
  const auto c45_kl =
      shear_scale * (adjoint_yz * backward_xz + adjoint_xz * backward_yz);
  const auto c46_kl =
      shear_scale * (adjoint_yz * backward_xy + adjoint_xy * backward_yz);
  const auto c55_kl = shear_scale * adjoint_xz * backward_xz;
  const auto c56_kl =
      shear_scale * (adjoint_xz * backward_xy + adjoint_xy * backward_xz);
  const auto c66_kl = shear_scale * adjoint_xy * backward_xy;

  return { rho_kl, c11_kl, c12_kl, c13_kl, c14_kl, c15_kl, c16_kl, c22_kl,
           c23_kl, c24_kl, c25_kl, c26_kl, c33_kl, c34_kl, c35_kl, c36_kl,
           c44_kl, c45_kl, c46_kl, c55_kl, c56_kl, c66_kl };
}

} // namespace medium_physics
} // namespace specfem
