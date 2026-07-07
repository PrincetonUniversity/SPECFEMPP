#include "specfem/enums.hpp"
#include "specfem/medium_physics.hpp"
#include "specfem/point.hpp"
#include <gtest/gtest.h>
#include <sstream>

namespace {

// Elastic anisotropic PSV Fréchet derivatives (2D), Tromp et al. 2005 Eq. 15:
//   K_c_ij = -c_ij * dt * ( eps^adj contraction eps^b )
//   K_rho  = -rho  * dt * ( ddot(s^adj) . s^b )
// with strain products
//   c11: eps_xx^adj eps_xx^b
//   c13: eps_xx^adj eps_zz^b + eps_zz^adj eps_xx^b
//   c15: 2 eps_xx^adj eps_xz^b + eps_xz^adj eps_xx^b
//   c33: eps_zz^adj eps_zz^b
//   c35: 2 eps_xz^b eps_zz^adj + eps_xz^adj eps_zz^b
//   c55: 4 eps_xz^adj eps_xz^b

TEST(FrechetDerivatives, ElasticAnisotropic2D_PSV_Basic) {
  static constexpr auto dimension = specfem::element::dimension_tag::dim2;
  static constexpr auto medium_tag = specfem::element::medium_tag::elastic_psv;
  static constexpr auto property_tag =
      specfem::element::property_tag::anisotropic;

  using Tags = specfem::tags::Tags<dimension, medium_tag, property_tag, false>;

  using PropertiesType = specfem::point::properties<Tags>;
  using VelocityType = specfem::point::velocity<Tags>;
  using AccelerationType = specfem::point::acceleration<Tags>;
  using DisplacementType = specfem::point::displacement<Tags>;
  using FieldDerivativesType = specfem::point::field_derivatives<Tags>;
  using KernelsType =
      specfem::point::kernels<dimension, medium_tag, property_tag, false>;

  // c11, c13, c15, c33, c35, c55, c12, c23, c25, rho
  const type_real c11 = 10.0, c13 = 2.0, c15 = 1.0;
  const type_real c33 = 20.0, c35 = 3.0, c55 = 5.0;
  const type_real c12 = 1.0, c23 = 2.0, c25 = 3.0;
  const type_real rho = 4.0;
  const PropertiesType properties(c11, c13, c15, c33, c35, c55, c12, c23, c25,
                                  rho);

  VelocityType adjoint_velocity; // unused by the implementation
  adjoint_velocity(0) = 0.0;
  adjoint_velocity(1) = 0.0;

  AccelerationType adjoint_acceleration;
  adjoint_acceleration(0) = 1.0;
  adjoint_acceleration(1) = 2.0;

  DisplacementType backward_displacement;
  backward_displacement(0) = 3.0;
  backward_displacement(1) = 4.0;

  FieldDerivativesType adjoint_derivatives;
  adjoint_derivatives.du(0, 0) = 1.0;
  adjoint_derivatives.du(1, 1) = 2.0;
  adjoint_derivatives.du(0, 1) = 3.0;
  adjoint_derivatives.du(1, 0) = 4.0;

  FieldDerivativesType backward_derivatives;
  backward_derivatives.du(0, 0) = 0.5;
  backward_derivatives.du(1, 1) = 0.6;
  backward_derivatives.du(0, 1) = 0.7;
  backward_derivatives.du(1, 0) = 0.8;

  const type_real dt = 0.5;

  const KernelsType kernels =
      specfem::medium_physics::compute_frechet_derivatives<Tags>(
          properties, adjoint_velocity, adjoint_acceleration,
          backward_displacement, adjoint_derivatives, backward_derivatives, dt);

  const type_real ad_xx = adjoint_derivatives.du(0, 0);
  const type_real ad_xz =
      0.5 * (adjoint_derivatives.du(1, 0) + adjoint_derivatives.du(0, 1));
  const type_real ad_zz = adjoint_derivatives.du(1, 1);
  const type_real b_xx = backward_derivatives.du(0, 0);
  const type_real b_xz =
      0.5 * (backward_derivatives.du(1, 0) + backward_derivatives.du(0, 1));
  const type_real b_zz = backward_derivatives.du(1, 1);

  const type_real rho_kl = -1.0 * properties.rho() * dt *
                           (adjoint_acceleration(0) * backward_displacement(0) +
                            adjoint_acceleration(1) * backward_displacement(1));
  const type_real c11_kl = -1.0 * (ad_xx * b_xx) * properties.c11() * dt;
  const type_real c13_kl =
      -1.0 * (ad_xx * b_zz + ad_zz * b_xx) * properties.c13() * dt;
  const type_real c15_kl =
      -1.0 * (2.0 * ad_xx * b_xz + ad_xz * b_xx) * properties.c15() * dt;
  const type_real c33_kl = -1.0 * (ad_zz * b_zz) * properties.c33() * dt;
  const type_real c35_kl =
      -1.0 * (2.0 * b_xz * ad_zz + ad_xz * b_zz) * properties.c35() * dt;
  const type_real c55_kl = -1.0 * (4.0 * ad_xz * b_xz) * properties.c55() * dt;

  const KernelsType expected(rho_kl, c11_kl, c13_kl, c15_kl, c33_kl, c35_kl,
                             c55_kl);

  std::ostringstream message;
  message << "Anisotropic PSV 2D Fréchet kernels are not equal to expected "
             "value: \n"
          << "Computed: " << kernels.print() << "\n"
          << "Expected: " << expected.print() << "\n";

  EXPECT_TRUE(kernels == expected) << message.str();
}

TEST(FrechetDerivatives, ElasticAnisotropic2D_PSV_ZeroFields) {
  static constexpr auto dimension = specfem::element::dimension_tag::dim2;
  static constexpr auto medium_tag = specfem::element::medium_tag::elastic_psv;
  static constexpr auto property_tag =
      specfem::element::property_tag::anisotropic;

  using Tags = specfem::tags::Tags<dimension, medium_tag, property_tag, false>;

  using PropertiesType = specfem::point::properties<Tags>;
  using VelocityType = specfem::point::velocity<Tags>;
  using AccelerationType = specfem::point::acceleration<Tags>;
  using DisplacementType = specfem::point::displacement<Tags>;
  using FieldDerivativesType = specfem::point::field_derivatives<Tags>;
  using KernelsType =
      specfem::point::kernels<dimension, medium_tag, property_tag, false>;

  const PropertiesType properties(10.0, 2.0, 1.0, 20.0, 3.0, 5.0, 1.0, 2.0, 3.0,
                                  4.0);

  VelocityType adjoint_velocity;
  adjoint_velocity(0) = 0.0;
  adjoint_velocity(1) = 0.0;
  AccelerationType adjoint_acceleration;
  adjoint_acceleration(0) = 0.0;
  adjoint_acceleration(1) = 0.0;
  DisplacementType backward_displacement;
  backward_displacement(0) = 0.0;
  backward_displacement(1) = 0.0;

  FieldDerivativesType adjoint_derivatives;
  adjoint_derivatives.du(0, 0) = 0.0;
  adjoint_derivatives.du(1, 1) = 0.0;
  adjoint_derivatives.du(0, 1) = 0.0;
  adjoint_derivatives.du(1, 0) = 0.0;
  FieldDerivativesType backward_derivatives;
  backward_derivatives.du(0, 0) = 0.0;
  backward_derivatives.du(1, 1) = 0.0;
  backward_derivatives.du(0, 1) = 0.0;
  backward_derivatives.du(1, 0) = 0.0;

  const type_real dt = 0.5;

  const KernelsType kernels =
      specfem::medium_physics::compute_frechet_derivatives<Tags>(
          properties, adjoint_velocity, adjoint_acceleration,
          backward_displacement, adjoint_derivatives, backward_derivatives, dt);

  const KernelsType expected(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0);

  std::ostringstream message;
  message << "Anisotropic PSV 2D Fréchet kernels should be zero for zero "
             "fields: \n"
          << "Computed: " << kernels.print() << "\n"
          << "Expected: " << expected.print() << "\n";

  EXPECT_TRUE(kernels == expected) << message.str();
}

} // namespace
