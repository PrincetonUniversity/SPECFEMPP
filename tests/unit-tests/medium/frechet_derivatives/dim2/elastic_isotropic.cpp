#include "specfem/enums.hpp"
#include "specfem/medium_physics.hpp"
#include "specfem/point.hpp"
#include <gtest/gtest.h>
#include <sstream>

namespace {

// Elastic isotropic Fréchet derivatives (2D), following Tromp et al. 2005.
//
// PSV (kap = lambdaplus2mu - mu, i.e. the 2D modulus lambda + mu):
//   K_kappa = -kap * dt * div(s^adj) * div(s^b)
//   K_mu    = -2 mu * dt * [ eps^adj : eps^b - 1/3 * div(s^adj) div(s^b) ]
//   K_rho   = -rho * dt * ( ddot(s^adj) . s^b )
//   K_rhop  = K_rho + K_kappa + K_mu
//   K_beta  = 2 ( K_mu - 4/3 mu/kap * K_kappa )
//   K_alpha = 2 ( 1 + 4/3 mu/kap ) * K_kappa

TEST(FrechetDerivatives, ElasticIsotropic2D_PSV_Basic) {
  static constexpr auto dimension = specfem::element::dimension_tag::dim2;
  static constexpr auto medium_tag = specfem::element::medium_tag::elastic_psv;
  static constexpr auto property_tag =
      specfem::element::property_tag::isotropic;

  using Tags = specfem::tags::Tags<dimension, medium_tag, property_tag, false>;

  using PropertiesType = specfem::point::properties<Tags>;
  using VelocityType = specfem::point::velocity<Tags>;
  using AccelerationType = specfem::point::acceleration<Tags>;
  using DisplacementType = specfem::point::displacement<Tags>;
  using FieldDerivativesType = specfem::point::field_derivatives<Tags>;
  using KernelsType = specfem::point::kernels<Tags>;

  const type_real kappa = 2.0;
  const type_real mu = 3.0;
  const type_real rho = 4.0;
  const PropertiesType properties(kappa, mu, rho);

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

  const type_real kap = properties.lambdaplus2mu() - properties.mu();
  const type_real ad_xx = adjoint_derivatives.du(0, 0);
  const type_real ad_xz =
      0.5 * (adjoint_derivatives.du(1, 0) + adjoint_derivatives.du(0, 1));
  const type_real ad_zz = adjoint_derivatives.du(1, 1);
  const type_real b_xx = backward_derivatives.du(0, 0);
  const type_real b_xz =
      0.5 * (backward_derivatives.du(1, 0) + backward_derivatives.du(0, 1));
  const type_real b_zz = backward_derivatives.du(1, 1);

  type_real kappa_kl = (ad_xx + ad_zz) * (b_xx + b_zz);
  type_real mu_kl =
      ad_xx * b_xx + ad_zz * b_zz + 2.0 * ad_xz * b_xz - (1.0 / 3.0) * kappa_kl;
  type_real rho_kl = adjoint_acceleration(0) * backward_displacement(0) +
                     adjoint_acceleration(1) * backward_displacement(1);

  kappa_kl = -1.0 * kap * dt * kappa_kl;
  mu_kl = -2.0 * properties.mu() * dt * mu_kl;
  rho_kl = -1.0 * properties.rho() * dt * rho_kl;

  const type_real rhop_kl = rho_kl + kappa_kl + mu_kl;
  const type_real beta_kl =
      2.0 * (mu_kl - (4.0 / 3.0) * properties.mu() / kap * kappa_kl);
  const type_real alpha_kl =
      2.0 * (1.0 + (4.0 / 3.0) * properties.mu() / kap) * kappa_kl;

  const KernelsType expected(rho_kl, mu_kl, kappa_kl, rhop_kl, alpha_kl,
                             beta_kl);

  std::ostringstream message;
  message << "PSV 2D Fréchet kernels are not equal to expected value: \n"
          << "Computed: " << kernels.print() << "\n"
          << "Expected: " << expected.print() << "\n";

  EXPECT_TRUE(kernels == expected) << message.str();
}

TEST(FrechetDerivatives, ElasticIsotropic2D_PSV_ZeroFields) {
  static constexpr auto dimension = specfem::element::dimension_tag::dim2;
  static constexpr auto medium_tag = specfem::element::medium_tag::elastic_psv;
  static constexpr auto property_tag =
      specfem::element::property_tag::isotropic;

  using Tags = specfem::tags::Tags<dimension, medium_tag, property_tag, false>;

  using PropertiesType = specfem::point::properties<Tags>;
  using VelocityType = specfem::point::velocity<Tags>;
  using AccelerationType = specfem::point::acceleration<Tags>;
  using DisplacementType = specfem::point::displacement<Tags>;
  using FieldDerivativesType = specfem::point::field_derivatives<Tags>;
  using KernelsType = specfem::point::kernels<Tags>;

  const PropertiesType properties(2.0, 3.0, 4.0);

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

  const KernelsType expected(0.0, 0.0, 0.0, 0.0, 0.0, 0.0);

  std::ostringstream message;
  message << "PSV 2D Fréchet kernels should be zero for zero fields: \n"
          << "Computed: " << kernels.print() << "\n"
          << "Expected: " << expected.print() << "\n";

  EXPECT_TRUE(kernels == expected) << message.str();
}

// Hardcoded-literal case with kappa=3, mu=3, rho=2, dt=1 so that
// kap = lambdaplus2mu - mu = (3 + 4/3*3) - 3 = 4, and unit strain/field values:
//   div^adj = div^b = 2, eps:eps = 2, ddot(s^adj).s^b = 1
//   K_kappa = -4*1*4 = -16,  K_mu = -2*3*1*(2/3) = -4,  K_rho = -2*1*1 = -2
//   K_rhop  = -22,  K_beta = 2*(-4 - 1*(-16)) = 24,  K_alpha = 2*2*(-16) = -64
TEST(FrechetDerivatives, ElasticIsotropic2D_PSV_Literal) {
  static constexpr auto dimension = specfem::element::dimension_tag::dim2;
  static constexpr auto medium_tag = specfem::element::medium_tag::elastic_psv;
  static constexpr auto property_tag =
      specfem::element::property_tag::isotropic;

  using Tags = specfem::tags::Tags<dimension, medium_tag, property_tag, false>;

  using PropertiesType = specfem::point::properties<Tags>;
  using VelocityType = specfem::point::velocity<Tags>;
  using AccelerationType = specfem::point::acceleration<Tags>;
  using DisplacementType = specfem::point::displacement<Tags>;
  using FieldDerivativesType = specfem::point::field_derivatives<Tags>;
  using KernelsType = specfem::point::kernels<Tags>;

  const PropertiesType properties(3.0, 3.0, 2.0);

  VelocityType adjoint_velocity;
  adjoint_velocity(0) = 0.0;
  adjoint_velocity(1) = 0.0;
  AccelerationType adjoint_acceleration;
  adjoint_acceleration(0) = 1.0;
  adjoint_acceleration(1) = 0.0;
  DisplacementType backward_displacement;
  backward_displacement(0) = 1.0;
  backward_displacement(1) = 0.0;

  FieldDerivativesType adjoint_derivatives;
  adjoint_derivatives.du(0, 0) = 1.0;
  adjoint_derivatives.du(1, 1) = 1.0;
  adjoint_derivatives.du(0, 1) = 0.0;
  adjoint_derivatives.du(1, 0) = 0.0;
  FieldDerivativesType backward_derivatives;
  backward_derivatives.du(0, 0) = 1.0;
  backward_derivatives.du(1, 1) = 1.0;
  backward_derivatives.du(0, 1) = 0.0;
  backward_derivatives.du(1, 0) = 0.0;

  const type_real dt = 1.0;

  const KernelsType kernels =
      specfem::medium_physics::compute_frechet_derivatives<Tags>(
          properties, adjoint_velocity, adjoint_acceleration,
          backward_displacement, adjoint_derivatives, backward_derivatives, dt);

  KernelsType expected;
  expected.rho() = -2.0;
  expected.mu() = -4.0;
  expected.kappa() = -16.0;
  expected.rhop() = -22.0;
  expected.alpha() = -64.0;
  expected.beta() = 24.0;

  std::ostringstream message;
  message << "PSV 2D Fréchet kernels are not equal to literal value: \n"
          << "Computed: " << kernels.print() << "\n"
          << "Expected: " << expected.print() << "\n";

  EXPECT_TRUE(kernels == expected) << message.str();
}

// SH waves: only shear modulus and density kernels are non-zero.
//   K_mu    = -2 mu * dt * 1/2 ( du^adj_y/dx du^b_y/dx + du^adj_y/dz du^b_y/dz
//   ) K_rho   = -rho * dt * ( ddot(u^adj_y) u^b_y ) K_kappa = K_alpha = 0,
//   K_rhop = K_rho + K_mu,  K_beta = 2 K_mu

TEST(FrechetDerivatives, ElasticIsotropic2D_SH_Basic) {
  static constexpr auto dimension = specfem::element::dimension_tag::dim2;
  static constexpr auto medium_tag = specfem::element::medium_tag::elastic_sh;
  static constexpr auto property_tag =
      specfem::element::property_tag::isotropic;

  using Tags = specfem::tags::Tags<dimension, medium_tag, property_tag, false>;

  using PropertiesType = specfem::point::properties<Tags>;
  using VelocityType = specfem::point::velocity<Tags>;
  using AccelerationType = specfem::point::acceleration<Tags>;
  using DisplacementType = specfem::point::displacement<Tags>;
  using FieldDerivativesType = specfem::point::field_derivatives<Tags>;
  using KernelsType = specfem::point::kernels<Tags>;

  const type_real kappa = 2.0;
  const type_real mu = 3.0;
  const type_real rho = 2.0;
  const PropertiesType properties(kappa, mu, rho);

  VelocityType adjoint_velocity; // unused by the implementation
  adjoint_velocity(0) = 0.0;
  AccelerationType adjoint_acceleration;
  adjoint_acceleration(0) = 5.0;
  DisplacementType backward_displacement;
  backward_displacement(0) = 6.0;

  FieldDerivativesType adjoint_derivatives;
  adjoint_derivatives.du(0, 0) = 1.0;
  adjoint_derivatives.du(0, 1) = 2.0;
  FieldDerivativesType backward_derivatives;
  backward_derivatives.du(0, 0) = 3.0;
  backward_derivatives.du(0, 1) = 4.0;

  const type_real dt = 0.5;

  const KernelsType kernels =
      specfem::medium_physics::compute_frechet_derivatives<Tags>(
          properties, adjoint_velocity, adjoint_acceleration,
          backward_displacement, adjoint_derivatives, backward_derivatives, dt);

  const type_real mu_kl =
      -2.0 * properties.mu() * dt * 0.5 *
      (adjoint_derivatives.du(0, 0) * backward_derivatives.du(0, 0) +
       adjoint_derivatives.du(0, 1) * backward_derivatives.du(0, 1));
  const type_real rho_kl = -1.0 * properties.rho() * dt *
                           (adjoint_acceleration(0) * backward_displacement(0));
  const type_real kappa_kl = 0.0;
  const type_real rhop_kl = rho_kl + kappa_kl + mu_kl;
  const type_real alpha_kl = 0.0;
  const type_real beta_kl = 2.0 * mu_kl;

  const KernelsType expected(rho_kl, mu_kl, kappa_kl, rhop_kl, alpha_kl,
                             beta_kl);

  std::ostringstream message;
  message << "SH 2D Fréchet kernels are not equal to expected value: \n"
          << "Computed: " << kernels.print() << "\n"
          << "Expected: " << expected.print() << "\n";

  EXPECT_TRUE(kernels == expected) << message.str();
}

TEST(FrechetDerivatives, ElasticIsotropic2D_SH_ZeroFields) {
  static constexpr auto dimension = specfem::element::dimension_tag::dim2;
  static constexpr auto medium_tag = specfem::element::medium_tag::elastic_sh;
  static constexpr auto property_tag =
      specfem::element::property_tag::isotropic;

  using Tags = specfem::tags::Tags<dimension, medium_tag, property_tag, false>;

  using PropertiesType = specfem::point::properties<Tags>;
  using VelocityType = specfem::point::velocity<Tags>;
  using AccelerationType = specfem::point::acceleration<Tags>;
  using DisplacementType = specfem::point::displacement<Tags>;
  using FieldDerivativesType = specfem::point::field_derivatives<Tags>;
  using KernelsType = specfem::point::kernels<Tags>;

  const PropertiesType properties(2.0, 3.0, 2.0);

  VelocityType adjoint_velocity;
  adjoint_velocity(0) = 0.0;
  AccelerationType adjoint_acceleration;
  adjoint_acceleration(0) = 0.0;
  DisplacementType backward_displacement;
  backward_displacement(0) = 0.0;

  FieldDerivativesType adjoint_derivatives;
  adjoint_derivatives.du(0, 0) = 0.0;
  adjoint_derivatives.du(0, 1) = 0.0;
  FieldDerivativesType backward_derivatives;
  backward_derivatives.du(0, 0) = 0.0;
  backward_derivatives.du(0, 1) = 0.0;

  const type_real dt = 0.5;

  const KernelsType kernels =
      specfem::medium_physics::compute_frechet_derivatives<Tags>(
          properties, adjoint_velocity, adjoint_acceleration,
          backward_displacement, adjoint_derivatives, backward_derivatives, dt);

  const KernelsType expected(0.0, 0.0, 0.0, 0.0, 0.0, 0.0);

  std::ostringstream message;
  message << "SH 2D Fréchet kernels should be zero for zero fields: \n"
          << "Computed: " << kernels.print() << "\n"
          << "Expected: " << expected.print() << "\n";

  EXPECT_TRUE(kernels == expected) << message.str();
}

} // namespace
