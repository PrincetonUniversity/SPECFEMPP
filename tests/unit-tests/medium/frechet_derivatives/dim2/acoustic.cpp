#include "specfem/enums.hpp"
#include "specfem/medium_physics.hpp"
#include "specfem/point.hpp"
#include <gtest/gtest.h>
#include <sstream>

namespace {

// Acoustic isotropic Fréchet derivatives (2D).
//
//   K_rho   = ( grad(phi^adj) . grad(phi^b) ) * (1/rho) * dt
//   K_kappa = ( ddot(phi^adj) * phi^b ) / kappa * dt
//
// The returned kernels object derives rhop = K_rho + K_kappa and
// alpha = 2 * K_kappa via the two-argument acoustic kernels constructor, so the
// expected object is built the same way.

TEST(FrechetDerivatives, AcousticIsotropic2D_Basic) {
  static constexpr auto dimension = specfem::element::dimension_tag::dim2;
  static constexpr auto medium_tag = specfem::element::medium_tag::acoustic;
  static constexpr auto property_tag =
      specfem::element::property_tag::isotropic;

  using Tags = specfem::tags::Tags<dimension, medium_tag, property_tag, false>;

  using PropertiesType = specfem::point::properties<Tags>;
  using VelocityType = specfem::point::velocity<Tags>;
  using AccelerationType = specfem::point::acceleration<Tags>;
  using DisplacementType = specfem::point::displacement<Tags>;
  using FieldDerivativesType = specfem::point::field_derivatives<Tags>;
  using KernelsType =
      specfem::point::kernels<dimension, medium_tag, property_tag, false>;

  const type_real rho_inverse = 2.5;
  const type_real kappa = 8.0;
  const PropertiesType properties(rho_inverse, kappa);

  // adjoint_velocity is unused by the implementation.
  VelocityType adjoint_velocity;
  adjoint_velocity(0) = 0.0;

  AccelerationType adjoint_acceleration;
  adjoint_acceleration(0) = 0.9;

  DisplacementType backward_displacement;
  backward_displacement(0) = -1.1;

  FieldDerivativesType adjoint_derivatives;
  adjoint_derivatives.du(0, 0) = 0.5;
  adjoint_derivatives.du(0, 1) = -0.3;

  FieldDerivativesType backward_derivatives;
  backward_derivatives.du(0, 0) = 1.2;
  backward_derivatives.du(0, 1) = 0.7;

  const type_real dt = 0.01;

  const KernelsType kernels =
      specfem::medium_physics::compute_frechet_derivatives<Tags>(
          properties, adjoint_velocity, adjoint_acceleration,
          backward_displacement, adjoint_derivatives, backward_derivatives, dt);

  const type_real rho_kl =
      (adjoint_derivatives.du(0, 0) * backward_derivatives.du(0, 0) +
       adjoint_derivatives.du(0, 1) * backward_derivatives.du(0, 1)) *
      properties.rho_inverse() * dt;
  const type_real kappa_kl =
      (adjoint_acceleration(0) * backward_displacement(0)) /
      properties.kappa() * dt;
  const KernelsType expected(rho_kl, kappa_kl);

  std::ostringstream message;
  message << "Acoustic 2D Fréchet kernels are not equal to expected value: \n"
          << "Computed: " << kernels.print() << "\n"
          << "Expected: " << expected.print() << "\n";

  EXPECT_TRUE(kernels == expected) << message.str();
}

TEST(FrechetDerivatives, AcousticIsotropic2D_ZeroFields) {
  static constexpr auto dimension = specfem::element::dimension_tag::dim2;
  static constexpr auto medium_tag = specfem::element::medium_tag::acoustic;
  static constexpr auto property_tag =
      specfem::element::property_tag::isotropic;

  using Tags = specfem::tags::Tags<dimension, medium_tag, property_tag, false>;

  using PropertiesType = specfem::point::properties<Tags>;
  using VelocityType = specfem::point::velocity<Tags>;
  using AccelerationType = specfem::point::acceleration<Tags>;
  using DisplacementType = specfem::point::displacement<Tags>;
  using FieldDerivativesType = specfem::point::field_derivatives<Tags>;
  using KernelsType =
      specfem::point::kernels<dimension, medium_tag, property_tag, false>;

  const PropertiesType properties(2.5, 8.0);

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

  const type_real dt = 0.01;

  const KernelsType kernels =
      specfem::medium_physics::compute_frechet_derivatives<Tags>(
          properties, adjoint_velocity, adjoint_acceleration,
          backward_displacement, adjoint_derivatives, backward_derivatives, dt);

  const KernelsType expected(0.0, 0.0);

  std::ostringstream message;
  message << "Acoustic 2D Fréchet kernels should be zero for zero fields: \n"
          << "Computed: " << kernels.print() << "\n"
          << "Expected: " << expected.print() << "\n";

  EXPECT_TRUE(kernels == expected) << message.str();
}

// Hardcoded-literal case: expected kernel values computed by hand, independent
// of the formula-mirroring above.
//   rho_kl   = (1*3 + 2*4) * 2 * 0.5 = 11
//   kappa_kl = (5 * 6) / 4 * 0.5     = 3.75
//   rhop     = 11 + 3.75 = 14.75,  alpha = 2 * 3.75 = 7.5
TEST(FrechetDerivatives, AcousticIsotropic2D_Literal) {
  static constexpr auto dimension = specfem::element::dimension_tag::dim2;
  static constexpr auto medium_tag = specfem::element::medium_tag::acoustic;
  static constexpr auto property_tag =
      specfem::element::property_tag::isotropic;

  using Tags = specfem::tags::Tags<dimension, medium_tag, property_tag, false>;

  using PropertiesType = specfem::point::properties<Tags>;
  using VelocityType = specfem::point::velocity<Tags>;
  using AccelerationType = specfem::point::acceleration<Tags>;
  using DisplacementType = specfem::point::displacement<Tags>;
  using FieldDerivativesType = specfem::point::field_derivatives<Tags>;
  using KernelsType =
      specfem::point::kernels<dimension, medium_tag, property_tag, false>;

  const type_real rho_inverse = 2.0;
  const type_real kappa = 4.0;
  const PropertiesType properties(rho_inverse, kappa);

  VelocityType adjoint_velocity;
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

  KernelsType expected;
  expected.rho() = 11.0;
  expected.kappa() = 3.75;
  expected.rhop() = 14.75;
  expected.alpha() = 7.5;

  std::ostringstream message;
  message << "Acoustic 2D Fréchet kernels are not equal to literal value: \n"
          << "Computed: " << kernels.print() << "\n"
          << "Expected: " << expected.print() << "\n";

  EXPECT_TRUE(kernels == expected) << message.str();
}

} // namespace
