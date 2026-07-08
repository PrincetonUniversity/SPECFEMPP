#include "specfem/enums.hpp"
#include "specfem/medium_physics.hpp"
#include "specfem/point.hpp"
#include <gtest/gtest.h>
#include <sstream>

namespace {

// Acoustic isotropic Fréchet derivatives (3D). The implementation is
// dimension-agnostic; here the density kernel sums the gradient product over
// all three spatial dimensions:
//   K_rho   = ( sum_{i<3} grad(phi^adj)_i grad(phi^b)_i ) * (1/rho) * dt
//   K_kappa = ( ddot(phi^adj) * phi^b ) / kappa * dt

TEST(FrechetDerivatives, AcousticIsotropic3D_Basic) {
  static constexpr auto dimension = specfem::element::dimension_tag::dim3;
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
  const type_real kappa = 5.0;
  const PropertiesType properties(rho_inverse, kappa);

  VelocityType adjoint_velocity; // unused by the implementation
  adjoint_velocity(0) = 0.0;
  AccelerationType adjoint_acceleration;
  adjoint_acceleration(0) = 7.0;
  DisplacementType backward_displacement;
  backward_displacement(0) = 8.0;

  FieldDerivativesType adjoint_derivatives;
  adjoint_derivatives.du(0, 0) = 1.0;
  adjoint_derivatives.du(0, 1) = 2.0;
  adjoint_derivatives.du(0, 2) = 3.0;
  FieldDerivativesType backward_derivatives;
  backward_derivatives.du(0, 0) = 4.0;
  backward_derivatives.du(0, 1) = 5.0;
  backward_derivatives.du(0, 2) = 6.0;

  const type_real dt = 0.1;

  const KernelsType kernels =
      specfem::medium_physics::compute_frechet_derivatives<Tags>(
          properties, adjoint_velocity, adjoint_acceleration,
          backward_displacement, adjoint_derivatives, backward_derivatives, dt);

  const type_real rho_kl =
      (adjoint_derivatives.du(0, 0) * backward_derivatives.du(0, 0) +
       adjoint_derivatives.du(0, 1) * backward_derivatives.du(0, 1) +
       adjoint_derivatives.du(0, 2) * backward_derivatives.du(0, 2)) *
      properties.rho_inverse() * dt;
  const type_real kappa_kl =
      (adjoint_acceleration(0) * backward_displacement(0)) /
      properties.kappa() * dt;
  const KernelsType expected(rho_kl, kappa_kl);

  std::ostringstream message;
  message << "Acoustic 3D Fréchet kernels are not equal to expected value: \n"
          << "Computed: " << kernels.print() << "\n"
          << "Expected: " << expected.print() << "\n";

  EXPECT_TRUE(kernels == expected) << message.str();
}

TEST(FrechetDerivatives, AcousticIsotropic3D_ZeroFields) {
  static constexpr auto dimension = specfem::element::dimension_tag::dim3;
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

  const PropertiesType properties(2.0, 5.0);

  VelocityType adjoint_velocity;
  adjoint_velocity(0) = 0.0;
  AccelerationType adjoint_acceleration;
  adjoint_acceleration(0) = 0.0;
  DisplacementType backward_displacement;
  backward_displacement(0) = 0.0;

  FieldDerivativesType adjoint_derivatives;
  adjoint_derivatives.du(0, 0) = 0.0;
  adjoint_derivatives.du(0, 1) = 0.0;
  adjoint_derivatives.du(0, 2) = 0.0;
  FieldDerivativesType backward_derivatives;
  backward_derivatives.du(0, 0) = 0.0;
  backward_derivatives.du(0, 1) = 0.0;
  backward_derivatives.du(0, 2) = 0.0;

  const type_real dt = 0.1;

  const KernelsType kernels =
      specfem::medium_physics::compute_frechet_derivatives<Tags>(
          properties, adjoint_velocity, adjoint_acceleration,
          backward_displacement, adjoint_derivatives, backward_derivatives, dt);

  const KernelsType expected(0.0, 0.0);

  std::ostringstream message;
  message << "Acoustic 3D Fréchet kernels should be zero for zero fields: \n"
          << "Computed: " << kernels.print() << "\n"
          << "Expected: " << expected.print() << "\n";

  EXPECT_TRUE(kernels == expected) << message.str();
}

} // namespace
