#include "specfem/enums.hpp"
#include "specfem/medium_physics.hpp"
#include "specfem/point.hpp"
#include <gtest/gtest.h>
#include <sstream>

namespace {

// Test 1: Stress computation with displacement gradients (classical + nu terms)
TEST(Stress, ElasticIsotropicCosserat3D_DisplacementGradients) {
  static constexpr auto dimension = specfem::element::dimension_tag::dim3;
  static constexpr auto property_tag =
      specfem::element::property_tag::isotropic_cosserat;
  static constexpr auto elasticTag = specfem::element::medium_tag::elastic_spin;

  using Tags = specfem::tags::Tags<dimension, elasticTag, property_tag, false>;
  using PropertiesType = specfem::point::properties<Tags>;
  using FieldDerivativesType = specfem::point::field_derivatives<Tags>;
  using StressType = specfem::point::stress<Tags>;

  const type_real rho = 4.0;
  const type_real kappa = 2.0;
  const type_real mu = 3.0;
  const type_real nu = 0.25;
  const type_real j = 0.01;
  const type_real lambda_c = 1.0;
  const type_real mu_c = 2.0;
  const type_real nu_c = 0.1;
  const PropertiesType properties(rho, kappa, mu, nu, j, lambda_c, mu_c, nu_c);

  const type_real lambda = properties.lambda();

  // Displacement gradients
  FieldDerivativesType field_derivatives;
  field_derivatives.du(0, 0) = 1.0; // du_x/dx
  field_derivatives.du(1, 1) = 2.0; // du_y/dy
  field_derivatives.du(2, 2) = 3.0; // du_z/dz
  field_derivatives.du(0, 1) = 4.0; // du_x/dy
  field_derivatives.du(1, 0) = 5.0; // du_y/dx
  field_derivatives.du(0, 2) = 6.0; // du_x/dz
  field_derivatives.du(2, 0) = 7.0; // du_z/dx
  field_derivatives.du(1, 2) = 8.0; // du_y/dz
  field_derivatives.du(2, 1) = 9.0; // du_z/dy

  // Rotation gradients (all zero in this test)
  field_derivatives.du(3, 0) = 0.0; // d(phi_x)/dx
  field_derivatives.du(3, 1) = 0.0; // d(phi_x)/dy
  field_derivatives.du(3, 2) = 0.0; // d(phi_x)/dz
  field_derivatives.du(4, 0) = 0.0; // d(phi_y)/dx
  field_derivatives.du(4, 1) = 0.0; // d(phi_y)/dy
  field_derivatives.du(4, 2) = 0.0; // d(phi_y)/dz
  field_derivatives.du(5, 0) = 0.0; // d(phi_z)/dx
  field_derivatives.du(5, 1) = 0.0; // d(phi_z)/dy
  field_derivatives.du(5, 2) = 0.0; // d(phi_z)/dz

  const StressType stress = specfem::medium_physics::compute_stress<Tags>(
      properties, field_derivatives);

  StressType expected_stress;

  // Classical force stress components (with nu asymmetry)
  const type_real div_u = 1.0 + 2.0 + 3.0;                   // = 6.0
  expected_stress.T(0, 0) = lambda * div_u + 2.0 * mu * 1.0; // sigma_xx
  expected_stress.T(1, 1) = lambda * div_u + 2.0 * mu * 2.0; // sigma_yy
  expected_stress.T(2, 2) = lambda * div_u + 2.0 * mu * 3.0; // sigma_zz

  // Shear stresses with nu coupling
  expected_stress.T(1, 0) = mu * (5.0 + 4.0) + nu * (5.0 - 4.0); // sigma_xy
  expected_stress.T(0, 1) = mu * (4.0 + 5.0) + nu * (4.0 - 5.0); // sigma_yx
  expected_stress.T(2, 0) = mu * (7.0 + 6.0) + nu * (7.0 - 6.0); // sigma_xz
  expected_stress.T(0, 2) = mu * (6.0 + 7.0) + nu * (6.0 - 7.0); // sigma_zx
  expected_stress.T(2, 1) = mu * (9.0 + 8.0) + nu * (9.0 - 8.0); // sigma_yz
  expected_stress.T(1, 2) = mu * (8.0 + 9.0) + nu * (8.0 - 9.0); // sigma_zy

  // Couple stress components (zero since rotation gradients are zero)
  expected_stress.T(3, 0) = 0.0; // sigma_c_xx
  expected_stress.T(4, 0) = 0.0; // sigma_c_xy
  expected_stress.T(5, 0) = 0.0; // sigma_c_xz
  expected_stress.T(3, 1) = 0.0; // sigma_c_yx
  expected_stress.T(4, 1) = 0.0; // sigma_c_yy
  expected_stress.T(5, 1) = 0.0; // sigma_c_yz
  expected_stress.T(3, 2) = 0.0; // sigma_c_zx
  expected_stress.T(4, 2) = 0.0; // sigma_c_zy
  expected_stress.T(5, 2) = 0.0; // sigma_c_zz

  std::ostringstream message;
  message << "3D stress tensor with displacement gradients is incorrect: \n"
          << "Computed:\n"
          << stress.print() << "\n"
          << "Expected:\n"
          << expected_stress.print() << "\n";

  EXPECT_TRUE(stress == expected_stress) << message.str();
}

// Test 2: Complete stress computation with both displacement and rotation
// gradients
TEST(Stress, ElasticIsotropicCosserat3D_DisplacementAndRotationGradients) {
  static constexpr auto dimension = specfem::element::dimension_tag::dim3;
  static constexpr auto property_tag =
      specfem::element::property_tag::isotropic_cosserat;
  static constexpr auto elasticTag = specfem::element::medium_tag::elastic_spin;

  using Tags = specfem::tags::Tags<dimension, elasticTag, property_tag, false>;
  using PropertiesType = specfem::point::properties<Tags>;
  using FieldDerivativesType = specfem::point::field_derivatives<Tags>;
  using StressType = specfem::point::stress<Tags>;

  const type_real rho = 4.0;
  const type_real kappa = 2.0;
  const type_real mu = 3.0;
  const type_real nu = 0.25;
  const type_real j = 0.01;
  const type_real lambda_c = 1.0;
  const type_real mu_c = 2.0;
  const type_real nu_c = 0.1;
  const PropertiesType properties(rho, kappa, mu, nu, j, lambda_c, mu_c, nu_c);

  const type_real lambda = properties.lambda();

  // Both displacement and rotation gradients
  FieldDerivativesType field_derivatives;
  // Displacement gradients
  field_derivatives.du(0, 0) = 1.0;
  field_derivatives.du(1, 1) = 2.0;
  field_derivatives.du(2, 2) = 3.0;
  field_derivatives.du(0, 1) = 4.0;
  field_derivatives.du(1, 0) = 5.0;
  field_derivatives.du(0, 2) = 6.0;
  field_derivatives.du(2, 0) = 7.0;
  field_derivatives.du(1, 2) = 8.0;
  field_derivatives.du(2, 1) = 9.0;

  // Rotation gradients
  field_derivatives.du(3, 0) = 0.1; // d(phi_x)/dx
  field_derivatives.du(3, 1) = 0.2; // d(phi_x)/dy
  field_derivatives.du(3, 2) = 0.3; // d(phi_x)/dz
  field_derivatives.du(4, 0) = 0.4; // d(phi_y)/dx
  field_derivatives.du(4, 1) = 0.5; // d(phi_y)/dy
  field_derivatives.du(4, 2) = 0.6; // d(phi_y)/dz
  field_derivatives.du(5, 0) = 0.7; // d(phi_z)/dx
  field_derivatives.du(5, 1) = 0.8; // d(phi_z)/dy
  field_derivatives.du(5, 2) = 0.9; // d(phi_z)/dz

  const StressType stress = specfem::medium_physics::compute_stress<Tags>(
      properties, field_derivatives);

  StressType expected_stress;

  // Classical force stress (same as before)
  const type_real div_u = 6.0;
  expected_stress.T(0, 0) = lambda * div_u + 2.0 * mu * 1.0;
  expected_stress.T(1, 1) = lambda * div_u + 2.0 * mu * 2.0;
  expected_stress.T(2, 2) = lambda * div_u + 2.0 * mu * 3.0;

  expected_stress.T(1, 0) = mu * (5.0 + 4.0) + nu * (5.0 - 4.0);
  expected_stress.T(0, 1) = mu * (4.0 + 5.0) + nu * (4.0 - 5.0);
  expected_stress.T(2, 0) = mu * (7.0 + 6.0) + nu * (7.0 - 6.0);
  expected_stress.T(0, 2) = mu * (6.0 + 7.0) + nu * (6.0 - 7.0);
  expected_stress.T(2, 1) = mu * (9.0 + 8.0) + nu * (9.0 - 8.0);
  expected_stress.T(1, 2) = mu * (8.0 + 9.0) + nu * (8.0 - 9.0);

  // Couple stress from rotation gradients
  const type_real div_phi = 0.1 + 0.5 + 0.9;                       // = 1.5
  expected_stress.T(3, 0) = lambda_c * div_phi + 2.0 * mu_c * 0.1; // sigma_c_xx
  expected_stress.T(4, 0) =
      mu_c * (0.4 + 0.2) + nu_c * (0.4 - 0.2); // sigma_c_xy
  expected_stress.T(5, 0) =
      mu_c * (0.7 + 0.3) + nu_c * (0.7 - 0.3); // sigma_c_xz
  expected_stress.T(3, 1) =
      mu_c * (0.2 + 0.4) + nu_c * (0.2 - 0.4);                     // sigma_c_yx
  expected_stress.T(4, 1) = lambda_c * div_phi + 2.0 * mu_c * 0.5; // sigma_c_yy
  expected_stress.T(5, 1) =
      mu_c * (0.8 + 0.6) + nu_c * (0.8 - 0.6); // sigma_c_yz
  expected_stress.T(3, 2) =
      mu_c * (0.3 + 0.7) + nu_c * (0.3 - 0.7); // sigma_c_zx
  expected_stress.T(4, 2) =
      mu_c * (0.6 + 0.8) + nu_c * (0.6 - 0.8);                     // sigma_c_zy
  expected_stress.T(5, 2) = lambda_c * div_phi + 2.0 * mu_c * 0.9; // sigma_c_zz

  std::ostringstream message;
  message << "3D stress tensor with rotation gradients is incorrect: \n"
          << "Computed:\n"
          << stress.print() << "\n"
          << "Expected:\n"
          << expected_stress.print() << "\n";

  EXPECT_TRUE(stress == expected_stress) << message.str();
}

// Test 3: Zero derivatives - should produce zero stress
TEST(Stress, ElasticIsotropicCosserat3D_ZeroDerivatives) {
  static constexpr auto dimension = specfem::element::dimension_tag::dim3;
  static constexpr auto property_tag =
      specfem::element::property_tag::isotropic_cosserat;
  static constexpr auto elasticTag = specfem::element::medium_tag::elastic_spin;

  using Tags = specfem::tags::Tags<dimension, elasticTag, property_tag, false>;
  using PropertiesType = specfem::point::properties<Tags>;
  using FieldDerivativesType = specfem::point::field_derivatives<Tags>;
  using StressType = specfem::point::stress<Tags>;

  const type_real rho = 4.0;
  const type_real kappa = 2.0;
  const type_real mu = 3.0;
  const type_real nu = 0.25;
  const type_real j = 0.01;
  const type_real lambda_c = 1.0;
  const type_real mu_c = 2.0;
  const type_real nu_c = 0.1;
  const PropertiesType properties(rho, kappa, mu, nu, j, lambda_c, mu_c, nu_c);

  FieldDerivativesType field_derivatives;
  // All derivatives zero
  for (int i = 0; i < 6; ++i) {
    for (int j = 0; j < 3; ++j) {
      field_derivatives.du(i, j) = 0.0;
    }
  }

  const StressType stress = specfem::medium_physics::compute_stress<Tags>(
      properties, field_derivatives);

  StressType expected_stress;
  // All stress components should be zero
  for (int i = 0; i < 6; ++i) {
    for (int j = 0; j < 3; ++j) {
      expected_stress.T(i, j) = 0.0;
    }
  }

  std::ostringstream message;
  message << "3D stress tensor should be zero for zero derivatives: \n"
          << "Computed:\n"
          << stress.print() << "\n"
          << "Expected:\n"
          << expected_stress.print() << "\n";

  EXPECT_TRUE(stress == expected_stress) << message.str();
}

// Test 4: Cosserat stress contribution from rotation field values
// Tests the asymmetric stress modification via actual rotation values (not
// gradients)
TEST(Stress, ElasticIsotropicCosserat3D_CosseratStressContribution) {
  static constexpr auto dimension = specfem::element::dimension_tag::dim3;
  static constexpr auto property_tag =
      specfem::element::property_tag::isotropic_cosserat;
  static constexpr auto elasticTag = specfem::element::medium_tag::elastic_spin;

  using Tags = specfem::tags::Tags<dimension, elasticTag, property_tag, false>;
  using PropertiesType = specfem::point::properties<Tags>;
  using FieldDerivativesType = specfem::point::field_derivatives<Tags>;
  using DisplacementType = specfem::point::displacement<Tags>;
  using StressType = specfem::point::stress<Tags>;

  const type_real rho = 4.0;
  const type_real kappa = 2.0;
  const type_real mu = 3.0;
  const type_real nu = 0.25;
  const type_real j = 0.01;
  const type_real lambda_c = 1.0;
  const type_real mu_c = 2.0;
  const type_real nu_c = 0.1;
  const PropertiesType properties(rho, kappa, mu, nu, j, lambda_c, mu_c, nu_c);

  const type_real lambda = properties.lambda();

  // Set up displacement field: [u_x, u_y, u_z, phi_x, phi_y, phi_z]
  DisplacementType u;
  u(0) = 0.0; // u_x
  u(1) = 0.0; // u_y
  u(2) = 0.0; // u_z
  u(3) = 0.1; // phi_x (rotation around x-axis)
  u(4) = 0.2; // phi_y (rotation around y-axis)
  u(5) = 0.3; // phi_z (rotation around z-axis)

  // Set up field derivatives with simple non-zero values
  FieldDerivativesType field_derivatives;
  field_derivatives.du(0, 0) = 1.0;
  field_derivatives.du(1, 1) = 1.0;
  field_derivatives.du(2, 2) = 1.0;
  field_derivatives.du(0, 1) = 0.0;
  field_derivatives.du(1, 0) = 0.0;
  field_derivatives.du(0, 2) = 0.0;
  field_derivatives.du(2, 0) = 0.0;
  field_derivatives.du(1, 2) = 0.0;
  field_derivatives.du(2, 1) = 0.0;
  // Zero rotation gradients
  for (int i = 3; i < 6; ++i) {
    for (int j = 0; j < 3; ++j) {
      field_derivatives.du(i, j) = 0.0;
    }
  }

  // Compute stress from gradients first
  StressType stress = specfem::medium_physics::compute_stress<Tags>(
      properties, field_derivatives);

  // Then apply cosserat_stress correction from rotation field values
  specfem::medium_physics::compute_cosserat_stress(properties, u, stress);

  // Expected stress: classical + cosserat correction
  StressType expected_stress;

  // Classical stress (only diagonal from du_ii=1)
  const type_real div_u = 3.0;
  expected_stress.T(0, 0) = lambda * div_u + 2.0 * mu * 1.0; // sigma_xx
  expected_stress.T(1, 1) = lambda * div_u + 2.0 * mu * 1.0; // sigma_yy
  expected_stress.T(2, 2) = lambda * div_u + 2.0 * mu * 1.0; // sigma_zz
  expected_stress.T(0, 1) = 0.0;
  expected_stress.T(1, 0) = 0.0;
  expected_stress.T(0, 2) = 0.0;
  expected_stress.T(2, 0) = 0.0;
  expected_stress.T(1, 2) = 0.0;
  expected_stress.T(2, 1) = 0.0;

  // Cosserat stress corrections from rotation field values:
  // T(0,1) += 2*nu*phi_z, T(1,0) -= 2*nu*phi_z
  // T(0,2) -= 2*nu*phi_y, T(2,0) += 2*nu*phi_y
  // T(1,2) += 2*nu*phi_x, T(2,1) -= 2*nu*phi_x
  expected_stress.T(0, 1) += 2.0 * nu * u(5); // += 2*nu*phi_z
  expected_stress.T(1, 0) -= 2.0 * nu * u(5); // -= 2*nu*phi_z
  expected_stress.T(0, 2) -= 2.0 * nu * u(4); // -= 2*nu*phi_y
  expected_stress.T(2, 0) += 2.0 * nu * u(4); // += 2*nu*phi_y
  expected_stress.T(1, 2) += 2.0 * nu * u(3); // += 2*nu*phi_x
  expected_stress.T(2, 1) -= 2.0 * nu * u(3); // -= 2*nu*phi_x

  // No couple stress since rotation gradients are zero
  expected_stress.T(3, 0) = 0.0;
  expected_stress.T(4, 0) = 0.0;
  expected_stress.T(5, 0) = 0.0;
  expected_stress.T(3, 1) = 0.0;
  expected_stress.T(4, 1) = 0.0;
  expected_stress.T(5, 1) = 0.0;
  expected_stress.T(3, 2) = 0.0;
  expected_stress.T(4, 2) = 0.0;
  expected_stress.T(5, 2) = 0.0;

  std::ostringstream message;
  message << "3D stress tensor with cosserat rotation coupling is incorrect: \n"
          << "Computed:\n"
          << stress.print() << "\n"
          << "Expected:\n"
          << expected_stress.print() << "\n";

  EXPECT_TRUE(stress == expected_stress) << message.str();
}

// Test 5: Couple stress acceleration contribution
// Tests the moment equilibrium calculation that produces angular accelerations
TEST(Stress, ElasticIsotropicCosserat3D_CoupleStressAcceleration) {
  static constexpr auto dimension = specfem::element::dimension_tag::dim3;
  static constexpr auto property_tag =
      specfem::element::property_tag::isotropic_cosserat;
  static constexpr auto elasticTag = specfem::element::medium_tag::elastic_spin;

  using Tags = specfem::tags::Tags<dimension, elasticTag, property_tag, false>;
  using PropertiesType = specfem::point::properties<Tags>;
  using FieldDerivativesType = specfem::point::field_derivatives<Tags>;
  using StressType = specfem::point::stress<Tags>;
  using JacobianMatrixType =
      specfem::point::jacobian_matrix<dimension, true, Tags::using_simd>;
  using AccelerationType = specfem::point::acceleration<Tags>;

  const type_real rho = 1.0;
  const type_real kappa = 1.0;
  const type_real mu = 1.0;
  const type_real nu = 0.5;
  const type_real j = 1.0;
  const type_real lambda_c = 1.0;
  const type_real mu_c = 1.0;
  const type_real nu_c = 0.1;
  const PropertiesType properties(rho, kappa, mu, nu, j, lambda_c, mu_c, nu_c);

  // Create a simple stress tensor with asymmetry
  FieldDerivativesType field_derivatives;
  // Create stress with known asymmetric terms
  field_derivatives.du(0, 0) = 1.0;
  field_derivatives.du(1, 1) = 1.0;
  field_derivatives.du(2, 2) = 1.0;
  // These create asymmetry: du(2,1) and du(1,2)
  field_derivatives.du(2, 1) = 2.0; // Creates sigma_zy
  field_derivatives.du(1, 2) = 1.0; // Creates sigma_yz
  // Zero all other displacement gradients except the ones above
  field_derivatives.du(0, 1) = 0.0;
  field_derivatives.du(1, 0) = 0.0;
  field_derivatives.du(0, 2) = 0.0;
  field_derivatives.du(2, 0) = 0.0;
  // Zero rotation gradients
  for (int i = 3; i < 6; ++i) {
    for (int j = 0; j < 3; ++j) {
      field_derivatives.du(i, j) = 0.0;
    }
  }

  const StressType stress = specfem::medium_physics::compute_stress<Tags>(
      properties, field_derivatives);

  // Create Jacobian matrix (identity for simplicity)
  JacobianMatrixType jacobian_matrix;
  jacobian_matrix.xix = 1.0;
  jacobian_matrix.xiy = 0.0;
  jacobian_matrix.xiz = 0.0;
  jacobian_matrix.etax = 0.0;
  jacobian_matrix.etay = 1.0;
  jacobian_matrix.etaz = 0.0;
  jacobian_matrix.gammax = 0.0;
  jacobian_matrix.gammay = 0.0;
  jacobian_matrix.gammaz = 1.0;
  jacobian_matrix.jacobian = 1.0; // det(J) = 1

  // Initialize acceleration
  AccelerationType acceleration;
  for (int i = 0; i < 6; ++i) {
    acceleration(i) = 0.0;
  }

  // Apply couple stress acceleration
  const type_real factor = 1.0;
  struct StressIntegrandAdapter {
    StressType::value_type F;

    static constexpr int rank() { return 2; }

    static constexpr int static_extent(const int dim) {
      return (dim == 0) ? StressType::components : StressType::dimension;
    }

    type_real operator()(const int i, const int j) const { return F(i, j); }

    StressType::value_type
    operator*(const JacobianMatrixType::tensor_type &tensor) const {
      return F * tensor;
    }
  };

  const StressIntegrandAdapter stress_integrand{
    stress.T * jacobian_matrix.tensor() * jacobian_matrix.jacobian
  };
  specfem::medium_physics::compute_cosserat_couple_stress(
      jacobian_matrix, properties, factor, stress_integrand, acceleration);

  // Expected angular accelerations from asymmetric stress
  // acceleration(3) -= (sigma_zy - sigma_yz) * factor / jacobian
  // acceleration(4) -= (sigma_xz - sigma_zx) * factor / jacobian
  // acceleration(5) -= (sigma_yx - sigma_xy) * factor / jacobian

  // From our setup:
  // sigma_yz = mu*(du(2,1)+du(1,2)) + nu*(du(2,1)-du(1,2)) = 3*mu + nu
  // sigma_zy = mu*(du(1,2)+du(2,1)) + nu*(du(1,2)-du(2,1)) = 3*mu - nu
  const type_real sigma_yz =
      mu * (2.0 + 1.0) + nu * (2.0 - 1.0); // = 3 + 0.5 = 3.5
  const type_real sigma_zy =
      mu * (1.0 + 2.0) + nu * (1.0 - 2.0); // = 3 - 0.5 = 2.5

  AccelerationType expected_acceleration;
  expected_acceleration(0) = 0.0;
  expected_acceleration(1) = 0.0;
  expected_acceleration(2) = 0.0;
  expected_acceleration(3) =
      -(sigma_zy - sigma_yz) * factor / jacobian_matrix.jacobian;
  expected_acceleration(4) = 0.0; // No xz/zx asymmetry
  expected_acceleration(5) = 0.0; // No xy/yx asymmetry

  std::ostringstream message;
  message << "3D couple stress acceleration is incorrect: \n"
          << "acceleration(3) computed: " << acceleration(3)
          << ", expected: " << expected_acceleration(3) << "\n"
          << "acceleration(4) computed: " << acceleration(4)
          << ", expected: " << expected_acceleration(4) << "\n"
          << "acceleration(5) computed: " << acceleration(5)
          << ", expected: " << expected_acceleration(5) << "\n";

  EXPECT_DOUBLE_EQ(acceleration(3), expected_acceleration(3)) << message.str();
  EXPECT_DOUBLE_EQ(acceleration(4), expected_acceleration(4)) << message.str();
  EXPECT_DOUBLE_EQ(acceleration(5), expected_acceleration(5)) << message.str();
}

} // namespace
