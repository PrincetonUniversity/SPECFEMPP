#include "enumerations/interface.hpp"
#include "medium/compute_stress.hpp"
#include "specfem/point.hpp"
#include <gtest/gtest.h>
#include <sstream>

namespace {

TEST(Stress, ElasticAnisotropic2D_PSV_Basic) {
  static constexpr auto dimension = specfem::dimension::type::dim2;
  static constexpr auto property_tag =
      specfem::element::property_tag::anisotropic;
  static constexpr auto PSVTag = specfem::element::medium_tag::elastic_psv;

  using PSVPropertiesType =
      specfem::point::properties<dimension, PSVTag, property_tag, false>;
  using PSVFieldDerivativesType =
      specfem::point::field_derivatives<dimension, PSVTag, false>;
  using PSVStressType = specfem::point::stress<dimension, PSVTag, false>;
  // c11, c13, c15, c33, c35, c55, rho
  const type_real c11 = 10.0, c13 = 2.0, c15 = 1.0;
  const type_real c33 = 20.0, c35 = 3.0, c55 = 5.0;
  const type_real rho = 4.0; // Density is not used in stress computation
  const type_real c12 = 1.0, c23 = 2.0, c25 = 3.0; // Additional coefficients
  const PSVPropertiesType properties(c11, c13, c15, c33, c35, c55, c12, c23,
                                     c25, rho);

  PSVFieldDerivativesType field_derivatives;
  field_derivatives.du(0, 0) = 1.0;
  field_derivatives.du(1, 1) = 2.0;
  field_derivatives.du(0, 1) = 3.0;
  field_derivatives.du(1, 0) = 4.0;

  const PSVStressType stress =
      specfem::medium::compute_stress(properties, field_derivatives);

  PSVStressType expected_stress;
  // sigma_xx
  expected_stress.T(0, 0) = c11 * 1.0 + c13 * 2.0 + c15 * (4.0 + 3.0);
  // sigma_zz
  expected_stress.T(1, 1) = c13 * 1.0 + c33 * 2.0 + c35 * (4.0 + 3.0);
  // sigma_xz and sigma_zx
  expected_stress.T(0, 1) = c15 * 1.0 + c35 * 2.0 + c55 * (4.0 + 3.0);
  expected_stress.T(1, 0) = expected_stress.T(0, 1);

  std::ostringstream message;
  message << "Anisotropic PSV stress tensor is not equal to expected value: \n"
          << "Computed: " << stress.print() << "\n"
          << "Expected: " << expected_stress.print() << "\n";

  EXPECT_TRUE(stress == expected_stress) << message.str();
}

TEST(Stress, ElasticAnisotropic2D_PSV_ZeroDerivatives) {
  static constexpr auto dimension = specfem::dimension::type::dim2;
  static constexpr auto property_tag =
      specfem::element::property_tag::anisotropic;
  static constexpr auto PSVTag = specfem::element::medium_tag::elastic_psv;

  using PSVPropertiesType =
      specfem::point::properties<dimension, PSVTag, property_tag, false>;
  using PSVFieldDerivativesType =
      specfem::point::field_derivatives<dimension, PSVTag, false>;
  using PSVStressType = specfem::point::stress<dimension, PSVTag, false>;
  const type_real c11 = 10.0, c13 = 2.0, c15 = 1.0;
  const type_real c33 = 20.0, c35 = 3.0, c55 = 5.0;
  const type_real rho = 4.0;
  const type_real c12 = 1.0, c23 = 2.0, c25 = 3.0; // Additional coefficients
  const PSVPropertiesType properties(c11, c13, c15, c33, c35, c55, c12, c23,
                                     c25, rho);

  PSVFieldDerivativesType field_derivatives;
  field_derivatives.du(0, 0) = 0.0;
  field_derivatives.du(1, 1) = 0.0;
  field_derivatives.du(0, 1) = 0.0;
  field_derivatives.du(1, 0) = 0.0;

  const PSVStressType stress =
      specfem::medium::compute_stress(properties, field_derivatives);

  PSVStressType expected_stress;
  expected_stress.T(0, 0) = 0.0;
  expected_stress.T(1, 1) = 0.0;
  expected_stress.T(0, 1) = 0.0;
  expected_stress.T(1, 0) = 0.0;

  std::ostringstream message;
  message
      << "Anisotropic PSV stress tensor should be zero for zero derivatives: \n"
      << "Computed: " << stress.print() << "\n"
      << "Expected: " << expected_stress.print() << "\n";

  EXPECT_TRUE(stress == expected_stress) << message.str();
}

// PSV: Anisotropic coefficients set to isotropic values
TEST(Stress, ElasticAnisotropic2D_PSV_IsotropicCoefficients) {
  static constexpr auto dimension = specfem::dimension::type::dim2;
  static constexpr auto property_tag =
      specfem::element::property_tag::anisotropic;
  static constexpr auto PSVTag = specfem::element::medium_tag::elastic_psv;

  using PSVPropertiesType =
      specfem::point::properties<dimension, PSVTag, property_tag, false>;
  using PSVFieldDerivativesType =
      specfem::point::field_derivatives<dimension, PSVTag, false>;
  using PSVStressType = specfem::point::stress<dimension, PSVTag, false>;
  // Isotropic properties
  const type_real kappa = 7.0;
  const type_real mu = 3.0;
  const type_real rho = 2.0;
  // Compute lambda
  const type_real lambda = kappa - (2.0 / 3.0) * mu;
  // Set anisotropic coefficients for isotropic case
  const type_real c11 = lambda + 2 * mu;
  const type_real c33 = c11;
  const type_real c13 = lambda;
  const type_real c15 = 0.0;
  const type_real c35 = 0.0;
  const type_real c55 = mu;
  const type_real c12 = lambda; // c12, c23, c25 can be set to lambda
  const type_real c23 = lambda;
  const type_real c25 = 0.0; // Additional coefficients
  const PSVPropertiesType properties(c11, c13, c15, c33, c35, c55, c12, c23,
                                     c25, rho);

  PSVFieldDerivativesType field_derivatives;
  field_derivatives.du(0, 0) = 1.0;
  field_derivatives.du(1, 1) = 2.0;
  field_derivatives.du(0, 1) = 3.0;
  field_derivatives.du(1, 0) = 4.0;

  const PSVStressType stress =
      specfem::medium::compute_stress(properties, field_derivatives);

  PSVStressType expected_stress;
  expected_stress.T(0, 0) = (lambda + 2 * mu) * 1.0 + lambda * 2.0;
  expected_stress.T(1, 1) = (lambda + 2 * mu) * 2.0 + lambda * 1.0;
  expected_stress.T(0, 1) = mu * (3.0 + 4.0);
  expected_stress.T(1, 0) = mu * (3.0 + 4.0);

  std::ostringstream message;
  message << "Anisotropic PSV (isotropic coefficients) stress tensor is not "
             "equal to expected value: \n"
          << "Computed: " << stress.print() << "\n"
          << "Expected: " << expected_stress.print() << "\n";

  EXPECT_TRUE(stress == expected_stress) << message.str();
}

TEST(Stress, ElasticAnisotropic2D_SH_Basic) {
  static constexpr auto dimension = specfem::dimension::type::dim2;
  static constexpr auto property_tag =
      specfem::element::property_tag::anisotropic;
  static constexpr auto SHTag = specfem::element::medium_tag::elastic_sh;

  using SHPropertiesType =
      specfem::point::properties<dimension, SHTag, property_tag, false>;
  using SHFieldDerivativesType =
      specfem::point::field_derivatives<dimension, SHTag, false>;
  using SHStressType = specfem::point::stress<dimension, SHTag, false>;
  const type_real c11 = 10.0, c13 = 2.0, c15 = 1.0;
  const type_real c33 = 20.0, c35 = 3.0, c55 = 5.0;
  const type_real rho = 4.0; // Density is not used in stress computation
  const type_real c12 = 1.0, c23 = 2.0, c25 = 3.0; // Additional coefficients
  const SHPropertiesType properties(c11, c13, c15, c33, c35, c55, c12, c23, c25,
                                    rho);

  SHFieldDerivativesType field_derivatives;
  field_derivatives.du(0, 0) = 1.0;
  field_derivatives.du(0, 1) = 2.0;

  const SHStressType stress =
      specfem::medium::compute_stress(properties, field_derivatives);

  SHStressType expected_stress;
  expected_stress.T(0, 0) = c55 * 1.0;
  expected_stress.T(0, 1) = c55 * 2.0;

  std::ostringstream message;
  message << "Anisotropic SH stress tensor is not equal to expected value: \n"
          << "Computed: " << stress.print() << "\n"
          << "Expected: " << expected_stress.print() << "\n";

  EXPECT_TRUE(stress == expected_stress) << message.str();
}

TEST(Stress, ElasticAnisotropic2D_SH_ZeroDerivatives) {
  static constexpr auto dimension = specfem::dimension::type::dim2;
  static constexpr auto property_tag =
      specfem::element::property_tag::anisotropic;
  static constexpr auto SHTag = specfem::element::medium_tag::elastic_sh;

  using SHPropertiesType =
      specfem::point::properties<dimension, SHTag, property_tag, false>;
  using SHFieldDerivativesType =
      specfem::point::field_derivatives<dimension, SHTag, false>;
  using SHStressType = specfem::point::stress<dimension, SHTag, false>;
  const type_real c11 = 10.0, c13 = 2.0, c15 = 1.0;
  const type_real c33 = 20.0, c35 = 3.0, c55 = 5.0;
  const type_real rho = 4.0; // Density is not used in stress computation
  const type_real c12 = 1.0, c23 = 2.0, c25 = 3.0; // Additional coefficients
  const SHPropertiesType properties(c11, c13, c15, c33, c35, c55, c12, c23, c25,
                                    rho);

  SHFieldDerivativesType field_derivatives;
  field_derivatives.du(0, 0) = 0.0;
  field_derivatives.du(0, 1) = 0.0;

  const SHStressType stress =
      specfem::medium::compute_stress(properties, field_derivatives);

  SHStressType expected_stress;
  expected_stress.T(0, 0) = 0.0;
  expected_stress.T(0, 1) = 0.0;

  std::ostringstream message;
  message
      << "Anisotropic SH stress tensor should be zero for zero derivatives: \n"
      << "Computed: " << stress.print() << "\n"
      << "Expected: " << expected_stress.print() << "\n";

  EXPECT_TRUE(stress == expected_stress) << message.str();
}

// SH: Anisotropic coefficients set to isotropic values
TEST(Stress, ElasticAnisotropic2D_SH_IsotropicCoefficients) {
  static constexpr auto dimension = specfem::dimension::type::dim2;
  static constexpr auto property_tag =
      specfem::element::property_tag::anisotropic;
  static constexpr auto SHTag = specfem::element::medium_tag::elastic_sh;

  using SHPropertiesType =
      specfem::point::properties<dimension, SHTag, property_tag, false>;
  using SHFieldDerivativesType =
      specfem::point::field_derivatives<dimension, SHTag, false>;
  using SHStressType = specfem::point::stress<dimension, SHTag, false>;
  // Isotropic properties
  const type_real kappa = 7.0;
  const type_real mu = 3.0;
  const type_real rho = 2.0;
  // Compute lambda
  const type_real lambda = kappa - (2.0 / 3.0) * mu;
  // Set anisotropic coefficients for isotropic case
  const type_real c11 = lambda + 2 * mu;
  const type_real c33 = c11;
  const type_real c13 = lambda;
  const type_real c15 = 0.0;
  const type_real c35 = 0.0;
  const type_real c55 = mu;
  const type_real c12 = lambda; // c12, c23, c25 can be set to lambda
  const type_real c23 = lambda;
  const type_real c25 = 0.0; // Additional coefficients
  const SHPropertiesType properties(c11, c13, c15, c33, c35, c55, c12, c23, c25,
                                    rho);

  SHFieldDerivativesType field_derivatives;
  field_derivatives.du(0, 0) = 1.2;
  field_derivatives.du(0, 1) = -0.8;

  const SHStressType stress =
      specfem::medium::compute_stress(properties, field_derivatives);

  SHStressType expected_stress;
  expected_stress.T(0, 0) = mu * 1.2;
  expected_stress.T(0, 1) = mu * -0.8;

  std::ostringstream message;
  message << "Anisotropic SH (isotropic coefficients) stress tensor is not "
             "equal to expected value: \n"
          << "Computed: " << stress.print() << "\n"
          << "Expected: " << expected_stress.print() << "\n";

  EXPECT_TRUE(stress == expected_stress) << message.str();
}
} // namespace
