#include "enumerations/interface.hpp"
#include "medium/compute_stress.hpp"
#include "specfem/point.hpp"
#include <gtest/gtest.h>
#include <sstream>

namespace {

TEST(Stress, ElasticIsotropicCosserat2D_Basic) {
  static constexpr auto dimension = specfem::dimension::type::dim2;
  static constexpr auto property_tag =
      specfem::element::property_tag::isotropic_cosserat;
  static constexpr auto CosseratTag =
      specfem::element::medium_tag::elastic_psv_t;

  using CosseratPropertiesType =
      specfem::point::properties<dimension, CosseratTag, property_tag, false>;
  using CosseratFieldDerivativesType =
      specfem::point::field_derivatives<dimension, CosseratTag, false>;
  using CosseratStressType =
      specfem::point::stress<dimension, CosseratTag, false>;
  // Set up properties (arbitrary but nonzero values)
  const type_real rho = 2.5;
  const type_real kappa = 7.0;
  const type_real mu = 3.0;
  const type_real nu = 1.2;
  const type_real j = 0.8;
  const type_real lambda_c = 0.5;
  const type_real mu_c = 0.7;
  const type_real nu_c = 0.9;
  const CosseratPropertiesType properties(rho, kappa, mu, nu, j, lambda_c, mu_c,
                                          nu_c);

  CosseratFieldDerivativesType field_derivatives;
  field_derivatives.du(0, 0) = 1.0;
  field_derivatives.du(1, 1) = 2.0;
  field_derivatives.du(0, 1) = 3.0;
  field_derivatives.du(1, 0) = 4.0;
  field_derivatives.du(2, 0) = 5.0;
  field_derivatives.du(2, 1) = 6.0;

  // Compute expected values using property accessors
  const type_real lambda = properties.lambda();
  // sigma_xx
  const type_real sigma_xx = lambda * (1.0 + 2.0) + 2.0 * mu * 1.0;
  // sigma_zz
  const type_real sigma_zz = lambda * (1.0 + 2.0) + 2.0 * mu * 2.0;
  // sigma_xz
  const type_real sigma_xz = mu * (4.0 + 3.0) + nu * (4.0 - 3.0);
  // sigma_zx
  const type_real sigma_zx = mu * (3.0 + 4.0) + nu * (3.0 - 4.0);
  // couple stress components
  const type_real sigma_c_xy = (mu_c + nu_c) * 5.0;
  const type_real sigma_c_zy = (mu_c + nu_c) * 6.0;

  CosseratStressType expected_stress;
  expected_stress.T(0, 0) = sigma_xx;
  expected_stress.T(1, 0) = sigma_xz;
  expected_stress.T(0, 1) = sigma_zx;
  expected_stress.T(1, 1) = sigma_zz;
  expected_stress.T(2, 0) = sigma_c_xy;
  expected_stress.T(2, 1) = sigma_c_zy;

  const CosseratStressType stress =
      specfem::medium::compute_stress(properties, field_derivatives);

  std::ostringstream message;
  message << "Cosserat stress tensor is not equal to expected value: \n"
          << "Computed: " << stress.print() << "\n"
          << "Expected: " << expected_stress.print() << "\n";

  EXPECT_TRUE(stress == expected_stress) << message.str();
}

TEST(Stress, ElasticIsotropicCosserat2D_ZeroDerivatives) {
  static constexpr auto dimension = specfem::dimension::type::dim2;
  static constexpr auto property_tag =
      specfem::element::property_tag::isotropic_cosserat;
  static constexpr auto CosseratTag =
      specfem::element::medium_tag::elastic_psv_t;

  using CosseratPropertiesType =
      specfem::point::properties<dimension, CosseratTag, property_tag, false>;
  using CosseratFieldDerivativesType =
      specfem::point::field_derivatives<dimension, CosseratTag, false>;
  using CosseratStressType =
      specfem::point::stress<dimension, CosseratTag, false>;
  const type_real rho = 2.5;
  const type_real kappa = 7.0;
  const type_real mu = 3.0;
  const type_real nu = 1.2;
  const type_real j = 0.8;
  const type_real lambda_c = 0.5;
  const type_real mu_c = 0.7;
  const type_real nu_c = 0.9;
  const CosseratPropertiesType properties(rho, kappa, mu, nu, j, lambda_c, mu_c,
                                          nu_c);

  CosseratFieldDerivativesType field_derivatives;
  for (int i = 0; i < 3; ++i) {
    for (int j = 0; j < 2; ++j) {
      field_derivatives.du(i, j) = 0.0;
    }
  }

  const CosseratStressType stress =
      specfem::medium::compute_stress(properties, field_derivatives);

  CosseratStressType expected_stress;
  for (int i = 0; i < 3; ++i) {
    for (int j = 0; j < 2; ++j) {
      expected_stress.T(i, j) = 0.0;
    }
  }

  std::ostringstream message;
  message << "Cosserat stress tensor should be zero for zero derivatives: \n"
          << "Computed: " << stress.print() << "\n"
          << "Expected: " << expected_stress.print() << "\n";

  EXPECT_TRUE(stress == expected_stress) << message.str();
}

TEST(Stress, ElasticIsotropicCosserat2D_SymmetricWhenNuZero) {
  static constexpr auto dimension = specfem::dimension::type::dim2;
  static constexpr auto property_tag =
      specfem::element::property_tag::isotropic_cosserat;
  static constexpr auto CosseratTag =
      specfem::element::medium_tag::elastic_psv_t;

  using CosseratPropertiesType =
      specfem::point::properties<dimension, CosseratTag, property_tag, false>;
  using CosseratFieldDerivativesType =
      specfem::point::field_derivatives<dimension, CosseratTag, false>;
  using CosseratStressType =
      specfem::point::stress<dimension, CosseratTag, false>;
  const type_real rho = 2.5;
  const type_real kappa = 7.0;
  const type_real mu = 3.0;
  const type_real nu = 0.0; // Key: nu = 0
  const type_real j = 0.8;
  const type_real lambda_c = 0.5;
  const type_real mu_c = 0.7;
  const type_real nu_c = 0.9;
  const CosseratPropertiesType properties(rho, kappa, mu, nu, j, lambda_c, mu_c,
                                          nu_c);

  CosseratFieldDerivativesType field_derivatives;
  field_derivatives.du(0, 0) = 1.0;
  field_derivatives.du(1, 1) = 2.0;
  field_derivatives.du(0, 1) = 3.0;
  field_derivatives.du(1, 0) = 4.0;
  field_derivatives.du(2, 0) = 5.0;
  field_derivatives.du(2, 1) = 6.0;

  const type_real lambda = properties.lambda();
  // sigma_xx
  const type_real sigma_xx = lambda * (1.0 + 2.0) + 2.0 * mu * 1.0;
  // sigma_zz
  const type_real sigma_zz = lambda * (1.0 + 2.0) + 2.0 * mu * 2.0;
  // sigma_xz
  const type_real sigma_xz = mu * (4.0 + 3.0); // nu = 0
  // sigma_zx
  const type_real sigma_zx = mu * (3.0 + 4.0); // nu = 0
  // couple stress components
  const type_real sigma_c_xy = (mu_c + nu_c) * 5.0;
  const type_real sigma_c_zy = (mu_c + nu_c) * 6.0;

  CosseratStressType expected_stress;
  expected_stress.T(0, 0) = sigma_xx;
  expected_stress.T(1, 0) = sigma_xz;
  expected_stress.T(0, 1) = sigma_zx;
  expected_stress.T(1, 1) = sigma_zz;
  expected_stress.T(2, 0) = sigma_c_xy;
  expected_stress.T(2, 1) = sigma_c_zy;

  const CosseratStressType stress =
      specfem::medium::compute_stress(properties, field_derivatives);

  std::ostringstream message;
  message << "Cosserat stress tensor should be symmetric when nu = 0: \n"
          << "Computed: " << stress.print() << "\n"
          << "Expected: " << expected_stress.print() << "\n";

  // Check symmetry: T(0,1) == T(1,0)
  EXPECT_TRUE(stress.T(0, 1) == stress.T(1, 0))
      << "Tensor is not symmetric: T(0,1) != T(1,0)";
  EXPECT_TRUE(stress == expected_stress) << message.str();
}

} // namespace
