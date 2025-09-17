#include "enumerations/interface.hpp"
#include "medium/compute_stress.hpp"
#include "specfem/point.hpp"
#include <gtest/gtest.h>
#include <sstream>

namespace {

TEST(Stress, ElasticIsotropic3D_Basic) {
  static constexpr auto dimension = specfem::dimension::type::dim3;
  static constexpr auto property_tag =
      specfem::element::property_tag::isotropic;
  static constexpr auto elasticTag = specfem::element::medium_tag::elastic;

  using PropertiesType =
      specfem::point::properties<dimension, elasticTag, property_tag, false>;
  using FieldDerivativesType =
      specfem::point::field_derivatives<dimension, elasticTag, false>;
  using StressType = specfem::point::stress<dimension, elasticTag, false>;
  const type_real kappa = 2.0;
  const type_real mu = 3.0;
  const type_real rho = 4.0; // Density is not used in stress computation
  const PropertiesType properties(kappa, mu, rho);

  const type_real lambda = properties.lambda();

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

  const StressType stress =
      specfem::medium::compute_stress(properties, field_derivatives);

  StressType expected_stress;
  expected_stress.T(0, 0) =
      (lambda + 2 * mu) * 1.0 + lambda * (2.0 + 3.0); // sigma_xx
  expected_stress.T(1, 1) =
      (lambda + 2 * mu) * 2.0 + lambda * (1.0 + 3.0); // sigma_yy
  expected_stress.T(2, 2) =
      (lambda + 2 * mu) * 3.0 + lambda * (1.0 + 2.0); // sigma_zz
  expected_stress.T(0, 1) = mu * (4.0 + 5.0);         // sigma_xy
  expected_stress.T(1, 0) = mu * (4.0 + 5.0);         // sigma_yx
  expected_stress.T(0, 2) = mu * (6.0 + 7.0);         // sigma_xz
  expected_stress.T(2, 0) = mu * (6.0 + 7.0);         // sigma_zx
  expected_stress.T(1, 2) = mu * (8.0 + 9.0);         // sigma_yz
  expected_stress.T(2, 1) = mu * (8.0 + 9.0);         // sigma_zy

  std::ostringstream message;
  message << "3D stress tensor is not equal to expected value: \n"
          << "Computed:\n"
          << stress.print() << "\n"
          << "Expected:\n"
          << expected_stress.print() << "\n";

  EXPECT_TRUE(stress == expected_stress) << message.str();
}

TEST(Stress, ElasticIsotropic3D_ZeroDerivatives) {
  static constexpr auto dimension = specfem::dimension::type::dim3;
  static constexpr auto property_tag =
      specfem::element::property_tag::isotropic;
  static constexpr auto elasticTag = specfem::element::medium_tag::elastic;

  using PropertiesType =
      specfem::point::properties<dimension, elasticTag, property_tag, false>;
  using FieldDerivativesType =
      specfem::point::field_derivatives<dimension, elasticTag, false>;
  using StressType = specfem::point::stress<dimension, elasticTag, false>;
  const type_real kappa = 2.0;
  const type_real mu = 3.0;
  const type_real rho = 4.0; // Density is not used in stress computation
  const PropertiesType properties(kappa, mu, rho);

  FieldDerivativesType field_derivatives;
  field_derivatives.du(0, 0) = 0.0;
  field_derivatives.du(1, 1) = 0.0;
  field_derivatives.du(2, 2) = 0.0;
  field_derivatives.du(0, 1) = 0.0;
  field_derivatives.du(1, 0) = 0.0;
  field_derivatives.du(0, 2) = 0.0;
  field_derivatives.du(2, 0) = 0.0;
  field_derivatives.du(1, 2) = 0.0;
  field_derivatives.du(2, 1) = 0.0;

  const StressType stress =
      specfem::medium::compute_stress(properties, field_derivatives);

  StressType expected_stress;
  expected_stress.T(0, 0) = 0.0;
  expected_stress.T(1, 1) = 0.0;
  expected_stress.T(2, 2) = 0.0;
  expected_stress.T(0, 1) = 0.0;
  expected_stress.T(1, 0) = 0.0;
  expected_stress.T(0, 2) = 0.0;
  expected_stress.T(2, 0) = 0.0;
  expected_stress.T(1, 2) = 0.0;
  expected_stress.T(2, 1) = 0.0;

  std::ostringstream message;
  message << "3D stress tensor should be zero for zero derivatives: \n"
          << "Computed:\n"
          << stress.print() << "\n"
          << "Expected:\n"
          << expected_stress.print() << "\n";

  EXPECT_TRUE(stress == expected_stress) << message.str();
}

} // namespace
