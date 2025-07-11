#include "enumerations/interface.hpp"
#include "medium/compute_stress.hpp"
#include "specfem/point.hpp"
#include <gtest/gtest.h>
#include <sstream>

namespace {

TEST(Stress, ElasticIsotropic2D_PSV_Basic) {
  static constexpr auto dimension = specfem::dimension::type::dim2;
  static constexpr auto property_tag =
      specfem::element::property_tag::isotropic;
  static constexpr auto PSVTag = specfem::element::medium_tag::elastic_psv;

  using PSVPropertiesType =
      specfem::point::properties<dimension, PSVTag, property_tag, false>;
  using PSVFieldDerivativesType =
      specfem::point::field_derivatives<dimension, PSVTag, false>;
  using PSVStressType = specfem::point::stress<dimension, PSVTag, false>;
  const type_real kappa = 2.0;
  const type_real mu = 3.0;
  const type_real rho = 4.0; // Density is not used in stress computation
  const PSVPropertiesType properties(kappa, mu, rho);

  const type_real lambda = properties.lambda();

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
  message << "PSV stress tensor is not equal to expected value: \n"
          << "Computed: " << stress.print() << "\n"
          << "Expected: " << expected_stress.print() << "\n";

  EXPECT_TRUE(stress == expected_stress) << message.str();
}

TEST(Stress, ElasticIsotropic2D_PSV_ZeroDerivatives) {
  static constexpr auto dimension = specfem::dimension::type::dim2;
  static constexpr auto property_tag =
      specfem::element::property_tag::isotropic;
  static constexpr auto PSVTag = specfem::element::medium_tag::elastic_psv;

  using PSVPropertiesType =
      specfem::point::properties<dimension, PSVTag, property_tag, false>;
  using PSVFieldDerivativesType =
      specfem::point::field_derivatives<dimension, PSVTag, false>;
  using PSVStressType = specfem::point::stress<dimension, PSVTag, false>;
  const type_real kappa = 2.0;
  const type_real mu = 3.0;
  const type_real rho = 4.0; // Density is not used in stress computation
  const PSVPropertiesType properties(kappa, mu, rho);

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
  message << "PSV stress tensor should be zero for zero derivatives: \n"
          << "Computed: " << stress.print() << "\n"
          << "Expected: " << expected_stress.print() << "\n";

  EXPECT_TRUE(stress == expected_stress) << message.str();
}

TEST(Stress, ElasticIsotropic2D_SH_Basic) {
  static constexpr auto dimension = specfem::dimension::type::dim2;
  static constexpr auto property_tag =
      specfem::element::property_tag::isotropic;
  static constexpr auto SHTag = specfem::element::medium_tag::elastic_sh;

  using SHPropertiesType =
      specfem::point::properties<dimension, SHTag, property_tag, false>;
  using SHFieldDerivativesType =
      specfem::point::field_derivatives<dimension, SHTag, false>;
  using SHStressType = specfem::point::stress<dimension, SHTag, false>;
  const type_real kappa = 2.0;
  const type_real mu = 3.0;
  const type_real rho = 4.0; // Density is not used in stress computation
  const SHPropertiesType properties(kappa, mu, rho);

  SHFieldDerivativesType field_derivatives;
  field_derivatives.du(0, 0) = 1.0;
  field_derivatives.du(0, 1) = 2.0;

  const SHStressType stress =
      specfem::medium::compute_stress(properties, field_derivatives);

  SHStressType expected_stress;
  expected_stress.T(0, 0) = mu * 1.0;
  expected_stress.T(0, 1) = mu * 2.0;

  std::ostringstream message;
  message << "SH stress tensor is not equal to expected value: \n"
          << "Computed: " << stress.print() << "\n"
          << "Expected: " << expected_stress.print() << "\n";

  EXPECT_TRUE(stress == expected_stress) << message.str();
}

TEST(Stress, ElasticIsotropic2D_SH_ZeroDerivatives) {
  static constexpr auto dimension = specfem::dimension::type::dim2;
  static constexpr auto property_tag =
      specfem::element::property_tag::isotropic;
  static constexpr auto SHTag = specfem::element::medium_tag::elastic_sh;

  using SHPropertiesType =
      specfem::point::properties<dimension, SHTag, property_tag, false>;
  using SHFieldDerivativesType =
      specfem::point::field_derivatives<dimension, SHTag, false>;
  using SHStressType = specfem::point::stress<dimension, SHTag, false>;
  const type_real kappa = 2.0;
  const type_real mu = 3.0;
  const type_real rho = 4.0; // Density is not used in stress computation
  const SHPropertiesType properties(kappa, mu, rho);

  SHFieldDerivativesType field_derivatives;
  field_derivatives.du(0, 0) = 0.0;
  field_derivatives.du(0, 1) = 0.0;

  const SHStressType stress =
      specfem::medium::compute_stress(properties, field_derivatives);

  SHStressType expected_stress;
  expected_stress.T(0, 0) = 0.0;
  expected_stress.T(0, 1) = 0.0;

  std::ostringstream message;
  message << "SH stress tensor should be zero for zero derivatives: \n"
          << "Computed: " << stress.print() << "\n"
          << "Expected: " << expected_stress.print() << "\n";

  EXPECT_TRUE(stress == expected_stress) << message.str();
}
} // namespace
