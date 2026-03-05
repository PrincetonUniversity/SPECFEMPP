#include "specfem/enums.hpp"
#include "specfem/medium_physics.hpp"
#include "specfem/point.hpp"
#include <gtest/gtest.h>
#include <sstream>

namespace {

TEST(Stress, AcousticIsotropic3D) {
  static constexpr auto dimension = specfem::element::dimension_tag::dim3;
  static constexpr auto medium_tag = specfem::element::medium_tag::acoustic;
  static constexpr auto property_tag =
      specfem::element::property_tag::isotropic;

  using PointPropertiesType = specfem::point::properties<
      specfem::tags::Tags<dimension, medium_tag, property_tag, false> >;
  using PointFieldDerivativesType =
      specfem::point::field_derivatives<dimension, medium_tag, false>;
  using PointStressType = specfem::point::stress<
      specfem::tags::Tags<dimension, medium_tag, false> >;

  const type_real rho_inverse = 2.0;
  const PointPropertiesType properties(rho_inverse);

  PointFieldDerivativesType field_derivatives;
  field_derivatives.du(0, 0) = 3.0;
  field_derivatives.du(0, 1) = 4.0;
  field_derivatives.du(0, 2) = 5.0;

  const PointStressType stress =
      specfem::medium_physics::compute_stress(properties, field_derivatives);

  PointStressType expected_stress;
  expected_stress.T(0, 0) = rho_inverse * field_derivatives.du(0, 0);
  expected_stress.T(0, 1) = rho_inverse * field_derivatives.du(0, 1);
  expected_stress.T(0, 2) = rho_inverse * field_derivatives.du(0, 2);

  std::ostringstream message;
  message << "Stress tensor is not equal to expected value: \n"
          << "Computed: " << stress.print() << "\n"
          << "Expected: " << expected_stress.print() << "\n";

  EXPECT_TRUE(stress == expected_stress) << message.str();
}

TEST(Stress, AcousticIsotropic3D_ZeroDerivatives) {
  static constexpr auto dimension = specfem::element::dimension_tag::dim3;
  static constexpr auto medium_tag = specfem::element::medium_tag::acoustic;
  static constexpr auto property_tag =
      specfem::element::property_tag::isotropic;

  using PointPropertiesType = specfem::point::properties<
      specfem::tags::Tags<dimension, medium_tag, property_tag, false> >;
  using PointFieldDerivativesType =
      specfem::point::field_derivatives<dimension, medium_tag, false>;
  using PointStressType = specfem::point::stress<
      specfem::tags::Tags<dimension, medium_tag, false> >;

  const type_real rho_inverse = 2.0;
  const PointPropertiesType properties(rho_inverse);

  PointFieldDerivativesType field_derivatives;
  field_derivatives.du(0, 0) = 0.0;
  field_derivatives.du(0, 1) = 0.0;
  field_derivatives.du(0, 2) = 0.0;

  const PointStressType stress =
      specfem::medium_physics::compute_stress(properties, field_derivatives);

  PointStressType expected_stress;
  expected_stress.T(0, 0) = 0.0;
  expected_stress.T(0, 1) = 0.0;
  expected_stress.T(0, 2) = 0.0;

  std::ostringstream message;
  message << "Stress tensor should be zero for zero derivatives: \n"
          << "Computed: " << stress.print() << "\n"
          << "Expected: " << expected_stress.print() << "\n";

  EXPECT_TRUE(stress == expected_stress) << message.str();
}

} // namespace
