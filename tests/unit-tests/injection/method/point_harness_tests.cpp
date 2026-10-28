// Tests of specfem::injection::evaluate_point, which runs a method's compute at
// one point described by explicit values.

#include "constant_force.hpp"
#include "specfem/injection/method/point_harness.hpp"
#include <gtest/gtest.h>

#include <stdexcept>

namespace specfem::injection_method_test {

using specfem::element::medium_tag;
using specfem::element::property_tag;

constexpr type_real tolerance = 1e-6;

using ElasticProperties = ConstantForce::ElasticContext::properties_type;
using AcousticProperties = ConstantForce::AcousticContext::properties_type;

TEST(InjectionPointHarness, ZeroIncidentFieldGivesZero) {
  const auto acceleration =
      specfem::injection::evaluate_point<ConstantForce, medium_tag::elastic,
                                         property_tag::isotropic>(
          ElasticProperties(1.0, 1.0, 1.0), { 0, 0, 1 }, 0.5,
          { 0, 0, 0, 0, 0, 0 });
  for (int icomp = 0; icomp < 3; ++icomp)
    EXPECT_NEAR(acceleration(icomp), 0.0, tolerance);
}

TEST(InjectionPointHarness, ElasticUsesVelocitySlotsAndWeight) {
  // Displacement occupies slots 0-2 and must be ignored; velocity is 3-5.
  const auto acceleration =
      specfem::injection::evaluate_point<ConstantForce, medium_tag::elastic,
                                         property_tag::isotropic>(
          ElasticProperties(1.0, 1.0, 1.0), { 0, 0, 1 }, 0.5,
          { 100, 200, 300, 1, 2, 3 });
  // scale (2) * weight (0.5) * velocity
  EXPECT_NEAR(acceleration(0), 1.0, tolerance);
  EXPECT_NEAR(acceleration(1), 2.0, tolerance);
  EXPECT_NEAR(acceleration(2), 3.0, tolerance);
}

TEST(InjectionPointHarness, ParametersReachCompute) {
  const auto acceleration =
      specfem::injection::evaluate_point<ConstantForce, medium_tag::acoustic,
                                         property_tag::isotropic>(
          AcousticProperties(1.0, 1.0), { 1, 0, 0 }, 0.25, { 4 }, 7, 1.5,
          ConstantForce::Parameters{ 3.0 });
  EXPECT_NEAR(acceleration(0), 3.0, tolerance);
}

TEST(InjectionPointHarness, WrongNumberOfIncidentValuesIsRejected) {
  EXPECT_THROW(
      (specfem::injection::evaluate_point<ConstantForce, medium_tag::elastic,
                                          property_tag::isotropic>(
          ElasticProperties(1.0, 1.0, 1.0), { 0, 0, 1 }, 1.0, { 1, 2, 3 })),
      std::invalid_argument);
}

} // namespace specfem::injection_method_test
