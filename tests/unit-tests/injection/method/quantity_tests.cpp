// Tests for quantity lists and the plugin request of a PointwiseSurfaceMethod.

#include "constant_force.hpp"
#include "specfem/injection/method/quantity_list.hpp"
#include <gtest/gtest.h>

#include <optional>
#include <vector>

namespace specfem::injection_method_test {

using specfem::element::medium_tag;
using specfem::element::property_tag;

using VelocityTraction =
    specfem::injection::QuantityList<specfem_quantity_velocity,
                                     specfem_quantity_traction>;

static_assert(specfem::injection::quantity_components<medium_tag::elastic>(
                  specfem_quantity_traction) == 3);
static_assert(specfem::injection::quantity_components<medium_tag::elastic>(
                  specfem_quantity_potential_rate) == 0);
static_assert(specfem::injection::quantity_components<medium_tag::acoustic>(
                  specfem_quantity_potential_rate) == 1);
static_assert(specfem::injection::quantity_components<medium_tag::acoustic>(
                  specfem_quantity_velocity) == 0);
static_assert(specfem::injection::quantity_components<medium_tag::elastic_spin>(
                  specfem_quantity_velocity) == 0);

static_assert(specfem::injection::QuantitySlot<medium_tag::elastic,
                                               specfem_quantity_velocity,
                                               VelocityTraction>::offset == 0);
static_assert(specfem::injection::QuantitySlot<medium_tag::elastic,
                                               specfem_quantity_traction,
                                               VelocityTraction>::offset == 3);
static_assert(!specfem::injection::QuantitySlot<medium_tag::elastic,
                                                specfem_quantity_displacement,
                                                VelocityTraction>::listed);
static_assert(specfem::injection::QuantityListTraits<
                  medium_tag::elastic, VelocityTraction>::components == 6);
static_assert(specfem::injection::QuantityListTraits<
              medium_tag::elastic, VelocityTraction>::supported);
static_assert(!specfem::injection::QuantityListTraits<
              medium_tag::acoustic, VelocityTraction>::supported);
static_assert(
    specfem::injection::QuantityListTraits<
        medium_tag::acoustic, specfem::injection::QuantityList<>>::components ==
    0);

using specfem::injection_impl::computes_at;
static_assert(
    computes_at<ConstantForce, medium_tag::elastic, property_tag::isotropic>);
static_assert(
    computes_at<ConstantForce, medium_tag::acoustic, property_tag::isotropic>);
static_assert(!computes_at<ConstantForce, medium_tag::elastic,
                           property_tag::anisotropic>);
static_assert(!computes_at<ConstantForce, medium_tag::elastic_spin,
                           property_tag::isotropic_cosserat>);

void expect_requests(
    const std::optional<std::vector<specfem::injection::QuantityRequest>>
        &actual,
    const std::vector<specfem::injection::QuantityRequest> &expected) {
  ASSERT_TRUE(actual.has_value());
  ASSERT_EQ(actual->size(), expected.size());
  for (std::size_t i = 0; i < expected.size(); ++i) {
    EXPECT_EQ((*actual)[i].quantity, expected[i].quantity);
    EXPECT_EQ((*actual)[i].components, expected[i].components);
  }
}

TEST(InjectionQuantities, RequestsFollowTheQuantityLists) {
  const auto policy = ConstantForce().quantities();
  expect_requests(policy(medium_tag::elastic, property_tag::isotropic),
                  { { specfem_quantity_displacement, 3 },
                    { specfem_quantity_velocity, 3 } });
  expect_requests(policy(medium_tag::acoustic, property_tag::isotropic),
                  { { specfem_quantity_potential_rate, 1 } });
}

TEST(InjectionQuantities, SystemsWithoutComputeAreUnsupported) {
  const auto policy = ConstantForce().quantities();
  EXPECT_FALSE(policy(medium_tag::elastic, property_tag::anisotropic));
  EXPECT_FALSE(
      policy(medium_tag::elastic_spin, property_tag::isotropic_cosserat));
}

TEST(InjectionQuantities, NameIsTheMethodName) {
  EXPECT_EQ(ConstantForce().name(), "constant-force");
}

} // namespace specfem::injection_method_test
