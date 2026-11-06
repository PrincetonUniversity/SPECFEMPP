#include "resolution_test_helpers.hpp"
#include "specfem/setup.hpp"
#include "specfem/source.hpp"
#include "test_macros.hpp"

#include <gtest/gtest.h>
#include <type_traits>
#include <variant>

namespace source_resolution_test {
using GeocentricForce =
    specfem::sources::ForceVector<specfem::coordinate_systems::GeocentricFrame>;
using CartesianForce =
    specfem::sources::ForceVector<specfem::coordinate_systems::CartesianFrame>;
using ForceSource =
    specfem::sources::force<specfem::element::dimension_tag::dim3>;
// set_cartesian_components only accepts the Cartesian-frame force.
static_assert(
    std::is_invocable_v<decltype(&ForceSource::set_cartesian_components),
                        ForceSource &, const CartesianForce &>);
static_assert(
    !std::is_invocable_v<decltype(&ForceSource::set_cartesian_components),
                         ForceSource &, const GeocentricForce &>);
} // namespace source_resolution_test

// A force constructed in the local geocentric frame is held as such until the
// assembly stage installs the Cartesian components. get_force_vector must throw
// meanwhile.
TEST(ForceResolution, GeocentricIsUnresolvedUntilSet) {
  using source_resolution_test::CartesianForce;
  using source_resolution_test::GeocentricForce;
  using specfem::element::dimension_tag;
  specfem::sources::force<dimension_tag::dim3> src(
      source_resolution_test::origin_coords(),
      GeocentricForce(type_real(1), type_real(2), type_real(3)),
      source_resolution_test::make_stf(),
      specfem::simulation::field_type::forward);
  src.set_medium_tag(specfem::element::medium_tag::elastic);

  ASSERT_TRUE(std::holds_alternative<GeocentricForce>(src.force_value()));
  EXPECT_THROW((void)src.get_force_vector(), std::bad_variant_access);

  src.set_cartesian_components(
      CartesianForce(type_real(10), type_real(20), type_real(30)));

  ASSERT_TRUE(std::holds_alternative<CartesianForce>(src.force_value()));

  const auto v = src.get_force_vector();
  EXPECT_REAL_EQ(v(0), type_real(10));
  EXPECT_REAL_EQ(v(1), type_real(20));
  EXPECT_REAL_EQ(v(2), type_real(30));
}

// A force constructed directly from Cartesian components is resolved from the
// start: get_force_vector works without assembly.
TEST(ForceResolution, CartesianCtorIsResolved) {
  specfem::sources::force<specfem::element::dimension_tag::dim3> src(
      0.0, 0.0, 0.0, type_real(1), type_real(2), type_real(3),
      source_resolution_test::make_stf(),
      specfem::simulation::field_type::forward);
  src.set_medium_tag(specfem::element::medium_tag::elastic);

  EXPECT_TRUE(std::holds_alternative<source_resolution_test::CartesianForce>(
      src.force_value()));
  EXPECT_NO_THROW((void)src.get_force_vector());
}
