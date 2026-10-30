#include "resolution_test_helpers.hpp"
#include "specfem/setup.hpp"
#include "specfem/source.hpp"
#include "test_macros.hpp"

#include <gtest/gtest.h>
#include <stdexcept>
#include <type_traits>
#include <variant>

namespace source_resolution_test {
using GeocentricForce =
    specfem::sources::ForceVector<specfem::coordinate_systems::GeocentricFrame>;
using CartesianForce =
    specfem::sources::ForceVector<specfem::coordinate_systems::CartesianFrame>;
using ForceQuantity =
    specfem::sources::LocalFrameQuantity<specfem::sources::ForceVector>;
// resolve() only accepts the Cartesian-frame force (compile-time check).
static_assert(std::is_invocable_v<decltype(&ForceQuantity::resolve),
                                  ForceQuantity &, const CartesianForce &>);
static_assert(!std::is_invocable_v<decltype(&ForceQuantity::resolve),
                                   ForceQuantity &, const GeocentricForce &>);
} // namespace source_resolution_test

// A force constructed in the local geocentric frame is unresolved until the
// assembly stage installs the Cartesian components. get_force_vector must throw
// while unresolved, and the as-read geocentric force must survive resolution.
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

  ASSERT_FALSE(src.force_quantity().is_resolved());
  EXPECT_THROW((void)src.get_force_vector(), std::runtime_error);

  ASSERT_TRUE(
      std::holds_alternative<GeocentricForce>(src.force_quantity().as_read()));

  src.set_cartesian_components(
      CartesianForce(type_real(10), type_real(20), type_real(30)));

  EXPECT_TRUE(src.force_quantity().is_resolved());
  ASSERT_TRUE(
      std::holds_alternative<GeocentricForce>(src.force_quantity().as_read()));

  const auto v = src.get_force_vector();
  EXPECT_REAL_EQ(v(0), type_real(10));
  EXPECT_REAL_EQ(v(1), type_real(20));
  EXPECT_REAL_EQ(v(2), type_real(30));
}

// A force constructed directly from Cartesian components is resolved from the
// start: get_force_vector works and the as-read value is Cartesian.
TEST(ForceResolution, CartesianCtorIsResolved) {
  specfem::sources::force<specfem::element::dimension_tag::dim3> src(
      0.0, 0.0, 0.0, type_real(1), type_real(2), type_real(3),
      source_resolution_test::make_stf(),
      specfem::simulation::field_type::forward);
  src.set_medium_tag(specfem::element::medium_tag::elastic);

  EXPECT_TRUE(src.force_quantity().is_resolved());
  EXPECT_TRUE(std::holds_alternative<source_resolution_test::CartesianForce>(
      src.force_quantity().as_read()));
  EXPECT_NO_THROW((void)src.get_force_vector());
}
