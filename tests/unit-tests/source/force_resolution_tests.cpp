#include "resolution_test_helpers.hpp"
#include "specfem/setup.hpp"
#include "specfem/source.hpp"
#include "test_macros.hpp"

#include <gtest/gtest.h>
#include <stdexcept>
#include <variant>

// A force constructed in the local East-North-Up frame is unresolved until the
// assembly stage installs the Cartesian components. get_force_vector must throw
// while unresolved, and the as-read geographic force must survive resolution.
TEST(ForceResolution, GeographicIsUnresolvedUntilSet) {
  using specfem::element::dimension_tag;
  specfem::sources::force<dimension_tag::dim3> src(
      source_resolution_test::origin_coords(),
      specfem::sources::GeographicForce{ type_real(1), type_real(2),
                                         type_real(3) },
      source_resolution_test::make_stf(),
      specfem::simulation::field_type::forward);
  src.set_medium_tag(specfem::element::medium_tag::elastic);

  ASSERT_FALSE(src.force_quantity().is_resolved());
  EXPECT_THROW((void)src.get_force_vector(), std::runtime_error);

  ASSERT_TRUE(std::holds_alternative<specfem::sources::GeographicForce>(
      src.force_quantity().as_read()));

  src.set_cartesian_components(specfem::sources::CartesianForce{
      type_real(10), type_real(20), type_real(30) });

  EXPECT_TRUE(src.force_quantity().is_resolved());
  ASSERT_TRUE(std::holds_alternative<specfem::sources::GeographicForce>(
      src.force_quantity().as_read()));

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
  EXPECT_TRUE(std::holds_alternative<specfem::sources::CartesianForce>(
      src.force_quantity().as_read()));
  EXPECT_NO_THROW((void)src.get_force_vector());
}
