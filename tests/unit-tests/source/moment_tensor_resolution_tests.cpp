#include "resolution_test_helpers.hpp"
#include "specfem/setup.hpp"
#include "specfem/source.hpp"
#include "test_macros.hpp"

#include <gtest/gtest.h>
#include <stdexcept>
#include <type_traits>
#include <variant>

namespace source_resolution_test {
using GeocentricTensor = specfem::sources::MomentTensor<
    specfem::coordinate_systems::GeocentricFrame>;
using CartesianTensor =
    specfem::sources::MomentTensor<specfem::coordinate_systems::CartesianFrame>;
using Quantity =
    specfem::sources::LocalFrameQuantity<specfem::sources::MomentTensor>;
// resolve() only accepts the Cartesian-frame tensor (compile-time check).
static_assert(std::is_invocable_v<decltype(&Quantity::resolve), Quantity &,
                                  const CartesianTensor &>);
static_assert(!std::is_invocable_v<decltype(&Quantity::resolve), Quantity &,
                                   const GeocentricTensor &>);
} // namespace source_resolution_test

// A moment tensor constructed in the geocentric frame is unresolved until the
// assembly stage installs the Cartesian components. get_source_tensor must
// throw while unresolved rather than returning a silently-wrong tensor, and the
// as-read geocentric tensor must survive resolution (provenance).
TEST(MomentTensorResolution, GeocentricIsUnresolvedUntilSet) {
  using source_resolution_test::CartesianTensor;
  using source_resolution_test::GeocentricTensor;
  using specfem::element::dimension_tag;
  specfem::sources::moment_tensor<dimension_tag::dim3> src(
      source_resolution_test::origin_coords(),
      GeocentricTensor(type_real(1), type_real(2), type_real(3), type_real(4),
                       type_real(5), type_real(6)),
      source_resolution_test::make_stf(),
      specfem::simulation::field_type::forward);
  src.set_medium_tag(specfem::element::medium_tag::elastic);

  ASSERT_FALSE(src.moment().is_resolved());
  EXPECT_THROW((void)src.get_source_tensor(), std::runtime_error);

  // The as-read geocentric tensor is preserved while unresolved.
  ASSERT_TRUE(std::holds_alternative<GeocentricTensor>(src.moment().as_read()));
  EXPECT_REAL_EQ(std::get<GeocentricTensor>(src.moment().as_read())(0, 0),
                 type_real(1));

  src.set_cartesian_components(CartesianTensor(type_real(10), type_real(20),
                                               type_real(30), type_real(40),
                                               type_real(50), type_real(60)));

  EXPECT_TRUE(src.moment().is_resolved());

  // The original geocentric tensor is still available after resolution.
  ASSERT_TRUE(std::holds_alternative<GeocentricTensor>(src.moment().as_read()));

  const auto t = src.get_source_tensor();
  EXPECT_REAL_EQ(t(0, 0), type_real(10)); // Mxx
  EXPECT_REAL_EQ(t(1, 1), type_real(20)); // Myy
  EXPECT_REAL_EQ(t(2, 2), type_real(30)); // Mzz
  EXPECT_REAL_EQ(t(0, 1), type_real(40)); // Mxy
  EXPECT_REAL_EQ(t(0, 2), type_real(50)); // Mxz
  EXPECT_REAL_EQ(t(1, 2), type_real(60)); // Myz
  // Symmetry
  EXPECT_REAL_EQ(t(1, 0), type_real(40));
  EXPECT_REAL_EQ(t(2, 0), type_real(50));
  EXPECT_REAL_EQ(t(2, 1), type_real(60));
}

// A moment tensor constructed directly from Cartesian components is resolved
// from the start: get_source_tensor works and the as-read value is Cartesian.
TEST(MomentTensorResolution, CartesianCtorIsResolved) {
  specfem::sources::moment_tensor<specfem::element::dimension_tag::dim3> src(
      0.0, 0.0, 0.0, type_real(1), type_real(2), type_real(3), type_real(4),
      type_real(5), type_real(6), source_resolution_test::make_stf(),
      specfem::simulation::field_type::forward);
  src.set_medium_tag(specfem::element::medium_tag::elastic);

  EXPECT_TRUE(src.moment().is_resolved());
  EXPECT_TRUE(std::holds_alternative<source_resolution_test::CartesianTensor>(
      src.moment().as_read()));
  EXPECT_NO_THROW((void)src.get_source_tensor());
}
