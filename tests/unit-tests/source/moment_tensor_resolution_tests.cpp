#include "specfem/coordinate_systems/cartesian.hpp"
#include "specfem/setup.hpp"
#include "specfem/source.hpp"
#include "specfem/source_time_functions.hpp"
#include "test_macros.hpp"

#include <gtest/gtest.h>
#include <memory>
#include <stdexcept>
#include <variant>

namespace {

constexpr auto dim3 = specfem::element::dimension_tag::dim3;

std::unique_ptr<specfem::source_time_functions::stf> make_stf() {
  return std::make_unique<specfem::source_time_functions::Ricker>(
      10, 0.01, 1.0, 0.0, 1.0, false);
}

std::unique_ptr<specfem::coordinate_systems::coordinates<dim3>>
origin_coords() {
  return std::make_unique<
      specfem::coordinate_systems::cartesian_coordinates<dim3>>(0.0, 0.0, 0.0);
}

} // namespace

// A moment tensor constructed in the geographic frame is unresolved until the
// assembly stage installs the Cartesian components. get_source_tensor must
// throw while unresolved rather than returning a silently-wrong tensor, and the
// as-read geographic tensor must survive resolution (provenance).
TEST(MomentTensorResolution, GeographicIsUnresolvedUntilSet) {
  specfem::sources::moment_tensor<dim3> src(
      origin_coords(),
      specfem::sources::GeographicMomentTensor{ type_real(1), type_real(2),
                                                type_real(3), type_real(4),
                                                type_real(5), type_real(6) },
      make_stf(), specfem::simulation::field_type::forward);
  src.set_medium_tag(specfem::element::medium_tag::elastic);

  ASSERT_FALSE(src.moment().is_resolved());
  EXPECT_THROW((void)src.get_source_tensor(), std::runtime_error);

  // The as-read geographic tensor is preserved while unresolved.
  ASSERT_TRUE(std::holds_alternative<specfem::sources::GeographicMomentTensor>(
      src.moment().as_read()));
  EXPECT_REAL_EQ(
      std::get<specfem::sources::GeographicMomentTensor>(src.moment().as_read())
          .Mrr,
      type_real(1));

  src.set_cartesian_components(specfem::sources::CartesianMomentTensor{
      type_real(10), type_real(20), type_real(30), type_real(40), type_real(50),
      type_real(60) });

  EXPECT_TRUE(src.moment().is_resolved());

  // The original geographic tensor is still available after resolution.
  ASSERT_TRUE(std::holds_alternative<specfem::sources::GeographicMomentTensor>(
      src.moment().as_read()));

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
  specfem::sources::moment_tensor<dim3> src(
      0.0, 0.0, 0.0, type_real(1), type_real(2), type_real(3), type_real(4),
      type_real(5), type_real(6), make_stf(),
      specfem::simulation::field_type::forward);
  src.set_medium_tag(specfem::element::medium_tag::elastic);

  EXPECT_TRUE(src.moment().is_resolved());
  EXPECT_TRUE(std::holds_alternative<specfem::sources::CartesianMomentTensor>(
      src.moment().as_read()));
  EXPECT_NO_THROW((void)src.get_source_tensor());
}
