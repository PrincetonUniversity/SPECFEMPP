#include "specfem/io/sources/impl/timing.hpp"
#include "../../SPECFEM_Environment.hpp"
#include "specfem/datetime.hpp"
#include "specfem/source.hpp"
#include "specfem/source_time_functions.hpp"
#include <gtest/gtest.h>
#include <memory>
#include <vector>

// Convenience aliases
constexpr auto dim2 = specfem::element::dimension_tag::dim2;
using Source2D = specfem::sources::source<dim2>;
using SourcePtr2D = std::shared_ptr<Source2D>;
using SourceVec2D = std::vector<SourcePtr2D>;
// Test constants
static constexpr int kNsteps = 100;
static constexpr type_real kDt = 0.01;
static constexpr type_real kFactor = 1.0e10;
static constexpr bool kUseTrick = false;
static const auto kWavefield = specfem::simulation::field_type::forward;

/// Helper: create a 2D force source with a Ricker STF.
/// Returns the source and the expected t0 value from the STF.
static SourcePtr2D make_force_2d(type_real f0, type_real tshift) {
  return std::make_shared<specfem::sources::force<dim2>>(
      0.0, 0.0, 0.0,
      std::make_unique<specfem::source_time_functions::Ricker>(
          kNsteps, kDt, f0, tshift, kFactor, kUseTrick),
      kWavefield);
}

/// Ricker t0 formula: t0 = -t0_factor * (1/f0) + tshift
/// Default t0_factor for Ricker is 1.2
static type_real ricker_t0(type_real f0, type_real tshift,
                           type_real t0_factor = 1.2) {
  return -t0_factor / f0 + tshift;
}

/// Convert seconds to chrono milliseconds using the same rounding as the
/// implementation (std::chrono::round), avoiding float truncation bugs.
static auto to_ms(type_real seconds) {
  return std::chrono::round<std::chrono::milliseconds>(
      std::chrono::duration<double>(static_cast<double>(seconds)));
}

// ============================================================================
// Test: No starttimes, single source, auto-detect t0
// ============================================================================
TEST(AdjustSourceTiming, SingleSourceNoStarttime) {
  const type_real f0 = 10.0;
  const type_real tshift = 5.0;

  SourceVec2D sources = { make_force_2d(f0, tshift) };

  auto [t0, starttime] =
      specfem::io::sources_impl::adjust_source_timing<dim2>(sources,
                                                            /*user_t0=*/0.0);

  // Expected: t0 = ricker_t0, no starttime
  const type_real expected_t0 = ricker_t0(f0, tshift);
  EXPECT_NEAR(t0, expected_t0, 1e-10);
  EXPECT_FALSE(starttime.has_value());
}

// ============================================================================
// Test: No starttimes, multiple sources, auto-detect t0
// ============================================================================
TEST(AdjustSourceTiming, MultipleSourcesNoStarttime) {
  // Source A: f0=10, tshift=5 -> t0 = -0.12 + 5.0 = 4.88
  // Source B: f0=1,  tshift=30 -> t0 = -1.2 + 30.0 = 28.8
  SourceVec2D sources = { make_force_2d(10.0, 5.0), make_force_2d(1.0, 30.0) };

  const type_real t0_A = ricker_t0(10.0, 5.0); // 4.88
  const type_real t0_B = ricker_t0(1.0, 30.0); // 28.8

  auto [t0, starttime] =
      specfem::io::sources_impl::adjust_source_timing<dim2>(sources,
                                                            /*user_t0=*/0.0);

  // min_t0 = 4.88, so t0 = 4.88
  EXPECT_NEAR(t0, t0_A, 1e-10);

  // Source A tshift unchanged (cur_t0 - min_t0 = 4.88 - 4.88 = 0)
  EXPECT_NEAR(sources[0]->get_tshift(), 0.0, 1e-10);
  // Source B tshift adjusted: cur_t0 - min_t0 = 28.8 - 4.88 = 23.92
  EXPECT_NEAR(sources[1]->get_tshift(), t0_B - t0_A, 1e-10);

  EXPECT_FALSE(starttime.has_value());
}

// ============================================================================
// Test: User-defined t0, no starttimes
// ============================================================================
TEST(AdjustSourceTiming, UserDefinedT0) {
  SourceVec2D sources = { make_force_2d(10.0, 5.0) };

  // User t0 must be <= min_t0 - min_tshift = 4.88 - 5.0 = -0.12
  const type_real user_t0 = -1.0;
  auto [t0, starttime] =
      specfem::io::sources_impl::adjust_source_timing<dim2>(sources, user_t0);

  EXPECT_NEAR(t0, user_t0, 1e-10);
  // tshift is NOT adjusted when user defines t0
  EXPECT_NEAR(sources[0]->get_tshift(), 5.0, 1e-10);
  EXPECT_FALSE(starttime.has_value());
}

// ============================================================================
// Test: User-defined t0 too large -> error
// ============================================================================
TEST(AdjustSourceTiming, UserDefinedT0TooLarge) {
  SourceVec2D sources = { make_force_2d(10.0, 5.0) };

  // min_t0 - min_tshift = 4.88 - 5.0 = -0.12
  // user_t0 = 1.0 > -0.12 -> should throw
  EXPECT_THROW(
      specfem::io::sources_impl::adjust_source_timing<dim2>(sources, 1.0),
      std::runtime_error);
}

// ============================================================================
// Test: Single source with starttime
// ============================================================================
TEST(AdjustSourceTiming, SingleSourceWithStarttime) {
  const type_real f0 = 10.0;
  const type_real tshift = 5.0;

  auto src = make_force_2d(f0, tshift);
  // Set UTC origin time: 2003-12-26T01:56:52.400
  auto origin = specfem::datetime::make(2003, 12, 26, 1, 56, 52.4);
  src->set_starttime(origin);

  SourceVec2D sources = { src };

  auto [t0, starttime] =
      specfem::io::sources_impl::adjust_source_timing<dim2>(sources,
                                                            /*user_t0=*/0.0);

  const type_real expected_t0 = ricker_t0(f0, tshift);
  EXPECT_NEAR(t0, expected_t0, 1e-10);
  ASSERT_TRUE(starttime.has_value());

  // UTC(t=t0) = origin_time - tshift + t0
  // After auto-adjust, single source: tshift becomes 0 (cur_t0 - min_t0 = 0)
  // So: starttime = origin - 0ms + t0_ms
  auto expected_starttime =
      origin - to_ms(sources[0]->get_tshift()) + to_ms(t0);
  EXPECT_EQ(*starttime, expected_starttime);
}

// ============================================================================
// Test: One source with starttime, others without (single starttime case)
// ============================================================================
TEST(AdjustSourceTiming, OneOfManyHasStarttime) {
  auto src_a = make_force_2d(10.0, 5.0);
  auto src_b = make_force_2d(1.0, 30.0);

  auto origin = specfem::datetime::make(2003, 12, 26, 1, 56, 52.4);
  src_a->set_starttime(origin);

  SourceVec2D sources = { src_a, src_b };

  auto [t0, starttime] =
      specfem::io::sources_impl::adjust_source_timing<dim2>(sources,
                                                            /*user_t0=*/0.0);

  const type_real t0_A = ricker_t0(10.0, 5.0);

  EXPECT_NEAR(t0, t0_A, 1e-10);
  ASSERT_TRUE(starttime.has_value());

  // Source A tshift adjusted to 0 (min_t0 source), source B adjusted normally
  auto expected_starttime =
      origin - to_ms(sources[0]->get_tshift()) + to_ms(t0);
  EXPECT_EQ(*starttime, expected_starttime);
}

// ============================================================================
// Test: All sources have same starttime
// ============================================================================
TEST(AdjustSourceTiming, AllSourcesSameStarttime) {
  auto src_a = make_force_2d(10.0, 5.0);
  auto src_b = make_force_2d(10.0, 5.0);

  auto origin = specfem::datetime::make(2003, 12, 26, 1, 56, 52.4);
  src_a->set_starttime(origin);
  src_b->set_starttime(origin);

  SourceVec2D sources = { src_a, src_b };

  auto [t0, starttime] =
      specfem::io::sources_impl::adjust_source_timing<dim2>(sources,
                                                            /*user_t0=*/0.0);

  ASSERT_TRUE(starttime.has_value());

  // Both sources have same origin and same tshift, so tshifts should be
  // identical after adjustment.
  // earliest_t0_utc = origin - to_ms(tshift) = origin - 5s
  // new_tshift = (origin - earliest_t0_utc) = (origin - (origin - 5s)) = 5s
  EXPECT_NEAR(sources[0]->get_tshift(), sources[1]->get_tshift(), 1e-10);
  EXPECT_NEAR(sources[0]->get_tshift(), 5.0, 1e-10);
}

// ============================================================================
// Test: All sources have different starttimes
// ============================================================================
TEST(AdjustSourceTiming, AllSourcesDifferentStarttimes) {
  auto src_a = make_force_2d(10.0, 0.0); // tshift=0
  auto src_b = make_force_2d(10.0, 0.0); // tshift=0

  // Source A fires 2 seconds before source B
  auto origin_a = specfem::datetime::make(2003, 12, 26, 1, 56, 50.0);
  auto origin_b = specfem::datetime::make(2003, 12, 26, 1, 56, 52.0);
  src_a->set_starttime(origin_a);
  src_b->set_starttime(origin_b);

  SourceVec2D sources = { src_a, src_b };

  auto [t0, starttime] =
      specfem::io::sources_impl::adjust_source_timing<dim2>(sources,
                                                            /*user_t0=*/0.0);

  ASSERT_TRUE(starttime.has_value());

  // Both have tshift=0 initially.
  // earliest_t0_utc = min(origin_a - 0, origin_b - 0) = origin_a
  // Source A new tshift = (origin_a - origin_a) = 0 seconds
  // Source B new tshift = (origin_b - origin_a) = 2 seconds
  EXPECT_NEAR(sources[0]->get_tshift(), 0.0, 1e-10);
  EXPECT_NEAR(sources[1]->get_tshift(), 2.0, 1e-10);

  // Simulation start: earliest_t0_utc + t0
  auto expected_starttime = origin_a + to_ms(t0);
  EXPECT_EQ(*starttime, expected_starttime);
}

// ============================================================================
// Test: Multiple starttimes with nonzero initial tshifts
// ============================================================================
TEST(AdjustSourceTiming, MultipleStarttimesWithTshifts) {
  auto src_a = make_force_2d(10.0, 1.0); // tshift=1.0
  auto src_b = make_force_2d(10.0, 3.0); // tshift=3.0

  // Source A origin: 2003-12-26T01:56:50.000
  // Source B origin: 2003-12-26T01:56:55.000
  auto origin_a = specfem::datetime::make(2003, 12, 26, 1, 56, 50.0);
  auto origin_b = specfem::datetime::make(2003, 12, 26, 1, 56, 55.0);
  src_a->set_starttime(origin_a);
  src_b->set_starttime(origin_b);

  SourceVec2D sources = { src_a, src_b };

  auto [t0, starttime] =
      specfem::io::sources_impl::adjust_source_timing<dim2>(sources,
                                                            /*user_t0=*/0.0);

  ASSERT_TRUE(starttime.has_value());

  // UTC(t=0) candidates:
  //   Source A: origin_a - tshift_a = 50.0 - 1.0 = 49.0s (in the second field)
  //   Source B: origin_b - tshift_b = 55.0 - 3.0 = 52.0s
  // earliest_t0_utc = 2003-12-26T01:56:49.000
  auto earliest = origin_a - to_ms(static_cast<type_real>(1.0));

  // New tshifts:
  //   Source A: origin_a - earliest = 50.0 - 49.0 = 1.0s
  //   Source B: origin_b - earliest = 55.0 - 49.0 = 6.0s
  EXPECT_NEAR(sources[0]->get_tshift(), 1.0, 1e-6);
  EXPECT_NEAR(sources[1]->get_tshift(), 6.0, 1e-6);

  // Simulation start
  auto expected_starttime = earliest + to_ms(t0);
  EXPECT_EQ(*starttime, expected_starttime);
}

// ============================================================================
// Test: Inconsistent starttimes (2 of 3) -> error
// ============================================================================
TEST(AdjustSourceTiming, InconsistentStarttimesThrows) {
  auto src_a = make_force_2d(10.0, 0.0);
  auto src_b = make_force_2d(10.0, 0.0);
  auto src_c = make_force_2d(10.0, 0.0);

  auto origin = specfem::datetime::make(2003, 12, 26, 1, 56, 52.4);
  src_a->set_starttime(origin);
  src_b->set_starttime(origin);
  // src_c has NO starttime — 2 of 3 is invalid

  SourceVec2D sources = { src_a, src_b, src_c };

  EXPECT_THROW(
      specfem::io::sources_impl::adjust_source_timing<dim2>(sources, 0.0),
      std::runtime_error);
}

// ============================================================================
// Test: No starttimes at all -> nullopt
// ============================================================================
TEST(AdjustSourceTiming, NoStarttimesReturnsNullopt) {
  SourceVec2D sources = { make_force_2d(10.0, 5.0), make_force_2d(1.0, 30.0) };

  auto [t0, starttime] =
      specfem::io::sources_impl::adjust_source_timing<dim2>(sources,
                                                            /*user_t0=*/0.0);

  EXPECT_FALSE(starttime.has_value());
}

int main(int argc, char *argv[]) {
  ::testing::InitGoogleTest(&argc, argv);
  ::testing::AddGlobalTestEnvironment(new SPECFEMEnvironment);
  return RUN_ALL_TESTS();
}
