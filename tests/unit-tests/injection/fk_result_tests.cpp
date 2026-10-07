#include "specfem/injection/fk/fk_result.hpp"
#include <Kokkos_Core.hpp>
#include <gtest/gtest.h>

using specfem::injection::fk::FkResult;

// ---------------------------------------------------------------------------
// FkResult with traction and pressure allocated
// ---------------------------------------------------------------------------

TEST(FkResult, FullAllocation) {
  FkResult result(5, 16, true, true);

  EXPECT_EQ(result.number_of_points(), 5);
  EXPECT_EQ(result.coefficient_count(), 16);
  EXPECT_TRUE(result.has_traction());
  EXPECT_TRUE(result.has_pressure());

  // Displacement: (5, 3, 16)
  EXPECT_EQ(result.displacement().extent(0), 5);
  EXPECT_EQ(result.displacement().extent(1), 3);
  EXPECT_EQ(result.displacement().extent(2), 16);

  // Traction: (5, 3, 16)
  EXPECT_EQ(result.traction().extent(0), 5);
  EXPECT_EQ(result.traction().extent(1), 3);
  EXPECT_EQ(result.traction().extent(2), 16);

  // Pressure: (5, 16)
  EXPECT_EQ(result.pressure().extent(0), 5);
  EXPECT_EQ(result.pressure().extent(1), 16);
}

// ---------------------------------------------------------------------------
// set_sampling / getters
// ---------------------------------------------------------------------------

TEST(FkResult, SetSampling) {
  FkResult result(5, 16, true, true);

  result.set_sampling(10, 0.001f, 2.5f);

  EXPECT_EQ(result.resampling_rate(), 10);
  // type_real may be float; use float-precision tolerance.
  EXPECT_NEAR(static_cast<double>(result.resampled_dt()), 0.001, 1.0e-5);
  EXPECT_NEAR(static_cast<double>(result.reference_time()), 2.5, 1.0e-5);
}

// ---------------------------------------------------------------------------
// FkResult without traction and pressure: Views have zero extent
// ---------------------------------------------------------------------------

TEST(FkResult, NoTractionNoPressure) {
  FkResult result(5, 16, false, false);

  EXPECT_FALSE(result.has_traction());
  EXPECT_FALSE(result.has_pressure());

  // Displacement is still allocated.
  EXPECT_EQ(result.displacement().extent(0), 5);

  // Traction and pressure Views are default-constructed → extent(0) == 0.
  EXPECT_EQ(result.traction().extent(0), 0);
  EXPECT_EQ(result.pressure().extent(0), 0);
}

// ---------------------------------------------------------------------------
// Default constructor: number_of_points and coefficient_count are 0
// ---------------------------------------------------------------------------

TEST(FkResult, DefaultConstruct) {
  FkResult result;

  EXPECT_EQ(result.number_of_points(), 0);
  EXPECT_EQ(result.coefficient_count(), 0);
  EXPECT_EQ(result.resampling_rate(), 0);
  EXPECT_NEAR(static_cast<double>(result.resampled_dt()), 0.0, 1.0e-12);
  EXPECT_NEAR(static_cast<double>(result.reference_time()), 0.0, 1.0e-12);
  EXPECT_FALSE(result.has_traction());
  EXPECT_FALSE(result.has_pressure());
}
