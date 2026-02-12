#include "specfem/assembly/info/impl/compute.hpp"
#include "specfem/utilities/is_close.hpp"
#include "test_macros.hpp"
#include <gtest/gtest.h>

using namespace specfem::assembly::info::impl;

class InfoComputeTests : public ::testing::Test {
protected:
  // Constants from specfem::constants
  static constexpr type_real NPTS_PER_WAVELENGTH = 5.0;
  static constexpr type_real COURANT_NUMBER = 0.5;
};

TEST_F(InfoComputeTests, ComputeAverageGllSpacing) {
  // Test with element size 100 and 4 intervals (ngll=5, so ngll-1=4)
  type_real element_size = 100.0;
  int ngll_minus_one = 4;

  type_real result = compute_average_gll_spacing(element_size, ngll_minus_one);
  type_real expected = 25.0; // 100 / 4

  EXPECT_TRUE(specfem::utilities::is_close(result, expected))
      << expected_got(expected, result);
}

TEST_F(InfoComputeTests, ComputeAverageGllSpacingSmallElement) {
  type_real element_size = 10.0;
  int ngll_minus_one = 5;

  type_real result = compute_average_gll_spacing(element_size, ngll_minus_one);
  type_real expected = 2.0; // 10 / 5

  EXPECT_TRUE(specfem::utilities::is_close(result, expected))
      << expected_got(expected, result);
}

TEST_F(InfoComputeTests, ComputeMinimumPeriod) {
  // avg_gll_spacing = 25m, min_velocity = 2500 m/s
  // minimum_period = (NPTS_PER_WAVELENGTH * avg_gll_spacing) / min_velocity
  //                = (5 * 25) / 2500 = 0.05 s
  type_real avg_gll_spacing = 25.0;
  type_real min_velocity = 2500.0;

  type_real result = compute_minimum_period(avg_gll_spacing, min_velocity);
  type_real expected = (NPTS_PER_WAVELENGTH * avg_gll_spacing) / min_velocity;

  EXPECT_TRUE(specfem::utilities::is_close(result, expected))
      << expected_got(expected, result);
}

TEST_F(InfoComputeTests, ComputeMinimumPeriodSlowVelocity) {
  // With slower velocity, period should be longer
  type_real avg_gll_spacing = 25.0;
  type_real min_velocity = 500.0;

  type_real result = compute_minimum_period(avg_gll_spacing, min_velocity);
  type_real expected = (NPTS_PER_WAVELENGTH * avg_gll_spacing) / min_velocity;

  EXPECT_TRUE(specfem::utilities::is_close(result, expected))
      << expected_got(expected, result);

  // Verify it's larger than with faster velocity
  type_real result_fast = compute_minimum_period(avg_gll_spacing, 2500.0);
  EXPECT_GT(result, result_fast);
}

TEST_F(InfoComputeTests, ComputeSuggestedTimestep) {
  // min_gll_distance = 5m, max_velocity = 5000 m/s
  // dt = COURANT_NUMBER * (min_gll_distance / max_velocity)
  //    = 0.5 * (5 / 5000) = 0.0005 s
  type_real min_gll_distance = 5.0;
  type_real max_velocity = 5000.0;

  type_real result = compute_suggested_timestep(min_gll_distance, max_velocity);
  type_real expected = COURANT_NUMBER * (min_gll_distance / max_velocity);

  EXPECT_TRUE(specfem::utilities::is_close(result, expected))
      << expected_got(expected, result);
}

TEST_F(InfoComputeTests, ComputeSuggestedTimestepHighVelocity) {
  // Higher velocity should give smaller timestep
  type_real min_gll_distance = 5.0;
  type_real max_velocity_low = 3000.0;
  type_real max_velocity_high = 6000.0;

  type_real result_low =
      compute_suggested_timestep(min_gll_distance, max_velocity_low);
  type_real result_high =
      compute_suggested_timestep(min_gll_distance, max_velocity_high);

  EXPECT_GT(result_low, result_high);
}

TEST_F(InfoComputeTests, ComputeSuggestedTimestepSmallDistance) {
  // Smaller GLL distance should give smaller timestep
  type_real min_gll_distance_large = 10.0;
  type_real min_gll_distance_small = 2.0;
  type_real max_velocity = 5000.0;

  type_real result_large =
      compute_suggested_timestep(min_gll_distance_large, max_velocity);
  type_real result_small =
      compute_suggested_timestep(min_gll_distance_small, max_velocity);

  EXPECT_GT(result_large, result_small);
}

TEST_F(InfoComputeTests, RealisticSeismicParameters) {
  // Test with realistic seismic parameters
  // Element size: 1000m, ngll=5 -> avg_gll_spacing = 250m
  // Velocities: vp=6000 m/s, vs=3500 m/s
  type_real element_size = 1000.0;
  int ngll_minus_one = 4;
  type_real min_velocity = 3500.0; // vs
  type_real max_velocity = 6000.0; // vp

  type_real avg_spacing =
      compute_average_gll_spacing(element_size, ngll_minus_one);
  EXPECT_TRUE(specfem::utilities::is_close(avg_spacing, type_real(250.0)))
      << expected_got(250.0, avg_spacing);

  // Minimum resolvable period
  type_real min_period = compute_minimum_period(avg_spacing, min_velocity);
  type_real expected_period = (NPTS_PER_WAVELENGTH * 250.0) / 3500.0;
  EXPECT_TRUE(specfem::utilities::is_close(min_period, expected_period))
      << expected_got(expected_period, min_period);

  // For min GLL distance, assume it's about 1/4 of average for GLL clustering
  type_real min_gll_distance = 62.5; // approximate
  type_real dt = compute_suggested_timestep(min_gll_distance, max_velocity);
  type_real expected_dt = COURANT_NUMBER * (62.5 / 6000.0);
  EXPECT_TRUE(specfem::utilities::is_close(dt, expected_dt))
      << expected_got(expected_dt, dt);
}
