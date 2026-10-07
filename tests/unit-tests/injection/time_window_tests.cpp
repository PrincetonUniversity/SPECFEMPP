#include "specfem/injection/fk/time_window.hpp"
#include <gtest/gtest.h>

using specfem::injection::fk::compute_fk_sizes;
using specfem::injection::fk::FkSizes;
using specfem::injection::fk::TimeWindow;

// ---------------------------------------------------------------------------
// Golden-value case 1 (derivation in comments)
//
// dt=0.01, nstep=1000, frequency_max=1.0, frequency_sampling=10.0,
// time_window_length=40.0
//
// dt_min_raw = 1/10 = 0.1
// NP_RESAMP  = floor(0.1/0.01) = 10
// dt_min     = 10*0.01 = 0.1
// tmax_samp  = (2*1000/10 + 1)*0.1 = 201*0.1 = 20.1
// tmax_use   = min(40, 20.1) = 20.1
// ceil(20.1/0.1) = 201
// ceil(log2(201)) = ceil(7.651) = 8  → NF_FOR_STORING exponent = 8
// NF_FOR_FFT = 1<<9 = 512
// NPOW_FOR_INTERP = 9
// NF_FOR_STORING = 1<<8 = 256
// tmax_fk = 0.1*511 = 51.1
// DF_FK = 1/51.1 ≈ 0.0195695...
// ---------------------------------------------------------------------------

TEST(TimeWindow, GoldenCase1) {
  TimeWindow window;
  window.dt = 0.01;
  window.nstep = 1000;
  window.frequency_max = 1.0;
  window.frequency_sampling = 10.0;
  window.time_window_length = 40.0;

  FkSizes sizes = compute_fk_sizes(window);

  EXPECT_EQ(sizes.resampling_rate, 10);
  EXPECT_EQ(sizes.number_of_stored_frequencies, 256);
  EXPECT_EQ(sizes.number_of_fft_samples, 512);
  EXPECT_EQ(sizes.interpolation_power, 9);
  // type_real may be float; cast to double for EXPECT_NEAR.
  EXPECT_NEAR(static_cast<double>(sizes.frequency_step), 1.0 / 51.1, 1.0e-5);
  EXPECT_NEAR(static_cast<double>(sizes.effective_time_window), 51.1, 1.0e-3);
}

// ---------------------------------------------------------------------------
// Clamp case: floor(dt_min_raw / dt) == 0 → NP_RESAMP clamped to 1
//
// dt=2.0, frequency_sampling=1.0
//   dt_min_raw = 1/1 = 1.0
//   NP_RESAMP  = floor(1.0/2.0) = 0 → clamped to 1
//   resampling_rate >= 1 must always hold.
// ---------------------------------------------------------------------------

TEST(TimeWindow, ClampResamplingRateToOne) {
  TimeWindow window;
  window.dt = 2.0;
  window.nstep = 100;
  window.frequency_max = 0.5;
  window.frequency_sampling = 1.0;
  window.time_window_length = 50.0;

  FkSizes sizes = compute_fk_sizes(window);

  EXPECT_GE(sizes.resampling_rate, 1);
  EXPECT_EQ(sizes.resampling_rate, 1);
}

// ---------------------------------------------------------------------------
// Sanity checks: sizes are always positive powers of two
// ---------------------------------------------------------------------------

TEST(TimeWindow, SizesArePositive) {
  TimeWindow window;
  window.dt = 0.05;
  window.nstep = 500;
  window.frequency_max = 2.0;
  window.frequency_sampling = 5.0;
  window.time_window_length = 30.0;

  FkSizes sizes = compute_fk_sizes(window);

  EXPECT_GE(sizes.resampling_rate, 1);
  EXPECT_GT(sizes.number_of_stored_frequencies, 0);
  EXPECT_GT(sizes.number_of_fft_samples, 0);
  EXPECT_GT(sizes.interpolation_power, 0);
  EXPECT_GT(static_cast<double>(sizes.frequency_step), 0.0);
  EXPECT_GT(static_cast<double>(sizes.effective_time_window), 0.0);

  // NF_FOR_FFT == 2 * NF_FOR_STORING.
  EXPECT_EQ(sizes.number_of_fft_samples,
            2 * sizes.number_of_stored_frequencies);
  // interpolation_power == log2(NF_FOR_FFT).
  EXPECT_EQ(1 << sizes.interpolation_power, sizes.number_of_fft_samples);
}
