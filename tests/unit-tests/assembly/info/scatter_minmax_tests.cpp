#include "specfem/assembly/info/impl/scatter_minmax.hpp"
#include "specfem/utilities/is_close.hpp"
#include "test_macros.hpp"
#include <Kokkos_Core.hpp>
#include <gtest/gtest.h>

using namespace specfem::assembly::info::impl;

class ScatterMinMaxTests : public ::testing::Test {
protected:
  static constexpr int N = 1000;
};

// Test LocalMinMax basic functionality in a parallel_for
TEST_F(ScatterMinMaxTests, LocalMinMaxBasic) {
  // Create a view with known values: 0, 1, 2, ..., N-1
  Kokkos::View<type_real *, Kokkos::DefaultExecutionSpace> data("data", N);

  Kokkos::parallel_for(
      "initialize_data", Kokkos::RangePolicy<Kokkos::DefaultExecutionSpace>(0, N),
      KOKKOS_LAMBDA(const int i) { data(i) = static_cast<type_real>(i); });
  Kokkos::fence();

  // Use LocalMinMax in a parallel_for with a parallel_reduce pattern
  type_real global_min, global_max;

  Kokkos::parallel_reduce(
      "find_minmax", Kokkos::RangePolicy<Kokkos::DefaultExecutionSpace>(0, N),
      KOKKOS_LAMBDA(const int i, type_real &lmin, type_real &lmax) {
        LocalMinMax<type_real> local;
        local.update(data(i));
        lmin = Kokkos::fmin(lmin, local.min_val);
        lmax = Kokkos::fmax(lmax, local.max_val);
      },
      Kokkos::Min<type_real>(global_min), Kokkos::Max<type_real>(global_max));

  Kokkos::fence();

  EXPECT_TRUE(specfem::utilities::is_close(global_min, type_real(0.0)))
      << expected_got(type_real(0.0), global_min);
  EXPECT_TRUE(specfem::utilities::is_close(global_max, type_real(N - 1)))
      << expected_got(type_real(N - 1), global_max);
}

// Test LocalMinMax update_min and update_max separately
TEST_F(ScatterMinMaxTests, LocalMinMaxSeparateUpdates) {
  Kokkos::View<type_real *, Kokkos::DefaultExecutionSpace> data("data", N);

  // Initialize with values: 100, 101, ..., 100+N-1
  Kokkos::parallel_for(
      "initialize_data", Kokkos::RangePolicy<Kokkos::DefaultExecutionSpace>(0, N),
      KOKKOS_LAMBDA(const int i) { data(i) = static_cast<type_real>(100 + i); });
  Kokkos::fence();

  type_real global_min, global_max;

  Kokkos::parallel_reduce(
      "find_minmax_separate",
      Kokkos::RangePolicy<Kokkos::DefaultExecutionSpace>(0, N),
      KOKKOS_LAMBDA(const int i, type_real &lmin, type_real &lmax) {
        LocalMinMax<type_real> local;
        local.update_min(data(i));
        local.update_max(data(i));
        lmin = Kokkos::fmin(lmin, local.min_val);
        lmax = Kokkos::fmax(lmax, local.max_val);
      },
      Kokkos::Min<type_real>(global_min), Kokkos::Max<type_real>(global_max));

  Kokkos::fence();

  EXPECT_TRUE(specfem::utilities::is_close(global_min, type_real(100.0)))
      << expected_got(type_real(100.0), global_min);
  EXPECT_TRUE(specfem::utilities::is_close(global_max, type_real(100 + N - 1)))
      << expected_got(type_real(100 + N - 1), global_max);
}

// Test ScatterMinMax with parallel_for
TEST_F(ScatterMinMaxTests, ScatterMinMaxBasic) {
  // Create data with known min/max: values from -50 to N-51
  Kokkos::View<type_real *, Kokkos::DefaultExecutionSpace> data("data", N);

  Kokkos::parallel_for(
      "initialize_data", Kokkos::RangePolicy<Kokkos::DefaultExecutionSpace>(0, N),
      KOKKOS_LAMBDA(const int i) { data(i) = static_cast<type_real>(i - 50); });
  Kokkos::fence();

  // Use ScatterMinMax
  ScatterMinMax<type_real> scatter("test");
  auto accessor = scatter.access();

  Kokkos::parallel_for(
      "scatter_minmax", Kokkos::RangePolicy<Kokkos::DefaultExecutionSpace>(0, N),
      KOKKOS_LAMBDA(const int i) { accessor.update(data(i)); });
  Kokkos::fence();

  scatter.contribute();

  Bounds bounds = scatter.get_bounds();

  EXPECT_TRUE(specfem::utilities::is_close(bounds.min, type_real(-50.0)))
      << expected_got(type_real(-50.0), bounds.min);
  EXPECT_TRUE(specfem::utilities::is_close(bounds.max, type_real(N - 51)))
      << expected_got(type_real(N - 51), bounds.max);
}

// Test ScatterMinMax with separate min/max updates
TEST_F(ScatterMinMaxTests, ScatterMinMaxSeparateUpdates) {
  // Create two data arrays: one for min updates, one for max updates
  Kokkos::View<type_real *, Kokkos::DefaultExecutionSpace> min_data("min_data",
                                                                    N);
  Kokkos::View<type_real *, Kokkos::DefaultExecutionSpace> max_data("max_data",
                                                                    N);

  // min_data: 0, 1, 2, ..., N-1 (min should be 0)
  // max_data: 1000, 1001, ..., 1000+N-1 (max should be 1000+N-1)
  Kokkos::parallel_for(
      "initialize_data", Kokkos::RangePolicy<Kokkos::DefaultExecutionSpace>(0, N),
      KOKKOS_LAMBDA(const int i) {
        min_data(i) = static_cast<type_real>(i);
        max_data(i) = static_cast<type_real>(1000 + i);
      });
  Kokkos::fence();

  ScatterMinMax<type_real> scatter("test_separate");
  auto accessor = scatter.access();

  Kokkos::parallel_for(
      "scatter_separate",
      Kokkos::RangePolicy<Kokkos::DefaultExecutionSpace>(0, N),
      KOKKOS_LAMBDA(const int i) {
        accessor.update_min(min_data(i));
        accessor.update_max(max_data(i));
      });
  Kokkos::fence();

  scatter.contribute();

  Bounds bounds = scatter.get_bounds();

  EXPECT_TRUE(specfem::utilities::is_close(bounds.min, type_real(0.0)))
      << expected_got(type_real(0.0), bounds.min);
  EXPECT_TRUE(specfem::utilities::is_close(bounds.max, type_real(1000 + N - 1)))
      << expected_got(type_real(1000 + N - 1), bounds.max);
}

// Test ScatterMinMax with negative values
TEST_F(ScatterMinMaxTests, ScatterMinMaxNegativeValues) {
  Kokkos::View<type_real *, Kokkos::DefaultExecutionSpace> data("data", N);

  // Values from -N/2 to N/2-1
  Kokkos::parallel_for(
      "initialize_data", Kokkos::RangePolicy<Kokkos::DefaultExecutionSpace>(0, N),
      KOKKOS_LAMBDA(const int i) {
        data(i) = static_cast<type_real>(i - N / 2);
      });
  Kokkos::fence();

  ScatterMinMax<type_real> scatter("test_negative");
  auto accessor = scatter.access();

  Kokkos::parallel_for(
      "scatter_negative",
      Kokkos::RangePolicy<Kokkos::DefaultExecutionSpace>(0, N),
      KOKKOS_LAMBDA(const int i) { accessor.update(data(i)); });
  Kokkos::fence();

  scatter.contribute();

  Bounds bounds = scatter.get_bounds();

  type_real expected_min = static_cast<type_real>(-N / 2);
  type_real expected_max = static_cast<type_real>(N - 1 - N / 2);

  EXPECT_TRUE(specfem::utilities::is_close(bounds.min, expected_min))
      << expected_got(expected_min, bounds.min);
  EXPECT_TRUE(specfem::utilities::is_close(bounds.max, expected_max))
      << expected_got(expected_max, bounds.max);
}

// Test Bounds helper methods
TEST_F(ScatterMinMaxTests, BoundsHelperMethods) {
  Kokkos::View<type_real *, Kokkos::DefaultExecutionSpace> data("data", N);

  // Values from 10 to 10+N-1
  Kokkos::parallel_for(
      "initialize_data", Kokkos::RangePolicy<Kokkos::DefaultExecutionSpace>(0, N),
      KOKKOS_LAMBDA(const int i) { data(i) = static_cast<type_real>(10 + i); });
  Kokkos::fence();

  ScatterMinMax<type_real> scatter("test_bounds");
  auto accessor = scatter.access();

  Kokkos::parallel_for(
      "scatter_bounds", Kokkos::RangePolicy<Kokkos::DefaultExecutionSpace>(0, N),
      KOKKOS_LAMBDA(const int i) { accessor.update(data(i)); });
  Kokkos::fence();

  scatter.contribute();

  Bounds bounds = scatter.get_bounds();

  type_real expected_min = 10.0;
  type_real expected_max = static_cast<type_real>(10 + N - 1);
  type_real expected_length = expected_max - expected_min;
  type_real expected_center = 0.5 * (expected_max + expected_min);
  type_real expected_ratio = expected_max / expected_min;

  EXPECT_TRUE(specfem::utilities::is_close(bounds.length(), expected_length))
      << expected_got(expected_length, bounds.length());
  EXPECT_TRUE(specfem::utilities::is_close(bounds.center(), expected_center))
      << expected_got(expected_center, bounds.center());
  EXPECT_TRUE(specfem::utilities::is_close(bounds.ratio(), expected_ratio))
      << expected_got(expected_ratio, bounds.ratio());
}

// Test with larger data set to stress parallel reduction
TEST_F(ScatterMinMaxTests, ScatterMinMaxLargeDataset) {
  constexpr int LARGE_N = 100000;

  Kokkos::View<type_real *, Kokkos::DefaultExecutionSpace> data("data", LARGE_N);

  // Use a pattern where min and max are at specific known locations
  // All values are 500.0 except: data[12345] = -999.0, data[67890] = 9999.0
  Kokkos::parallel_for(
      "initialize_large",
      Kokkos::RangePolicy<Kokkos::DefaultExecutionSpace>(0, LARGE_N),
      KOKKOS_LAMBDA(const int i) {
        if (i == 12345) {
          data(i) = static_cast<type_real>(-999.0);
        } else if (i == 67890) {
          data(i) = static_cast<type_real>(9999.0);
        } else {
          data(i) = static_cast<type_real>(500.0);
        }
      });
  Kokkos::fence();

  ScatterMinMax<type_real> scatter("test_large");
  auto accessor = scatter.access();

  Kokkos::parallel_for(
      "scatter_large",
      Kokkos::RangePolicy<Kokkos::DefaultExecutionSpace>(0, LARGE_N),
      KOKKOS_LAMBDA(const int i) { accessor.update(data(i)); });
  Kokkos::fence();

  scatter.contribute();

  Bounds bounds = scatter.get_bounds();

  EXPECT_TRUE(specfem::utilities::is_close(bounds.min, type_real(-999.0)))
      << expected_got(type_real(-999.0), bounds.min);
  EXPECT_TRUE(specfem::utilities::is_close(bounds.max, type_real(9999.0)))
      << expected_got(type_real(9999.0), bounds.max);
}
