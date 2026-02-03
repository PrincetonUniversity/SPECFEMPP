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

  Kokkos::parallel_for(
      "scatter_minmax", Kokkos::RangePolicy<Kokkos::DefaultExecutionSpace>(0, N),
      KOKKOS_LAMBDA(const int i) { 
        auto accessor = scatter.access();
        accessor.update(data(i)); 
      });
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

  Kokkos::parallel_for(
      "scatter_separate",
      Kokkos::RangePolicy<Kokkos::DefaultExecutionSpace>(0, N),
      KOKKOS_LAMBDA(const int i) {
        auto accessor = scatter.access();
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

  Kokkos::parallel_for(
      "scatter_negative",
      Kokkos::RangePolicy<Kokkos::DefaultExecutionSpace>(0, N),
      KOKKOS_LAMBDA(const int i) { 
        auto accessor = scatter.access();
        accessor.update(data(i)); 
      });
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

  Kokkos::parallel_for(
      "scatter_bounds", Kokkos::RangePolicy<Kokkos::DefaultExecutionSpace>(0, N),
      KOKKOS_LAMBDA(const int i) { 
        auto accessor = scatter.access();
        accessor.update(data(i)); 
      });
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

  Kokkos::parallel_for(
      "scatter_large",
      Kokkos::RangePolicy<Kokkos::DefaultExecutionSpace>(0, LARGE_N),
      KOKKOS_LAMBDA(const int i) { 
        auto accessor = scatter.access();
        accessor.update(data(i)); 
      });
  Kokkos::fence();

  scatter.contribute();

  Bounds bounds = scatter.get_bounds();

  EXPECT_TRUE(specfem::utilities::is_close(bounds.min, type_real(-999.0)))
      << expected_got(type_real(-999.0), bounds.min);
  EXPECT_TRUE(specfem::utilities::is_close(bounds.max, type_real(9999.0)))
      << expected_got(type_real(9999.0), bounds.max);
}

// Test ScatterMinMax with N values (multi-value tracking)
TEST_F(ScatterMinMaxTests, ScatterMinMaxMultiValue) {
  constexpr int NUM_BUCKETS = 10;
  constexpr int ITEMS_PER_BUCKET = 100;

  // Create data where each bucket has values: bucket_id*100 to bucket_id*100 + ITEMS_PER_BUCKET - 1
  Kokkos::View<type_real **, Kokkos::DefaultExecutionSpace> data(
      "data", NUM_BUCKETS, ITEMS_PER_BUCKET);

  Kokkos::parallel_for(
      "initialize_multi",
      Kokkos::MDRangePolicy<Kokkos::Rank<2>, Kokkos::DefaultExecutionSpace>(
          {0, 0}, {NUM_BUCKETS, ITEMS_PER_BUCKET}),
      KOKKOS_LAMBDA(const int bucket, const int item) {
        data(bucket, item) = static_cast<type_real>(bucket * 100 + item);
      });
  Kokkos::fence();

  // Use ScatterMinMax with N buckets
  ScatterMinMax<type_real> scatter("test_multi", NUM_BUCKETS);

  EXPECT_EQ(scatter.size(), NUM_BUCKETS);

  Kokkos::parallel_for(
      "scatter_multi",
      Kokkos::MDRangePolicy<Kokkos::Rank<2>, Kokkos::DefaultExecutionSpace>(
          {0, 0}, {NUM_BUCKETS, ITEMS_PER_BUCKET}),
      KOKKOS_LAMBDA(const int bucket, const int item) {
        auto accessor = scatter.access();
        accessor.update(bucket, data(bucket, item));
      });
  Kokkos::fence();

  scatter.contribute();

  // Check each bucket's bounds
  for (int bucket = 0; bucket < NUM_BUCKETS; ++bucket) {
    Bounds bounds = scatter.get_bounds(bucket);
    type_real expected_min = static_cast<type_real>(bucket * 100);
    type_real expected_max = static_cast<type_real>(bucket * 100 + ITEMS_PER_BUCKET - 1);

    EXPECT_TRUE(specfem::utilities::is_close(bounds.min, expected_min))
        << "Bucket " << bucket << ": " << expected_got(expected_min, bounds.min);
    EXPECT_TRUE(specfem::utilities::is_close(bounds.max, expected_max))
        << "Bucket " << bucket << ": " << expected_got(expected_max, bounds.max);
  }
}

// Test ScatterMinMax get_all_bounds
TEST_F(ScatterMinMaxTests, ScatterMinMaxGetAllBounds) {
  constexpr int NUM_BUCKETS = 5;

  ScatterMinMax<type_real> scatter("test_all_bounds", NUM_BUCKETS);

  // Update each bucket with a single known value
  Kokkos::parallel_for(
      "scatter_all_bounds",
      Kokkos::RangePolicy<Kokkos::DefaultExecutionSpace>(0, NUM_BUCKETS),
      KOKKOS_LAMBDA(const int bucket) {
        auto accessor = scatter.access();
        // Each bucket gets value bucket * 10
        accessor.update(bucket, static_cast<type_real>(bucket * 10));
      });
  Kokkos::fence();

  scatter.contribute();

  std::vector<Bounds> all_bounds = scatter.get_all_bounds();

  EXPECT_EQ(all_bounds.size(), NUM_BUCKETS);

  for (int bucket = 0; bucket < NUM_BUCKETS; ++bucket) {
    type_real expected_val = static_cast<type_real>(bucket * 10);
    EXPECT_TRUE(specfem::utilities::is_close(all_bounds[bucket].min, expected_val))
        << "Bucket " << bucket << " min: " << expected_got(expected_val, all_bounds[bucket].min);
    EXPECT_TRUE(specfem::utilities::is_close(all_bounds[bucket].max, expected_val))
        << "Bucket " << bucket << " max: " << expected_got(expected_val, all_bounds[bucket].max);
  }
}

// Test ScatterMinMax multi-value with separate min/max updates
TEST_F(ScatterMinMaxTests, ScatterMinMaxMultiValueSeparateUpdates) {
  constexpr int NUM_BUCKETS = 4;

  ScatterMinMax<type_real> scatter("test_multi_separate", NUM_BUCKETS);

  // Update min and max separately for each bucket
  Kokkos::parallel_for(
      "scatter_multi_separate",
      Kokkos::RangePolicy<Kokkos::DefaultExecutionSpace>(0, NUM_BUCKETS),
      KOKKOS_LAMBDA(const int bucket) {
        auto accessor = scatter.access();
        // Min value: -bucket
        accessor.update_min(bucket, static_cast<type_real>(-bucket));
        // Max value: bucket * 100
        accessor.update_max(bucket, static_cast<type_real>(bucket * 100));
      });
  Kokkos::fence();

  scatter.contribute();

  for (int bucket = 0; bucket < NUM_BUCKETS; ++bucket) {
    Bounds bounds = scatter.get_bounds(bucket);
    type_real expected_min = static_cast<type_real>(-bucket);
    type_real expected_max = static_cast<type_real>(bucket * 100);

    EXPECT_TRUE(specfem::utilities::is_close(bounds.min, expected_min))
        << "Bucket " << bucket << " min: " << expected_got(expected_min, bounds.min);
    EXPECT_TRUE(specfem::utilities::is_close(bounds.max, expected_max))
        << "Bucket " << bucket << " max: " << expected_got(expected_max, bounds.max);
  }
}
