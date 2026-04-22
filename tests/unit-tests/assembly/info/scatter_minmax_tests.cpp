#include "specfem/assembly/info/impl/scatter_minmax.hpp"
#include "specfem/utilities/is_close.hpp"
#include "test_macros.hpp"
#include <Kokkos_Core.hpp>
#include <gtest/gtest.h>

using namespace specfem::assembly::info::impl;

// ============================================================================
// Functors for parallel operations (avoid CUDA extended lambda restrictions)
// ============================================================================

// Initialize view with values: offset, offset+1, ..., offset+N-1
template <typename ViewType> struct InitializeSequential {
  ViewType data;
  type_real offset;

  InitializeSequential(ViewType data_, type_real offset_ = 0)
      : data(data_), offset(offset_) {}

  KOKKOS_INLINE_FUNCTION
  void operator()(const int i) const {
    data(i) = static_cast<type_real>(i) + offset;
  }
};

// Initialize 2D view with values: bucket*100 + item
template <typename ViewType> struct InitializeMulti {
  ViewType data;

  InitializeMulti(ViewType data_) : data(data_) {}

  KOKKOS_INLINE_FUNCTION
  void operator()(const int bucket, const int item) const {
    data(bucket, item) = static_cast<type_real>(bucket * 100 + item);
  }
};

// Initialize with special pattern for large dataset test
template <typename ViewType> struct InitializeLargePattern {
  ViewType data;

  InitializeLargePattern(ViewType data_) : data(data_) {}

  KOKKOS_INLINE_FUNCTION
  void operator()(const int i) const {
    if (i == 12345) {
      data(i) = static_cast<type_real>(-999.0);
    } else if (i == 67890) {
      data(i) = static_cast<type_real>(9999.0);
    } else {
      data(i) = static_cast<type_real>(500.0);
    }
  }
};

// Initialize two views: min_data with i, max_data with 1000+i
template <typename ViewType> struct InitializeTwoViews {
  ViewType min_data;
  ViewType max_data;

  InitializeTwoViews(ViewType min_data_, ViewType max_data_)
      : min_data(min_data_), max_data(max_data_) {}

  KOKKOS_INLINE_FUNCTION
  void operator()(const int i) const {
    min_data(i) = static_cast<type_real>(i);
    max_data(i) = static_cast<type_real>(1000 + i);
  }
};

// Reduce to find min/max using LocalMinMax
template <typename ViewType> struct FindMinMaxReduce {
  ViewType data;

  FindMinMaxReduce(ViewType data_) : data(data_) {}

  KOKKOS_INLINE_FUNCTION
  void operator()(const int i, type_real &lmin, type_real &lmax) const {
    LocalMinMax<type_real> local;
    local.update(data(i));
    lmin = Kokkos::fmin(lmin, local.min_val);
    lmax = Kokkos::fmax(lmax, local.max_val);
  }
};

// Reduce using separate update_min/update_max
template <typename ViewType> struct FindMinMaxSeparateReduce {
  ViewType data;

  FindMinMaxSeparateReduce(ViewType data_) : data(data_) {}

  KOKKOS_INLINE_FUNCTION
  void operator()(const int i, type_real &lmin, type_real &lmax) const {
    LocalMinMax<type_real> local;
    local.update_min(data(i));
    local.update_max(data(i));
    lmin = Kokkos::fmin(lmin, local.min_val);
    lmax = Kokkos::fmax(lmax, local.max_val);
  }
};

// Update ScatterMinMax with data values
template <typename ViewType> struct ScatterUpdate {
  ViewType data;
  ScatterMinMax<type_real> scatter;

  ScatterUpdate(ViewType data_, ScatterMinMax<type_real> scatter_)
      : data(data_), scatter(scatter_) {}

  KOKKOS_INLINE_FUNCTION
  void operator()(const int i) const {
    auto accessor = scatter.access();
    accessor.update(data(i));
  }
};

// Update ScatterMinMax with separate min/max from two views
template <typename ViewType> struct ScatterUpdateSeparate {
  ViewType min_data;
  ViewType max_data;
  ScatterMinMax<type_real> scatter;

  ScatterUpdateSeparate(ViewType min_data_, ViewType max_data_,
                        ScatterMinMax<type_real> scatter_)
      : min_data(min_data_), max_data(max_data_), scatter(scatter_) {}

  KOKKOS_INLINE_FUNCTION
  void operator()(const int i) const {
    auto accessor = scatter.access();
    accessor.update_min(min_data(i));
    accessor.update_max(max_data(i));
  }
};

// Update ScatterMinMax with 2D data (bucket, item)
// Iterates over buckets in parallel; accumulates items into a local min/max
// first so that the scatter write (to a shared address) happens only once per
// bucket, avoiding the SIMD read-modify-write issue where Intel's aggressive
// vectorizer stores the last SIMD lane instead of the true minimum.
template <typename ViewType> struct ScatterUpdateMulti {
  ViewType data;
  ScatterMinMax<type_real> scatter;
  int items_per_bucket;

  ScatterUpdateMulti(ViewType data_, ScatterMinMax<type_real> scatter_,
                     int items_per_bucket_)
      : data(data_), scatter(scatter_), items_per_bucket(items_per_bucket_) {}

  KOKKOS_INLINE_FUNCTION
  void operator()(const int bucket) const {
    LocalMinMax<type_real> local;
    for (size_t item = 0; item < items_per_bucket; ++item) {
      local.update(data(bucket, item));
    }
    auto accessor = scatter.access();
    accessor.update_min(bucket, local.min_val);
    accessor.update_max(bucket, local.max_val);
  }
};

// Update ScatterMinMax with bucket index (single value per bucket)
struct ScatterUpdateBucket {
  ScatterMinMax<type_real> scatter;

  ScatterUpdateBucket(ScatterMinMax<type_real> scatter_) : scatter(scatter_) {}

  KOKKOS_INLINE_FUNCTION
  void operator()(const int bucket) const {
    auto accessor = scatter.access();
    accessor.update(bucket, static_cast<type_real>(bucket * 10));
  }
};

// Update ScatterMinMax with separate min/max per bucket
struct ScatterUpdateBucketSeparate {
  ScatterMinMax<type_real> scatter;

  ScatterUpdateBucketSeparate(ScatterMinMax<type_real> scatter_)
      : scatter(scatter_) {}

  KOKKOS_INLINE_FUNCTION
  void operator()(const int bucket) const {
    auto accessor = scatter.access();
    accessor.update_min(bucket, static_cast<type_real>(-bucket));
    accessor.update_max(bucket, static_cast<type_real>(bucket * 100));
  }
};

// ============================================================================
// Test fixture and tests
// ============================================================================

class ScatterMinMaxTests : public ::testing::Test {
protected:
  static constexpr int N = 1000;
};

// Test LocalMinMax basic functionality in a parallel_for
TEST_F(ScatterMinMaxTests, LocalMinMaxBasic) {
  // Create a view with known values: 0, 1, 2, ..., N-1
  Kokkos::View<type_real *, Kokkos::DefaultExecutionSpace> data("data", N);

  Kokkos::parallel_for("initialize_data",
                       Kokkos::RangePolicy<Kokkos::DefaultExecutionSpace>(0, N),
                       InitializeSequential<decltype(data)>(data, 0));
  Kokkos::fence();

  // Use LocalMinMax in a parallel_for with a parallel_reduce pattern
  type_real global_min, global_max;

  Kokkos::parallel_reduce(
      "find_minmax", Kokkos::RangePolicy<Kokkos::DefaultExecutionSpace>(0, N),
      FindMinMaxReduce<decltype(data)>(data),
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
  Kokkos::parallel_for("initialize_data",
                       Kokkos::RangePolicy<Kokkos::DefaultExecutionSpace>(0, N),
                       InitializeSequential<decltype(data)>(data, 100));
  Kokkos::fence();

  type_real global_min, global_max;

  Kokkos::parallel_reduce(
      "find_minmax_separate",
      Kokkos::RangePolicy<Kokkos::DefaultExecutionSpace>(0, N),
      FindMinMaxSeparateReduce<decltype(data)>(data),
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

  Kokkos::parallel_for("initialize_data",
                       Kokkos::RangePolicy<Kokkos::DefaultExecutionSpace>(0, N),
                       InitializeSequential<decltype(data)>(data, -50));
  Kokkos::fence();

  // Use ScatterMinMax
  ScatterMinMax<type_real> scatter("test");

  Kokkos::parallel_for("scatter_minmax",
                       Kokkos::RangePolicy<Kokkos::DefaultExecutionSpace>(0, N),
                       ScatterUpdate<decltype(data)>(data, scatter));
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
      "initialize_data",
      Kokkos::RangePolicy<Kokkos::DefaultExecutionSpace>(0, N),
      InitializeTwoViews<decltype(min_data)>(min_data, max_data));
  Kokkos::fence();

  ScatterMinMax<type_real> scatter("test_separate");

  Kokkos::parallel_for(
      "scatter_separate",
      Kokkos::RangePolicy<Kokkos::DefaultExecutionSpace>(0, N),
      ScatterUpdateSeparate<decltype(min_data)>(min_data, max_data, scatter));
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
  Kokkos::parallel_for("initialize_data",
                       Kokkos::RangePolicy<Kokkos::DefaultExecutionSpace>(0, N),
                       InitializeSequential<decltype(data)>(data, -N / 2));
  Kokkos::fence();

  ScatterMinMax<type_real> scatter("test_negative");

  Kokkos::parallel_for("scatter_negative",
                       Kokkos::RangePolicy<Kokkos::DefaultExecutionSpace>(0, N),
                       ScatterUpdate<decltype(data)>(data, scatter));
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
  Kokkos::parallel_for("initialize_data",
                       Kokkos::RangePolicy<Kokkos::DefaultExecutionSpace>(0, N),
                       InitializeSequential<decltype(data)>(data, 10));
  Kokkos::fence();

  ScatterMinMax<type_real> scatter("test_bounds");

  Kokkos::parallel_for("scatter_bounds",
                       Kokkos::RangePolicy<Kokkos::DefaultExecutionSpace>(0, N),
                       ScatterUpdate<decltype(data)>(data, scatter));
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

  Kokkos::View<type_real *, Kokkos::DefaultExecutionSpace> data("data",
                                                                LARGE_N);

  // Use a pattern where min and max are at specific known locations
  // All values are 500.0 except: data[12345] = -999.0, data[67890] = 9999.0
  Kokkos::parallel_for(
      "initialize_large",
      Kokkos::RangePolicy<Kokkos::DefaultExecutionSpace>(0, LARGE_N),
      InitializeLargePattern<decltype(data)>(data));
  Kokkos::fence();

  ScatterMinMax<type_real> scatter("test_large");

  Kokkos::parallel_for(
      "scatter_large",
      Kokkos::RangePolicy<Kokkos::DefaultExecutionSpace>(0, LARGE_N),
      ScatterUpdate<decltype(data)>(data, scatter));
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

  // Create data where each bucket has values: bucket_id*100 to bucket_id*100 +
  // ITEMS_PER_BUCKET - 1
  Kokkos::View<type_real **, Kokkos::DefaultExecutionSpace> data(
      "data", NUM_BUCKETS, ITEMS_PER_BUCKET);

  Kokkos::parallel_for(
      "initialize_multi",
      Kokkos::MDRangePolicy<Kokkos::Rank<2>, Kokkos::DefaultExecutionSpace>(
          { 0, 0 }, { NUM_BUCKETS, ITEMS_PER_BUCKET }),
      InitializeMulti<decltype(data)>(data));
  Kokkos::fence();

  // Use ScatterMinMax with N buckets
  ScatterMinMax<type_real> scatter("test_multi", NUM_BUCKETS);

  EXPECT_EQ(scatter.size(), NUM_BUCKETS);

  Kokkos::parallel_for(
      "scatter_multi",
      Kokkos::RangePolicy<Kokkos::DefaultExecutionSpace>(0, NUM_BUCKETS),
      ScatterUpdateMulti<decltype(data)>(data, scatter, ITEMS_PER_BUCKET));
  Kokkos::fence();

  scatter.contribute();

  // Check each bucket's bounds
  for (int bucket = 0; bucket < NUM_BUCKETS; ++bucket) {
    Bounds bounds = scatter.get_bounds(bucket);
    type_real expected_min = static_cast<type_real>(bucket * 100);
    type_real expected_max =
        static_cast<type_real>(bucket * 100 + ITEMS_PER_BUCKET - 1);

    EXPECT_TRUE(specfem::utilities::is_close(bounds.min, expected_min))
        << "Bucket " << bucket << ": "
        << expected_got(expected_min, bounds.min);
    EXPECT_TRUE(specfem::utilities::is_close(bounds.max, expected_max))
        << "Bucket " << bucket << ": "
        << expected_got(expected_max, bounds.max);
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
      ScatterUpdateBucket(scatter));
  Kokkos::fence();

  scatter.contribute();

  std::vector<Bounds> all_bounds = scatter.get_all_bounds();

  EXPECT_EQ(all_bounds.size(), NUM_BUCKETS);

  for (int bucket = 0; bucket < NUM_BUCKETS; ++bucket) {
    type_real expected_val = static_cast<type_real>(bucket * 10);
    EXPECT_TRUE(
        specfem::utilities::is_close(all_bounds[bucket].min, expected_val))
        << "Bucket " << bucket
        << " min: " << expected_got(expected_val, all_bounds[bucket].min);
    EXPECT_TRUE(
        specfem::utilities::is_close(all_bounds[bucket].max, expected_val))
        << "Bucket " << bucket
        << " max: " << expected_got(expected_val, all_bounds[bucket].max);
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
      ScatterUpdateBucketSeparate(scatter));
  Kokkos::fence();

  scatter.contribute();

  for (int bucket = 0; bucket < NUM_BUCKETS; ++bucket) {
    Bounds bounds = scatter.get_bounds(bucket);
    type_real expected_min = static_cast<type_real>(-bucket);
    type_real expected_max = static_cast<type_real>(bucket * 100);

    EXPECT_TRUE(specfem::utilities::is_close(bounds.min, expected_min))
        << "Bucket " << bucket
        << " min: " << expected_got(expected_min, bounds.min);
    EXPECT_TRUE(specfem::utilities::is_close(bounds.max, expected_max))
        << "Bucket " << bucket
        << " max: " << expected_got(expected_max, bounds.max);
  }
}
