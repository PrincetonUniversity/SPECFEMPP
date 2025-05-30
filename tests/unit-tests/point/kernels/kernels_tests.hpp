#include <Kokkos_Core.hpp>
#include <gtest/gtest.h>

// Test fixture for point kernels tests
class PointKernelsTest : public ::testing::Test {
protected:
  void SetUp() override {
    // Initialize Kokkos if needed for tests
    if (!Kokkos::is_initialized())
      Kokkos::initialize();
  }

  void TearDown() override {
    // Finalize Kokkos if needed
    if (Kokkos::is_initialized())
      Kokkos::finalize();
  }
};
