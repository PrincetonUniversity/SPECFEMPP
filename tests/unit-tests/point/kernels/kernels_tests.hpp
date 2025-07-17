#include "datatypes/simd.hpp"
#include "specfem_setup.hpp"
#include <Kokkos_Core.hpp>
#include <gtest/gtest.h>

template <bool UseSIMD> class PointKernelsTestUntyped : public ::testing::Test {
protected:
  using simd_type = specfem::datatype::simd<type_real, UseSIMD>;
  using value_type = typename simd_type::datatype;
  using mask_type = typename simd_type::mask_type;

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

// For better naming
struct Serial : std::integral_constant<bool, false> {};
struct SIMD : std::integral_constant<bool, true> {};

using TestTypes = ::testing::Types<Serial, SIMD>;

template <typename T>
class PointKernelsTest : public PointKernelsTestUntyped<T::value> {};

TYPED_TEST_SUITE(PointKernelsTest, TestTypes);
