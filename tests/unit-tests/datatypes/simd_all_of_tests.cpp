// test_all_of.cpp
#include "datatypes/simd.hpp"
#include "specfem_setup.hpp"
#include <gtest/gtest.h>

// Test fixture for all_of function
template <bool UseSIMD> class AllOfTest : public ::testing::Test {
protected:
  using simd_type = specfem::datatype::simd<type_real, UseSIMD>;
  using value_type = typename simd_type::datatype;
  using mask_type = typename simd_type::mask_type;

  value_type create_value(type_real val) {
    if constexpr (UseSIMD) {
      return value_type(val);
    } else {
      return val;
    }
  }
};

struct Serial : std::integral_constant<bool, false> {};
struct SIMD : std::integral_constant<bool, true> {};

// Test both SIMD and scalar cases
using TestTypes = ::testing::Types<Serial, SIMD>;
template <typename T>
class Datatype_SIMD_Test_Typed : public AllOfTest<T::value> {};
