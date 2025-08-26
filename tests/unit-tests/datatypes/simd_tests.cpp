#include "datatypes/simd.hpp"
#include "specfem_setup.hpp"
#include "test_macros.hpp"
#include <gtest/gtest.h>

// Test fixture for basic SIMD functionality
template <bool UseSIMD> class Datatype_SIMD_Test : public ::testing::Test {
protected:
  using simd_type = specfem::datatype::simd<type_real, UseSIMD>;
  using value_type = typename simd_type::datatype;
  using mask_type = typename simd_type::mask_type;

  static constexpr int expected_size =
      UseSIMD ?
#ifdef SPECFEM_ENABLE_SIMD
              Kokkos::Experimental::simd<type_real>::size()
#else
              1
#endif
              : 1;

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
class Datatype_SIMD_Test_Typed : public Datatype_SIMD_Test<T::value> {};

TYPED_TEST_SUITE(Datatype_SIMD_Test_Typed, TestTypes);

// Test SIMD type traits
TYPED_TEST(Datatype_SIMD_Test_Typed, TypeTraits) {
  using simd_type = typename Datatype_SIMD_Test_Typed<TypeParam>::simd_type;
  constexpr bool using_simd = TypeParam::value;

  // Check base type
  EXPECT_TRUE((std::is_same_v<typename simd_type::base_type, type_real>));

  // Check using_simd flag
  EXPECT_EQ(simd_type::using_simd, using_simd);

  // Check size
  EXPECT_EQ(simd_type::size(), this->expected_size);
}

// Test construction and basic operations
TYPED_TEST(Datatype_SIMD_Test_Typed, ConstructionAndAccess) {
  using value_type = typename Datatype_SIMD_Test_Typed<TypeParam>::value_type;
  constexpr bool using_simd = TypeParam::value;

  // Default construction
  value_type val1;

  // Broadcast construction
  value_type val2(static_cast<type_real>(5.0));

  if constexpr (using_simd) {
    // SIMD case - check all lanes
    for (int i = 0; i < this->expected_size; ++i) {
      EXPECT_REAL_EQ(val2[i], static_cast<type_real>(5.0));
    }
  } else {
    // Scalar case
    EXPECT_REAL_EQ(val2, static_cast<type_real>(5.0));
  }
}

// Test arithmetic operations
TYPED_TEST(Datatype_SIMD_Test_Typed, ArithmeticOperations) {
  using value_type = typename Datatype_SIMD_Test_Typed<TypeParam>::value_type;
  constexpr bool using_simd = TypeParam::value;

  value_type a(static_cast<type_real>(10.0));
  value_type b(static_cast<type_real>(3.0));

  // Addition
  auto sum = a + b;
  if constexpr (using_simd) {
    for (int i = 0; i < this->expected_size; ++i) {
      EXPECT_REAL_EQ(sum[i], static_cast<type_real>(13.0));
    }
  } else {
    EXPECT_REAL_EQ(sum, static_cast<type_real>(13.0));
  }

  // Subtraction
  auto diff = a - b;
  if constexpr (using_simd) {
    for (int i = 0; i < this->expected_size; ++i) {
      EXPECT_REAL_EQ(diff[i], static_cast<type_real>(7.0));
    }
  } else {
    EXPECT_REAL_EQ(diff, static_cast<type_real>(7.0));
  }

  // Multiplication
  auto prod = a * b;
  if constexpr (using_simd) {
    for (int i = 0; i < this->expected_size; ++i) {
      EXPECT_REAL_EQ(prod[i], static_cast<type_real>(30.0));
    }
  } else {
    EXPECT_REAL_EQ(prod, static_cast<type_real>(30.0));
  }

  // Division
  auto quot = a / b;
  if constexpr (using_simd) {
    for (int i = 0; i < this->expected_size; ++i) {
      EXPECT_REAL_EQ(quot[i], static_cast<type_real>(10.0 / 3.0));
    }
  } else {
    EXPECT_REAL_EQ(quot, static_cast<type_real>(10.0 / 3.0));
  }
}

// Test comparison operations
TYPED_TEST(Datatype_SIMD_Test_Typed, ComparisonOperations) {
  using value_type = typename Datatype_SIMD_Test_Typed<TypeParam>::value_type;
  using mask_type = typename Datatype_SIMD_Test_Typed<TypeParam>::mask_type;

  value_type a(static_cast<type_real>(5.0));
  value_type b(static_cast<type_real>(5.0));
  value_type c(static_cast<type_real>(3.0));

  // Equal
  auto mask_eq = (a == b);
  EXPECT_TRUE(specfem::datatype::all_of(mask_eq));

  // Not equal
  auto mask_neq = (a != c);
  EXPECT_TRUE(specfem::datatype::all_of(mask_neq));

  // Greater than
  auto mask_gt = (a > c);
  EXPECT_TRUE(specfem::datatype::all_of(mask_gt));

  // Less than
  auto mask_lt = (c < a);
  EXPECT_TRUE(specfem::datatype::all_of(mask_lt));
}

// Test with integer types
TEST(Datatype_SIMD_Test, IntegerTypes) {
  // Test scalar integer
  using scalar_int = specfem::datatype::simd<int, false>;
  EXPECT_EQ(scalar_int::size(), 1);
  EXPECT_FALSE(scalar_int::using_simd);

  scalar_int::datatype val1 = 42;
  scalar_int::datatype val2 = 10;
  auto sum = val1 + val2;
  EXPECT_EQ(sum, 52);

  // Test SIMD integer
  using simd_int = specfem::datatype::simd<int, true>;
  EXPECT_TRUE(simd_int::using_simd);
  EXPECT_GE(simd_int::size(), 1);

  simd_int::datatype vec1(42);
  simd_int::datatype vec2(10);
  auto vec_sum = vec1 + vec2;

  for (int i = 0; i < simd_int::size(); ++i) {
    EXPECT_EQ(vec_sum[i], 52);
  }
}

// SIMD-specific test: lane manipulation
TEST(Datatype_SIMD_Test, LaneManipulation) {
  constexpr bool use_simd = true;
  using simd_type = specfem::datatype::simd<type_real, use_simd>;
  using value_type = typename simd_type::datatype;

  if (simd_type::size() > 1) {
    value_type vec([](std::size_t lane) -> type_real {
      return static_cast<type_real>(lane);
    });

    // Verify each lane
    for (int i = 0; i < simd_type::size(); ++i) {
      EXPECT_REAL_EQ(vec[i], static_cast<type_real>(i));
    }

    // Test operations preserve lane values
    value_type vec2(1.0);
    auto result = vec + vec2;

    for (int i = 0; i < simd_type::size(); ++i) {
      EXPECT_REAL_EQ(result[i], static_cast<type_real>(i) + 1.0);
    }
  }
}

// Test SIMD and non-SIMD comparison
TEST(Datatype_SIMD_Test, CrossTypeComparison) {
  // Test scalar equality
  {
    using scalar_type = specfem::datatype::simd<type_real, false>::datatype;
    scalar_type a = 1.0;
    scalar_type b = 1.0;

    bool result = (a == b);
    EXPECT_TRUE(result);
  }

  // Test SIMD equality
  {
    using simd_type = specfem::datatype::simd<type_real, true>;
    simd_type::datatype a(1.0);
    simd_type::datatype b(1.0);

    auto result = (a == b);
    EXPECT_TRUE((specfem::datatype::all_of(result)));
  }

  // We don't test cross-type comparison because it's not supported
  // If we tried: scalar_val == simd_val, it would be a compilation error
  // which is the behavior we want
}

// Test basic functionality with comparison masks
TYPED_TEST(Datatype_SIMD_Test_Typed, AllOfBasicComparison) {
  auto val1 = this->create_value(5.0);
  auto val2 = this->create_value(5.0);
  auto val3 = this->create_value(3.0);

  // Test all true
  auto mask_true = (val1 == val2);
  EXPECT_TRUE((specfem::datatype::all_of(mask_true)));

  // Test all false
  auto mask_false = (val1 == val3);
  EXPECT_FALSE((specfem::datatype::all_of(mask_false)));
}

// SIMD-specific test: mixed values in different lanes
TEST(Datatype_SIMD_Test, AllOfSIMDMixedLanes) {
  constexpr bool use_simd = true;
  using simd_type = specfem::datatype::simd<type_real, use_simd>;
  using value_type = typename simd_type::datatype;
  using mask_type = typename simd_type::mask_type;

  if (simd_type::size() > 1) {
    value_type val1(5.0);
    value_type val2([](std::size_t lane) -> type_real {
      return (lane % 2 == 0) ? 5.0 : 3.0; // Mixed values
    });
    auto mask_mixed = (val1 == val2);

    // Should be false because not all lanes are equal
    EXPECT_FALSE((specfem::datatype::all_of(mask_mixed)));
  } else {
    GTEST_SKIP() << "SIMD tests skipped because simd_size <= 1";
    return;
  }
}

// Test with floating-point tolerance comparison
TYPED_TEST(Datatype_SIMD_Test_Typed, AllOfToleranceComparison) {
  auto val1 = this->create_value(1.0);
  auto val2 = this->create_value(1.0 + 1e-7);

  // Exact comparison might pass or fail depending on compiler optimization and
  // precision So we won't assert anything about it directly

  // What we really want to test is that we can detect small differences
  // correctly
  auto diff_small = this->create_value(1e-8); // Smaller than our difference
  auto diff_large = this->create_value(1e-6); // Larger than our difference

  // Check that the difference is larger than the small tolerance
  auto mask_greater_than_small = (val2 - val1) > diff_small;
  EXPECT_TRUE((specfem::datatype::all_of(mask_greater_than_small)))
      << "Difference should be detected as larger than small tolerance";

  // Check that the difference is smaller than the large tolerance
  auto mask_less_than_large = (val2 - val1) < diff_large;
  EXPECT_TRUE((specfem::datatype::all_of(mask_less_than_large)))
      << "Difference should be detected as smaller than large tolerance";
}

// Test direct floating-point comparison behavior
TYPED_TEST(Datatype_SIMD_Test_Typed, AllOfDirectFloatingPointComparison) {
  // Use values that are guaranteed to be representable exactly in floating
  // point
  auto val1 = this->create_value(1.0);

  // Add a value that's guaranteed to change the bit pattern
  // 2^-23 is the machine epsilon for float, so 2^-22 is definitely
  // distinguishable
  auto val2 = this->create_value(1.0 + std::pow(2.0, -22));

  // This direct comparison should definitely fail
  auto mask_exact = (val1 == val2);
  EXPECT_FALSE((specfem::datatype::all_of(mask_exact)))
      << "Direct comparison of distinguishable values should fail";
}

int main(int argc, char **argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
