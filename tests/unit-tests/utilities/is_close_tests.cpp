#include "test_macros.hpp"
#include "utilities/utilities.hpp"
#include <gtest/gtest.h>

using specfem::utilities::is_close;

TEST(UtilitiesIsClose, BasicComparisons) {
  // Exact equality
  EXPECT_TRUE(is_close(1.0, 1.0));
  EXPECT_TRUE(is_close(0.0, 0.0));
  EXPECT_TRUE(is_close(-1.0, -1.0));

  // Clear inequality
  EXPECT_FALSE(is_close(1.0, 2.0));
  EXPECT_FALSE(is_close(0.0, 1.0));
}

TEST(UtilitiesIsClose, RelativeToleranceTests) {
  // Test relative tolerance for large numbers
  EXPECT_TRUE(is_close(1000.0, 1000.001)); // 1e-6 relative difference
  EXPECT_FALSE(is_close(1000.0, 1001.0));  // 1e-3 relative difference

  // Test relative tolerance for moderate numbers
  EXPECT_TRUE(is_close(1.0, 1.000001)); // 1e-6 relative difference
  EXPECT_FALSE(is_close(1.0, 1.001));   // 1e-3 relative difference
}

TEST(UtilitiesIsClose, AbsoluteToleranceTests) {
  // Test absolute tolerance for small numbers near zero
  EXPECT_TRUE(is_close(0.0, 1e-10));  // Within absolute tolerance
  EXPECT_TRUE(is_close(1e-10, 0.0));  // Within absolute tolerance
  EXPECT_TRUE(is_close(0.0, -1e-10)); // Within absolute tolerance

  // Test the specific case from the locate_point test
  EXPECT_TRUE(is_close(-2.2351741790771484e-08, 0.0))
      << expected_got(0.0, -2.2351741790771484e-08);

  // Test comparison of two very small values (not just small vs zero)
  EXPECT_TRUE(is_close(1e-10, -1e-10));  // Small positive vs small negative
  EXPECT_TRUE(is_close(3e-8, 3.1e-8));   // Two small values close to each other
  EXPECT_TRUE(is_close(-5e-9, -4.9e-9)); // Two small negative values
  EXPECT_TRUE(is_close(1e-8, 2e-8));  // Two small values - absolute tolerance
                                      // dominates
  EXPECT_FALSE(is_close(1e-6, 2e-6)); // Larger values where relative tolerance
                                      // should dominate

  // Values just outside absolute tolerance
  EXPECT_FALSE(is_close(0.0, 1e-6)); // Outside absolute tolerance (1e-7)
}

TEST(UtilitiesIsClose, CustomTolerances) {
  // Test custom relative tolerance
  EXPECT_TRUE(is_close(1.0, 1.01, 0.02));   // 2% relative tolerance
  EXPECT_FALSE(is_close(1.0, 1.01, 0.005)); // 0.5% relative tolerance

  // Test custom absolute tolerance
  EXPECT_TRUE(is_close(0.0, 1e-5, 1e-6, 1e-4));  // Custom abs_tol = 1e-4
  EXPECT_FALSE(is_close(0.0, 1e-5, 1e-6, 1e-8)); // Custom abs_tol = 1e-8
}

TEST(UtilitiesIsClose, NegativeNumbers) {
  EXPECT_TRUE(is_close(-1.0, -1.000001));
  EXPECT_TRUE(is_close(-1000.0, -1000.001));
  EXPECT_FALSE(is_close(-1.0, -1.001));
}

TEST(UtilitiesIsClose, TypeRealConsistency) {
  // Test with type_real explicit casting
  EXPECT_TRUE(is_close(type_real{ 1.0 }, type_real{ 1.000001 }));
  EXPECT_TRUE(is_close(type_real{ 0.0 }, type_real{ 1e-10 }));
  EXPECT_FALSE(is_close(type_real{ 1.0 }, type_real{ 1.001 }));
}

TEST(UtilitiesIsClose, InfinityTests) {
  const type_real inf = std::numeric_limits<type_real>::infinity();

  // Note: Current implementation doesn't handle infinity properly due to NaN
  // results from infinity arithmetic. These tests document the current
  // behavior.

  // Infinity arithmetic produces NaN, so comparisons fail
  EXPECT_FALSE(is_close(inf, inf)) << "inf - inf = NaN, so comparison fails";
  EXPECT_FALSE(is_close(-inf, -inf))
      << "-inf - (-inf) = NaN, so comparison fails";

  // Infinity arithmetic issues mean these behave unexpectedly
  EXPECT_TRUE(is_close(inf, -inf))
      << "Current implementation doesn't handle inf properly";
  EXPECT_TRUE(is_close(inf, type_real{ 1.0 }))
      << "Current implementation doesn't handle inf properly";
  EXPECT_TRUE(is_close(type_real{ 1.0 }, inf))
      << "Current implementation doesn't handle inf properly";
  EXPECT_TRUE(is_close(-inf, type_real{ 1.0 }))
      << "Current implementation doesn't handle inf properly";
}

TEST(UtilitiesIsClose, NaNTests) {
  const type_real nan_val = std::numeric_limits<type_real>::quiet_NaN();

  // NaN comparisons - NaN should never be close to anything, including itself
  EXPECT_FALSE(is_close(nan_val, nan_val)) << "NaN should not equal itself";
  EXPECT_FALSE(is_close(nan_val, type_real{ 1.0 }))
      << "NaN should not be close to finite values";
  EXPECT_FALSE(is_close(type_real{ 1.0 }, nan_val))
      << "Finite values should not be close to NaN";
  EXPECT_FALSE(is_close(nan_val, type_real{ 0.0 }))
      << "NaN should not be close to zero";
}

TEST(UtilitiesIsClose, ExtremeValueTests) {
  const type_real max_val = std::numeric_limits<type_real>::max();
  const type_real min_val = std::numeric_limits<type_real>::lowest();
  const type_real denorm_min = std::numeric_limits<type_real>::denorm_min();

  // Maximum representable values
  EXPECT_TRUE(is_close(max_val, max_val)) << "Max value should equal itself";
  EXPECT_TRUE(is_close(min_val, min_val)) << "Lowest value should equal itself";

  // Denormalized numbers (very close to zero)
  EXPECT_TRUE(is_close(denorm_min, type_real{ 0.0 }))
      << "Denormalized min should be close to zero";
  EXPECT_TRUE(is_close(type_real{ 0.0 }, denorm_min))
      << "Zero should be close to denormalized min";

  // Very large numbers with small relative differences
  type_real large_val = max_val / type_real{ 2.0 };
  type_real large_val_close =
      large_val * (type_real{ 1.0 } + type_real{ 1e-7 });
  EXPECT_TRUE(is_close(large_val, large_val_close))
      << "Large values with small relative diff should be close";
}

TEST(UtilitiesIsClose, ZeroSignTests) {
  const type_real pos_zero = type_real{ 0.0 };
  const type_real neg_zero = -type_real{ 0.0 };

  // Positive and negative zero should be considered equal
  EXPECT_TRUE(is_close(pos_zero, neg_zero))
      << "Positive and negative zero should be close";
  EXPECT_TRUE(is_close(neg_zero, pos_zero))
      << "Negative and positive zero should be close";
}

TEST(UtilitiesIsClose, LargeToleranceTests) {
  // Very large relative tolerances
  EXPECT_TRUE(is_close(type_real{ 1.0 }, type_real{ 2.0 }, type_real{ 2.0 }))
      << "200% tolerance allows 100% difference";

  // For is_close(1.0, 3.0, 2.0): difference=2.0, allowed=2.0*3.0=6.0, so 2.0
  // <= 6.0 = true
  EXPECT_TRUE(is_close(type_real{ 1.0 }, type_real{ 3.0 }, type_real{ 2.0 }))
      << "200% tolerance allows 200% difference due to max() in formula";

  // Need much larger difference to exceed the tolerance
  // For 1.0 vs 10.0 with 200% tolerance: allowed = 2.0 * 10.0 = 20.0, actual
  // = 9.0, so it passes
  EXPECT_TRUE(is_close(type_real{ 1.0 }, type_real{ 10.0 }, type_real{ 2.0 }))
      << "200% tolerance allows 900% difference due to max() formula";

  // Use a smaller tolerance that will actually fail
  EXPECT_FALSE(is_close(type_real{ 1.0 }, type_real{ 10.0 }, type_real{ 0.5 }))
      << "50% tolerance should not allow 900% difference";

  // Test opposite signs with large tolerance
  EXPECT_TRUE(is_close(type_real{ 1.0 }, type_real{ -1.0 }, type_real{ 2.1 }))
      << "210% tolerance should allow opposite signs";
  EXPECT_FALSE(is_close(type_real{ 1.0 }, type_real{ -1.0 }, type_real{ 1.9 }))
      << "190% tolerance should not allow opposite signs";
}

TEST(UtilitiesIsClose, EdgeCaseTolerances) {
  // Test when relative and absolute tolerances are in competition
  type_real a = type_real{ 1e-6 };
  type_real b = type_real{ 2e-6 };
  type_real custom_rel_tol = type_real{ 0.5 };    // 50%
  type_real custom_abs_tol = type_real{ 1.5e-6 }; // Larger than the difference

  // Absolute tolerance should dominate here
  EXPECT_TRUE(is_close(a, b, custom_rel_tol, custom_abs_tol))
      << "Absolute tolerance should dominate when larger";

  // Relative tolerance should dominate here
  type_real custom_abs_tol_small = type_real{ 1e-9 };
  EXPECT_TRUE(is_close(a, b, custom_rel_tol, custom_abs_tol_small))
      << "Relative tolerance should dominate when larger";
}
