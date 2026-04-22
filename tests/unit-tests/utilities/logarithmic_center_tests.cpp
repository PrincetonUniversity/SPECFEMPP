#include "specfem/utilities.hpp"
#include "test_macros.hpp"
#include <cmath>
#include <gtest/gtest.h>

using specfem::utilities::is_close;
using specfem::utilities::logarithmic_center;

TEST(UtilitiesLogarithmicCenter, SymmetricInputs) {
  // The logarithmic center of f and f is f itself
  EXPECT_TRUE(is_close(logarithmic_center(100.0, 100.0), type_real(100.0)))
      << expected_got(type_real(100.0), logarithmic_center(100.0, 100.0));

  EXPECT_TRUE(is_close(logarithmic_center(1.0, 1.0), type_real(1.0)))
      << expected_got(type_real(1.0), logarithmic_center(1.0, 1.0));

  EXPECT_TRUE(is_close(logarithmic_center(0.01, 0.01), type_real(0.01)))
      << expected_got(type_real(0.01), logarithmic_center(0.01, 0.01));
}

TEST(UtilitiesLogarithmicCenter, KnownGeometricMean) {
  // logarithmic_center(f1, f2) == sqrt(f1 * f2)
  // 10^(0.5*(log10(1) + log10(100))) = 10^(0.5*2) = 10
  EXPECT_TRUE(is_close(logarithmic_center(1.0, 100.0), type_real(10.0)))
      << expected_got(type_real(10.0), logarithmic_center(1.0, 100.0));

  // 10^(0.5*(log10(10) + log10(1000))) = 10^(0.5*4) = 100
  EXPECT_TRUE(is_close(logarithmic_center(10.0, 1000.0), type_real(100.0)))
      << expected_got(type_real(100.0), logarithmic_center(10.0, 1000.0));

  // 10^(0.5*(log10(0.1) + log10(10))) = 10^(0.5*0) = 1
  EXPECT_TRUE(is_close(logarithmic_center(0.1, 10.0), type_real(1.0)))
      << expected_got(type_real(1.0), logarithmic_center(0.1, 10.0));
}

TEST(UtilitiesLogarithmicCenter, Commutativity) {
  // The result must be the same regardless of argument order
  const type_real f1 = 5.0;
  const type_real f2 = 500.0;

  EXPECT_TRUE(is_close(logarithmic_center(f1, f2), logarithmic_center(f2, f1)))
      << expected_got(logarithmic_center(f2, f1), logarithmic_center(f1, f2));

  const type_real f3 = 0.02;
  const type_real f4 = 200.0;

  EXPECT_TRUE(is_close(logarithmic_center(f3, f4), logarithmic_center(f4, f3)))
      << expected_got(logarithmic_center(f4, f3), logarithmic_center(f3, f4));
}
