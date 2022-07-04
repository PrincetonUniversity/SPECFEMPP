#include "../../include/gll_library.h"
#include "../../include/gll_utils.h"
#include <gtest/gtest.h>
#include <iostream>
#include <stdexcept>

TEST(GLL_tests, PNLEG) {
  const auto &pnleg = gll_library::pnleg;
  double tol = 1e-6;

  try {
    gll_library::pnleg(-1.0, 0);
    FAIL();
  } catch (const std::invalid_argument &err) {
    EXPECT_STREQ(err.what(), "value of n > 0");
  }

  EXPECT_NEAR(pnleg(-1.0, 1), -1.0, tol);
  EXPECT_NEAR(pnleg(0.0, 1), 0.0, tol);
  EXPECT_NEAR(pnleg(1.0, 1), 1.0, tol);
  EXPECT_NEAR(pnleg(-1.0, 2), 1.0, tol);
  EXPECT_NEAR(pnleg(0.0, 2), -0.5, tol);
  EXPECT_NEAR(pnleg(1.0, 2), 1.0, tol);
  EXPECT_NEAR(pnleg(-1.0, 5), -1.0, tol);
  EXPECT_NEAR(pnleg(0.0, 5), 0.0, tol);
  EXPECT_NEAR(pnleg(1.0, 5), 1.0, tol);
}

TEST(GLL_tests, PNDLEG) {
  const auto &pndleg = gll_library::pndleg;

  double tol = 1e-6;
  try {
    gll_library::pndleg(-1.0, 0);
    FAIL();
  } catch (const std::invalid_argument &err) {
    EXPECT_STREQ(err.what(), "value of n > 0");
  }

  EXPECT_NEAR(pndleg(-1.0, 1), 1.0, tol);
  EXPECT_NEAR(pndleg(0.0, 1), 1.0, tol);
  EXPECT_NEAR(pndleg(1.0, 1), 1.0, tol);
  EXPECT_NEAR(pndleg(0.0, 2), 0.0, tol);
  EXPECT_NEAR(pndleg(-0.2852315, 5), 0.0, tol);
  EXPECT_NEAR(pndleg(0.2852315, 5), 0.0, tol);
  EXPECT_NEAR(pndleg(-0.7650553, 5), 0.0, tol);
  EXPECT_NEAR(pndleg(0.7650553, 5), 0.0, tol);
}

TEST(GLL_tests, JACOBF) {

  double tol = 1e-6, p, pd;
  const auto &jacobf = gll_utils::jacobf;

  // Tests for Legendre polynnomials
  std::tie(p, pd) = jacobf(1, 0.0, 0.0, 0.0);
  EXPECT_NEAR(p, 0.0, tol);

  std::tie(p, pd) = jacobf(1, 0.0, 0.0, 1.0);
  EXPECT_NEAR(p, 1.0, tol);

  std::tie(p, pd) = jacobf(2, 0.0, 0.0, 0.0);
  EXPECT_NEAR(p, -0.5, tol);

  std::tie(p, pd) = jacobf(5, 0.0, 0.0, -0.2852315);
  EXPECT_NEAR(pd, 0.0, tol);

  std::tie(p, pd) = jacobf(5, 0.0, 0.0, 0.7650553);
  EXPECT_NEAR(pd, 0.0, tol);
}

TEST(GLL_tests, JACG) {

  double tol = 1e-6;
  std::vector<double> r1;
  const auto &jacg = gll_utils::jacg;

  r1 = jacg(5, 0.0, 0.0);

  EXPECT_NEAR(r1[0], -0.538469310, tol);
  EXPECT_NEAR(r1[1], -0.9061798459, tol);
  EXPECT_NEAR(r1[2], 0, tol);
  EXPECT_NEAR(r1[3], 0.538469310, tol);
  EXPECT_NEAR(r1[4], 0.9061798459, tol);

  std::vector<double> r2;
  r2 = jacg(3, 0.0, 0.0);
  EXPECT_NEAR(r2[0], -0.77459666924, tol);
  EXPECT_NEAR(r2[1], 0, tol);
  EXPECT_NEAR(r2[2], 0.77459666924, tol);
}
