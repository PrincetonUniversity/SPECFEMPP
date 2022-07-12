#include "../../include/gll_library.h"
#include "../../include/gll_utils.h"
#include "Kokkos_Environment.hpp"
#include <Kokkos_Core.hpp>
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
  std::tie(p, pd, std::ignore) = jacobf(1, 0.0, 0.0, 0.0);
  EXPECT_NEAR(p, 0.0, tol);

  std::tie(p, pd, std::ignore) = jacobf(1, 0.0, 0.0, 1.0);
  EXPECT_NEAR(p, 1.0, tol);

  std::tie(p, pd, std::ignore) = jacobf(2, 0.0, 0.0, 0.0);
  EXPECT_NEAR(p, -0.5, tol);

  std::tie(p, pd, std::ignore) = jacobf(5, 0.0, 0.0, -0.2852315);
  EXPECT_NEAR(pd, 0.0, tol);

  std::tie(p, pd, std::ignore) = jacobf(5, 0.0, 0.0, 0.7650553);
  EXPECT_NEAR(pd, 0.0, tol);
}

TEST(GLL_tests, JACG) {

  double tol = 1e-6;
  const auto &jacg = gll_utils::jacg;

  HostArray<double> r3("r3", 3);
  ASSERT_DEATH(jacg(r3, 2, 0.0, 0.0), "");

  HostArray<double> r1("r1", 5);
  jacg(r1, 5, 0.0, 0.0);
  EXPECT_NEAR(r1(0), -0.9061798459, tol);
  EXPECT_NEAR(r1(1), -0.538469310, tol);
  EXPECT_NEAR(r1(2), 0, tol);
  EXPECT_NEAR(r1(3), 0.538469310, tol);
  EXPECT_NEAR(r1(4), 0.9061798459, tol);

  HostArray<double> r2("r2", 3);
  jacg(r2, 3, 0.0, 0.0);
  EXPECT_NEAR(r2(0), -0.77459666924, tol);
  EXPECT_NEAR(r2(1), 0, tol);
  EXPECT_NEAR(r2(2), 0.77459666924, tol);
}

TEST(GLL_tests, JACW) {

  double tol = 1e-6;
  const auto &jacg = gll_utils::jacg;
  const auto &jacw = gll_utils::jacw;

  HostArray<double> r2("r1", 3);
  jacg(r2, 3, 1.0, 1.0);
  EXPECT_NEAR(r2(0), -0.6546536707, tol);
  EXPECT_NEAR(r2(1), 0, tol);
  EXPECT_NEAR(r2(2), 0.6546536707, tol);

  HostArray<double> w2("r2", 3);
  jacw(r2, w2, 3, 1.0, 1.0);
  std::array<double, 3> reference = { 0.5444444444, 0.7111111111,
                                      0.5444444444 };
  for (int i = 0; i < 3; i++) {
    double result = reference[i] * (1.0 - r2(i) * r2(i));
    EXPECT_NEAR(w2(i), result, tol);
  }
}

TEST(GLL_tests, ZWGJD) {
  // This test checks for the special case of np == 1
  const double tol = 1e-6;
  const auto &zwgjd = gll_utils::zwgjd;

  HostArray<double> r1("r1", 1);
  HostArray<double> w1("w1", 1);
  zwgjd(r1, w1, 1, 1.0, 1.0);

  EXPECT_NEAR(r1(0), 0.0, tol);
  EXPECT_NEAR(w1(0), 1.333333, tol);
}

TEST(GLL_tests, ZWGLJD) {

  double tol = 1e-6;
  const auto &zwgljd = gll_library::zwgljd;

  HostArray<double> z1("z1", 3);
  HostArray<double> w1("w1", 3);
  zwgljd(z1, w1, 3, 0.0, 0.0);
  EXPECT_NEAR(z1(0), -1.0, tol);
  EXPECT_NEAR(z1(1), 0.0, tol);
  EXPECT_NEAR(z1(2), 1.0, tol);
  EXPECT_NEAR(w1(0), 0.333333, tol);
  EXPECT_NEAR(w1(1), 1.333333, tol);
  EXPECT_NEAR(w1(2), 0.333333, tol);

  HostArray<double> z2("z1", 5);
  HostArray<double> w2("w1", 5);
  zwgljd(z2, w2, 5, 0.0, 0.0);
  EXPECT_NEAR(z2(0), -1.0, tol);
  EXPECT_NEAR(z2(1), -0.6546536707, tol);
  EXPECT_NEAR(z2(2), 0.0, tol);
  EXPECT_NEAR(z2(3), 0.6546536707, tol);
  EXPECT_NEAR(z2(4), 1.0, tol);
  EXPECT_NEAR(w2(0), 0.1, tol);
  EXPECT_NEAR(w2(1), 0.5444444444, tol);
  EXPECT_NEAR(w2(2), 0.7111111111, tol);
  EXPECT_NEAR(w2(3), 0.5444444444, tol);
  EXPECT_NEAR(w2(4), 0.1, tol);

  zwgljd(z2, w2, 5, 0.0, 1.0);
  EXPECT_NEAR(z2(0), -1.0, tol);
  EXPECT_NEAR(z2(1), -0.5077876295, tol);
  EXPECT_NEAR(z2(2), 0.1323008207, tol);
  EXPECT_NEAR(z2(3), 0.7088201421, tol);
  EXPECT_NEAR(z2(4), 1.0, tol);
  EXPECT_NEAR(w2(0), 0.01333333333, tol);
  EXPECT_NEAR(w2(1), 0.2896566946, tol);
  EXPECT_NEAR(w2(2), 0.7360043695, tol);
  EXPECT_NEAR(w2(3), 0.794338936, tol);
  EXPECT_NEAR(w2(4), 0.1666666667, tol);

  HostArray<double> z3("z1", 7);
  HostArray<double> w3("w1", 7);
  zwgljd(z3, w3, 7, 0.0, 0.0);
  EXPECT_NEAR(z3(0), -1.0, tol);
  EXPECT_NEAR(z3(1), -0.8302238962, tol);
  EXPECT_NEAR(z3(2), -0.4688487934, tol);
  EXPECT_NEAR(z3(3), 0.0, tol);
  EXPECT_NEAR(z3(4), 0.4688487934, tol);
  EXPECT_NEAR(z3(5), 0.8302238962, tol);
  EXPECT_NEAR(z3(6), 1.0, tol);
  EXPECT_NEAR(w3(0), 0.0476190476, tol);
  EXPECT_NEAR(w3(1), 0.2768260473, tol);
  EXPECT_NEAR(w3(2), 0.4317453812, tol);
  EXPECT_NEAR(w3(3), 0.4876190476, tol);
  EXPECT_NEAR(w3(4), 0.4317453812, tol);
  EXPECT_NEAR(w3(5), 0.2768260473, tol);
  EXPECT_NEAR(w3(6), 0.0476190476, tol);

  zwgljd(z3, w3, 7, 0.0, 1.0);
  EXPECT_NEAR(z3(0), -1.0, tol);
  EXPECT_NEAR(z3(1), -0.7401236486, tol);
  EXPECT_NEAR(z3(2), -0.3538526341, tol);
  EXPECT_NEAR(z3(3), 0.09890279315, tol);
  EXPECT_NEAR(z3(4), 0.5288423045, tol);
  EXPECT_NEAR(z3(5), 0.8508465697, tol);
  EXPECT_NEAR(z3(6), 1.0, tol);
  EXPECT_NEAR(w3(0), 0.003401360544, tol);
  EXPECT_NEAR(w3(1), 0.08473655296, tol);
  EXPECT_NEAR(w3(2), 0.2803032119, tol);
  EXPECT_NEAR(w3(3), 0.5016469619, tol);
  EXPECT_NEAR(w3(4), 0.5945754451, tol);
  EXPECT_NEAR(w3(5), 0.4520031342, tol);
  EXPECT_NEAR(w3(6), 0.08333333333, tol);
}

int main(int argc, char *argv[]) {
  ::testing::InitGoogleTest(&argc, argv);
  ::testing::AddGlobalTestEnvironment(new KokkosEnvironment);
  return RUN_ALL_TESTS();
}
