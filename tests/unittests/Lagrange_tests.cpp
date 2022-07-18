#include "../../include/gll_library.h"
#include "../../include/lagrange_poly.h"
#include "Kokkos_Environment.hpp"
#include <Kokkos_Core.hpp>
#include <gtest/gtest.h>
#include <iostream>
#include <stdexcept>

TEST(lagrange_tests, LAGRANGE_TESTS) {
  const auto &compute_lagrange_interpolants =
      Lagrange::compute_lagrange_interpolants;
  const auto &zwgljd = gll_library::zwgljd;
  const auto &compute_lagrange_derivatives_GLL =
      Lagrange::compute_lagrange_derivatives_GLL;
  int ngll = 5;
  double degpoly = ngll - 1;
  double tol = 1e-6;
  Kokkos::View<double *, Kokkos::LayoutRight, Kokkos::HostSpace> z1("r1", ngll);
  Kokkos::View<double *, Kokkos::LayoutRight, Kokkos::HostSpace> w1("w1", ngll);
  Kokkos::View<double *, Kokkos::LayoutRight, Kokkos::HostSpace> h1("h1", ngll);
  Kokkos::View<double *, Kokkos::LayoutRight, Kokkos::HostSpace> h1_prime(
      "h1_prime", ngll);
  Kokkos::View<double **, Kokkos::LayoutRight, Kokkos::HostSpace> h1_prime_xx(
      "h1_prime_xx", ngll, ngll);

  zwgljd(z1, w1, ngll, 0.0, 0.0);
  compute_lagrange_derivatives_GLL(h1_prime_xx, z1, ngll);

  for (int i = 0; i < ngll; i++) {
    compute_lagrange_interpolants(h1, h1_prime, z1(i), ngll, z1);
    for (int j = 0; j < ngll; j++) {
      EXPECT_NEAR(h1_prime_xx(i, j), h1_prime(j), tol);
      if (i == j) {
        EXPECT_NEAR(h1(j), 1.0, tol);
        if (i == 0) {
          double result = -1.0 * static_cast<double>(degpoly) *
                          (static_cast<double>(degpoly) + 1.0) * 0.25;
          EXPECT_NEAR(h1_prime(j), result, tol) << i;
        } else if (i == degpoly) {
          double result = 1.0 * static_cast<double>(degpoly) *
                          (static_cast<double>(degpoly) + 1.0) * 0.25;
          EXPECT_NEAR(h1_prime(j), result, tol) << i;
        } else {
          double result = 0.0;
          EXPECT_NEAR(h1_prime(j), result, tol) << i;
        }
      } else {
        EXPECT_NEAR(h1(j), 0.0, tol);
      }
    }
  }
}

int main(int argc, char *argv[]) {
  ::testing::InitGoogleTest(&argc, argv);
  ::testing::AddGlobalTestEnvironment(new KokkosEnvironment);
  return RUN_ALL_TESTS();
}
