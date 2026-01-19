#include "../SPECFEM_Environment.hpp"
#include "specfem/quadrature.hpp"
#include "specfem/quadrature/gll/gll_library.hpp"
#include <Kokkos_Core.hpp>
#include <gtest/gtest.h>
#include <iostream>
#include <stdexcept>

TEST(lagrange_tests, LAGRANGE_TESTS) {
  /**
   *  This test checks if compute_lagrange_interpolants and
   * compute_lagrange_derivatives_GLL give the same value at GLL points
   *
   */
  int ngll = 5;
  type_real degpoly = ngll - 1;
  type_real tol = 1e-6;

  auto [h_z1, h_w1] =
      specfem::quadrature::gll::gll_library::zwgljd(ngll, 0.0, 0.0);
  auto h_hprime_xx =
      specfem::quadrature::gll::Lagrange::compute_lagrange_derivatives_GLL(
          h_z1, ngll);

  for (int i = 0; i < ngll; i++) {
    auto [h_h1, h_h1_prime] =
        specfem::quadrature::gll::Lagrange::compute_lagrange_interpolants(
            h_z1(i), ngll, h_z1);
    for (int j = 0; j < ngll; j++) {
      EXPECT_NEAR(h_hprime_xx(j, i), h_h1_prime(j), tol);
      if (i == j) {
        EXPECT_NEAR(h_h1(j), 1.0, tol);
        if (i == 0) {
          type_real result = -1.0 * static_cast<type_real>(degpoly) *
                             (static_cast<type_real>(degpoly) + 1.0) * 0.25;
          EXPECT_NEAR(h_h1_prime(j), result, tol) << i;
        } else if (i == degpoly) {
          type_real result = 1.0 * static_cast<type_real>(degpoly) *
                             (static_cast<type_real>(degpoly) + 1.0) * 0.25;
          EXPECT_NEAR(h_h1_prime(j), result, tol) << i;
        } else {
          type_real result = 0.0;
          EXPECT_NEAR(h_h1_prime(j), result, tol) << i;
        }
      } else {
        EXPECT_NEAR(h_h1(j), 0.0, tol);
      }
    }
  }
}

int main(int argc, char *argv[]) {
  ::testing::InitGoogleTest(&argc, argv);
  ::testing::AddGlobalTestEnvironment(new SPECFEMEnvironment);
  return RUN_ALL_TESTS();
}
