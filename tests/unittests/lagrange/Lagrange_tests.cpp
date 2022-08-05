#include "../../include/gll_library.h"
#include "../../include/lagrange_poly.h"
#include "Kokkos_Environment.hpp"
#include <Kokkos_Core.hpp>
#include <gtest/gtest.h>
#include <iostream>
#include <stdexcept>

using DeviceView1d = specfem::DeviceView1d<type_real>;
using DeviceView2d = specfem::DeviceView2d<type_real>;
using HostMirror1d = specfem::HostMirror1d<type_real>;
using HostMirror2d = specfem::HostMirror2d<type_real>;

TEST(lagrange_tests, LAGRANGE_TESTS) {
  /**
   *  This test checks if compute_lagrange_interpolants and
   * compute_lagrange_derivatives_GLL give the same value at GLL points
   *
   */
  const auto &compute_lagrange_interpolants =
      Lagrange::compute_lagrange_interpolants;
  const auto &zwgljd = gll_library::zwgljd;
  const auto &compute_lagrange_derivatives_GLL =
      Lagrange::compute_lagrange_derivatives_GLL;
  int ngll = 5;
  type_real degpoly = ngll - 1;
  type_real tol = 1e-6;

  DeviceView1d z1("Lagrange_test::z1", ngll);
  HostMirror1d h_z1 = Kokkos::create_mirror_view(z1);

  DeviceView1d w1("Lagrange_test::w1", ngll);
  HostMirror1d h_w1 = Kokkos::create_mirror_view(w1);

  DeviceView2d hprime_xx("Lagrange_test::hprime_xx", ngll, ngll);
  HostMirror2d h_hprime_xx = Kokkos::create_mirror_view(hprime_xx);

  DeviceView1d h1("Lagrange_test::h1", ngll);
  HostMirror1d h_h1 = Kokkos::create_mirror_view(h1);

  DeviceView1d h1_prime("Lagrange_test::h1_prime", ngll);
  HostMirror1d h_h1_prime = Kokkos::create_mirror_view(h1_prime);

  zwgljd(h_z1, h_w1, ngll, 0.0, 0.0);
  compute_lagrange_derivatives_GLL(h_hprime_xx, h_z1, ngll);

  for (int i = 0; i < ngll; i++) {
    compute_lagrange_interpolants(h_h1, h_h1_prime, h_z1(i), ngll, z1);
    for (int j = 0; j < ngll; j++) {
      EXPECT_NEAR(h_hprime_xx(i, j), h_h1_prime(j), tol);
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

// TEST(lagrange_tests, GLJ_DERIVS_TEST){
//   const auto& compute_jacobi_derivatives_GLJ =
//   Lagrange::compute_jacobi_derivatives_GLJ; const auto& jacobf =
//   gll_utils::jacobf; const auto &zwgljd = gll_library::zwgljd; const
//   type_real alpha = 0.0, beta = 1.0; const int nglj = 5; type_real tol =
//   1e-6; type_real pd; HostMirror1d z1("r1", nglj); HostMirror1d w1("w1",
//   nglj); Kokkos::View<type_real **, Kokkos::LayoutRight, Kokkos::HostSpace>
//   h1_primeBar_xx(
//       "h1_primeBar_xx", nglj, nglj);

//   zwgljd(z1, w1, nglj, alpha, beta);
//   compute_jacobi_derivatives_GLJ(h1_primeBar_xx, z1, nglj);

//   for (int i = 0; i < nglj-2; i++){
//     for (int j = 0; j < nglj-2; j++){
//       std::cout << i << " " << j << std::endl;
//       std::tie(std::ignore, pd, std::ignore) = jacobf(j, alpha, beta, z1(i));
//       EXPECT_NEAR(h1_primeBar_xx(i+1, j-1), pd, tol);
//     }
//   }
// }

int main(int argc, char *argv[]) {
  ::testing::InitGoogleTest(&argc, argv);
  ::testing::AddGlobalTestEnvironment(new KokkosEnvironment);
  return RUN_ALL_TESTS();
}
