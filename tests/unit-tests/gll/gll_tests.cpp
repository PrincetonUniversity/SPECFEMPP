#include "../SPECFEM_Environment.hpp"
#include "specfem/quadrature/gll/gll_library.hpp"
#include "specfem/quadrature/gll/gll_utils.hpp"
#include <Kokkos_Core.hpp>
#include <gtest/gtest.h>
#include <iostream>
#include <stdexcept>

using DeviceView1d = specfem::kokkos::DeviceView1d<type_real>;
using HostMirror1d = specfem::kokkos::HostMirror1d<type_real>;

TEST(GLL_tests, PNLEG) {
  const auto &pnleg = specfem::quadrature::gll::gll_library::pnleg;
  type_real tol = 1e-6;

  try {
    specfem::quadrature::gll::gll_library::pnleg(-1.0, 0);
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
  const auto &pndleg = specfem::quadrature::gll::gll_library::pndleg;

  type_real tol = 1e-6;
  try {
    specfem::quadrature::gll::gll_library::pndleg(-1.0, 0);
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

  type_real tol = 1e-6, p, pd;
  const auto &jacobf = specfem::quadrature::gll::gll_utils::jacobf;

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

  type_real tol = 1e-6;
  const auto &jacg = specfem::quadrature::gll::gll_utils::jacg;

  DeviceView1d r1("gll_tests::specfem::quadrature::gll::gll_utils::r1", 5);
  HostMirror1d h_r1 = Kokkos::create_mirror_view(r1);
  jacg(h_r1, 5, 0.0, 0.0);
  EXPECT_NEAR(h_r1(0), -0.9061798459, tol);
  EXPECT_NEAR(h_r1(1), -0.538469310, tol);
  EXPECT_NEAR(h_r1(2), 0, tol);
  EXPECT_NEAR(h_r1(3), 0.538469310, tol);
  EXPECT_NEAR(h_r1(4), 0.9061798459, tol);

  DeviceView1d r2("gll_tests::specfem::quadrature::gll::gll_utils::r2", 3);
  HostMirror1d h_r2 = Kokkos::create_mirror_view(r2);
  jacg(h_r2, 3, 0.0, 0.0);
  EXPECT_NEAR(h_r2(0), -0.77459666924, tol);
  EXPECT_NEAR(h_r2(1), 0, tol);
  EXPECT_NEAR(h_r2(2), 0.77459666924, tol);
}

TEST(GLL_tests, JACW) {

  type_real tol = 1e-6;
  const auto &jacg = specfem::quadrature::gll::gll_utils::jacg;
  const auto &jacw = specfem::quadrature::gll::gll_utils::jacw;

  DeviceView1d r2("gll_tests::specfem::quadrature::gll::gll_utils::r2", 3);
  HostMirror1d h_r2 = Kokkos::create_mirror_view(r2);
  jacg(h_r2, 3, 1.0, 1.0);
  EXPECT_NEAR(h_r2(0), -0.6546536707, tol);
  EXPECT_NEAR(h_r2(1), 0, tol);
  EXPECT_NEAR(h_r2(2), 0.6546536707, tol);

  DeviceView1d w2("gll_tests::specfem::quadrature::gll::gll_utils::w2", 3);
  HostMirror1d h_w2 = Kokkos::create_mirror_view(w2);
  jacw(h_r2, h_w2, 3, 1.0, 1.0);
  std::array<type_real, 3> reference = { 0.5444444444, 0.7111111111,
                                         0.5444444444 };
  for (int i = 0; i < 3; i++) {
    type_real result = reference[i] * (1.0 - h_r2(i) * h_r2(i));
    EXPECT_NEAR(h_w2(i), result, tol);
  }
}

TEST(GLL_tests, ZWGJD) {
  // This test checks for the special case of np == 1
  const type_real tol = 1e-6;
  const auto &zwgjd = specfem::quadrature::gll::gll_utils::zwgjd;

  DeviceView1d r1("gll_tests::specfem::quadrature::gll::gll_utils::r1", 1);
  HostMirror1d h_r1 = Kokkos::create_mirror_view(r1);
  DeviceView1d w1("gll_tests::specfem::quadrature::gll::gll_utils::w1", 1);
  HostMirror1d h_w1 = Kokkos::create_mirror_view(w1);
  zwgjd(h_r1, h_w1, 1, 1.0, 1.0);

  EXPECT_NEAR(h_r1(0), 0.0, tol);
  EXPECT_NEAR(h_w1(0), 1.333333, tol);
}

TEST(GLL_tests, ZWGLJD) {

  type_real tol = 1e-6;

  DeviceView1d z1("gll_tests::specfem::quadrature::gll::gll_library::z1", 3);
  HostMirror1d h_z1 = Kokkos::create_mirror_view(z1);
  DeviceView1d w1("gll_tests::specfem::quadrature::gll::gll_library::w1", 3);
  HostMirror1d h_w1 = Kokkos::create_mirror_view(w1);
  specfem::quadrature::gll::gll_library::zwgljd(h_z1, h_w1, 3, 0.0, 0.0);
  EXPECT_NEAR(h_z1(0), -1.0, tol);
  EXPECT_NEAR(h_z1(1), 0.0, tol);
  EXPECT_NEAR(h_z1(2), 1.0, tol);
  EXPECT_NEAR(h_w1(0), 0.333333, tol);
  EXPECT_NEAR(h_w1(1), 1.333333, tol);
  EXPECT_NEAR(h_w1(2), 0.333333, tol);

  DeviceView1d z2("gll_tests::specfem::quadrature::gll::gll_library::z2", 5);
  HostMirror1d h_z2 = Kokkos::create_mirror_view(z2);
  DeviceView1d w2("gll_tests::specfem::quadrature::gll::gll_library::w2", 5);
  HostMirror1d h_w2 = Kokkos::create_mirror_view(w2);
  specfem::quadrature::gll::gll_library::zwgljd(h_z2, h_w2, 5, 0.0, 0.0);
  EXPECT_NEAR(h_z2(0), -1.0, tol);
  EXPECT_NEAR(h_z2(1), -0.6546536707, tol);
  EXPECT_NEAR(h_z2(2), 0.0, tol);
  EXPECT_NEAR(h_z2(3), 0.6546536707, tol);
  EXPECT_NEAR(h_z2(4), 1.0, tol);
  EXPECT_NEAR(h_w2(0), 0.1, tol);
  EXPECT_NEAR(h_w2(1), 0.5444444444, tol);
  EXPECT_NEAR(h_w2(2), 0.7111111111, tol);
  EXPECT_NEAR(h_w2(3), 0.5444444444, tol);
  EXPECT_NEAR(h_w2(4), 0.1, tol);

  specfem::quadrature::gll::gll_library::zwgljd(h_z2, h_w2, 5, 0.0, 1.0);
  EXPECT_NEAR(h_z2(0), -1.0, tol);
  EXPECT_NEAR(h_z2(1), -0.5077876295, tol);
  EXPECT_NEAR(h_z2(2), 0.1323008207, tol);
  EXPECT_NEAR(h_z2(3), 0.7088201421, tol);
  EXPECT_NEAR(h_z2(4), 1.0, tol);
  EXPECT_NEAR(h_w2(0), 0.01333333333, tol);
  EXPECT_NEAR(h_w2(1), 0.2896566946, tol);
  EXPECT_NEAR(h_w2(2), 0.7360043695, tol);
  EXPECT_NEAR(h_w2(3), 0.794338936, tol);
  EXPECT_NEAR(h_w2(4), 0.1666666667, tol);

  DeviceView1d z3("gll_tests::specfem::quadrature::gll::gll_library::z3", 7);
  HostMirror1d h_z3 = Kokkos::create_mirror_view(z3);
  DeviceView1d w3("gll_tests::specfem::quadrature::gll::gll_library::w3", 7);
  HostMirror1d h_w3 = Kokkos::create_mirror_view(w3);
  specfem::quadrature::gll::gll_library::zwgljd(h_z3, h_w3, 7, 0.0, 0.0);
  EXPECT_NEAR(h_z3(0), -1.0, tol);
  EXPECT_NEAR(h_z3(1), -0.8302238962, tol);
  EXPECT_NEAR(h_z3(2), -0.4688487934, tol);
  EXPECT_NEAR(h_z3(3), 0.0, tol);
  EXPECT_NEAR(h_z3(4), 0.4688487934, tol);
  EXPECT_NEAR(h_z3(5), 0.8302238962, tol);
  EXPECT_NEAR(h_z3(6), 1.0, tol);
  EXPECT_NEAR(h_w3(0), 0.0476190476, tol);
  EXPECT_NEAR(h_w3(1), 0.2768260473, tol);
  EXPECT_NEAR(h_w3(2), 0.4317453812, tol);
  EXPECT_NEAR(h_w3(3), 0.4876190476, tol);
  EXPECT_NEAR(h_w3(4), 0.4317453812, tol);
  EXPECT_NEAR(h_w3(5), 0.2768260473, tol);
  EXPECT_NEAR(h_w3(6), 0.0476190476, tol);

  specfem::quadrature::gll::gll_library::zwgljd(h_z3, h_w3, 7, 0.0, 1.0);
  EXPECT_NEAR(h_z3(0), -1.0, tol);
  EXPECT_NEAR(h_z3(1), -0.7401236486, tol);
  EXPECT_NEAR(h_z3(2), -0.3538526341, tol);
  EXPECT_NEAR(h_z3(3), 0.09890279315, tol);
  EXPECT_NEAR(h_z3(4), 0.5288423045, tol);
  EXPECT_NEAR(h_z3(5), 0.8508465697, tol);
  EXPECT_NEAR(h_z3(6), 1.0, tol);
  EXPECT_NEAR(h_w3(0), 0.003401360544, tol);
  EXPECT_NEAR(h_w3(1), 0.08473655296, tol);
  EXPECT_NEAR(h_w3(2), 0.2803032119, tol);
  EXPECT_NEAR(h_w3(3), 0.5016469619, tol);
  EXPECT_NEAR(h_w3(4), 0.5945754451, tol);
  EXPECT_NEAR(h_w3(5), 0.4520031342, tol);
  EXPECT_NEAR(h_w3(6), 0.08333333333, tol);
}

int main(int argc, char *argv[]) {
  ::testing::InitGoogleTest(&argc, argv);
  ::testing::AddGlobalTestEnvironment(new SPECFEMEnvironment);
  return RUN_ALL_TESTS();
}
