#include "gll_library.hpp"
#include "gll_utils.hpp"
#include "specfem_setup.hpp"
#include <cmath>
#include <iostream>
#include <stdexcept>

using HostMirror1d = specfem::kokkos::HostMirror1d<type_real>;
using HostView1d = specfem::kokkos::HostView1d<type_real>;

type_real specfem::quadrature::gll::gll_library::pnleg(const type_real z,
                                                       const int n) {
  // Generate Lagendre polynomials using recurrance relation
  // (l+1)P_(l+1)(x)-(2l+1)xP_l(x)+lP_(l-1)(x)=0

  if (n == 0)
    throw std::invalid_argument("value of n > 0");

  type_real p1, p2, p3, double_k;

  p1 = 1.0;
  p2 = z;
  p3 = p2;

  for (int k = 1; k < n; k++) {
    double_k = static_cast<double>(k);
    p3 = ((2.0 * double_k + 1.0) * z * p2 - double_k * p1) / (double_k + 1.0);
    p1 = p2;
    p2 = p3;
  }

  return p3;
}

type_real specfem::quadrature::gll::gll_library::pnglj(const type_real z,
                                                       const int n) {

  if (n == 0)
    throw std::invalid_argument("value of n > 0");

  type_real glj_value;

  if (std::abs(z + 1.0) > 1e-9) {
    glj_value =
        (gll_library::pnleg(z, n) + gll_library::pnleg(z, n + 1)) / (1.0 + z);
  } else {
    glj_value = (static_cast<double>(n) + 1.0) * std::pow(-1.0, n);
  }

  return glj_value;
}

type_real specfem::quadrature::gll::gll_library::pndleg(const type_real z,
                                                        const int n) {

  if (n == 0)
    throw std::invalid_argument("value of n > 0");

  type_real p1, p2, p1d, p2d, p3, p3d, double_k;

  p1 = 1.0;
  p2 = z;
  p1d = 0.0;
  p2d = 1.0;
  p3d = 1.0;

  for (int k = 1; k < n; k++) {
    double_k = static_cast<double>(k);
    p3 = ((2.0 * double_k + 1.0) * z * p2 - double_k * p1) / (double_k + 1.0);
    p3d = ((2.0 * double_k + 1.0) * p2 + (2.0 * double_k + 1.0) * z * p2d -
           double_k * p1d) /
          (double_k + 1.0);
    p1 = p2;
    p2 = p3;
    p1d = p2d;
    p2d = p3d;
  }

  return p3d;
}

type_real specfem::quadrature::gll::gll_library::pndglj(const type_real z,
                                                        const int n) {

  if (n == 0)
    throw std::invalid_argument("value of n > 0");

  type_real glj_deriv;

  if (std::abs(z + 1.0) > 1e-9) {
    glj_deriv = (gll_library::pndleg(z, n) + gll_library::pndleg(z, n + 1)) /
                    (1.0 + z) -
                (gll_library::pnleg(z, n) + gll_library::pnleg(z, n + 1)) /
                    ((1.0 + z) * (1.0 + z));
  } else {
    glj_deriv = gll_library::pnleg(-1.0, n) + gll_library::pnleg(-1.0, n + 1);
  }

  return glj_deriv;
}

void specfem::quadrature::gll::gll_library::zwgljd(HostMirror1d z,
                                                   HostMirror1d w, const int np,
                                                   const type_real alpha,
                                                   const type_real beta) {

  type_real p, pd;

  assert(np > 2);
  assert(alpha > -1.0 && beta > -1.0);
  assert(z.rank == 1);
  assert(w.rank == 1);
  assert(z.extent(0) == np);
  assert(w.extent(0) == np);

  if (np > 2) {
    auto z_view = Kokkos::subview(z, Kokkos::pair(1, np - 1));
    auto w_view = Kokkos::subview(w, Kokkos::pair(1, np - 1));
    gll_utils::zwgjd(z_view, w_view, np - 2, alpha + 1.0, beta + 1.0);
  }

  // start and end point at exactly -1 and 1
  z(0) = -1.0;
  z(np - 1) = 1.0;

  // note: Jacobi polynomials with (alpha,beta) equal to zero become Legendre
  // polynomials.
  //       for Legendre polynomials, if number of points is odd, the middle
  //       abscissa is exactly zero
  if (std::abs(alpha) < 1e-9 && std::abs(beta) < 1e-9) {
    if (np % 2 != 0)
      z((np - 1) / 2) = 0.0;
  }

  for (int i = 1; i < np - 1; i++) {
    w(i) = w(i) / (1.0 - z(i) * z(i));
  }

  std::tie(p, pd, std::ignore) = gll_utils::jacobf(np - 1, alpha, beta, z(0));
  w(0) = gll_utils::endw1(np - 1, alpha, beta) / (2.0 * pd);

  std::tie(p, pd, std::ignore) =
      gll_utils::jacobf(np - 1, alpha, beta, z(np - 1));
  w(np - 1) = gll_utils::endw2(np - 1, alpha, beta) / (2.0 * pd);

  return;
}

std::tuple<HostView1d, HostView1d>
specfem::quadrature::gll::gll_library::zwgljd(const int np,
                                              const type_real alpha,
                                              const type_real beta) {

  type_real p, pd;

  assert(np > 2);
  assert(alpha > -1.0 && beta > -1.0);

  HostView1d z("specfem::gll_library::z", np);
  HostView1d w("specfem::gll_library::z", np);

  if (np > 2) {
    auto z_view = Kokkos::subview(z, Kokkos::pair(1, np - 1));
    auto w_view = Kokkos::subview(w, Kokkos::pair(1, np - 1));
    gll_utils::zwgjd(z_view, w_view, np - 2, alpha + 1.0, beta + 1.0);
  }

  // start and end point at exactly -1 and 1
  z(0) = -1.0;
  z(np - 1) = 1.0;

  // note: Jacobi polynomials with (alpha,beta) equal to zero become Legendre
  // polynomials.
  //       for Legendre polynomials, if number of points is odd, the middle
  //       abscissa is exactly zero
  if (std::abs(alpha) < 1e-9 && std::abs(beta) < 1e-9) {
    if (np % 2 != 0)
      z((np - 1) / 2) = 0.0;
  }

  for (int i = 1; i < np - 1; i++) {
    w(i) = w(i) / (1.0 - z(i) * z(i));
  }

  std::tie(p, pd, std::ignore) = gll_utils::jacobf(np - 1, alpha, beta, z(0));
  w(0) = gll_utils::endw1(np - 1, alpha, beta) / (2.0 * pd);

  std::tie(p, pd, std::ignore) =
      gll_utils::jacobf(np - 1, alpha, beta, z(np - 1));
  w(np - 1) = gll_utils::endw2(np - 1, alpha, beta) / (2.0 * pd);

  return std::make_tuple(z, w);
}
