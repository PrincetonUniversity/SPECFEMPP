#include "../include/gll_utils.h"
#include <Kokkos_Core.hpp>
#include <Kokkos_Sort.hpp>
#include <algorithm>
#include <cassert>
#include <cmath>
#include <iostream>
#include <stdexcept>
#include <tuple>

double update_x(
    double &x, const double p, const double pd,
    const Kokkos::View<double *, Kokkos::LayoutRight, Kokkos::HostSpace> xjac,
    const int j) {
  int recsum = 0.0;
  double delx;
  for (int i = 1; i < j; i++) {
    recsum = recsum + 1.0 / (x - xjac(xjac.extent(0) - i));
  }
  delx = -p / (pd - recsum * p);
  x = x + delx;
  return delx;
}

void gll_utils::jacg(
    Kokkos::View<double *, Kokkos::LayoutRight, Kokkos::HostSpace> xjac,
    const int np, const double alpha, const double beta) {

  double xlast, dth, x, x1, x2, delx;
  double p, pd;
  int k, jmin, jm, n;
  const int k_max = 10;
  const double eps = 1e-12;

  assert(xjac.rank == 1);
  assert(xjac.extent(0) == np);

  xlast = 0.0;
  n = np - 1;
  dth = 4.0 * std::atan(1.0) / (2.0 * static_cast<double>(n) + 2.0);

  for (int j = 1; j <= np; j++) {
    if (j == 1) {
      x = std::cos((2.0 * (static_cast<double>(j) - 1.0) + 1.0) * dth);
    } else {
      x1 = std::cos((2.0 * (static_cast<double>(j) - 1.0) + 1.0) * dth);
      x2 = xlast;
      x = (x1 + x2) / 2.0;
    }

    for (int k = 0; k < k_max; k++) {
      std::tie(p, pd, std::ignore) = gll_utils::jacobf(np, alpha, beta, x);
      delx = update_x(x, p, pd, xjac, j);
      if (std::abs(delx) < eps)
        break;
    }

    if (np - j < 0 || np - j > np - 1) {
      char err_msg[80];
      sprintf(err_msg,
              "Index xjac is out of range: xjac.extent(0) == %i & np-j = %i",
              np, np - j);
      throw std::out_of_range(err_msg);
    }
    xjac(np - j) = x;
    xlast = x;
  }

  Kokkos::sort(xjac);

  return;
}

std::tuple<double, double, double> gll_utils::jacobf(const int n,
                                                     const double alpha,
                                                     const double beta,
                                                     const double x) {

  double p1 = 0.0, p1d = 0.0, pm1, pm1d, pm2, pm2d;

  double double_k, a1, a2, b3, a3, a4;
  int k;

  p1 = 1.0;
  p1d = 0.0;

  if (n == 0)
    return std::make_tuple(p1, p1d, 0.0);

  pm1 = p1;
  pm1d = p1d;
  p1 = (alpha - beta + (alpha + beta + 2.0) * x) / 2.0;
  p1d = (alpha + beta + 2.0) / 2.0;

  if (n == 1)
    return std::make_tuple(p1, p1d, pm1d);

  for (int k = 2; k <= n; k++) {
    double_k = static_cast<double>(k);
    pm2 = pm1;
    pm2d = pm1d;
    pm1 = p1;
    pm1d = p1d;
    a1 = 2.0 * double_k * (double_k + alpha + beta) *
         (2.0 * double_k + alpha + beta - 2.0);
    a2 = (2.0 * double_k + alpha + beta - 1.0) * (alpha * alpha - beta * beta);
    b3 = (2.0 * double_k + alpha + beta - 2.0);
    a3 = b3 * (b3 + 1.0) * (b3 + 2.0);
    a4 = 2.0 * (double_k + alpha - 1.0) * (double_k + beta - 1.0) *
         (2.0 * double_k + alpha + beta);
    p1 = ((a2 + a3 * x) * pm1 - a4 * pm2) / a1;
    p1d = ((a2 + a3 * x) * pm1d - a4 * pm2d + a3 * pm1) / a1;
  }

  return std::make_tuple(p1, p1d, pm1d);
}

double gll_utils::calc_gammaf(const double x) {

  const double pi = 3.1415926535897930;

  double gammaf = 1.0;

  if (x == -0.5)
    gammaf = -2.0 * std::sqrt(pi);
  if (x == 0.5)
    gammaf = std::sqrt(pi);
  if (x == 1.0)
    gammaf = 1.0;
  if (x == 2.0)
    gammaf = 1.0;
  if (x == 1.5)
    gammaf = std::sqrt(pi) / 2.0;
  if (x == 2.5)
    gammaf = 1.5 * std::sqrt(pi) / 2.0;
  if (x == 3.5)
    gammaf = 2.5 * 1.5 * std::sqrt(pi) / 2.0;
  if (x == 3.0)
    gammaf = 2.0;
  if (x == 4.0)
    gammaf = 6.0;
  if (x == 5.0)
    gammaf = 24.0;
  if (x == 6.0)
    gammaf = 120.0;

  return gammaf;
}

double gll_utils::calc_pnormj(const int n, const double alpha,
                              const double beta) {

  double double_n, apb1, prod, double_i, pnormj;

  double_n = static_cast<double>(n);
  apb1 = alpha + beta + 1.0;

  if (n <= 1) {
    prod = gll_utils::calc_gammaf(double_n + alpha) *
           gll_utils::calc_gammaf(double_n + beta);
    prod = prod / (gll_utils::calc_gammaf(double_n) *
                   gll_utils::calc_gammaf(double_n + alpha + beta));
    pnormj = prod * std::pow(2.0, apb1) / (2.0 * double_n + apb1);
    return pnormj;
  }

  prod =
      gll_utils::calc_gammaf(alpha + 1.0) * gll_utils::calc_gammaf(beta + 1.0);
  prod = prod / (2.0 * (1.0 + apb1) * gll_utils::calc_gammaf(apb1 + 1.0));
  prod = prod * (1.0 + alpha) * (2.0 + alpha);
  prod = prod * (1.0 + beta) * (2.0 + beta);

  for (int i = 3; i < n + 1; i++) {
    double_i = static_cast<double>(i);
    prod = prod * (double_i + alpha) * (double_i + beta) /
           (double_i * (double_i + alpha + beta));
  }

  pnormj = prod * std::pow(2.0, apb1) / (2.0 * double_n + apb1);
  return pnormj;
}

void gll_utils::jacw(
    Kokkos::View<double *, Kokkos::LayoutRight, Kokkos::HostSpace> z,
    Kokkos::View<double *, Kokkos::LayoutRight, Kokkos::HostSpace> w,
    const int np, const int alpha, const int beta) {

  double p, pd, pm1d;

  int n = np - 1;

  assert(z.rank == 1);
  assert(w.rank == 1);
  assert(z.extent(0) == np);
  assert(w.extent(0) == np);

  double fac1 = static_cast<double>(n + 1) + alpha + beta + 1.0;
  double fac2 = fac1 + static_cast<double>(n + 1);
  double fac3 = fac2 + 1.0;
  double fnorm = gll_utils::calc_pnormj(n + 1, alpha, beta);
  double rcoef =
      (fnorm * fac2 * fac3) / (2.0 * fac1 * static_cast<double>(n + 2));
  for (int i = 0; i < np; i++) {
    std::tie(p, pd, pm1d) = gll_utils::jacobf(n + 2, alpha, beta, z(i));
    w(i) = -rcoef / ((p * pm1d));
  }
  return;
}

void gll_utils::zwgjd(
    Kokkos::View<double *, Kokkos::LayoutRight, Kokkos::HostSpace> z,
    Kokkos::View<double *, Kokkos::LayoutRight, Kokkos::HostSpace> w,
    const int np, const int alpha, const int beta) {

  // calculate the GLL points and weights in the open interval (-1.0, 1.0)

  assert(z.rank == 1);
  assert(w.rank == 1);
  assert(z.extent(0) == np);
  assert(w.extent(0) == np);

  // condition 1 gets hits only when ngll == 3
  if (np == 1) {
    z(0) = (beta - alpha) / (alpha + beta + 2.0);
    w(0) = gll_utils::calc_gammaf(alpha + 1.0) *
           gll_utils::calc_gammaf(beta + 1.0) /
           gll_utils::calc_gammaf(alpha + beta + 2.0) *
           std::pow(2.0, alpha + beta + 1.0);
    return;
  }

  // functions to calculate GLL poitns and weights
  gll_utils::jacg(z, np, alpha, beta);
  gll_utils::jacw(z, w, np, alpha, beta);
  return;
}

double gll_utils::endw1(const int n, const double alpha, const double beta) {

  // Calculates the weight contrinution at xi == -1.0
  double apb, f1, fint1, fint2, f2, double_i, abn, abnn, a1, a2, a3, f3;

  // I dont think np == 0 && np == 1 are ever reached based on where these
  // function is called

  f3 = 0.0;
  apb = alpha + beta;
  if (n == 0) {
    return 0.0;
  }
  f1 = gll_utils::calc_gammaf(alpha + 2.0) *
       gll_utils::calc_gammaf(beta + 1.0) / gll_utils::calc_gammaf(apb + 3.0);
  f1 = f1 * (apb + 2.0) * std::pow(2.0, (apb + 2.0)) / 2.0;
  if (n == 1) {
    return f1;
  }
  fint1 = gll_utils::calc_gammaf(alpha + 2.0) *
          gll_utils::calc_gammaf(beta + 1.0) /
          gll_utils::calc_gammaf(apb + 3.0);
  fint1 = fint1 * std::pow(2.0, (apb + 2.0));
  fint2 = gll_utils::calc_gammaf(alpha + 2.0) *
          gll_utils::calc_gammaf(beta + 2.0) /
          gll_utils::calc_gammaf(apb + 4.0);
  fint2 = fint2 * std::pow(2.0, (apb + 3.0));
  f2 = (-2.0 * (beta + 2.0) * fint1 + (apb + 4.0) * fint2) * (apb + 3.0) / 4.0;
  if (n == 2) {
    return f2;
  }
  for (int i = 3; i <= n; i++) {
    double_i = static_cast<double>(i - 1);
    abn = alpha + beta + double_i;
    abnn = abn + double_i;
    a1 = -(2.0 * (double_i + alpha) * (double_i + beta)) /
         (abn * abnn * (abnn + 1.0));
    a2 = (2.0 * (alpha - beta)) / (abnn * (abnn + 2.0));
    a3 = (2.0 * (abn + 1.0)) / ((abnn + 2.0) * (abnn + 1.0));
    f3 = -(a2 * f2 + a1 * f1) / a3;
    f1 = f2;
    f2 = f3;
  }

  return f3;
}

double gll_utils::endw2(const int n, const double alpha, const double beta) {

  // Calculates the weight contribution at xi == 1.0
  double apb, f1, fint1, fint2, f2, double_i, abn, abnn, a1, a2, a3, f3;

  // I dont think np == 0 && np == 1 are ever reached based on where these
  // function is called

  f3 = 0.0;
  apb = alpha + beta;
  if (n == 0) {
    return 0.0;
  }
  f1 = gll_utils::calc_gammaf(alpha + 1.0) *
       gll_utils::calc_gammaf(beta + 2.0) / gll_utils::calc_gammaf(apb + 3.0);
  f1 = f1 * (apb + 2.0) * std::pow(2.0, (apb + 2.0)) / 2.0;
  if (n == 1) {
    return f1;
  }
  fint1 = gll_utils::calc_gammaf(alpha + 1.0) *
          gll_utils::calc_gammaf(beta + 2.0) /
          gll_utils::calc_gammaf(apb + 3.0);
  fint1 = fint1 * std::pow(2.0, (apb + 2.0));
  fint2 = gll_utils::calc_gammaf(alpha + 2.0) *
          gll_utils::calc_gammaf(beta + 2.0) /
          gll_utils::calc_gammaf(apb + 4.0);
  fint2 = fint2 * std::pow(2.0, (apb + 3.0));
  f2 = (2.0 * (alpha + 2.0) * fint1 - (apb + 4.0) * fint2) * (apb + 3.0) / 4.0;
  if (n == 2) {
    return f2;
  }
  for (int i = 3; i <= n; i++) {
    double_i = static_cast<double>(i - 1);
    abn = alpha + beta + double_i;
    abnn = abn + double_i;
    a1 = -(2.0 * (double_i + alpha) * (double_i + beta)) /
         (abn * abnn * (abnn + 1.0));
    a2 = (2.0 * (alpha - beta)) / (abnn * (abnn + 2.0));
    a3 = (2.0 * (abn + 1.0)) / ((abnn + 2.0) * (abnn + 1.0));
    f3 = -(a2 * f2 + a1 * f1) / a3;
    f1 = f2;
    f2 = f3;
  }

  return f3;
}
