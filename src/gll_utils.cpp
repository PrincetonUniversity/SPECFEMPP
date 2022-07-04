#include "../include/gll_utils.h"
#include <algorithm>
#include <cmath>
#include <iostream>
#include <stdexcept>
#include <tuple>
#include <vector>

double update_x(double &x, const double p, const double pd,
                const std::vector<double> &xjac, const int j) {
  int recsum = 0.0;
  double delx;
  for (int i = 0; i < j - 1; i++) {
    recsum = recsum + 1.0 / (x - xjac[xjac.size() - i - 1]);
  }
  delx = -p / (pd - recsum * p);
  x = x + delx;
  return delx;
}

std::vector<double> &gll_utils::jacg(const int np, const double alpha,
                                     const double beta) {

  std::vector<double> xjac(np, 0.0);
  double xlast, dth, x, x1, x2, recsum, delx, xmin, swap;
  double p, pd;
  int k, jmin, jm, n;
  const int k_max = 10;
  const double eps = 1e-12;

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

    for (int k = 0; k < k_max && std::abs(delx) < eps; k++) {
      std::tie(p, pd) = gll_utils::jacobf(np, alpha, beta, x);
      delx = update_x(x, p, pd, xjac, j);
    }
    try {
      xjac.at(np - j) = x;
      xlast = x;
    } catch (std::out_of_range const &e) {
      throw;
    }
  }

  std::sort(xjac.begin(), xjac.end());

  return xjac;
}

std::tuple<double, double> gll_utils::jacobf(const int n, const double alpha,
                                             const double beta,
                                             const double x) {

  double p1 = 0.0, p1d = 0.0, pm1, pm1d, pm2, pm2d;

  double double_k, a1, a2, b3, a3, a4;
  int k;

  p1 = 1.0;
  p1d = 0.0;

  if (n == 0)
    return std::make_tuple(p1, p1d);

  pm1 = p1;
  pm1d = p1d;
  p1 = (alpha - beta + (alpha + beta + 2.0) * x) / 2.0;
  p1d = (alpha + beta + 2.0) / 2.0;

  if (n == 1)
    return std::make_tuple(p1, p1d);

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

  return std::make_tuple(p1, p1d);
}
