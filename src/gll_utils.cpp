#include "../include/gll_utils.h"
#include <Kokkos_Core.hpp>
#include <Kokkos_Sort.hpp>
#include <algorithm>
#include <cassert>
#include <cmath>
#include <iostream>
#include <stdexcept>
#include <tuple>

double update_x(double &x, const double p, const double pd,
                const HostArray<double> &xjac, const int j) {
  int recsum = 0.0;
  double delx;
  for (int i = 1; i < j; i++) {
    recsum = recsum + 1.0 / (x - xjac(xjac.extent(0) - i));
  }
  delx = -p / (pd - recsum * p);
  x = x + delx;
  return delx;
}

void gll_utils::jacg(HostArray<double> &xjac, const int np, const double alpha,
                     const double beta) {

  double xlast, dth, x, x1, x2, recsum, delx, xmin, swap;
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
      std::tie(p, pd) = gll_utils::jacobf(np, alpha, beta, x);
      delx = update_x(x, p, pd, xjac, j);
      if (std::abs(delx) < eps)
        break;
    }
    try {
      xjac(np - j) = x;
      xlast = x;
    } catch (std::out_of_range const &e) {
      throw;
    }
  }

  Kokkos::sort(xjac);

  return;
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

// double gll_utils::calc_gammaf(const double x){

//   const double pi = 3.141592653589793d0;

//   double gammaf = 1.0;

//   if (x == -0.5) gammaf = -2.0*std::sqrt(pi);
//   if (x == 0.5) gammaf =  std::sqrt(pi);
//   if (x == 1.0) gammaf =  1.0;
//   if (x == 2.0) gammaf =  1.0;
//   if (x == 1.5) gammaf =  std::sqrt(pi)/2.0;
//   if (x == 2.5) gammaf =  1.5*std::sqrt(pi)/2.0;
//   if (x == 3.5) gammaf =  2.5*1.5*std::sqrt(pi)/2.0;
//   if (x == 3.d ) gammaf =  2.0;
//   if (x == 4.d ) gammaf = 6.0;
//   if (x == 5.d ) gammaf = 24.0;
//   if (x == 6.d ) gammaf = 120.0;

//   return gammaf;
// }

// double gll_utils::calc_pnormj(const int n, const double alpha, const double
// beta){

//   double double_n, const, prod, dindx, frac;

//   double_n    = static_cast<double>(n);
//   const = alpha+beta+1.0;

//   if (n <= 1) then
//     prod   =
//     gll_utils::gammaf(double_n+alpha)*gll_utils::gammaf(double_n+beta); prod
//     =
//     prod/(gll_utils::gammaf(double_n)*gll_utils::gammaf(double_n+alpha+beta));
//     pnormj = prod * 2.0**const/(2.0*double_n+const);
//     return pnormj;
//   endif

//   prod  = gll_utils::gammaf(alpha+1.0)*gll_utils::gammaf(beta+1.0);
//   prod  = prod/(2.0*(1.0+const)*gll_utils::gammaf(const+1.0));
//   prod  = prod*(1.0+alpha)*(2.0+alpha);
//   prod  = prod*(1.0+beta)*(2.0+beta);

//   for (int i = 3; i <= n; i++){
//     dindx = static_cast<double>(i);
//     frac  = (dindx+alpha)*(dindx+beta)/(dindx*(dindx+alpha+beta));
//     prod  = prod*frac;
//   }

//   pnormj = prod * std::pow(2.0, const)/(2.0*double_n+const);

//   return pnormj;
// }
