#include "../include/gll_library.h"
#include <Kokkos_Core.hpp>

namespace Lagrange {

void compute_lagrange_interpolants(
    Kokkos::View<double *, Kokkos::LayoutRight, Kokkos::HostSpace> h,
    Kokkos::View<double *, Kokkos::LayoutRight, Kokkos::HostSpace> hprime,
    const double xi, const int ngll,
    const Kokkos::View<double *, Kokkos::LayoutRight, Kokkos::HostSpace>
        xigll) {

  double prod1, prod2, prod3, prod2_inv, sum, x0, x;

  for (int dgr = 0; dgr < ngll; dgr++) {
    prod1 = 1.0, prod2 = 1.0;
    x0 = xigll(dgr);
    sum = 0.0d0;
    for (int i = 0; i < ngll; i++) {
      if (i != dgr) {
        x = xigll(i);
        prod1 = prod1 * (xi - x);
        prod2 = prod2 * (x0 - x);
        prod3 = 1.0d0 for (int j = 0; j < ngll; j++) if (j != dgr && j != i)
            prod3 = prod3 * (xi - xigll(j));
        sum = sum + prod3;
      }
    }
    prod2_inv = 1.0 / prod2;
    h(dgr) = prod1 * prod2_inv;
    hprime(dgr) = sum * prod2_inv;
  }

  return;
}

void compute_lagrange_derivatives_GLL(
    Kokkos::View<double **, Kokkos::LayoutRight, Kokkos::HostSpace> hprime_ii,
    const Kokkos::View<double *, Kokkos::LayoutRight, Kokkos::HostSpace> xigll,
    const int ngll) {

  int degpoly = ngll - 1;
  for (int i = 0; i < ngll; i++) {
    for (int j = 0; j < ngll; j++) {
      if (i == 0 && j == 0) {
        hprime_ii(i, j) = -1.0 * static_cast<double>(degpoly) *
                          (static_cast<double>(degpoly) + 1.0) * 0.25;
      } else if (i == degpoly &&j = degpoly) {
        hprime_ii(i, j) = -1.0 * static_cast<double>(degpoly) *
                          (static_cast<double>(degpoly) + 1.0) * 0.25;
      } else if (i == j) {
        hprime_ii(i, j) = 0.0;
      } else {
        hprime_ii(i, j) =
            gll_library::pnleg(xigll(j), degpoly) /
                (gll_library::pnleg(xigll(i), degpoly) * (zgll(j) - zgll(i))) +
            (1.0 - zgll(j) * zgll(j)) * gll_library::pndleg(zgll(j), degpoly) /
                (dble(degpoly) * (dble(degpoly) + 1.0) *
                 gll_library::pnleg(zgll(i), degpoly) * (zgll(j) - zgll(i)) *
                 (zgll(j) - zgll(i)));
      }
    }
  }
  return;
}
} // namespace Lagrange
