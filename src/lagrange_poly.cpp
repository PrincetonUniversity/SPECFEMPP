#include "../include/gll_library.h"
#include <Kokkos_Core.hpp>

namespace Lagrange {

void compute_lagrange_interpolants(
    Kokkos::View<double *, Kokkos::LayoutRight, Kokkos::HostSpace> h,
    Kokkos::View<double *, Kokkos::LayoutRight, Kokkos::HostSpace> hprime,
    const double xi, const int ngll,
    const Kokkos::View<double *, Kokkos::LayoutRight, Kokkos::HostSpace>
        xigll) {

  assert(xigll.extent(0) == ngll);
  assert(h.extent(0) == ngll);
  assert(hprime.extent(0) == ngll);
  double prod1, prod2, prod3, prod2_inv, sum, x0, x;

  for (int dgr = 0; dgr < ngll; dgr++) {
    prod1 = 1.0, prod2 = 1.0;
    x0 = xigll(dgr);
    sum = 0.00;
    for (int i = 0; i < ngll; i++) {
      if (i != dgr) {
        x = xigll(i);
        prod1 = prod1 * (xi - x);
        prod2 = prod2 * (x0 - x);
        prod3 = 1.0;
        for (int j = 0; j < ngll; j++)
          if (j != dgr && j != i)
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

  assert(xigll.extent(0) == ngll);
  assert(hprime_ii.extent(0) == ngll && hprime_ii.extent(1) == ngll);
  int degpoly = ngll - 1;
  for (int i = 0; i < ngll; i++) {
    for (int j = 0; j < ngll; j++) {
      if (i == 0 && j == 0) {
        hprime_ii(i, j) = -1.0 * static_cast<double>(degpoly) *
                          (static_cast<double>(degpoly) + 1.0) * 0.25;
      } else if (i == degpoly && j == degpoly) {
        hprime_ii(i, j) = 1.0 * static_cast<double>(degpoly) *
                          (static_cast<double>(degpoly) + 1.0) * 0.25;
      } else if (i == j) {
        hprime_ii(i, j) = 0.0;
      } else {
        hprime_ii(i, j) = gll_library::pnleg(xigll(i), degpoly) /
                              (gll_library::pnleg(xigll(j), degpoly) *
                               (xigll(i) - xigll(j))) +
                          (1.0 - xigll(i) * xigll(i)) *
                              gll_library::pndleg(xigll(i), degpoly) /
                              (static_cast<double>(degpoly) *
                               (static_cast<double>(degpoly) + 1.0) *
                               gll_library::pnleg(xigll(j), degpoly) *
                               (xigll(i) - xigll(j)) * (xigll(i) - xigll(j)));
      }
    }
  }
  return;
}

void compute_jacobi_derivatives_GLJ(
    Kokkos::View<double **, Kokkos::LayoutRight, Kokkos::HostSpace>
        hprimeBar_ii,
    const Kokkos::View<double *, Kokkos::LayoutRight, Kokkos::HostSpace> xiglj,
    const int nglj) {

  assert(xiglj.extent(0) == nglj);
  assert(hprimeBar_ii.extent(0) == nglj && hprimeBar_ii.extent(1) == nglj);
  int degpoly = nglj - 1;
  for (int i = 0; i < nglj; i++) {
    for (int j = 0; j < nglj; j++) {
      if (i == 0 && j == 0) {
        hprimeBar_ii(i, j) = -1.0 * static_cast<double>(degpoly) *
                             (static_cast<double>(degpoly) + 2.0) / 6.0;
      } else if (i == 0 && 0 < j && j < degpoly) {
        hprimeBar_ii(i, j) =
            2.0 * std::pow(-1.0, degpoly) *
            gll_library::pnglj(xiglj(j), degpoly) /
            ((1.0 + xiglj(j)) * (static_cast<double>(degpoly) + 1.0));
      } else if (i == 0 && j == degpoly) {
        hprimeBar_ii(i, j) =
            std::pow(-1, degpoly) / (static_cast<double>(degpoly) + 1.0);
      } else if (0 < i && i < degpoly && j == 0) {
        hprimeBar_ii(i, j) =
            std::pow(-1, degpoly + 1) * (static_cast<double>(degpoly) + 1.0) /
            (2.0 * gll_library::pnglj(xiglj(i), degpoly) * (1.0 + xiglj(i)));
      } else if (0 < i && i < degpoly && 0 < j && j < degpoly && i != j) {
        hprimeBar_ii(i, j) = 1.0 / (xiglj(j) - xiglj(i)) *
                             gll_library::pnglj(xiglj(j), degpoly) /
                             gll_library::pnglj(xiglj(i), degpoly);
      } else if (0 < i && i < degpoly && i == j) {
        hprimeBar_ii(i, j) = -1.0 / (2.0 * (1.0 + xiglj(i)));
      } else if (0 < i && i < degpoly && j == degpoly) {
        hprimeBar_ii(i, j) =
            1.0 / (gll_library::pnglj(xiglj(i), degpoly) * (1.0 - xiglj(i)));
      } else if (i == degpoly && j == 0) {
        hprimeBar_ii(i, j) = std::pow(-1, degpoly + 1) *
                             (static_cast<double>(degpoly) + 1.0) / 4.0;
      } else if (i == degpoly && 0 < j && j < degpoly) {
        hprimeBar_ii(i, j) =
            -1.0 / (1.0 - xiglj(j)) * gll_library::pnglj(xiglj(j), degpoly);
      } else if (i == degpoly && j == degpoly) {
        hprimeBar_ii(i, j) = (static_cast<double>(degpoly) *
                                  (static_cast<double>(degpoly) + 2.0) -
                              1.0) /
                             4.0;
      } else {
        throw std::runtime_error("Problem in poly_deriv_GLJ: in a perfect "
                                 "world this would NEVER appear");
      }
    }
  }
  return;
}
} // namespace Lagrange
