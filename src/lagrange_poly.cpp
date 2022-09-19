#include "../include/lagrange_poly.h"
#include "../include/config.h"
#include "../include/gll_library.h"
#include <Kokkos_Core.hpp>

using HostMirror1d = specfem::HostMirror1d<type_real>;
using HostMirror2d = specfem::HostMirror2d<type_real>;

void Lagrange::compute_lagrange_interpolants(HostMirror1d h,
                                             HostMirror1d hprime,
                                             const type_real xi, const int ngll,
                                             const HostMirror1d xigll) {

  assert(xigll.extent(0) == ngll);
  assert(h.extent(0) == ngll);
  assert(hprime.extent(0) == ngll);
  type_real prod1, prod2, prod3, prod2_inv, sum, x0, x;

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

void Lagrange::compute_lagrange_derivatives_GLL(HostMirror2d hprime_ii,
                                                const HostMirror1d xigll,
                                                const int ngll) {

  assert(xigll.extent(0) == ngll);
  assert(hprime_ii.extent(0) == ngll && hprime_ii.extent(1) == ngll);
  int degpoly = ngll - 1;
  for (int i = 0; i < ngll; i++) {
    for (int j = 0; j < ngll; j++) {
      if (i == 0 && j == 0) {
        hprime_ii(i, j) = -1.0 * static_cast<type_real>(degpoly) *
                          (static_cast<type_real>(degpoly) + 1.0) * 0.25;
      } else if (i == degpoly && j == degpoly) {
        hprime_ii(i, j) = 1.0 * static_cast<type_real>(degpoly) *
                          (static_cast<type_real>(degpoly) + 1.0) * 0.25;
      } else if (i == j) {
        hprime_ii(i, j) = 0.0;
      } else {
        hprime_ii(i, j) = gll_library::pnleg(xigll(i), degpoly) /
                              (gll_library::pnleg(xigll(j), degpoly) *
                               (xigll(i) - xigll(j))) +
                          (1.0 - xigll(i) * xigll(i)) *
                              gll_library::pndleg(xigll(i), degpoly) /
                              (static_cast<type_real>(degpoly) *
                               (static_cast<type_real>(degpoly) + 1.0) *
                               gll_library::pnleg(xigll(j), degpoly) *
                               (xigll(i) - xigll(j)) * (xigll(i) - xigll(j)));
      }
    }
  }
  return;
}

void Lagrange::compute_jacobi_derivatives_GLJ(HostMirror2d hprimeBar_ii,
                                              const HostMirror1d xiglj,
                                              const int nglj) {

  assert(xiglj.extent(0) == nglj);
  assert(hprimeBar_ii.extent(0) == nglj && hprimeBar_ii.extent(1) == nglj);
  int degpoly = nglj - 1;
  for (int i = 0; i < nglj; i++) {
    for (int j = 0; j < nglj; j++) {
      std::cout << gll_library::pnglj(xiglj(i), degpoly) << " " << xiglj(i)
                << std::endl;
      if (j == 0 && i == 0) {
        hprimeBar_ii(i, j) = -1.0 * static_cast<type_real>(degpoly) *
                             (static_cast<type_real>(degpoly) + 2.0) / 6.0;
      } else if (j == 0 && 0 < i && i < degpoly) {
        hprimeBar_ii(i, j) =
            2.0 * std::pow(-1.0, degpoly) *
            gll_library::pnglj(xiglj(i), degpoly) /
            ((1.0 + xiglj(i)) * (static_cast<type_real>(degpoly) + 1.0));
      } else if (j == 0 && i == degpoly) {
        hprimeBar_ii(i, j) =
            std::pow(-1, degpoly) / (static_cast<type_real>(degpoly) + 1.0);
      } else if (0 < j && j < degpoly && i == 0) {
        hprimeBar_ii(i, j) =
            std::pow(-1, degpoly + 1) *
            (static_cast<type_real>(degpoly) + 1.0) /
            (2.0 * gll_library::pnglj(xiglj(j), degpoly) * (1.0 + xiglj(j)));
      } else if (0 < j && j < degpoly && 0 < i && i < degpoly && i != j) {
        hprimeBar_ii(i, j) = 1.0 / (xiglj(i) - xiglj(j)) *
                             gll_library::pnglj(xiglj(i), degpoly) /
                             gll_library::pnglj(xiglj(j), degpoly);
      } else if (0 < j && j < degpoly && i == j) {
        hprimeBar_ii(i, j) = -1.0 / (2.0 * (1.0 + xiglj(j)));
      } else if (0 < j && j < degpoly && i == degpoly) {
        hprimeBar_ii(i, j) =
            1.0 / (gll_library::pnglj(xiglj(i), degpoly) * (1.0 - xiglj(j)));
      } else if (j == degpoly && i == 0) {
        hprimeBar_ii(i, j) = std::pow(-1, degpoly + 1) *
                             (static_cast<type_real>(degpoly) + 1.0) / 4.0;
      } else if (j == degpoly && 0 < i && i < degpoly) {
        hprimeBar_ii(i, j) =
            -1.0 / (1.0 - xiglj(i)) * gll_library::pnglj(xiglj(i), degpoly);
      } else if (j == degpoly && i == degpoly) {
        hprimeBar_ii(i, j) = (static_cast<type_real>(degpoly) *
                                  (static_cast<type_real>(degpoly) + 2.0) -
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
