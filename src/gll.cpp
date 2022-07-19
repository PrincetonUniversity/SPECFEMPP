#include "../include/gll.h"
#include "../include/gll_library.h"
#include "../include/lagrange_poly.h"
#include <Kokkos_Core.hpp>
#include <iostream>
#include <stdexcept>

void gll::gll::set_allocations() {
  xigll = Kokkos::View<double *, Kokkos::LayoutRight, Kokkos::HostSpace>(
      "xigll", ngllx);
  zigll = Kokkos::View<double *, Kokkos::LayoutRight, Kokkos::HostSpace>(
      "zigll", ngllz);
  wxgll = Kokkos::View<double *, Kokkos::LayoutRight, Kokkos::HostSpace>(
      "wxgll", ngllx);
  wzgll = Kokkos::View<double *, Kokkos::LayoutRight, Kokkos::HostSpace>(
      "wzgll", ngllz);
  hprime_xx = Kokkos::View<double **, Kokkos::LayoutRight, Kokkos::HostSpace>(
      "hprime_xx", ngllx, ngllx);
  hprime_zz = Kokkos::View<double **, Kokkos::LayoutRight, Kokkos::HostSpace>(
      "hprime_zz", ngllz, ngllz);
}

gll::gll::gll() : alpha(0.0), beta(0.0), ngllx(5), ngllz(5) {
  this->set_allocations();
};

gll::gll::gll(const double alpha, const double beta)
    : alpha(alpha), beta(beta), ngllx(5), ngllz(5) {
  if ((alpha <= -1.0) || (beta <= -1.0))
    throw std::invalid_argument("alpha and beta must be greater than -1");
  this->set_allocations();
}

gll::gll::gll(const double alpha, const double beta, const int ngll)
    : alpha(alpha), beta(beta), ngllx(ngll), ngllz(ngll) {
  if (ngll <= 1)
    throw std::invalid_argument("Minimum number of Gauss-Labatto points is 2");
  if (ngll <= 2)
    throw std::invalid_argument(
        "Minimum number of Gauss-Lobatto points for the SEM is 3");
  if ((alpha <= -1.0) || (beta <= -1.0))
    throw std::invalid_argument("alpha and beta must be greater than -1");
  this->set_allocations();
}

gll::gll::gll(const double alpha, const double beta, const int ngllx,
              const int ngllz)
    : alpha(alpha), beta(beta), ngllx(ngllx), ngllz(ngllz) {
  if (ngllx <= 1)
    throw std::invalid_argument("Minimum number of Gauss-Labatto points is 2");
  if (ngllx <= 2)
    throw std::invalid_argument(
        "Minimum number of Gauss-Lobatto points for the SEM is 3");
  if ((alpha <= -1.0) || (beta <= -1.0))
    throw std::invalid_argument("alpha and beta must be greater than -1");
  if (ngllx != ngllz) {
    throw std::invalid_argument(
        "ngllz != ngllx: Cannot handle unstructured meshes due to mismatch in "
        "polynomials at boundaries");
  }
  this->set_allocations();
}

void gll::gll::set_derivation_matrices() {
  gll_library::zwgljd(this->xigll, this->wxgll, this->ngllx, this->alpha,
                      this->beta);
  gll_library::zwgljd(this->zigll, this->wzgll, this->ngllz, this->alpha,
                      this->beta);
  Lagrange::compute_lagrange_derivatives_GLL(this->hprime_xx, this->xigll,
                                             this->ngllx);
  Lagrange::compute_lagrange_derivatives_GLL(this->hprime_zz, this->xigll,
                                             this->ngllx);
}

Kokkos::View<double *, Kokkos::LayoutRight, Kokkos::HostSpace>
gll::gll::get_xigll() const {
  return this->xigll;
}

Kokkos::View<double *, Kokkos::LayoutRight, Kokkos::HostSpace>
gll::gll::get_zigll() const {
  return this->zigll;
}

Kokkos::View<double *, Kokkos::LayoutRight, Kokkos::HostSpace>
gll::gll::get_wxgll() const {
  return this->wxgll;
}

Kokkos::View<double *, Kokkos::LayoutRight, Kokkos::HostSpace>
gll::gll::get_wzgll() const {
  return this->wzgll;
}

Kokkos::View<double **, Kokkos::LayoutRight, Kokkos::HostSpace>
gll::gll::get_hprime_xx() const {
  return this->hprime_xx;
}

Kokkos::View<double **, Kokkos::LayoutRight, Kokkos::HostSpace>
gll::gll::get_hprime_zz() const {
  return this->hprime_zz;
}
