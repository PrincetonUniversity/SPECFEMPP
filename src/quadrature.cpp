#include "../include/quadrature.h"
#include "../include/config.h"
#include "../include/gll_library.h"
#include "../include/kokkos_abstractions.h"
#include "../include/lagrange_poly.h"
#include <Kokkos_Core.hpp>
#include <iostream>
#include <stdexcept>

using DeviceView1d = specfem::kokkos::DeviceView1d<type_real>;
using DeviceView2d = specfem::kokkos::DeviceView2d<type_real>;
using HostMirror1d = specfem::kokkos::HostMirror1d<type_real>;
using HostMirror2d = specfem::kokkos::HostMirror2d<type_real>;

void specfem::quadrature::quadrature::set_allocations() {
  xi = specfem::kokkos::DeviceView1d<type_real>(
      "specfem::quadrature::quadrature::DeviceView1d::xi", N);
  h_xi = Kokkos::create_mirror_view(xi);
  w = specfem::kokkos::DeviceView1d<type_real>(
      "specfem::quadrature::quadrature::DeviceView1d::w", N);
  h_w = Kokkos::create_mirror_view(w);
  hprime = specfem::kokkos::DeviceView2d<type_real>(
      "specfem::quadrature::quadrature::DeviceView1d::hprime", N, N);
  h_hprime = Kokkos::create_mirror_view(hprime);
}

void specfem::quadrature::quadrature::sync_views() {
  Kokkos::deep_copy(xi, h_xi);
  Kokkos::deep_copy(w, h_w);
  Kokkos::deep_copy(hprime, h_hprime);
}

specfem::quadrature::quadrature::quadrature() : alpha(0.0), beta(0.0), N(5) {
  this->set_allocations();
  this->set_derivation_matrices();
};

specfem::quadrature::quadrature::quadrature(const type_real alpha,
                                            const type_real beta)
    : alpha(alpha), beta(beta), N(5) {
  if ((alpha <= -1.0) || (beta <= -1.0))
    throw std::invalid_argument("alpha and beta must be greater than -1");
  this->set_allocations();
  this->set_derivation_matrices();
}

specfem::quadrature::quadrature::quadrature(const type_real alpha,
                                            const type_real beta,
                                            const int ngll)
    : alpha(alpha), beta(beta), N(ngll) {
  if (ngll <= 1)
    throw std::invalid_argument("Minimum number of Gauss-Labatto points is 2");
  if (ngll <= 2)
    throw std::invalid_argument(
        "Minimum number of Gauss-Lobatto points for the SEM is 3");
  if ((alpha <= -1.0) || (beta <= -1.0))
    throw std::invalid_argument("alpha and beta must be greater than -1");
  this->set_allocations();
  this->set_derivation_matrices();
}

void specfem::quadrature::quadrature::set_derivation_matrices() {
  gll_library::zwgljd(this->h_xi, this->h_w, this->N, this->alpha, this->beta);
  Lagrange::compute_lagrange_derivatives_GLL(this->h_hprime, this->h_xi,
                                             this->N);
  this->sync_views();
}

DeviceView1d specfem::quadrature::quadrature::get_xi() const {
  return this->xi;
}

DeviceView1d specfem::quadrature::quadrature::get_w() const { return this->w; }

DeviceView2d specfem::quadrature::quadrature::get_hprime() const {
  return this->hprime;
}

HostMirror1d specfem::quadrature::quadrature::get_hxi() const {
  return this->h_xi;
}

HostMirror1d specfem::quadrature::quadrature::get_hw() const {
  return this->h_w;
}

HostMirror2d specfem::quadrature::quadrature::get_hhprime() const {
  return this->h_hprime;
}

int specfem::quadrature::quadrature::get_N() const { return this->N; };
