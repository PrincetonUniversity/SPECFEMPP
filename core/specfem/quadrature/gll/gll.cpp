#include "specfem/quadrature/gll.hpp"
#include "gll_library.hpp"
#include "kokkos_abstractions.h"
#include "specfem_setup.hpp"
#include <Kokkos_Core.hpp>
#include <iostream>
#include <stdexcept>

using DeviceView1d = Kokkos::View<type_real *, Kokkos::LayoutRight,
                                  Kokkos::DefaultExecutionSpace>;
using DeviceView2d = Kokkos::View<type_real **, Kokkos::LayoutRight,
                                  Kokkos::DefaultExecutionSpace>;
using HostMirror1d =
    Kokkos::View<type_real *, Kokkos::LayoutRight, Kokkos::HostSpace>;
using HostMirror2d =
    Kokkos::View<type_real **, Kokkos::LayoutRight, Kokkos::HostSpace>;

void specfem::quadrature::gll::gll::set_allocations() {
  xi = DeviceView1d("specfem::quadrature::quadrature::DeviceView1d::xi", N);
  h_xi = Kokkos::create_mirror_view(xi);
  w = DeviceView1d("specfem::quadrature::quadrature::DeviceView1d::w", N);
  h_w = Kokkos::create_mirror_view(w);
  hprime = DeviceView2d("specfem::quadrature::quadrature::DeviceView1d::hprime",
                        N, N);
  h_hprime = Kokkos::create_mirror_view(hprime);
}

void specfem::quadrature::gll::gll::sync_views() {
  Kokkos::deep_copy(xi, h_xi);
  Kokkos::deep_copy(w, h_w);
  Kokkos::deep_copy(hprime, h_hprime);
}

specfem::quadrature::gll::gll::gll() : alpha(0.0), beta(0.0), N(5) {
  this->set_allocations();
  this->set_derivation_matrices();
};

specfem::quadrature::gll::gll::gll(const type_real alpha, const type_real beta)
    : alpha(alpha), beta(beta), N(5) {
  if ((alpha <= -1.0) || (beta <= -1.0))
    throw std::invalid_argument("alpha and beta must be greater than -1");
  this->set_allocations();
  this->set_derivation_matrices();
}

specfem::quadrature::gll::gll::gll(const type_real alpha, const type_real beta,
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

void specfem::quadrature::gll::gll::set_derivation_matrices() {
  specfem::quadrature::gll::gll_library::zwgljd(this->h_xi, this->h_w, this->N,
                                                this->alpha, this->beta);
  specfem::quadrature::gll::Lagrange::compute_lagrange_derivatives_GLL(
      this->h_hprime, this->h_xi, this->N);
  this->sync_views();
}

std::string specfem::quadrature::gll::gll::to_string() const {
  std::ostringstream message;
  message << "- GLL\n"
          << "    alpha = " << this->alpha << "\n"
          << "    beta = " << this->beta << "\n"
          << "    Number of GLL points = " << this->N << "\n";
  return message.str();
}

void specfem::quadrature::gll::gll::print(std::ostream &message) const {
  message << this->to_string();
}
