#ifndef _SPECFEM_COMPUTE_KERNELS_IMPL_KERNELS_CONTAINER_HPP_
#define _SPECFEM_COMPUTE_KERNELS_IMPL_KERNELS_CONTAINER_HPP_

#include "enumerations/medium.hpp"
#include "kokkos_abstractions.h"
#include "point/kernels.hpp"
#include <Kokkos_Core.hpp>

namespace specfem {
namespace compute {
namespace impl {
namespace kernels {
template <specfem::element::medium_tag MediumTag,
          specfem::element::property_tag PropertyTag>
class kernels_container;

template <>
class kernels_container<specfem::element::medium_tag::elastic,
                        specfem::element::property_tag::isotropic> {
public:
  constexpr static auto value_type = specfem::element::medium_tag::elastic;
  constexpr static auto property_type =
      specfem::element::property_tag::isotropic;
  specfem::kokkos::DeviceView3d<type_real> rho;
  specfem::kokkos::HostMirror3d<type_real> h_rho;
  specfem::kokkos::DeviceView3d<type_real> mu;
  specfem::kokkos::HostMirror3d<type_real> h_mu;
  specfem::kokkos::DeviceView3d<type_real> kappa;
  specfem::kokkos::HostMirror3d<type_real> h_kappa;
  specfem::kokkos::DeviceView3d<type_real> rhop;
  specfem::kokkos::HostMirror3d<type_real> h_rhop;
  specfem::kokkos::DeviceView3d<type_real> alpha;
  specfem::kokkos::HostMirror3d<type_real> h_alpha;
  specfem::kokkos::DeviceView3d<type_real> beta;
  specfem::kokkos::HostMirror3d<type_real> h_beta;

  kernels_container() = default;

  kernels_container(const int nspec, const int ngllz, const int ngllx)
      : rho("specfem::compute::impl::kernels::elastic::rho", nspec, ngllz,
            ngllx),
        mu("specfem::compute::impl::kernels::elastic::mu", nspec, ngllz, ngllx),
        kappa("specfem::compute::impl::kernels::elastic::kappa", nspec, ngllz,
              ngllx),
        rhop("specfem::compute::impl::kernels::elastic::rhop", nspec, ngllz,
             ngllx),
        alpha("specfem::compute::impl::kernels::elastic::alpha", nspec, ngllz,
              ngllx),
        beta("specfem::compute::impl::kernels::elastic::beta", nspec, ngllz,
             ngllx),
        h_rho(Kokkos::create_mirror_view(rho)),
        h_mu(Kokkos::create_mirror_view(mu)),
        h_kappa(Kokkos::create_mirror_view(kappa)),
        h_rhop(Kokkos::create_mirror_view(rhop)),
        h_alpha(Kokkos::create_mirror_view(alpha)),
        h_beta(Kokkos::create_mirror_view(beta)) {

    Kokkos::parallel_for(
        "specfem::compute::impl::kernels::elastic::initialize",
        Kokkos::MDRangePolicy<Kokkos::Rank<3> >({ 0, 0, 0 },
                                                { nspec, ngllz, ngllx }),
        KOKKOS_LAMBDA(const int ispec, const int iz, const int ix) {
          rho(ispec, iz, ix) = 0.0;
          mu(ispec, iz, ix) = 0.0;
          kappa(ispec, iz, ix) = 0.0;
          rhop(ispec, iz, ix) = 0.0;
          alpha(ispec, iz, ix) = 0.0;
          beta(ispec, iz, ix) = 0.0;
        });
  }

  KOKKOS_INLINE_FUNCTION void load_device_kernels(
      const int ispec, const int iz, const int ix,
      specfem::point::kernels<value_type, property_type> &kernels) const {
    kernels.rho = rho(ispec, iz, ix);
    kernels.mu = mu(ispec, iz, ix);
    kernels.kappa = kappa(ispec, iz, ix);
    kernels.rhop = rhop(ispec, iz, ix);
    kernels.alpha = alpha(ispec, iz, ix);
    kernels.beta = beta(ispec, iz, ix);
  }

  void load_host_kernels(
      const int ispec, const int iz, const int ix,
      specfem::point::kernels<value_type, property_type> &kernels) const {
    kernels.rho = h_rho(ispec, iz, ix);
    kernels.mu = h_mu(ispec, iz, ix);
    kernels.kappa = h_kappa(ispec, iz, ix);
    kernels.rhop = h_rhop(ispec, iz, ix);
    kernels.alpha = h_alpha(ispec, iz, ix);
    kernels.beta = h_beta(ispec, iz, ix);
  }

  KOKKOS_INLINE_FUNCTION void update_kernels_on_device(
      const int ispec, const int iz, const int ix,
      const specfem::point::kernels<value_type, property_type> &kernels) {
    rho(ispec, iz, ix) = kernels.rho;
    mu(ispec, iz, ix) = kernels.mu;
    kappa(ispec, iz, ix) = kernels.kappa;
    rhop(ispec, iz, ix) = kernels.rhop;
    alpha(ispec, iz, ix) = kernels.alpha;
    beta(ispec, iz, ix) = kernels.beta;
  }

  void update_kernels_on_host(
      const int ispec, const int iz, const int ix,
      const specfem::point::kernels<value_type, property_type> &kernels) {
    h_rho(ispec, iz, ix) = kernels.rho;
    h_mu(ispec, iz, ix) = kernels.mu;
    h_kappa(ispec, iz, ix) = kernels.kappa;
    h_rhop(ispec, iz, ix) = kernels.rhop;
    h_alpha(ispec, iz, ix) = kernels.alpha;
    h_beta(ispec, iz, ix) = kernels.beta;
  }

  KOKKOS_INLINE_FUNCTION void add_kernels_on_device(
      const int ispec, const int iz, const int ix,
      const specfem::point::kernels<value_type, property_type> &kernels) {
    rho(ispec, iz, ix) += kernels.rho;
    mu(ispec, iz, ix) += kernels.mu;
    kappa(ispec, iz, ix) += kernels.kappa;
    rhop(ispec, iz, ix) += kernels.rhop;
    alpha(ispec, iz, ix) += kernels.alpha;
    beta(ispec, iz, ix) += kernels.beta;
  }

  void add_kernels_on_host(
      const int ispec, const int iz, const int ix,
      const specfem::point::kernels<value_type, property_type> &kernels) {
    h_rho(ispec, iz, ix) += kernels.rho;
    h_mu(ispec, iz, ix) += kernels.mu;
    h_kappa(ispec, iz, ix) += kernels.kappa;
    h_rhop(ispec, iz, ix) += kernels.rhop;
    h_alpha(ispec, iz, ix) += kernels.alpha;
    h_beta(ispec, iz, ix) += kernels.beta;
  }

  void copy_to_host() {
    Kokkos::deep_copy(h_rho, rho);
    Kokkos::deep_copy(h_mu, mu);
    Kokkos::deep_copy(h_kappa, kappa);
    Kokkos::deep_copy(h_rhop, rhop);
    Kokkos::deep_copy(h_alpha, alpha);
    Kokkos::deep_copy(h_beta, beta);
  }

  void copy_to_device() {
    Kokkos::deep_copy(rho, h_rho);
    Kokkos::deep_copy(mu, h_mu);
    Kokkos::deep_copy(kappa, h_kappa);
    Kokkos::deep_copy(rhop, h_rhop);
    Kokkos::deep_copy(alpha, h_alpha);
    Kokkos::deep_copy(beta, h_beta);
  }
};

template <>
class kernels_container<specfem::element::medium_tag::acoustic,
                        specfem::element::property_tag::isotropic> {
public:
  constexpr static auto value_type = specfem::element::medium_tag::acoustic;
  constexpr static auto property_type =
      specfem::element::property_tag::isotropic;
  specfem::kokkos::DeviceView3d<type_real> rho;
  specfem::kokkos::HostMirror3d<type_real> h_rho;
  specfem::kokkos::DeviceView3d<type_real> kappa;
  specfem::kokkos::HostMirror3d<type_real> h_kappa;
  specfem::kokkos::DeviceView3d<type_real> rho_prime;
  specfem::kokkos::HostMirror3d<type_real> h_rho_prime;
  specfem::kokkos::DeviceView3d<type_real> alpha;
  specfem::kokkos::HostMirror3d<type_real> h_alpha;

  kernels_container() = default;

  kernels_container(const int nspec, const int ngllz, const int ngllx)
      : rho("specfem::compute::impl::kernels::acoustic::rho", nspec, ngllz,
            ngllx),
        kappa("specfem::compute::impl::kernels::acoustic::kappa", nspec, ngllz,
              ngllx),
        rho_prime("specfem::compute::impl::kernels::acoustic::rho_prime", nspec,
                  ngllz, ngllx),
        alpha("specfem::compute::impl::kernels::acoustic::alpha", nspec, ngllz,
              ngllx),
        h_rho(Kokkos::create_mirror_view(rho)),
        h_kappa(Kokkos::create_mirror_view(kappa)),
        h_rho_prime(Kokkos::create_mirror_view(rho_prime)),
        h_alpha(Kokkos::create_mirror_view(alpha)) {}

  KOKKOS_INLINE_FUNCTION void load_device_kernels(
      const int ispec, const int iz, const int ix,
      specfem::point::kernels<value_type, property_type> &kernels) const {
    kernels.rho = rho(ispec, iz, ix);
    kernels.kappa = kappa(ispec, iz, ix);
    kernels.rho_prime = rho_prime(ispec, iz, ix);
    kernels.alpha = alpha(ispec, iz, ix);
  }

  void load_host_kernels(
      const int ispec, const int iz, const int ix,
      specfem::point::kernels<value_type, property_type> &kernels) const {
    kernels.rho = h_rho(ispec, iz, ix);
    kernels.kappa = h_kappa(ispec, iz, ix);
    kernels.rho_prime = h_rho_prime(ispec, iz, ix);
    kernels.alpha = h_alpha(ispec, iz, ix);
  }

  KOKKOS_INLINE_FUNCTION void update_kernels_on_device(
      const int ispec, const int iz, const int ix,
      const specfem::point::kernels<value_type, property_type> &kernels) {
    rho(ispec, iz, ix) = kernels.rho;
    kappa(ispec, iz, ix) = kernels.kappa;
    rho_prime(ispec, iz, ix) = kernels.rho_prime;
    alpha(ispec, iz, ix) = kernels.alpha;
  }

  void update_kernels_on_host(
      const int ispec, const int iz, const int ix,
      const specfem::point::kernels<value_type, property_type> &kernels) {
    h_rho(ispec, iz, ix) = kernels.rho;
    h_kappa(ispec, iz, ix) = kernels.kappa;
    h_rho_prime(ispec, iz, ix) = kernels.rho_prime;
    h_alpha(ispec, iz, ix) = kernels.alpha;
  }

  KOKKOS_INLINE_FUNCTION void add_kernels_on_device(
      const int ispec, const int iz, const int ix,
      const specfem::point::kernels<value_type, property_type> &kernels) {
    rho(ispec, iz, ix) += kernels.rho;
    kappa(ispec, iz, ix) += kernels.kappa;
    rho_prime(ispec, iz, ix) += kernels.rho_prime;
    alpha(ispec, iz, ix) += kernels.alpha;
  }

  void add_kernels_on_host(
      const int ispec, const int iz, const int ix,
      const specfem::point::kernels<value_type, property_type> &kernels) {
    h_rho(ispec, iz, ix) += kernels.rho;
    h_kappa(ispec, iz, ix) += kernels.kappa;
    h_rho_prime(ispec, iz, ix) += kernels.rho_prime;
    h_alpha(ispec, iz, ix) += kernels.alpha;
  }

  void copy_to_host() {
    Kokkos::deep_copy(h_rho, rho);
    Kokkos::deep_copy(h_kappa, kappa);
    Kokkos::deep_copy(h_rho_prime, rho_prime);
    Kokkos::deep_copy(h_alpha, alpha);
  }

  void copy_to_device() {
    Kokkos::deep_copy(rho, h_rho);
    Kokkos::deep_copy(kappa, h_kappa);
    Kokkos::deep_copy(rho_prime, h_rho_prime);
    Kokkos::deep_copy(alpha, h_alpha);
  }
};

} // namespace kernels
} // namespace impl
} // namespace compute
} // namespace specfem

#endif /* _SPECFEM_COMPUTE_KERNELS_IMPL_KERNELS_CONTAINER_HPP_ */
