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

  kernels_container() = default;

  kernels_container(const int nspec, const int ngllz, const int ngllx)
      : rho("specfem::compute::impl::kernels::elastic::rho", nspec, ngllz,
            ngllx),
        mu("specfem::compute::impl::kernels::elastic::mu", nspec, ngllz, ngllx),
        kappa("specfem::compute::impl::kernels::elastic::kappa", nspec, ngllz,
              ngllx),
        h_rho(Kokkos::create_mirror_view(rho)),
        h_mu(Kokkos::create_mirror_view(mu)),
        h_kappa(Kokkos::create_mirror_view(kappa)) {}

  KOKKOS_FUNCTION void update_kernels(
      const int ispec, const int iz, const int ix,
      const specfem::point::kernels<value_type, property_type> &kernels) const {
    rho(ispec, iz, ix) = kernels.rho;
    mu(ispec, iz, ix) = kernels.mu;
    kappa(ispec, iz, ix) = kernels.kappa;
  }

  void sync_views() {
    Kokkos::deep_copy(h_rho, rho);
    Kokkos::deep_copy(h_mu, mu);
    Kokkos::deep_copy(h_kappa, kappa);
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

  KOKKOS_FUNCTION void update_kernels(
      const int ispec, const int iz, const int ix,
      const specfem::point::kernels<value_type, property_type> &kernels) const {
    rho(ispec, iz, ix) = kernels.rho;
    kappa(ispec, iz, ix) = kernels.kappa;
    rho_prime(ispec, iz, ix) = kernels.rho * kernels.kappa;
    alpha(ispec, iz, ix) = 2.0 * kernels.kappa;
  }

  void sync_views() {
    Kokkos::deep_copy(h_rho, rho);
    Kokkos::deep_copy(h_kappa, kappa);
    Kokkos::deep_copy(h_rho_prime, rho_prime);
    Kokkos::deep_copy(h_alpha, alpha);
  }
};

} // namespace kernels
} // namespace impl
} // namespace compute
} // namespace specfem

#endif /* _SPECFEM_COMPUTE_KERNELS_IMPL_KERNELS_CONTAINER_HPP_ */
