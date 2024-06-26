#ifndef _COMPUTE_PROPERTIES_IMPL_HPP
#define _COMPUTE_PROPERTIES_IMPL_HPP

#include "point/interface.hpp"

namespace specfem {
namespace compute {
namespace impl {

namespace properties {

template <specfem::element::medium_tag type,
          specfem::element::property_tag property>
struct properties_container {

  static_assert("Material type not implemented");
};

template <>
struct properties_container<specfem::element::medium_tag::elastic,
                            specfem::element::property_tag::isotropic> {

  constexpr static auto dimension = specfem::dimension::type::dim2;
  constexpr static auto value_type = specfem::element::medium_tag::elastic;
  constexpr static auto property_type =
      specfem::element::property_tag::isotropic;

  using ViewType = typename Kokkos::View<type_real ***, Kokkos::LayoutLeft,
                                         Kokkos::DefaultExecutionSpace>;

  int nspec; ///< total number of acoustic spectral elements
  int ngllz; ///< number of quadrature points in z dimension
  int ngllx; ///< number of quadrature points in x dimension
  ViewType rho;
  ViewType::HostMirror h_rho;
  ViewType mu;
  ViewType::HostMirror h_mu;
  ViewType lambdaplus2mu;
  ViewType::HostMirror h_lambdaplus2mu;

  properties_container() = default;

  properties_container(const int nspec, const int ngllz, const int ngllx)
      : nspec(nspec), ngllz(ngllz), ngllx(ngllx),
        rho("specfem::compute::properties::rho", nspec, ngllz, ngllx),
        h_rho(Kokkos::create_mirror_view(rho)),
        mu("specfem::compute::properties::mu", nspec, ngllz, ngllx),
        h_mu(Kokkos::create_mirror_view(mu)),
        lambdaplus2mu("specfem::compute::properties::lambdaplus2mu", nspec,
                      ngllz, ngllx),
        h_lambdaplus2mu(Kokkos::create_mirror_view(lambdaplus2mu)) {}

  KOKKOS_INLINE_FUNCTION void load_device_properties(
      const int &ispec, const int &iz, const int &ix,
      specfem::point::properties<dimension, value_type, property_type>
          &property) const {
    property.rho = rho(ispec, iz, ix);
    property.mu = mu(ispec, iz, ix);
    property.lambdaplus2mu = lambdaplus2mu(ispec, iz, ix);
    property.lambda = property.lambdaplus2mu - 2 * property.mu;
    property.rho_vp = sqrt(property.rho * property.lambdaplus2mu);
    property.rho_vs = sqrt(property.rho * property.mu);
  }

  void load_host_properties(
      const int &ispec, const int &iz, const int &ix,
      specfem::point::properties<dimension, value_type, property_type>
          &property) const {
    property.rho = h_rho(ispec, iz, ix);
    property.mu = h_mu(ispec, iz, ix);
    property.lambdaplus2mu = h_lambdaplus2mu(ispec, iz, ix);
    property.lambda = property.lambdaplus2mu - 2 * property.mu;
    property.rho_vp = sqrt(property.rho * property.lambdaplus2mu);
    property.rho_vs = sqrt(property.rho * property.mu);
  }

  void copy_to_device() {
    Kokkos::deep_copy(rho, h_rho);
    Kokkos::deep_copy(mu, h_mu);
    Kokkos::deep_copy(lambdaplus2mu, h_lambdaplus2mu);
  }

  void copy_to_host() {
    Kokkos::deep_copy(h_rho, rho);
    Kokkos::deep_copy(h_mu, mu);
    Kokkos::deep_copy(h_lambdaplus2mu, lambdaplus2mu);
  }

  void assign(const int ispec, const int iz, const int ix,
              const specfem::point::properties<dimension, value_type,
                                               property_type> &property) const {
    h_rho(ispec, iz, ix) = property.rho;
    h_mu(ispec, iz, ix) = property.mu;
    h_lambdaplus2mu(ispec, iz, ix) = property.lambdaplus2mu;
  }
};

template <>
struct properties_container<specfem::element::medium_tag::acoustic,
                            specfem::element::property_tag::isotropic> {

  constexpr static auto dimension = specfem::dimension::type::dim2;
  constexpr static auto value_type = specfem::element::medium_tag::acoustic;
  constexpr static auto property_type =
      specfem::element::property_tag::isotropic;

  using ViewType = typename Kokkos::View<type_real ***, Kokkos::LayoutLeft,
                                         Kokkos::DefaultExecutionSpace>;

  int nspec; ///< total number of acoustic spectral elements
  int ngllz; ///< number of quadrature points in z dimension
  int ngllx; ///< number of quadrature points in x dimension
  ViewType rho_inverse;
  ViewType::HostMirror h_rho_inverse;
  ViewType lambdaplus2mu_inverse;
  ViewType::HostMirror h_lambdaplus2mu_inverse;
  ViewType kappa;
  ViewType::HostMirror h_kappa;

  properties_container() = default;

  properties_container(const int nspec, const int ngllz, const int ngllx)
      : nspec(nspec), ngllz(ngllz), ngllx(ngllx),
        rho_inverse("specfem::compute::properties::rho_inverse", nspec, ngllz,
                    ngllx),
        h_rho_inverse(Kokkos::create_mirror_view(rho_inverse)),
        lambdaplus2mu_inverse(
            "specfem::compute::properties::lambdaplus2mu_inverse", nspec, ngllz,
            ngllx),
        h_lambdaplus2mu_inverse(
            Kokkos::create_mirror_view(lambdaplus2mu_inverse)),
        kappa("specfem::compute::properties::kappa", nspec, ngllz, ngllx),
        h_kappa(Kokkos::create_mirror_view(kappa)) {}

  KOKKOS_INLINE_FUNCTION void load_device_properties(
      const int &ispec, const int &iz, const int &ix,
      specfem::point::properties<dimension, value_type, property_type>
          &property) const {
    property.rho_inverse = rho_inverse(ispec, iz, ix);
    property.lambdaplus2mu_inverse = lambdaplus2mu_inverse(ispec, iz, ix);
    property.kappa = kappa(ispec, iz, ix);
    property.rho_vpinverse =
        sqrt(property.rho_inverse * property.lambdaplus2mu_inverse);
  }

  void load_host_properties(
      const int &ispec, const int &iz, const int &ix,
      specfem::point::properties<dimension, value_type, property_type>
          &property) const {
    property.rho_inverse = h_rho_inverse(ispec, iz, ix);
    property.lambdaplus2mu_inverse = h_lambdaplus2mu_inverse(ispec, iz, ix);
    property.kappa = h_kappa(ispec, iz, ix);
    property.rho_vpinverse =
        sqrt(property.rho_inverse * property.lambdaplus2mu_inverse);
  }

  void copy_to_device() {
    Kokkos::deep_copy(rho_inverse, h_rho_inverse);
    Kokkos::deep_copy(lambdaplus2mu_inverse, h_lambdaplus2mu_inverse);
    Kokkos::deep_copy(kappa, h_kappa);
  }

  void copy_to_host() {
    Kokkos::deep_copy(h_rho_inverse, rho_inverse);
    Kokkos::deep_copy(h_lambdaplus2mu_inverse, lambdaplus2mu_inverse);
    Kokkos::deep_copy(h_kappa, kappa);
  }

  void assign(const int ispec, const int iz, const int ix,
              const specfem::point::properties<dimension, value_type,
                                               property_type> &property) const {
    h_rho_inverse(ispec, iz, ix) = property.rho_inverse;
    h_lambdaplus2mu_inverse(ispec, iz, ix) = property.lambdaplus2mu_inverse;
    h_kappa(ispec, iz, ix) = property.kappa;
  }
};

} // namespace properties

} // namespace impl
} // namespace compute
} // namespace specfem

#endif /* _COMPUTE_PROPERTIES_IMPL_HPP */
