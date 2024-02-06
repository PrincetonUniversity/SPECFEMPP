#ifndef _COMPUTE_PROPERTIES_IMPL_HPP
#define _COMPUTE_PROPERTIES_IMPL_HPP

#include "point/interface.hpp"

namespace specfem {
namespace compute {
namespace impl {

namespace properties {

template <specfem::enums::element::type type,
          specfem::enums::element::property_tag property>
struct properties_container {

  static_assert("Material type not implemented");
};

template <>
struct properties_container<specfem::enums::element::type::elastic,
                            specfem::enums::element::property_tag::isotropic> {

  constexpr static auto value_type = specfem::enums::element::type::elastic;
  constexpr static auto property_type =
      specfem::enums::element::property_tag::isotropic;

  int nspec; ///< total number of acoustic spectral elements
  int ngllz; ///< number of quadrature points in z dimension
  int ngllx; ///< number of quadrature points in x dimension
  specfem::kokkos::DeviceView3d<type_real> rho;
  specfem::kokkos::HostMirror3d<type_real> h_rho;
  specfem::kokkos::DeviceView3d<type_real> mu;
  specfem::kokkos::HostMirror3d<type_real> h_mu;
  specfem::kokkos::DeviceView3d<type_real> lambdaplus2mu;
  specfem::kokkos::HostMirror3d<type_real> h_lambdaplus2mu;

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

  template <typename ExecSpace>
  KOKKOS_INLINE_FUNCTION specfem::point::properties<value_type, property_type>
  load_properties(const int &ispec, const int &iz, const int &ix) const {
    if constexpr (std::is_same_v<ExecSpace, specfem::kokkos::DevExecSpace>) {
      return specfem::point::properties<value_type, property_type>(
          lambdaplus2mu(ispec, iz, ix), mu(ispec, iz, ix), rho(ispec, iz, ix));
    } else {
      return specfem::point::properties<value_type, property_type>(
          h_lambdaplus2mu(ispec, iz, ix), h_mu(ispec, iz, ix),
          h_rho(ispec, iz, ix));
    }
  }

  void copy_to_device() {
    Kokkos::deep_copy(rho, h_rho);
    Kokkos::deep_copy(mu, h_mu);
    Kokkos::deep_copy(lambdaplus2mu, h_lambdaplus2mu);
  }

  void assign(
      const int ispec, const int iz, const int ix,
      const specfem::point::properties<value_type, property_type> &property) {
    h_rho(ispec, iz, ix) = property.rho;
    h_mu(ispec, iz, ix) = property.mu;
    h_lambdaplus2mu(ispec, iz, ix) = property.lambdaplus2mu;
  }
};

template <>
struct properties_container<specfem::enums::element::type::acoustic,
                            specfem::enums::element::property_tag::isotropic> {

  constexpr static auto value_type = specfem::enums::element::type::acoustic;
  constexpr static auto property_type =
      specfem::enums::element::property_tag::isotropic;

  int nspec; ///< total number of acoustic spectral elements
  int ngllz; ///< number of quadrature points in z dimension
  int ngllx; ///< number of quadrature points in x dimension
  specfem::kokkos::DeviceView3d<type_real> rho_inverse;
  specfem::kokkos::HostMirror3d<type_real> h_rho_inverse;
  specfem::kokkos::DeviceView3d<type_real> lambdaplus2mu_inverse;
  specfem::kokkos::HostMirror3d<type_real> h_lambdaplus2mu_inverse;
  specfem::kokkos::DeviceView3d<type_real> kappa;
  specfem::kokkos::HostMirror3d<type_real> h_kappa;

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

  template <typename ExecSpace>
  KOKKOS_INLINE_FUNCTION specfem::point::properties<value_type, property_type>
  load_properties(const int &ispec, const int &iz, const int &ix) const {
    if constexpr (std::is_same_v<ExecSpace, specfem::kokkos::DevExecSpace>) {
      return specfem::point::properties<value_type, property_type>(
          lambdaplus2mu_inverse(ispec, iz, ix), rho_inverse(ispec, iz, ix),
          kappa(ispec, iz, ix));
    } else {
      return specfem::point::properties<value_type, property_type>(
          h_lambdaplus2mu_inverse(ispec, iz, ix), h_rho_inverse(ispec, iz, ix),
          h_kappa(ispec, iz, ix));
    }
  }

  void copy_to_device() {
    Kokkos::deep_copy(rho_inverse, h_rho_inverse);
    Kokkos::deep_copy(lambdaplus2mu_inverse, h_lambdaplus2mu_inverse);
    Kokkos::deep_copy(kappa, h_kappa);
  }

  void assign(
      const int ispec, const int iz, const int ix,
      const specfem::point::properties<value_type, property_type> &property) {
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
