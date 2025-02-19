#pragma once

#include "medium/properties_container.hpp"
#include "point/interface.hpp"
#include <Kokkos_SIMD.hpp>

namespace specfem {
namespace medium {

template <>
struct properties_container<specfem::element::medium_tag::elastic_sv,
                            specfem::element::property_tag::isotropic> {

  constexpr static auto dimension = specfem::dimension::type::dim2;
  constexpr static auto value_type = specfem::element::medium_tag::elastic_sv;
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

  template <
      typename PointProperties,
      typename std::enable_if_t<!PointProperties::simd::using_simd, int> = 0>
  KOKKOS_FORCEINLINE_FUNCTION void
  load_device_properties(const specfem::point::index<dimension> &index,
                         PointProperties &property) const {

    static_assert(PointProperties::dimension == dimension,
                  "Dimension mismatch");
    static_assert(PointProperties::medium_tag == value_type,
                  "Medium tag mismatch");
    static_assert(PointProperties::property_tag == property_type,
                  "Property tag mismatch");

    const int ispec = index.ispec;
    const int iz = index.iz;
    const int ix = index.ix;

    property.rho = rho(ispec, iz, ix);
    property.mu = mu(ispec, iz, ix);
    property.lambdaplus2mu = lambdaplus2mu(ispec, iz, ix);
    property.lambda = property.lambdaplus2mu - 2 * property.mu;
    property.rho_vp = sqrt(property.rho * property.lambdaplus2mu);
    property.rho_vs = sqrt(property.rho * property.mu);
  }

  template <
      typename PointProperties,
      typename std::enable_if_t<PointProperties::simd::using_simd, int> = 0>
  KOKKOS_FORCEINLINE_FUNCTION void
  load_device_properties(const specfem::point::simd_index<dimension> &index,
                         PointProperties &property) const {

    static_assert(PointProperties::dimension == dimension,
                  "Dimension mismatch");
    static_assert(PointProperties::medium_tag == value_type,
                  "Medium tag mismatch");
    static_assert(PointProperties::property_tag == property_type,
                  "Property tag mismatch");

    using simd = typename PointProperties::simd;
    using mask_type = typename simd::mask_type;
    using tag_type = typename simd::tag_type;

    const int ispec = index.ispec;
    const int iz = index.iz;
    const int ix = index.ix;

    mask_type mask([&](std::size_t lane) { return index.mask(lane); });

    Kokkos::Experimental::where(mask, property.rho)
        .copy_from(&rho(ispec, iz, ix), tag_type());
    Kokkos::Experimental::where(mask, property.mu)
        .copy_from(&mu(ispec, iz, ix), tag_type());
    Kokkos::Experimental::where(mask, property.lambdaplus2mu)
        .copy_from(&lambdaplus2mu(ispec, iz, ix), tag_type());

    property.lambda = property.lambdaplus2mu - 2 * property.mu;
    property.rho_vp = Kokkos::sqrt(property.rho * property.lambdaplus2mu);
    property.rho_vs = Kokkos::sqrt(property.rho * property.mu);
  }

  template <
      typename PointProperties,
      typename std::enable_if_t<!PointProperties::simd::using_simd, int> = 0>
  inline void
  load_host_properties(const specfem::point::index<dimension> &index,
                       PointProperties &property) const {

    static_assert(PointProperties::dimension == dimension,
                  "Dimension mismatch");
    static_assert(PointProperties::medium_tag == value_type,
                  "Medium tag mismatch");
    static_assert(PointProperties::property_tag == property_type,
                  "Property tag mismatch");

    const int ispec = index.ispec;
    const int iz = index.iz;
    const int ix = index.ix;

    property.rho = h_rho(ispec, iz, ix);
    property.mu = h_mu(ispec, iz, ix);
    property.lambdaplus2mu = h_lambdaplus2mu(ispec, iz, ix);
    property.lambda = property.lambdaplus2mu - 2 * property.mu;
    property.rho_vp = sqrt(property.rho * property.lambdaplus2mu);
    property.rho_vs = sqrt(property.rho * property.mu);
  }

  template <
      typename PointProperties,
      typename std::enable_if_t<PointProperties::simd::using_simd, int> = 0>
  inline void
  load_host_properties(const specfem::point::simd_index<dimension> &index,
                       PointProperties &property) const {

    static_assert(PointProperties::dimension == dimension,
                  "Dimension mismatch");
    static_assert(PointProperties::medium_tag == value_type,
                  "Medium tag mismatch");
    static_assert(PointProperties::property_tag == property_type,
                  "Property tag mismatch");

    using simd = typename PointProperties::simd;
    using mask_type = typename simd::mask_type;
    using tag_type = typename simd::tag_type;

    const int ispec = index.ispec;
    const int iz = index.iz;
    const int ix = index.ix;

    mask_type mask([&](std::size_t lane) { return index.mask(lane); });

    Kokkos::Experimental::where(mask, property.rho)
        .copy_from(&h_rho(ispec, iz, ix), tag_type());
    Kokkos::Experimental::where(mask, property.mu)
        .copy_from(&h_mu(ispec, iz, ix), tag_type());
    Kokkos::Experimental::where(mask, property.lambdaplus2mu)
        .copy_from(&h_lambdaplus2mu(ispec, iz, ix), tag_type());

    property.lambda = property.lambdaplus2mu - 2 * property.mu;
    property.rho_vp = Kokkos::sqrt(property.rho * property.lambdaplus2mu);
    property.rho_vs = Kokkos::sqrt(property.rho * property.mu);
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

  template <
      typename PointProperties,
      typename std::enable_if_t<!PointProperties::simd::using_simd, int> = 0>
  inline void assign(const specfem::point::index<dimension> &index,
                     const PointProperties &property) const {

    static_assert(PointProperties::dimension == dimension,
                  "Dimension mismatch");
    static_assert(PointProperties::medium_tag == value_type,
                  "Medium tag mismatch");
    static_assert(PointProperties::property_tag == property_type,
                  "Property tag mismatch");

    const int ispec = index.ispec;
    const int iz = index.iz;
    const int ix = index.ix;

    h_rho(ispec, iz, ix) = property.rho;
    h_mu(ispec, iz, ix) = property.mu;
    h_lambdaplus2mu(ispec, iz, ix) = property.lambdaplus2mu;
  }

  template <
      typename PointProperties,
      typename std::enable_if_t<PointProperties::simd::using_simd, int> = 0>
  inline void assign(const specfem::point::simd_index<dimension> &index,
                     const PointProperties &property) const {

    static_assert(PointProperties::dimension == dimension,
                  "Dimension mismatch");
    static_assert(PointProperties::medium_tag == value_type,
                  "Medium tag mismatch");
    static_assert(PointProperties::property_tag == property_type,
                  "Property tag mismatch");

    using simd = typename PointProperties::simd;
    using mask_type = typename simd::mask_type;
    using tag_type = typename simd::tag_type;

    const int ispec = index.ispec;
    const int iz = index.iz;
    const int ix = index.ix;

    mask_type mask([&, this](std::size_t lane) { return index.mask(lane); });

    Kokkos::Experimental::where(mask, property.rho)
        .copy_to(&h_rho(ispec, iz, ix), tag_type());
    Kokkos::Experimental::where(mask, property.mu)
        .copy_to(&h_mu(ispec, iz, ix), tag_type());
    Kokkos::Experimental::where(mask, property.lambdaplus2mu)
        .copy_to(&h_lambdaplus2mu(ispec, iz, ix), tag_type());
  }
};

} // namespace medium
} // namespace specfem
