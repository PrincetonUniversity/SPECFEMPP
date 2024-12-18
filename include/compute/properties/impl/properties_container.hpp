#ifndef _COMPUTE_PROPERTIES_IMPL_HPP
#define _COMPUTE_PROPERTIES_IMPL_HPP

#include "point/interface.hpp"
#include <Kokkos_SIMD.hpp>

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

template <>
struct properties_container<specfem::element::medium_tag::elastic,
                            specfem::element::property_tag::anisotropic> {

  constexpr static auto dimension = specfem::dimension::type::dim2;
  constexpr static auto value_type = specfem::element::medium_tag::elastic;
  constexpr static auto property_type =
      specfem::element::property_tag::anisotropic;

  using ViewType = typename Kokkos::View<type_real ***, Kokkos::LayoutLeft,
                                         Kokkos::DefaultExecutionSpace>;

  int nspec; ///< total number of acoustic spectral elements
  int ngllz; ///< number of quadrature points in z dimension
  int ngllx; ///< number of quadrature points in x dimension

  ViewType rho;
  ViewType::HostMirror h_rho;
  ViewType c11;
  ViewType::HostMirror h_c11;
  ViewType c13;
  ViewType::HostMirror h_c13;
  ViewType c15;
  ViewType::HostMirror h_c15;
  ViewType c33;
  ViewType::HostMirror h_c33;
  ViewType c35;
  ViewType::HostMirror h_c35;
  ViewType c55;
  ViewType::HostMirror h_c55;
  ViewType c12;
  ViewType::HostMirror h_c12;
  ViewType c23;
  ViewType::HostMirror h_c23;
  ViewType c25;
  ViewType::HostMirror h_c25;

  properties_container() = default;

  properties_container(const int nspec, const int ngllz, const int ngllx)
      : nspec(nspec), ngllz(ngllz), ngllx(ngllx),
        rho("specfem::compute::properties::rho", nspec, ngllz, ngllx),
        h_rho(Kokkos::create_mirror_view(rho)),
        c11("specfem::compute::properties::c11", nspec, ngllz, ngllx),
        h_c11(Kokkos::create_mirror_view(c11)),
        c12("specfem::compute::properties::c12", nspec, ngllz, ngllx),
        h_c12(Kokkos::create_mirror_view(c12)),
        c13("specfem::compute::properties::c13", nspec, ngllz, ngllx),
        h_c13(Kokkos::create_mirror_view(c13)),
        c15("specfem::compute::properties::c15", nspec, ngllz, ngllx),
        h_c15(Kokkos::create_mirror_view(c15)),
        c33("specfem::compute::properties::c33", nspec, ngllz, ngllx),
        h_c33(Kokkos::create_mirror_view(c33)),
        c35("specfem::compute::properties::c35", nspec, ngllz, ngllx),
        h_c35(Kokkos::create_mirror_view(c35)),
        c55("specfem::compute::properties::c55", nspec, ngllz, ngllx),
        h_c55(Kokkos::create_mirror_view(c55)),
        c23("specfem::compute::properties::c23", nspec, ngllz, ngllx),
        h_c23(Kokkos::create_mirror_view(c23)),
        c25("specfem::compute::properties::c25", nspec, ngllz, ngllx),
        h_c25(Kokkos::create_mirror_view(c25)) {}

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
    property.c11 = c11(ispec, iz, ix);
    property.c12 = c12(ispec, iz, ix);
    property.c13 = c13(ispec, iz, ix);
    property.c15 = c15(ispec, iz, ix);
    property.c33 = c33(ispec, iz, ix);
    property.c35 = c35(ispec, iz, ix);
    property.c55 = c55(ispec, iz, ix);
    property.c23 = c23(ispec, iz, ix);
    property.c25 = c25(ispec, iz, ix);
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
    Kokkos::Experimental::where(mask, property.c11)
        .copy_from(&c11(ispec, iz, ix), tag_type());
    Kokkos::Experimental::where(mask, property.c12)
        .copy_from(&c12(ispec, iz, ix), tag_type());
    Kokkos::Experimental::where(mask, property.c13)
        .copy_from(&c13(ispec, iz, ix), tag_type());
    Kokkos::Experimental::where(mask, property.c15)
        .copy_from(&c15(ispec, iz, ix), tag_type());
    Kokkos::Experimental::where(mask, property.c33)
        .copy_from(&c33(ispec, iz, ix), tag_type());
    Kokkos::Experimental::where(mask, property.c35)
        .copy_from(&c35(ispec, iz, ix), tag_type());
    Kokkos::Experimental::where(mask, property.c55)
        .copy_from(&c55(ispec, iz, ix), tag_type());
    Kokkos::Experimental::where(mask, property.c23)
        .copy_from(&c23(ispec, iz, ix), tag_type());
    Kokkos::Experimental::where(mask, property.c25)
        .copy_from(&c25(ispec, iz, ix), tag_type());
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
    property.c11 = h_c11(ispec, iz, ix);
    property.c12 = h_c12(ispec, iz, ix);
    property.c13 = h_c13(ispec, iz, ix);
    property.c15 = h_c15(ispec, iz, ix);
    property.c33 = h_c33(ispec, iz, ix);
    property.c35 = h_c35(ispec, iz, ix);
    property.c55 = h_c55(ispec, iz, ix);
    property.c23 = h_c23(ispec, iz, ix);
    property.c25 = h_c25(ispec, iz, ix);
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
    Kokkos::Experimental::where(mask, property.c11)
        .copy_from(&h_c11(ispec, iz, ix), tag_type());
    Kokkos::Experimental::where(mask, property.c13)
        .copy_from(&h_c13(ispec, iz, ix), tag_type());
    Kokkos::Experimental::where(mask, property.c15)
        .copy_from(&h_c15(ispec, iz, ix), tag_type());
    Kokkos::Experimental::where(mask, property.c33)
        .copy_from(&h_c33(ispec, iz, ix), tag_type());
    Kokkos::Experimental::where(mask, property.c35)
        .copy_from(&h_c35(ispec, iz, ix), tag_type());
    Kokkos::Experimental::where(mask, property.c55)
        .copy_from(&h_c55(ispec, iz, ix), tag_type());
    Kokkos::Experimental::where(mask, property.c23)
        .copy_from(&h_c23(ispec, iz, ix), tag_type());
    Kokkos::Experimental::where(mask, property.c25)
        .copy_from(&h_c25(ispec, iz, ix), tag_type());
  }

  void copy_to_device() {
    Kokkos::deep_copy(rho, h_rho);
    Kokkos::deep_copy(c11, h_c11);
    Kokkos::deep_copy(c13, h_c13);
    Kokkos::deep_copy(c15, h_c15);
    Kokkos::deep_copy(c33, h_c33);
    Kokkos::deep_copy(c35, h_c35);
    Kokkos::deep_copy(c55, h_c55);
    Kokkos::deep_copy(c23, h_c23);
    Kokkos::deep_copy(c25, h_c25);
  }

  void copy_to_host() {
    Kokkos::deep_copy(h_rho, rho);
    Kokkos::deep_copy(h_c11, c11);
    Kokkos::deep_copy(h_c13, c13);
    Kokkos::deep_copy(h_c15, c15);
    Kokkos::deep_copy(h_c33, c33);
    Kokkos::deep_copy(h_c35, c35);
    Kokkos::deep_copy(h_c55, c55);
    Kokkos::deep_copy(h_c23, c23);
    Kokkos::deep_copy(h_c25, c25);
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
    h_c11(ispec, iz, ix) = property.c11;
    h_c13(ispec, iz, ix) = property.c13;
    h_c15(ispec, iz, ix) = property.c15;
    h_c33(ispec, iz, ix) = property.c33;
    h_c35(ispec, iz, ix) = property.c35;
    h_c55(ispec, iz, ix) = property.c55;
    h_c23(ispec, iz, ix) = property.c23;
    h_c25(ispec, iz, ix) = property.c25;
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
    Kokkos::Experimental::where(mask, property.c11)
        .copy_to(&h_c11(ispec, iz, ix), tag_type());
    Kokkos::Experimental::where(mask, property.c13)
        .copy_to(&h_c13(ispec, iz, ix), tag_type());
    Kokkos::Experimental::where(mask, property.c15)
        .copy_to(&h_c15(ispec, iz, ix), tag_type());
    Kokkos::Experimental::where(mask, property.c33)
        .copy_to(&h_c33(ispec, iz, ix), tag_type());
    Kokkos::Experimental::where(mask, property.c35)
        .copy_to(&h_c35(ispec, iz, ix), tag_type());
    Kokkos::Experimental::where(mask, property.c55)
        .copy_to(&h_c55(ispec, iz, ix), tag_type());
    Kokkos::Experimental::where(mask, property.c23)
        .copy_to(&h_c23(ispec, iz, ix), tag_type());
    Kokkos::Experimental::where(mask, property.c25)
        .copy_to(&h_c25(ispec, iz, ix), tag_type());
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

    property.rho_inverse = rho_inverse(ispec, iz, ix);
    property.lambdaplus2mu_inverse = lambdaplus2mu_inverse(ispec, iz, ix);
    property.kappa = kappa(ispec, iz, ix);
    property.rho_vpinverse =
        sqrt(property.rho_inverse * property.lambdaplus2mu_inverse);
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

    mask_type mask([&, this](std::size_t lane) { return index.mask(lane); });

    Kokkos::Experimental::where(mask, property.rho_inverse)
        .copy_from(&rho_inverse(ispec, iz, ix), tag_type());
    Kokkos::Experimental::where(mask, property.lambdaplus2mu_inverse)
        .copy_from(&lambdaplus2mu_inverse(ispec, iz, ix), tag_type());
    Kokkos::Experimental::where(mask, property.kappa)
        .copy_from(&kappa(ispec, iz, ix), tag_type());

    property.rho_vpinverse =
        Kokkos::sqrt(property.rho_inverse * property.lambdaplus2mu_inverse);
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

    property.rho_inverse = h_rho_inverse(ispec, iz, ix);
    property.lambdaplus2mu_inverse = h_lambdaplus2mu_inverse(ispec, iz, ix);
    property.kappa = h_kappa(ispec, iz, ix);
    property.rho_vpinverse =
        sqrt(property.rho_inverse * property.lambdaplus2mu_inverse);
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

    mask_type mask([&, this](std::size_t lane) { return index.mask(lane); });

    Kokkos::Experimental::where(mask, property.rho_inverse)
        .copy_from(&h_rho_inverse(ispec, iz, ix), tag_type());
    Kokkos::Experimental::where(mask, property.lambdaplus2mu_inverse)
        .copy_from(&h_lambdaplus2mu_inverse(ispec, iz, ix), tag_type());
    Kokkos::Experimental::where(mask, property.kappa)
        .copy_from(&h_kappa(ispec, iz, ix), tag_type());

    property.rho_vpinverse =
        Kokkos::sqrt(property.rho_inverse * property.lambdaplus2mu_inverse);
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

    h_rho_inverse(ispec, iz, ix) = property.rho_inverse;
    h_lambdaplus2mu_inverse(ispec, iz, ix) = property.lambdaplus2mu_inverse;
    h_kappa(ispec, iz, ix) = property.kappa;
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

    Kokkos::Experimental::where(mask, property.rho_inverse)
        .copy_to(&h_rho_inverse(ispec, iz, ix), tag_type());
    Kokkos::Experimental::where(mask, property.lambdaplus2mu_inverse)
        .copy_to(&h_lambdaplus2mu_inverse(ispec, iz, ix), tag_type());
    Kokkos::Experimental::where(mask, property.kappa)
        .copy_to(&h_kappa(ispec, iz, ix), tag_type());
  }
};

} // namespace properties

} // namespace impl
} // namespace compute
} // namespace specfem

#endif /* _COMPUTE_PROPERTIES_IMPL_HPP */
