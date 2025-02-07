#pragma once

#include "medium/properties_container.hpp"
#include "point/interface.hpp"
#include <Kokkos_SIMD.hpp>

namespace specfem {
namespace medium {

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
  ViewType kappa;
  ViewType::HostMirror h_kappa;

  properties_container() = default;

  properties_container(const int nspec, const int ngllz, const int ngllx)
      : nspec(nspec), ngllz(ngllz), ngllx(ngllx),
        rho_inverse("specfem::compute::properties::rho_inverse", nspec, ngllz,
                    ngllx),
        h_rho_inverse(Kokkos::create_mirror_view(rho_inverse)),
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
    property.kappa = kappa(ispec, iz, ix);
    property.kappa_inverse = static_cast<type_real>(1.0) / property.kappa;
    property.rho_vpinverse =
        sqrt(property.rho_inverse * property.kappa_inverse);
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
    Kokkos::Experimental::where(mask, property.kappa)
        .copy_from(&kappa(ispec, iz, ix), tag_type());

    property.kappa_inverse = static_cast<type_real>(1.0) / property.kappa;
    property.rho_vpinverse =
        Kokkos::sqrt(property.rho_inverse * property.kappa_inverse);
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
    property.kappa = h_kappa(ispec, iz, ix);
    property.kappa_inverse = static_cast<type_real>(1.0) / property.kappa;
    property.rho_vpinverse =
        sqrt(property.rho_inverse * property.kappa_inverse);
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
    Kokkos::Experimental::where(mask, property.kappa)
        .copy_from(&h_kappa(ispec, iz, ix), tag_type());

    property.kappa_inverse = static_cast<type_real>(1.0) / property.kappa;
    property.rho_vpinverse =
        Kokkos::sqrt(property.rho_inverse * property.kappa_inverse);
  }

  void copy_to_device() {
    Kokkos::deep_copy(rho_inverse, h_rho_inverse);
    Kokkos::deep_copy(kappa, h_kappa);
  }

  void copy_to_host() {
    Kokkos::deep_copy(h_rho_inverse, rho_inverse);
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
    Kokkos::Experimental::where(mask, property.kappa)
        .copy_to(&h_kappa(ispec, iz, ix), tag_type());
  }
};

} // namespace medium
} // namespace specfem
