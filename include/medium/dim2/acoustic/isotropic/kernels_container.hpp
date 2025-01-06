#pragma once

#include "enumerations/medium.hpp"
#include "kokkos_abstractions.h"
#include "point/coordinates.hpp"
#include "point/kernels.hpp"
#include <Kokkos_Core.hpp>

namespace specfem {
namespace medium {

template <>
class kernels_container<specfem::element::medium_tag::acoustic,
                        specfem::element::property_tag::isotropic> {
public:
  constexpr static auto value_type = specfem::element::medium_tag::acoustic;
  constexpr static auto property_type =
      specfem::element::property_tag::isotropic;
  int nspec;
  int ngllz;
  int ngllx;

  using ViewType = Kokkos::View<type_real ***, Kokkos::LayoutLeft,
                                Kokkos::DefaultExecutionSpace>;
  ViewType rho;
  ViewType::HostMirror h_rho;
  ViewType kappa;
  ViewType::HostMirror h_kappa;
  ViewType rho_prime;
  ViewType::HostMirror h_rho_prime;
  ViewType alpha;
  ViewType::HostMirror h_alpha;

  kernels_container() = default;

  kernels_container(const int nspec, const int ngllz, const int ngllx)
      : nspec(nspec), ngllz(ngllz), ngllx(ngllx),
        rho("specfem::medium::acoustic::rho", nspec, ngllz, ngllx),
        kappa("specfem::medium::acoustic::kappa", nspec, ngllz, ngllx),
        rho_prime("specfem::medium::acoustic::rho_prime", nspec, ngllz, ngllx),
        alpha("specfem::medium::acoustic::alpha", nspec, ngllz, ngllx),
        h_rho(Kokkos::create_mirror_view(rho)),
        h_kappa(Kokkos::create_mirror_view(kappa)),
        h_rho_prime(Kokkos::create_mirror_view(rho_prime)),
        h_alpha(Kokkos::create_mirror_view(alpha)) {

    initialize();
  }

  template <
      typename PointKernelType,
      typename std::enable_if_t<!PointKernelType::simd::using_simd, int> = 0>
  KOKKOS_INLINE_FUNCTION void load_device_kernels(
      const specfem::point::index<PointKernelType::dimension> &index,
      PointKernelType &kernels) const {

    static_assert(PointKernelType::medium_tag == value_type);
    static_assert(PointKernelType::property_tag == property_type);

    const int ispec = index.ispec;
    const int iz = index.iz;
    const int ix = index.ix;

    kernels.rho = rho(ispec, iz, ix);
    kernels.kappa = kappa(ispec, iz, ix);
    kernels.rhop = rho_prime(ispec, iz, ix);
    kernels.alpha = alpha(ispec, iz, ix);
  }

  template <
      typename PointKernelType,
      typename std::enable_if_t<PointKernelType::simd::using_simd, int> = 0>
  KOKKOS_INLINE_FUNCTION void load_device_kernels(
      const specfem::point::simd_index<PointKernelType::dimension> &index,
      PointKernelType &kernels) const {

    static_assert(PointKernelType::medium_tag == value_type);
    static_assert(PointKernelType::property_tag == property_type);

    using simd_type = typename PointKernelType::simd::datatype;
    using mask_type = typename PointKernelType::simd::mask_type;
    using tag_type = typename PointKernelType::simd::tag_type;

    mask_type mask([&](std::size_t lane) { return index.mask(lane); });

    const int ispec = index.ispec;
    const int iz = index.iz;
    const int ix = index.ix;

    Kokkos::Experimental::where(mask, kernels.rho)
        .copy_from(&rho(ispec, iz, ix), tag_type());
    Kokkos::Experimental::where(mask, kernels.kappa)
        .copy_from(&kappa(ispec, iz, ix), tag_type());
    Kokkos::Experimental::where(mask, kernels.rhop)
        .copy_from(&rho_prime(ispec, iz, ix), tag_type());
    Kokkos::Experimental::where(mask, kernels.alpha)
        .copy_from(&alpha(ispec, iz, ix), tag_type());
  }

  template <
      typename PointKernelType,
      typename std::enable_if_t<!PointKernelType::simd::using_simd, int> = 0>
  void load_host_kernels(
      const specfem::point::index<PointKernelType::dimension> &index,
      PointKernelType &kernels) const {

    static_assert(PointKernelType::medium_tag == value_type);
    static_assert(PointKernelType::property_tag == property_type);

    const int ispec = index.ispec;
    const int iz = index.iz;
    const int ix = index.ix;

    kernels.rho = h_rho(ispec, iz, ix);
    kernels.kappa = h_kappa(ispec, iz, ix);
    kernels.rhop = h_rho_prime(ispec, iz, ix);
    kernels.alpha = h_alpha(ispec, iz, ix);
  }

  template <
      typename PointKernelType,
      typename std::enable_if_t<PointKernelType::simd::using_simd, int> = 0>
  void load_host_kernels(
      const specfem::point::simd_index<PointKernelType::dimension> &index,
      PointKernelType &kernels) const {

    static_assert(PointKernelType::medium_tag == value_type);
    static_assert(PointKernelType::property_tag == property_type);

    using simd_type = typename PointKernelType::simd::datatype;
    using mask_type = typename PointKernelType::simd::mask_type;
    using tag_type = typename PointKernelType::simd::tag_type;

    mask_type mask([&](std::size_t lane) { return index.mask(lane); });

    const int ispec = index.ispec;
    const int iz = index.iz;
    const int ix = index.ix;

    Kokkos::Experimental::where(mask, kernels.rho)
        .copy_from(&h_rho(ispec, iz, ix), tag_type());
    Kokkos::Experimental::where(mask, kernels.kappa)
        .copy_from(&h_kappa(ispec, iz, ix), tag_type());
    Kokkos::Experimental::where(mask, kernels.rhop)
        .copy_from(&h_rho_prime(ispec, iz, ix), tag_type());
    Kokkos::Experimental::where(mask, kernels.alpha)
        .copy_from(&h_alpha(ispec, iz, ix), tag_type());
  }

  template <
      typename PointKernelType,
      typename std::enable_if_t<!PointKernelType::simd::using_simd, int> = 0>
  KOKKOS_INLINE_FUNCTION void update_kernels_on_device(
      const specfem::point::index<PointKernelType::dimension> &index,
      const PointKernelType &kernels) const {

    static_assert(PointKernelType::medium_tag == value_type);
    static_assert(PointKernelType::property_tag == property_type);

    const int ispec = index.ispec;
    const int iz = index.iz;
    const int ix = index.ix;

    rho(ispec, iz, ix) = kernels.rho;
    kappa(ispec, iz, ix) = kernels.kappa;
    rho_prime(ispec, iz, ix) = kernels.rhop;
    alpha(ispec, iz, ix) = kernels.alpha;
  }

  template <
      typename PointKernelType,
      typename std::enable_if_t<PointKernelType::simd::using_simd, int> = 0>
  KOKKOS_INLINE_FUNCTION void update_kernels_on_device(
      const specfem::point::simd_index<PointKernelType::dimension> &index,
      const PointKernelType &kernels) const {

    static_assert(PointKernelType::medium_tag == value_type);
    static_assert(PointKernelType::property_tag == property_type);

    using simd_type = typename PointKernelType::simd::datatype;
    using mask_type = typename PointKernelType::simd::mask_type;
    using tag_type = typename PointKernelType::simd::tag_type;

    mask_type mask([&](std::size_t lane) { return index.mask(lane); });

    const int ispec = index.ispec;
    const int iz = index.iz;
    const int ix = index.ix;

    Kokkos::Experimental::where(mask, kernels.rho)
        .copy_to(&rho(ispec, iz, ix), tag_type());
    Kokkos::Experimental::where(mask, kernels.kappa)
        .copy_to(&kappa(ispec, iz, ix), tag_type());
    Kokkos::Experimental::where(mask, kernels.rhop)
        .copy_to(&rho_prime(ispec, iz, ix), tag_type());
    Kokkos::Experimental::where(mask, kernels.alpha)
        .copy_to(&alpha(ispec, iz, ix), tag_type());
  }

  template <
      typename PointKernelType,
      typename std::enable_if_t<!PointKernelType::simd::using_simd, int> = 0>
  void update_kernels_on_host(
      const specfem::point::index<PointKernelType::dimension> &index,
      const PointKernelType &kernels) const {

    static_assert(PointKernelType::medium_tag == value_type);
    static_assert(PointKernelType::property_tag == property_type);

    const int ispec = index.ispec;
    const int iz = index.iz;
    const int ix = index.ix;

    h_rho(ispec, iz, ix) = kernels.rho;
    h_kappa(ispec, iz, ix) = kernels.kappa;
    h_rho_prime(ispec, iz, ix) = kernels.rhop;
    h_alpha(ispec, iz, ix) = kernels.alpha;
  }

  template <
      typename PointKernelType,
      typename std::enable_if_t<PointKernelType::simd::using_simd, int> = 0>
  void update_kernels_on_host(
      const specfem::point::simd_index<PointKernelType::dimension> &index,
      const PointKernelType &kernels) const {

    static_assert(PointKernelType::medium_tag == value_type);
    static_assert(PointKernelType::property_tag == property_type);

    using simd_type = typename PointKernelType::simd::datatype;
    using mask_type = typename PointKernelType::simd::mask_type;
    using tag_type = typename PointKernelType::simd::tag_type;

    mask_type mask([&](std::size_t lane) { return index.mask(lane); });

    const int ispec = index.ispec;
    const int iz = index.iz;
    const int ix = index.ix;

    Kokkos::Experimental::where(mask, kernels.rho)
        .copy_to(&h_rho(ispec, iz, ix), tag_type());
    Kokkos::Experimental::where(mask, kernels.kappa)
        .copy_to(&h_kappa(ispec, iz, ix), tag_type());
    Kokkos::Experimental::where(mask, kernels.rhop)
        .copy_to(&h_rho_prime(ispec, iz, ix), tag_type());
    Kokkos::Experimental::where(mask, kernels.alpha)
        .copy_to(&h_alpha(ispec, iz, ix), tag_type());
  }

  template <
      typename PointKernelType,
      typename std::enable_if_t<!PointKernelType::simd::using_simd, int> = 0>
  KOKKOS_INLINE_FUNCTION void add_kernels_on_device(
      const specfem::point::index<PointKernelType::dimension> &index,
      const PointKernelType &kernels) const {

    static_assert(PointKernelType::medium_tag == value_type);
    static_assert(PointKernelType::property_tag == property_type);

    const int ispec = index.ispec;
    const int iz = index.iz;
    const int ix = index.ix;

    rho(ispec, iz, ix) += kernels.rho;
    kappa(ispec, iz, ix) += kernels.kappa;
    rho_prime(ispec, iz, ix) += kernels.rhop;
    alpha(ispec, iz, ix) += kernels.alpha;
  }

  template <
      typename PointKernelType,
      typename std::enable_if_t<PointKernelType::simd::using_simd, int> = 0>
  KOKKOS_INLINE_FUNCTION void add_kernels_on_device(
      const specfem::point::simd_index<PointKernelType::dimension> &index,
      const PointKernelType &kernels) const {

    static_assert(PointKernelType::medium_tag == value_type);
    static_assert(PointKernelType::property_tag == property_type);

    using simd_type = typename PointKernelType::simd::datatype;
    using mask_type = typename PointKernelType::simd::mask_type;
    using tag_type = typename PointKernelType::simd::tag_type;

    mask_type mask([&](std::size_t lane) { return index.mask(lane); });

    const int ispec = index.ispec;
    const int iz = index.iz;
    const int ix = index.ix;

    simd_type lhs;

    Kokkos::Experimental::where(mask, lhs).copy_from(&rho(ispec, iz, ix),
                                                     tag_type());
    lhs += kernels.rho;
    Kokkos::Experimental::where(mask, lhs).copy_to(&rho(ispec, iz, ix),
                                                   tag_type());

    Kokkos::Experimental::where(mask, lhs).copy_from(&kappa(ispec, iz, ix),
                                                     tag_type());
    lhs += kernels.kappa;
    Kokkos::Experimental::where(mask, lhs).copy_to(&kappa(ispec, iz, ix),
                                                   tag_type());

    Kokkos::Experimental::where(mask, lhs).copy_from(&rho_prime(ispec, iz, ix),
                                                     tag_type());
    lhs += kernels.rhop;
    Kokkos::Experimental::where(mask, lhs).copy_to(&rho_prime(ispec, iz, ix),
                                                   tag_type());

    Kokkos::Experimental::where(mask, lhs).copy_from(&alpha(ispec, iz, ix),
                                                     tag_type());
    lhs += kernels.alpha;
    Kokkos::Experimental::where(mask, lhs).copy_to(&alpha(ispec, iz, ix),
                                                   tag_type());
  }

  template <
      typename PointKernelType,
      typename std::enable_if_t<!PointKernelType::simd::using_simd, int> = 0>
  void add_kernels_on_host(
      const specfem::point::index<PointKernelType::dimension> &index,
      const PointKernelType &kernels) const {

    static_assert(PointKernelType::medium_tag == value_type);
    static_assert(PointKernelType::property_tag == property_type);

    const int ispec = index.ispec;
    const int iz = index.iz;
    const int ix = index.ix;

    h_rho(ispec, iz, ix) += kernels.rho;
    h_kappa(ispec, iz, ix) += kernels.kappa;
    h_rho_prime(ispec, iz, ix) += kernels.rhop;
    h_alpha(ispec, iz, ix) += kernels.alpha;
  }

  template <
      typename PointKernelType,
      typename std::enable_if_t<PointKernelType::simd::using_simd, int> = 0>
  void add_kernels_on_host(
      const specfem::point::simd_index<PointKernelType::dimension> &index,
      const PointKernelType &kernels) const {

    static_assert(PointKernelType::medium_tag == value_type);
    static_assert(PointKernelType::property_tag == property_type);

    using simd_type = typename PointKernelType::simd::datatype;
    using mask_type = typename PointKernelType::simd::mask_type;
    using tag_type = typename PointKernelType::simd::tag_type;

    mask_type mask([&](std::size_t lane) { return index.mask(lane); });

    const int ispec = index.ispec;
    const int iz = index.iz;
    const int ix = index.ix;

    simd_type lhs;

    Kokkos::Experimental::where(mask, lhs).copy_from(&h_rho(ispec, iz, ix),
                                                     tag_type());
    lhs += kernels.rho;
    Kokkos::Experimental::where(mask, lhs).copy_to(&h_rho(ispec, iz, ix),
                                                   tag_type());

    Kokkos::Experimental::where(mask, lhs).copy_from(&h_kappa(ispec, iz, ix),
                                                     tag_type());
    lhs += kernels.kappa;
    Kokkos::Experimental::where(mask, lhs).copy_to(&h_kappa(ispec, iz, ix),
                                                   tag_type());

    Kokkos::Experimental::where(mask, lhs).copy_from(
        &h_rho_prime(ispec, iz, ix), tag_type());
    lhs += kernels.rhop;
    Kokkos::Experimental::where(mask, lhs).copy_to(&h_rho_prime(ispec, iz, ix),
                                                   tag_type());

    Kokkos::Experimental::where(mask, lhs).copy_from(&h_alpha(ispec, iz, ix),
                                                     tag_type());
    lhs += kernels.alpha;
    Kokkos::Experimental::where(mask, lhs).copy_to(&h_alpha(ispec, iz, ix),
                                                   tag_type());
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

  void initialize() {
    Kokkos::parallel_for(
        "specfem::medium::acoustic::initialize",
        Kokkos::MDRangePolicy<Kokkos::Rank<3> >({ 0, 0, 0 },
                                                { nspec, ngllz, ngllx }),
        KOKKOS_CLASS_LAMBDA(const int ispec, const int iz, const int ix) {
          this->rho(ispec, iz, ix) = 0.0;
          this->kappa(ispec, iz, ix) = 0.0;
          this->rho_prime(ispec, iz, ix) = 0.0;
          this->alpha(ispec, iz, ix) = 0.0;
        });
  }
};

} // namespace medium
} // namespace specfem
