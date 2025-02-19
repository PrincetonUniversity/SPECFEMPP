#pragma once

#include "enumerations/medium.hpp"
#include "kokkos_abstractions.h"
#include "point/coordinates.hpp"
#include "point/kernels.hpp"
#include <Kokkos_Core.hpp>

namespace specfem {
namespace medium {

template <>
class kernels_container<specfem::element::medium_tag::elastic_sv,
                        specfem::element::property_tag::isotropic> {
public:
  constexpr static auto value_type = specfem::element::medium_tag::elastic_sv;
  constexpr static auto property_type =
      specfem::element::property_tag::isotropic;
  int nspec;
  int ngllz;
  int ngllx;

  using ViewType = Kokkos::View<type_real ***, Kokkos::LayoutLeft,
                                Kokkos::DefaultExecutionSpace>;

  ViewType rho;
  ViewType::HostMirror h_rho;
  ViewType mu;
  ViewType::HostMirror h_mu;
  ViewType kappa;
  ViewType::HostMirror h_kappa;
  ViewType rhop;
  ViewType::HostMirror h_rhop;
  ViewType alpha;
  ViewType::HostMirror h_alpha;
  ViewType beta;
  ViewType::HostMirror h_beta;

  kernels_container() = default;

  kernels_container(const int nspec, const int ngllz, const int ngllx)
      : nspec(nspec), ngllz(ngllz), ngllx(ngllx),
        rho("specfem::medium::elastic::rho", nspec, ngllz, ngllx),
        mu("specfem::medium::elastic::mu", nspec, ngllz, ngllx),
        kappa("specfem::medium::elastic::kappa", nspec, ngllz, ngllx),
        rhop("specfem::medium::elastic::rhop", nspec, ngllz, ngllx),
        alpha("specfem::medium::elastic::alpha", nspec, ngllz, ngllx),
        beta("specfem::medium::elastic::beta", nspec, ngllz, ngllx),
        h_rho(Kokkos::create_mirror_view(rho)),
        h_mu(Kokkos::create_mirror_view(mu)),
        h_kappa(Kokkos::create_mirror_view(kappa)),
        h_rhop(Kokkos::create_mirror_view(rhop)),
        h_alpha(Kokkos::create_mirror_view(alpha)),
        h_beta(Kokkos::create_mirror_view(beta)) {

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
    kernels.mu = mu(ispec, iz, ix);
    kernels.kappa = kappa(ispec, iz, ix);
    kernels.rhop = rhop(ispec, iz, ix);
    kernels.alpha = alpha(ispec, iz, ix);
    kernels.beta = beta(ispec, iz, ix);
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
    Kokkos::Experimental::where(mask, kernels.mu)
        .copy_from(&mu(ispec, iz, ix), tag_type());
    Kokkos::Experimental::where(mask, kernels.kappa)
        .copy_from(&kappa(ispec, iz, ix), tag_type());
    Kokkos::Experimental::where(mask, kernels.rhop)
        .copy_from(&rhop(ispec, iz, ix), tag_type());
    Kokkos::Experimental::where(mask, kernels.alpha)
        .copy_from(&alpha(ispec, iz, ix), tag_type());
    Kokkos::Experimental::where(mask, kernels.beta)
        .copy_from(&beta(ispec, iz, ix), tag_type());
  }

  template <
      typename PointKernelType,
      typename std::enable_if_t<!PointKernelType::simd::using_simd, int> = 0>
  void
  load_host_kernels(specfem::point::index<PointKernelType::dimension> &index,
                    PointKernelType &kernels) const {

    static_assert(PointKernelType::medium_tag == value_type);
    static_assert(PointKernelType::property_tag == property_type);

    const int ispec = index.ispec;
    const int iz = index.iz;
    const int ix = index.ix;

    kernels.rho = h_rho(ispec, iz, ix);
    kernels.mu = h_mu(ispec, iz, ix);
    kernels.kappa = h_kappa(ispec, iz, ix);
    kernels.rhop = h_rhop(ispec, iz, ix);
    kernels.alpha = h_alpha(ispec, iz, ix);
    kernels.beta = h_beta(ispec, iz, ix);
  }

  template <
      typename PointKernelType,
      typename std::enable_if_t<PointKernelType::simd::using_simd, int> = 0>
  void load_host_kernels(
      specfem::point::simd_index<PointKernelType::dimension> &index,
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
    Kokkos::Experimental::where(mask, kernels.mu)
        .copy_from(&h_mu(ispec, iz, ix), tag_type());
    Kokkos::Experimental::where(mask, kernels.kappa)
        .copy_from(&h_kappa(ispec, iz, ix), tag_type());
    Kokkos::Experimental::where(mask, kernels.rhop)
        .copy_from(&h_rhop(ispec, iz, ix), tag_type());
    Kokkos::Experimental::where(mask, kernels.alpha)
        .copy_from(&h_alpha(ispec, iz, ix), tag_type());
    Kokkos::Experimental::where(mask, kernels.beta)
        .copy_from(&h_beta(ispec, iz, ix), tag_type());
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
    mu(ispec, iz, ix) = kernels.mu;
    kappa(ispec, iz, ix) = kernels.kappa;
    rhop(ispec, iz, ix) = kernels.rhop;
    alpha(ispec, iz, ix) = kernels.alpha;
    beta(ispec, iz, ix) = kernels.beta;
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
    Kokkos::Experimental::where(mask, kernels.mu)
        .copy_to(&mu(ispec, iz, ix), tag_type());
    Kokkos::Experimental::where(mask, kernels.kappa)
        .copy_to(&kappa(ispec, iz, ix), tag_type());
    Kokkos::Experimental::where(mask, kernels.rhop)
        .copy_to(&rhop(ispec, iz, ix), tag_type());
    Kokkos::Experimental::where(mask, kernels.alpha)
        .copy_to(&alpha(ispec, iz, ix), tag_type());
    Kokkos::Experimental::where(mask, kernels.beta)
        .copy_to(&beta(ispec, iz, ix), tag_type());
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
    h_mu(ispec, iz, ix) = kernels.mu;
    h_kappa(ispec, iz, ix) = kernels.kappa;
    h_rhop(ispec, iz, ix) = kernels.rhop;
    h_alpha(ispec, iz, ix) = kernels.alpha;
    h_beta(ispec, iz, ix) = kernels.beta;
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
    Kokkos::Experimental::where(mask, kernels.mu)
        .copy_to(&h_mu(ispec, iz, ix), tag_type());
    Kokkos::Experimental::where(mask, kernels.kappa)
        .copy_to(&h_kappa(ispec, iz, ix), tag_type());
    Kokkos::Experimental::where(mask, kernels.rhop)
        .copy_to(&h_rhop(ispec, iz, ix), tag_type());
    Kokkos::Experimental::where(mask, kernels.alpha)
        .copy_to(&h_alpha(ispec, iz, ix), tag_type());
    Kokkos::Experimental::where(mask, kernels.beta)
        .copy_to(&h_beta(ispec, iz, ix), tag_type());
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
    mu(ispec, iz, ix) += kernels.mu;
    kappa(ispec, iz, ix) += kernels.kappa;
    rhop(ispec, iz, ix) += kernels.rhop;
    alpha(ispec, iz, ix) += kernels.alpha;
    beta(ispec, iz, ix) += kernels.beta;
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

    Kokkos::Experimental::where(mask, lhs).copy_from(&mu(ispec, iz, ix),
                                                     tag_type());
    lhs += kernels.mu;
    Kokkos::Experimental::where(mask, lhs).copy_to(&mu(ispec, iz, ix),
                                                   tag_type());

    Kokkos::Experimental::where(mask, lhs).copy_from(&kappa(ispec, iz, ix),
                                                     tag_type());
    lhs += kernels.kappa;
    Kokkos::Experimental::where(mask, lhs).copy_to(&kappa(ispec, iz, ix),
                                                   tag_type());

    Kokkos::Experimental::where(mask, lhs).copy_from(&rhop(ispec, iz, ix),
                                                     tag_type());
    lhs += kernels.rhop;
    Kokkos::Experimental::where(mask, lhs).copy_to(&rhop(ispec, iz, ix),
                                                   tag_type());

    Kokkos::Experimental::where(mask, lhs).copy_from(&alpha(ispec, iz, ix),
                                                     tag_type());
    lhs += kernels.alpha;
    Kokkos::Experimental::where(mask, lhs).copy_to(&alpha(ispec, iz, ix),
                                                   tag_type());

    Kokkos::Experimental::where(mask, lhs).copy_from(&beta(ispec, iz, ix),
                                                     tag_type());
    lhs += kernels.beta;
    Kokkos::Experimental::where(mask, lhs).copy_to(&beta(ispec, iz, ix),
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
    h_mu(ispec, iz, ix) += kernels.mu;
    h_kappa(ispec, iz, ix) += kernels.kappa;
    h_rhop(ispec, iz, ix) += kernels.rhop;
    h_alpha(ispec, iz, ix) += kernels.alpha;
    h_beta(ispec, iz, ix) += kernels.beta;
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

    Kokkos::Experimental::where(mask, lhs).copy_from(&h_mu(ispec, iz, ix),
                                                     tag_type());
    lhs += kernels.mu;
    Kokkos::Experimental::where(mask, lhs).copy_to(&h_mu(ispec, iz, ix),
                                                   tag_type());

    Kokkos::Experimental::where(mask, lhs).copy_from(&h_kappa(ispec, iz, ix),
                                                     tag_type());
    lhs += kernels.kappa;
    Kokkos::Experimental::where(mask, lhs).copy_to(&h_kappa(ispec, iz, ix),
                                                   tag_type());

    Kokkos::Experimental::where(mask, lhs).copy_from(&h_rhop(ispec, iz, ix),
                                                     tag_type());
    lhs += kernels.rhop;
    Kokkos::Experimental::where(mask, lhs).copy_to(&h_rhop(ispec, iz, ix),
                                                   tag_type());

    Kokkos::Experimental::where(mask, lhs).copy_from(&h_alpha(ispec, iz, ix),
                                                     tag_type());
    lhs += kernels.alpha;
    Kokkos::Experimental::where(mask, lhs).copy_to(&h_alpha(ispec, iz, ix),
                                                   tag_type());

    Kokkos::Experimental::where(mask, lhs).copy_from(&h_beta(ispec, iz, ix),
                                                     tag_type());
    lhs += kernels.beta;
    Kokkos::Experimental::where(mask, lhs).copy_to(&h_beta(ispec, iz, ix),
                                                   tag_type());
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

  void initialize() {
    Kokkos::parallel_for(
        "specfem::medium::elastic::isotropic::initialize",
        Kokkos::MDRangePolicy<Kokkos::Rank<3> >({ 0, 0, 0 },
                                                { nspec, ngllz, ngllx }),
        KOKKOS_CLASS_LAMBDA(const int ispec, const int iz, const int ix) {
          this->rho(ispec, iz, ix) = 0.0;
          this->mu(ispec, iz, ix) = 0.0;
          this->kappa(ispec, iz, ix) = 0.0;
          this->rhop(ispec, iz, ix) = 0.0;
          this->alpha(ispec, iz, ix) = 0.0;
          this->beta(ispec, iz, ix) = 0.0;
        });
  }
};

} // namespace medium
} // namespace specfem
